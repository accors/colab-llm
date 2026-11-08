#!/usr/bin/env python3
"""Tool-calling benchmark for the colab-llm endpoint (stdlib only).

Runs a fixed set of agent-style requests (tool choice, argument extraction,
answering from a tool result, recovering from a tool error, not calling a tool
when none is needed) plus multi-turn agent loops (the bench plays the tools and
keeps reasoning in history, up to 8 turns), and records pass/fail, latency,
completion tokens, reasoning length, leaked tags and truncation.

  python3 bench/tool_bench.py --label embedded -n 2      # run, save, summarize
  python3 bench/tool_bench.py --compare embedded froggeric

Results go to ~/.local/state/colab-llm/bench-<label>.json. Connection settings
come from the same places as chat.py (env, .env, ~/.config/colab-llm/client.env).

To A/B two chat templates: run with the current config, change CHAT_TEMPLATE_*
in config.env, `colab-llm restart`, run again with another label, compare.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chat import load_env  # noqa: E402

STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "colab-llm"


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


WEATHER = fn("get_weather", "查询城市当前天气", {"city": {"type": "string", "description": "城市名"}}, ["city"])
CALC = fn("calculate", "计算数学表达式，返回数值结果", {"expression": {"type": "string"}}, ["expression"])
SEARCH = fn("search_files", "在项目目录中按内容和文件名模式搜索文件",
            {"pattern": {"type": "string", "description": "要搜索的文本"},
             "glob": {"type": "string", "description": "文件名模式，如 *.py"}}, ["pattern"])
ALL = [WEATHER, CALC, SEARCH]


def tool_turn(call_id, name, args, result):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def expect_call(name, check_args=None):
    def check(msg):
        calls = msg.get("tool_calls") or []
        if not calls:
            return False, "no tool call"
        hit = [c for c in calls if c.get("function", {}).get("name") == name]
        if not hit:
            return False, f"called {[c.get('function', {}).get('name') for c in calls]}"
        try:
            args = json.loads(hit[0]["function"].get("arguments") or "{}")
        except ValueError:
            return False, "arguments not JSON"
        if check_args and not check_args(args):
            return False, f"bad args {args}"
        return True, ""
    return check


def expect_answer(*must_contain):
    def check(msg):
        if msg.get("tool_calls"):
            return False, "unexpected tool call"
        text = (msg.get("content") or "").strip()
        if not text:
            return False, "empty content"
        missing = [m for m in must_contain if m not in text]
        return (not missing), (f"missing {missing}" if missing else "")
    return check


def expect_any(*checks):
    def check(msg):
        reasons = []
        for c in checks:
            ok, why = c(msg)
            if ok:
                return True, ""
            reasons.append(why)
        return False, " / ".join(reasons)
    return check


CASES = [
    ("weather", [{"role": "user", "content": "墨尔本现在天气怎么样？"}], ALL,
     expect_call("get_weather", lambda a: "墨尔本" in a.get("city", "") or "Melbourne" in a.get("city", ""))),
    ("calc", [{"role": "user", "content": "帮我算一下 (17.5*4.2)-3^3 等于多少"}], ALL,
     expect_call("calculate", lambda a: "17.5" in a.get("expression", ""))),
    ("pick_tool", [{"role": "user", "content": "在我的项目里找出所有包含 TODO 的 .py 文件"}], ALL,
     expect_call("search_files", lambda a: "TODO" in a.get("pattern", ""))),
    ("two_cities", [{"role": "user", "content": "比较一下北京和上海现在的天气"}], ALL,
     expect_call("get_weather")),
    ("use_result", [{"role": "user", "content": "墨尔本现在多少度？"}]
     + tool_turn("c1", "get_weather", {"city": "墨尔本"}, '{"city":"墨尔本","temp_c":17,"condition":"多云"}'), ALL,
     expect_answer("17")),
    ("tool_error", [{"role": "user", "content": "查一下 Melborne 的天气"}]
     + tool_turn("c2", "get_weather", {"city": "Melborne"}, '{"error":"city not found: Melborne"}'), ALL,
     expect_any(expect_call("get_weather", lambda a: a.get("city", "").lower() in ("melbourne", "墨尔本")), expect_answer())),
    ("no_tool", [{"role": "user", "content": "用一句话解释什么是 KV cache。"}], ALL, expect_answer()),
    ("english", [{"role": "user", "content": "What is 2^20? Use the calculator."}], ALL,
     expect_call("calculate")),
]

LEAKS = ("<think>", "</think>", "<tool_call>", "<function=")


def request(url, key, model, messages, tools, max_tokens, extra):
    body = {"model": model, "messages": messages, "tools": tools, "max_tokens": max_tokens, **extra}
    req = urllib.request.Request(
        f"{url}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    return data, time.time() - t0


# ---------------------------------------------------------------- agent loops
# Multi-turn cases: the bench plays the tools (deterministic fake results) and
# feeds them back, keeping each assistant turn, including reasoning_content,
# in the history, until the model answers or MAX_TURNS is hit. These are the
# situations chat-template fixes target (history with thinking, tool errors,
# long chains); single-shot cases above do not exercise them.

READ = fn("read_file", "读取项目里的文件内容", {"path": {"type": "string"}}, ["path"])
MAX_TURNS = 8

WEATHER_DB = {"墨尔本": (17, "多云"), "melbourne": (17, "多云"), "悉尼": (23, "晴"), "sydney": (23, "晴"),
              "北京": (12, "晴"), "上海": (15, "小雨")}
FILES = {
    "src/app.py": "def main():\n    db = connect()\n    # TODO: 加上数据库连接重试，最多 3 次，间隔指数退避\n    serve(db)\n",
    "src/db.py": "def connect():\n    # TODO: 关闭空闲超过 10 分钟的连接\n    return Pool()\n",
}


def safe_calc(expr):
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
           ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")

    value = ev(ast.parse(expr.replace("^", "**").replace("×", "*").replace("÷", "/"), mode="eval").body)
    return int(value) if float(value).is_integer() else round(value, 6)


class FakeTools:
    def __init__(self, flaky_weather=False):
        self.flaky = flaky_weather
        self.weather_calls = 0

    def __call__(self, name, args):
        if name == "get_weather":
            self.weather_calls += 1
            if self.flaky and self.weather_calls == 1:
                return {"error": "weather service temporarily unavailable, please retry"}
            city = str(args.get("city", "")).strip()
            hit = WEATHER_DB.get(city) or WEATHER_DB.get(city.lower())
            return {"city": city, "temp_c": hit[0], "condition": hit[1]} if hit else {"error": f"city not found: {city}"}
        if name == "calculate":
            try:
                return {"expression": args.get("expression"), "result": safe_calc(str(args.get("expression", "")))}
            except Exception as e:  # noqa: BLE001
                return {"error": f"cannot evaluate: {e}"}
        if name == "search_files":
            if "TODO" in str(args.get("pattern", "")):
                return {"matches": ["src/app.py:3: # TODO: 加上数据库连接重试，最多 3 次，间隔指数退避",
                                    "src/db.py:2: # TODO: 关闭空闲超过 10 分钟的连接"]}
            return {"matches": []}
        if name == "read_file":
            path = str(args.get("path", "")).lstrip("./")
            return {"path": path, "content": FILES[path]} if path in FILES else {"error": f"no such file: {path}"}
        return {"error": f"unknown tool {name}"}


def final_has(*needles):
    def check(text, calls):
        missing = [n for n in needles if n not in text]
        return (not missing), (f"final answer missing {missing}" if missing else "")
    return check


AGENT_CASES = [
    ("loop_compare", "帮我查一下墨尔本和悉尼现在的天气，再用计算器算一下两地温差，最后告诉我哪边更暖和、暖和几度。",
     [WEATHER, CALC], lambda: FakeTools(), final_has("悉尼", "6")),
    ("loop_code", "找出项目里所有的 TODO，打开第一个文件看看那个 TODO 的上下文，然后告诉我具体要做什么。",
     [SEARCH, READ, CALC], lambda: FakeTools(), final_has("重试")),
    ("loop_retry", "墨尔本现在多少度？", [WEATHER], lambda: FakeTools(flaky_weather=True), final_has("17")),
    ("loop_chain", "依次计算：先算 12*7，再把结果加 58，再把结果除以 2。每一步都必须调用计算器，最后告诉我结果。",
     [CALC], lambda: FakeTools(), final_has("71")),
]


def run_agent_case(url, key, model, case, max_tokens, extra):
    name, prompt, tools, make_tools, check = case
    execute = make_tools()
    messages = [{"role": "user", "content": prompt}]
    seen, tokens, reasoning, secs, n_calls, truncated = {}, 0, 0, 0.0, 0, False
    for turn in range(1, MAX_TURNS + 1):
        data, dt = request(url, key, model, messages, tools, max_tokens, extra)
        secs += dt
        choice = data["choices"][0]
        msg = choice["message"]
        tokens += (data.get("usage") or {}).get("completion_tokens") or 0
        reasoning += len(msg.get("reasoning_content") or "")
        truncated |= choice.get("finish_reason") == "length"
        content = msg.get("content") or ""
        stats = {"turns": turn, "completion_tokens": tokens, "reasoning_chars": reasoning,
                 "secs": round(secs, 2), "truncated": truncated, "tool_calls": n_calls}
        leaked = [t for t in LEAKS if t in content]
        if leaked:
            return False, f"turn {turn}: leaked {leaked}", stats
        calls = msg.get("tool_calls") or []
        if not calls:
            if not content.strip():
                return False, f"turn {turn}: empty answer (stalled)", stats
            ok, why = check(content, n_calls)
            return ok, why, stats
        assistant = {"role": "assistant", "content": content, "tool_calls": calls}
        if msg.get("reasoning_content"):
            assistant["reasoning_content"] = msg["reasoning_content"]
        messages.append(assistant)
        for c in calls:
            n_calls += 1
            fname = c.get("function", {}).get("name", "")
            raw = c.get("function", {}).get("arguments") or "{}"
            sig = (fname, raw)
            seen[sig] = seen.get(sig, 0) + 1
            if seen[sig] > 2:
                stats["tool_calls"] = n_calls
                return False, f"turn {turn}: repeated {fname}({raw}) 3 times", stats
            try:
                result = execute(fname, json.loads(raw))
            except ValueError:
                result = {"error": "arguments are not valid JSON"}
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""),
                             "content": json.dumps(result, ensure_ascii=False)})
    stats["tool_calls"] = n_calls
    return False, f"no final answer after {MAX_TURNS} turns", stats


def run(args):
    url, key, model = load_env()
    extra = {}
    if args.temperature is not None:
        extra["temperature"] = args.temperature
    if args.effort:
        extra["chat_template_kwargs"] = {"reasoning_effort": args.effort}
    try:
        models = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"{url}/models", headers={"Authorization": f"Bearer {key}"}), timeout=15).read())
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"endpoint not reachable ({e}); run: colab-llm ensure")
    n_cases = len(CASES) + (0 if args.single_only else len(AGENT_CASES))
    print(f"endpoint {url} model {models['data'][0]['id']} | {n_cases} cases x {args.n}")

    rows = []
    for i in range(args.n):
        for name, messages, tools, check in CASES:
            try:
                data, secs = request(url, key, model, messages, tools, args.max_tokens, extra)
            except (urllib.error.URLError, OSError) as e:
                rows.append({"case": name, "rep": i, "ok": False, "why": f"request failed: {e}"})
                print(f"  {name:<13} rep {i}  FAIL  request failed: {e}")
                continue
            choice = data["choices"][0]
            msg = choice["message"]
            ok, why = check(msg)
            content = msg.get("content") or ""
            leaked = [t for t in LEAKS if t in content]
            if leaked:
                ok, why = False, f"leaked {leaked} into content"
            usage = data.get("usage") or {}
            row = {
                "case": name, "rep": i, "ok": ok, "why": why, "secs": round(secs, 2),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_chars": len(msg.get("reasoning_content") or ""),
                "truncated": choice.get("finish_reason") == "length",
                "tool_calls": len(msg.get("tool_calls") or []),
            }
            rows.append(row)
            print(f"  {name:<13} rep {i}  {'ok  ' if ok else 'FAIL'}  {secs:6.1f}s  "
                  f"{row['completion_tokens'] or 0:5} tok  reasoning {row['reasoning_chars']:5} ch"
                  f"{'  TRUNCATED' if row['truncated'] else ''}{'  ' + why if why else ''}")
        if args.single_only:
            continue
        for case in AGENT_CASES:
            name = case[0]
            try:
                ok, why, st = run_agent_case(url, key, model, case, args.max_tokens, extra)
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                rows.append({"case": name, "rep": i, "ok": False, "why": f"request failed: {e}"})
                print(f"  {name:<13} rep {i}  FAIL  request failed: {e}")
                continue
            rows.append({"case": name, "rep": i, "ok": ok, "why": why, **st})
            print(f"  {name:<13} rep {i}  {'ok  ' if ok else 'FAIL'}  {st['secs']:6.1f}s  "
                  f"{st['completion_tokens']:5} tok  reasoning {st['reasoning_chars']:5} ch  "
                  f"{st['turns']} turns {st['tool_calls']} calls"
                  f"{'  TRUNCATED' if st['truncated'] else ''}{'  ' + why if why else ''}")
    STATE.mkdir(parents=True, exist_ok=True)
    out = STATE / f"bench-{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "time": time.time(), "settings": {
        "n": args.n, "max_tokens": args.max_tokens, **extra}, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"saved {out}")
    summarize([load(args.label)])


def load(label):
    path = STATE / f"bench-{label}.json"
    if not path.exists():
        sys.exit(f"no results for '{label}' ({path})")
    return json.loads(path.read_text())


def stats(rows):
    done = [r for r in rows if "secs" in r]
    tok = [r["completion_tokens"] or 0 for r in done]
    return {
        "pass": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
        "tok_mean": statistics.mean(tok) if tok else 0,
        "tok_max": max(tok) if tok else 0,
        "reason_mean": statistics.mean([r["reasoning_chars"] for r in done]) if done else 0,
        "secs_mean": statistics.mean([r["secs"] for r in done]) if done else 0,
        "truncated": sum(r.get("truncated", False) for r in done),
    }


def summarize(results):
    labels = [r["label"] for r in results]
    print("\n" + f"{'case':<13}" + "".join(f"{lab:>39}" for lab in labels))
    print(f"{'':<13}" + "".join(f"{'pass  tok(mean/max)  reason   sec   ':>39}" for _ in labels))
    present = {x["case"] for r in results for x in r["rows"]}
    names = [c[0] for c in CASES] + [c[0] for c in AGENT_CASES]
    for name in [n for n in names if n in present] + ["ALL"]:
        line = f"{name:<13}"
        for r in results:
            rows = r["rows"] if name == "ALL" else [x for x in r["rows"] if x["case"] == name]
            if not rows:
                line += f"{'-':>39}"
                continue
            s = stats(rows)
            line += f"{s['pass']:>7} {s['tok_mean']:7.0f}/{s['tok_max']:<6} {s['reason_mean']:7.0f} {s['secs_mean']:5.1f}" \
                    + (f" T{s['truncated']}" if s["truncated"] else "   ")
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="name for this run, e.g. embedded / froggeric")
    ap.add_argument("-n", type=int, default=2, help="repetitions per case (default 2)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=None, help="default: server/model default")
    ap.add_argument("--effort", choices=["low", "medium", "xhigh"], help="send chat_template_kwargs.reasoning_effort")
    ap.add_argument("--single-only", action="store_true", help="skip the multi-turn agent-loop cases")
    ap.add_argument("--compare", nargs="+", metavar="LABEL", help="compare saved runs")
    args = ap.parse_args()
    if args.compare:
        summarize([load(lab) for lab in args.compare])
    elif args.label:
        run(args)
    else:
        ap.error("give --label to run, or --compare LABEL LABEL")


if __name__ == "__main__":
    main()

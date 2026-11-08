#!/usr/bin/env python3
"""Tool-calling benchmark for the colab-llm endpoint (stdlib only).

Runs a fixed set of agent-style requests (tool choice, argument extraction,
answering from a tool result, recovering from a tool error, not calling a tool
when none is needed) and records, per request: pass/fail, latency, completion
tokens, reasoning length, leaked tags and truncation.

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
    print(f"endpoint {url} model {models['data'][0]['id']} | {len(CASES)} cases x {args.n}")

    rows = []
    for i in range(args.n):
        for name, messages, tools, check in CASES:
            try:
                data, secs = request(url, key, model, messages, tools, args.max_tokens, extra)
            except (urllib.error.URLError, OSError) as e:
                rows.append({"case": name, "rep": i, "ok": False, "why": f"request failed: {e}"})
                print(f"  {name:<11} rep {i}  FAIL  request failed: {e}")
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
            print(f"  {name:<11} rep {i}  {'ok  ' if ok else 'FAIL'}  {secs:6.1f}s  "
                  f"{row['completion_tokens'] or 0:5} tok  reasoning {row['reasoning_chars']:5} ch"
                  f"{'  TRUNCATED' if row['truncated'] else ''}{'  ' + why if why else ''}")
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
    print("\n" + f"{'case':<11}" + "".join(f"{lab:>39}" for lab in labels))
    print(f"{'':<11}" + "".join(f"{'pass  tok(mean/max)  reason   sec   ':>39}" for _ in labels))
    for name in [c[0] for c in CASES] + ["ALL"]:
        line = f"{name:<11}"
        for r in results:
            rows = r["rows"] if name == "ALL" else [x for x in r["rows"] if x["case"] == name]
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

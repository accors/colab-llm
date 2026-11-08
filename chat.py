#!/usr/bin/env python3
"""调用 colab-llm 启动的 llama-server（OpenAI 兼容接口）。只用标准库。

用法：
  python3 chat.py              # 交互式多轮对话（输入 /clear 清空历史，/exit 退出）
  python3 chat.py "你好"        # 单次提问
  colab-llm chat "你好"         # 同上

连接信息按顺序读取：环境变量 > 同目录 .env > ~/.config/colab-llm/client.env
（client.env 由 `colab-llm init` 生成）：
  COLAB_LLM_URL=http://127.0.0.1:8080/v1
  COLAB_LLM_KEY=...
  COLAB_LLM_MODEL=qwen3.8-flash-next
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def env_files():
    if os.environ.get("COLAB_LLM_ENV"):
        return [Path(os.environ["COLAB_LLM_ENV"])]
    conf = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return [Path(__file__).with_name(".env"), conf / "colab-llm" / "client.env"]


def load_env():
    for env_file in env_files():
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip())
    missing = [k for k in ("COLAB_LLM_URL", "COLAB_LLM_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"缺少 {', '.join(missing)}：先运行 colab-llm init")
    return (os.environ["COLAB_LLM_URL"].rstrip("/"), os.environ["COLAB_LLM_KEY"],
            os.environ.get("COLAB_LLM_MODEL", "qwen3.8-flash-next"))


def stream_chat(url, key, model, messages, max_tokens=4096):
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "stream": True,
                         "max_tokens": max_tokens,
                         "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "colab-llm-chat/1.0"},
    )
    t0, first, n, answer, in_think = time.time(), None, 0, [], False
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                n = chunk["usage"].get("completion_tokens", n)
            for c in chunk.get("choices", []):
                d = c.get("delta", {})
                think, text = d.get("reasoning_content"), d.get("content")
                if (think or text) and first is None:
                    first = time.time()
                if think:
                    if not in_think:
                        print("\033[2m[思考] ", end="")
                        in_think = True
                    print(think, end="", flush=True)
                if text:
                    if in_think:
                        print("\033[0m\n")
                        in_think = False
                    print(text, end="", flush=True)
                    answer.append(text)
    if in_think:
        print("\033[0m")
    dt = time.time() - (first or t0)
    print(f"\n\033[2m--- 首字 {(first or t0) - t0:.2f}s | {n} tokens | {n / dt if dt else 0:.1f} tok/s ---\033[0m")
    return "".join(answer)


def main():
    url, key, model = load_env()
    try:
        if len(sys.argv) > 1:
            stream_chat(url, key, model, [{"role": "user", "content": " ".join(sys.argv[1:])}])
            return
        history = []
        print(f"已连接 {url}（{model}）。/clear 清空历史，/exit 退出。")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in ("/exit", "/quit"):
                break
            if q == "/clear":
                history = []
                print("历史已清空")
                continue
            if not q:
                continue
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": stream_chat(url, key, model, history)})
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:300]}"
                 + ("\n（401 = key 不对：client.env 和 Colab 上的 key 不一致，运行 colab-llm restart）" if e.code == 401 else ""))
    except urllib.error.URLError as e:
        sys.exit(f"连不上 {url}: {e.reason}\n（服务没起来或 ssh 隧道断了：colab-llm status）")
    except OSError as e:
        sys.exit(f"连接 {url} 出错: {e}\n（服务没起来或 ssh 隧道断了：colab-llm status）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""在 Mac 上调用 Colab 里的 llama-server（OpenAI 兼容接口）。只用标准库。

用法：
  python3 chat.py              # 交互式多轮对话（输入 /clear 清空历史，/exit 退出）
  python3 chat.py "你好"        # 单次提问

连接信息从同目录的 .env 读取（Colab 第 5 个 cell 会打印出来）：
  COLAB_LLM_URL=https://xxx.trycloudflare.com/v1
  COLAB_LLM_KEY=...
  COLAB_LLM_MODEL=qwen3.8-27b
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_env():
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip())
    missing = [k for k in ("COLAB_LLM_URL", "COLAB_LLM_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"缺少 {', '.join(missing)}：把 Colab 第 5 个 cell 的输出粘贴进 {env_file}")
    return (os.environ["COLAB_LLM_URL"].rstrip("/"), os.environ["COLAB_LLM_KEY"],
            os.environ.get("COLAB_LLM_MODEL", "qwen3.8-27b"))


def stream_chat(url, key, model, messages, max_tokens=4096):
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "stream": True,
                         "max_tokens": max_tokens,
                         "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
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
                 + ("\n（401 = key 不对；Colab 重新运行第 3 步后 key 会变，要更新 .env）" if e.code == 401 else ""))
    except urllib.error.URLError as e:
        sys.exit(f"连不上 {url}: {e.reason}\n（Colab 运行时断了，或隧道地址变了，重新运行第 5 步并更新 .env）")


if __name__ == "__main__":
    main()

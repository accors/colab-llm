# colab-llm

用 Google Colab 的 GPU 跑 GGUF 大模型（llama.cpp），通过 Cloudflare 隧道得到一个固定的 OpenAI 兼容地址，在本地任意客户端里调用。

```
VS Code ──运行 cell──▶ Colab GPU [llama-server ◀─ cloudflared] ──隧道──▶ Cloudflare (llm.example.com) ◀── 本地客户端
```

## 文件

| 文件 | 在哪运行 | 作用 |
| --- | --- | --- |
| `notebooks/01_download_to_drive.ipynb` | Colab（CPU） | 从 Hugging Face 下载 GGUF 到 Google Drive，只需一次 |
| `scripts/setup_tunnel.sh` | Mac | 创建 Cloudflare 固定隧道，生成 `secrets.json`（给 Colab）和 `.env`（给本地），只需一次 |
| `notebooks/02_serve.ipynb` | Colab（GPU） | 复制模型、编译 llama.cpp、启动 llama-server、连上隧道，每次使用都运行 |
| `chat.py` | Mac | 命令行对话客户端，只用标准库，读取 `.env` |

## 快速开始

```bash
# 1. Colab（CPU 运行时）：运行 notebooks/01_download_to_drive.ipynb

# 2. Mac：配置固定隧道（需要一个托管在 Cloudflare 的域名）
./scripts/setup_tunnel.sh llm.example.com

# 3. Colab（GPU 运行时）：依次运行 notebooks/02_serve.ipynb 的第 1–5 步

# 4. Mac：对话
python3 chat.py
```

其他客户端（Magpie、Cherry Studio、Chatbox、Open WebUI 等）选「OpenAI 兼容」，Base URL 填 `https://llm.example.com/v1`，API Key 用 `.env` 里的 `COLAB_LLM_KEY`，模型名 `qwen3.8-27b`。

没有域名：跳过第 2 步，`02_serve.ipynb` 第 5 步会自动改用临时的 `trycloudflare.com` 地址，并打印出要填进 `.env` 的内容。

用完关机：运行 `02_serve.ipynb` 最后一个 cell 里的 `runtime.unassign()`，否则 GPU 会继续消耗计算单元。

## 换模型

改 `01_download_to_drive.ipynb` 和 `02_serve.ipynb` 里的 `REPO`、`FILE`、`DRIVE_DIR`。显存不够就选小一号的量化（Q6_K、Q4_K_M）。

## 安全

- `.env` 和 `secrets.json` 已在 `.gitignore` 中，不要提交。
- `secrets.json` 里的隧道 token 能控制你的子域名指向哪里，不要分享存放它的 Drive 文件夹。泄露后用 `cloudflared tunnel delete colab-llm` 删除重建。

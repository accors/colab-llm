---
name: colab-llm
description: Start, use, check and stop the self-hosted Qwen3.8-Flash-Next endpoint that runs on a Google Colab A100 80GB via this repo (OpenAI-compatible at http://127.0.0.1:8080/v1). Use when a task needs this local LLM, when asked to bring the Colab model up/down, check its status or cost, or debug it.
---

# colab-llm

This repo runs a GGUF model with llama.cpp on a Colab GPU, driven by the Colab
CLI from this machine. The server listens only on the VM's 127.0.0.1 and is
forwarded here with `ssh -L`. Everything goes through one script:
`bin/colab-llm` (also on PATH as `colab-llm`).

**It costs money.** The A100 80GB burns about **8.88 compute units (CU) per
hour**, idle or not. Colab Pro adds 100 CU/month (≈11 h of A100). Treat
starting it as a spending decision.

## Rules

1. **Do not start the GPU without a reason.** Only run `ensure` / `up` when the
   task actually needs the model, or the user asked. If it is not already
   running and the job is small, ask the user first.
2. **Check the balance first:** `colab-llm usage`. If fewer than ~10 CU are
   left, ask before starting (one start ≈ 0.6–1 CU of setup, then 8.88 CU/h).
3. **Stop it when done:** `colab-llm down`. Do not rely on the idle timer
   (15 min) unless the user wants the endpoint to stay warm.
4. **Never print, log, commit or send secrets:** `~/.config/colab-llm/api_key`
   and `~/.config/colab-llm/hf_token`. Read the key from the file at request
   time; do not paste it into prompts, commits or chat.
5. **Do not edit `config.env` or the remote scripts** unless the user asked.
   After a config change, apply it with `colab-llm restart`.
6. Never run `colab ssh` / `ssh colab-llm-*` by hand against a session that may
   not exist: the Colab CLI then auto-creates a new (billable) runtime. Use the
   `colab-llm` subcommands, which check first.

## Common tasks

Get an endpoint (idempotent; starts only if not serving):

```bash
colab-llm ensure
# prints:
# COLAB_LLM_URL=http://127.0.0.1:8080/v1
# COLAB_LLM_MODEL=qwen3.8-flash-next
# COLAB_LLM_KEY_FILE=/home/<user>/.config/colab-llm/api_key
```

A cold start took 5 min 39 s when measured (2026-11): create the A100 session, download
83.6 GB from Hugging Face (285 s), then load the model (23 s). Budget
up to 10 min. `ensure` blocks until the endpoint is healthy; give it a timeout of at
least 30 minutes.

Call it (OpenAI-compatible):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/.config/colab-llm/api_key)" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}]}'
```

```python
from pathlib import Path
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1",
                api_key=(Path.home() / ".config/colab-llm/api_key").read_text().strip())
```

Quick smoke test: `colab-llm chat "say hi"` (streams and prints tok/s).

Check state and cost: `colab-llm status` — session, supervisor, endpoint health,
watchdog idle timer and session age, tokens served, CU balance.

Stop: `colab-llm down` (releases the GPU and shows the balance).

Reload after editing `config.env` (model, context, extra llama-server flags):
`colab-llm restart` (keeps the VM, re-downloads only if the model changed).

## Facts that matter for planning

| Item | Value |
| --- | --- |
| Model | Qwen3.8-Flash-Next, 125B MoE, 6B active, GSQ-RCO IQ3_S GGUF (`config.env`: `HF_REPO`, `QUANT`) |
| Context | 262,144 tokens, one slot (`CTX`, `PARALLEL`) |
| VRAM | ~55 GB weights on GPU; the 28.8 GB n-gram table is memory-mapped from disk |
| Endpoint | `http://127.0.0.1:${LOCAL_PORT:-8080}/v1`, model name `qwen3.8-flash-next` (`ALIAS`) |
| Auto-stop | after `IDLE_MINUTES` (15) without tokens, or `MAX_HOURS` (8) |
| Cost | A100 80GB ≈ 8.88 CU/h; T4 ≈ 1.19 CU/h |

The model is not uncensored by default; switching `HF_REPO` to an abliterated
build is a user decision, not an agent one.

## When something fails

| Symptom | Look at | Usual cause / fix |
| --- | --- | --- |
| `up` stops with `got ... < MIN_VRAM_GIB` | output | Colab gave a 40 GB A100. Retry later; the session was already stopped. |
| `serve failed` | `colab-llm logs serve` | Download or load error; the session is stopped automatically (use `up --keep` to debug, then `down`). |
| Server up on Colab but local port dead | `colab-llm logs supervisor` | ssh tunnel; the supervisor retries every 5 s. |
| Requests hang or error | `colab-llm logs server` | llama-server log on the VM. |
| Endpoint vanished | `colab-llm logs supervisor` | Watchdog stopped it (`idle` / `max` / `server_dead`) or Colab reclaimed the VM. `colab-llm ensure` again. |
| `BIN_STORE=hf needs ...` | `config.env`, `~/.config/colab-llm/hf_token` | HF token or `HF_BIN_REPO` missing. Ask the user. |
| No llama-server build in the HF repo | — | `colab-llm prep --gpu T4` (≈15 min on a T4, ≈0.3 CU). Ask before running. |
| 401 from the endpoint | — | Key mismatch: `colab-llm restart` re-sends the key. |

Do not try to fix Colab account, OAuth or quota problems yourself; report them
to the user with the exact error.

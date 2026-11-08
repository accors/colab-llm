# Agent notes

This repo serves a self-hosted LLM on a Google Colab A100 80GB and exposes it
locally at `http://127.0.0.1:8080/v1` (OpenAI-compatible). Every action goes
through `bin/colab-llm`.

Before using, starting or stopping it, read
[`.claude/skills/colab-llm/SKILL.md`](.claude/skills/colab-llm/SKILL.md).
The short version:

- Starting it spends money (~8.88 compute units/hour). Do not start it
  without need; check `colab-llm usage` first; run `colab-llm down` when done.
- Get an endpoint with `colab-llm ensure` (idempotent). Read the API key
  from `~/.config/colab-llm/api_key` at request time; never print or commit
  it or `~/.config/colab-llm/hf_token`.
- Do not run `colab ssh` by hand; it can create a new billable runtime.

## Developing this repo

- `bin/colab-llm` (bash) runs on the local machine; `remote/*.py` run on the
  Colab VM via `colab exec`, with `remote/common.py` prepended.
- Lint before committing: `shellcheck -S warning bin/colab-llm` and
  `cat remote/common.py remote/serve.py | ruff check --select F,E9 -`
  (same for `prep.py` and `watchdog.py`).

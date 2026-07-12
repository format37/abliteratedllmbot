# abliteratedllmbot

A Telegram bot that talks to a **local, uncensored (abliterated) LLM served by
[Ollama](https://ollama.com)**. It long-polls Telegram, streams answers **live**
into a single edited message, shows the model's reasoning separately, and renders
rich Markdown (the [pantherabot](https://github.com/format37/pantherabot) way).

> ⚠️ **This fronts an uncensored model.** Keep `allowed_user_ids` restricted to
> people you trust. The default is a single owner. Do not expose it publicly.

## Features

- **Polling** (no webhook, no public URL needed).
- **Streaming** — answer appears token-by-token in one live-edited message,
  throttled and rolled over near Telegram's 4096-char limit.
- **Reasoning shown separately** — inline `<think>…</think>` is split out into its
  own message; capability-gated and suppressed when empty.
- **Multi-turn memory** per chat, `/reset` to clear, configurable cap.
- **Rich formatting** — `sendRichMessage` (32768 chars) via an optional local Bot
  API server, degrading to MarkdownV2 → `.txt` → plain.
- **Images** — text-only models reply with a note (optional vision-model fallback).
- **Uncensored-model hardening** — strips leaked chat-template tokens
  (`<|im_start|>`, `<|endoftext|>`, …) that this class of GGUF emits.

See **[SPEC.md](./SPEC.md)** for the full design and **[PLAN.md](./PLAN.md)** for
the implementation milestones.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a model pulled, e.g.:
  ```bash
  ollama pull qwen3.6-abliterated:35b-a3b
  ```
- A Telegram bot token from [@BotFather](https://t.me/BotFather).
- (Optional) the local Bot API server for true rich messages.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt         # or: pip install -e .

cp config.example.yaml config.yaml      # then set telegram.allowed_user_ids
cp .env.example .env                     # then set TELEGRAM_BOT_TOKEN

python -m abliteratedllmbot              # uses ./config.yaml
```

Your Telegram user ID goes in `config.yaml → telegram.allowed_user_ids`
(get it from [@userinfobot](https://t.me/userinfobot)).

## Configuration

All non-secret settings live in `config.yaml` (**gitignored**); the annotated
canonical schema is [`config.example.yaml`](./config.example.yaml). The only
secret, `TELEGRAM_BOT_TOKEN`, lives in `.env`. Any nested key can be overridden by
an env var using the `__` delimiter, e.g. `OLLAMA__MODEL=qwen3.5:27b`.

Rich messages are opt-in: set `telegram.base_url` to your local Bot API server
(e.g. `http://localhost:8081/bot`). Without it the bot runs on the official
Telegram API and finalizes answers as MarkdownV2.

## Deployment

- **systemd:** edit and install [`deploy/abliteratedllmbot.service`](./deploy/abliteratedllmbot.service).
- **Docker:** `docker compose up --build -d` (reaches host Ollama via
  `host.docker.internal`; set `ollama.host` accordingly).

## Development

```bash
pip install -e '.[dev]'
pytest -q                    # pure-logic tests (no bot / no Ollama)
pytest -q -m ollama          # integration tests against a live Ollama (opt-in)
```

Build order and Definitions of Done are in [PLAN.md](./PLAN.md). Start with
Milestone 1a (`tests/test_reasoning.py`) — it's pure and already green.

## License

MIT — see [LICENSE](./LICENSE).

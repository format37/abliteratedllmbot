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

Both paths need the same two local files, neither of which is committed:

```bash
cp config.example.yaml config.yaml      # then set telegram.allowed_user_ids
cp .env.example .env                     # then set TELEGRAM_BOT_TOKEN
```

Your Telegram user ID goes in `config.yaml → telegram.allowed_user_ids`
(get it from [@userinfobot](https://t.me/userinfobot)). Ollama must already be
running on the host with your model pulled.

### Docker Compose (recommended)

```bash
docker compose up --build -d      # build + run in the background
docker compose logs -f            # follow; expect "Ollama reachable"
docker compose down               # stop
```

That's it — no Python toolchain, no venv. `restart: unless-stopped` brings the
bot back after a reboot, and logs are capped at 3×10 MB.

`config.yaml` and `.env` are bind-mounted, never baked into the image, so
editing them and running `docker compose restart` is enough to pick up changes
(a code change still needs `--build`).

**Reaching Ollama from inside the container.** `config.yaml` ships with
`ollama.host: http://localhost:11434`, and inside a container `localhost` is the
*container itself* — so that address would fail. Compose therefore sets
`OLLAMA__HOST=http://host.docker.internal:11434` for you (env overrides yaml, and
only that one key), which is mapped to the host via `extra_hosts`. You don't have
to change `config.yaml`, and it keeps working for a bare-metal run.

If Ollama is itself a container on a shared Docker network, point the bot at it
by service name instead — set this in `.env` and Compose will use it:

```bash
OLLAMA__HOST=http://ollama:11434
```

Ollama must be listening on more than loopback for the host mapping to work
(`OLLAMA_HOST=0.0.0.0`); check with `curl http://localhost:11434/api/version`.

**File ownership.** The container runs as `1000:1000` rather than root, so
anything it writes into the bind-mounted `./data` (i.e. `history.db`, once
`memory.persist` is on) stays owned by your host user — otherwise Docker creates
those files as `root` and a later bare-metal run can't write them. `1000` is the
first user on most Linux hosts; if `id -u` prints something else, tell Compose
once:

```bash
echo "UID=$(id -u)" >> .env && echo "GID=$(id -g)" >> .env
```

(Docker Desktop on macOS/Windows maps ownership itself and ignores this.) If you
already have a root-owned `data/` from an earlier run, reset it without sudo via
`docker run --rm -v ./data:/d alpine chown -R $(id -u):$(id -g) /d`.

### Local Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt         # or: pip install -e .

python -m abliteratedllmbot              # uses ./config.yaml
```

## Configuration

All non-secret settings live in `config.yaml` (**gitignored**); the annotated
canonical schema is [`config.example.yaml`](./config.example.yaml). The only
secret, `TELEGRAM_BOT_TOKEN`, lives in `.env`. Any nested key can be overridden by
an env var using the `__` delimiter, e.g. `OLLAMA__MODEL=qwen3.5:27b`.

Rich messages are opt-in: set `telegram.base_url` to your local Bot API server
(e.g. `http://localhost:8081/bot`). Without it the bot runs on the official
Telegram API and finalizes answers as MarkdownV2.

## Deployment

- **Docker Compose** — see [Quick start](#docker-compose-recommended) above;
  `restart: unless-stopped` already survives reboots, so there's nothing else to
  install.
- **systemd** (bare metal) — edit the paths in
  [`deploy/abliteratedllmbot.service`](./deploy/abliteratedllmbot.service), then:
  ```bash
  sudo cp deploy/abliteratedllmbot.service /etc/systemd/system/
  sudo systemctl daemon-reload && sudo systemctl enable --now abliteratedllmbot
  ```

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

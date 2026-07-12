# abliteratedllmbot — Technical Specification

A Telegram bot that fronts a **local, uncensored (abliterated) LLM served by Ollama**. Long‑polling, single‑owner by default, streams answers live into one edited message, shows the model's reasoning separately, and renders rich Markdown the way [`pantherabot`](../pantherabot) does.

> This document **freezes the contracts** (config keys, the stream‑delta type, the concurrency policy, the formatting chain). It is the single source of truth the implementation session builds against. Where a fact was verified against the live machine it is marked **[V]** and detailed in [Appendix A](#appendix-a--verified-facts).

---

## 1. Goals & scope

| # | Requirement | Decision |
|---|---|---|
| 1 | Telegram mode | **Polling** (`getUpdates`); no webhook |
| 2 | Model | Local Ollama, configurable; default `qwen3.6-abliterated:35b-a3b` at `http://localhost:11434` |
| 3 | Access | Only `allowed_user_ids` (default `[106129214]`); others get **reply‑once** denial |
| 4 | Repo | Public git; **`.env` and `config.yaml` gitignored**; ship `*.example` templates |
| 5 | Streaming | **Edit one message live**, throttled; roll over near the 4096 limit |
| 6 | Memory | **Multi‑turn per chat**; `/reset` clears; configurable cap; oldest‑turn trimming |
| 7 | Images | Default **reject‑with‑note** (default model is text‑only); optional vision‑model fallback |
| 8 | Reasoning | **Show thinking separately**; capability‑gated; suppressed when empty |
| 9 | Typing | `typing` chat action during the cold‑load / pre‑first‑token gap |
| 10 | Formatting | pantherabot **rich** path: `sendRichMessage` → `.txt` → MarkdownV2 → plain |

**Non‑goals for v1:** group chats, inline queries, tool calling, SQLite persistence, `/model` switching, audio/voice. All are v1.1 candidates (see [PLAN.md](./PLAN.md) Milestone 6).

---

## 2. The abliterated model — what the design must absorb

Everything below was verified live (Ollama 0.20.5) and **drives non‑obvious design choices**:

- **[V] No structured thinking.** `POST /api/chat` with `think:true` on `qwen3.6-abliterated:35b-a3b` returns HTTP `{"error":"…does not support thinking"}` and **zero** stream lines. `think:true` must **never** be sent to a model whose `/api/show` capabilities lack `"thinking"`.
- **[V] Inline `<think>` blocks instead.** The abliterated model wraps replies in an inline `<think>…</think>` block inside `message.content` (often **empty**, e.g. `<think>\n\n</think>\n\nHi there!`). So "show thinking separately" is implemented by **parsing inline tags out of the content stream**, not by reading `message.thinking`.
- **[V] Special‑token leakage.** Even at `done_reason:"stop"` the GGUF can emit chat‑template tokens into visible content (e.g. `…OK<|endoftext|><|im_start|>…`). A **sanitizer** + sampler `stop` sequences are mandatory, not optional.
- **[V] Big context, slow cold start.** `context_length` = **262144**; first request cold‑loads the 35B MoE in **~156 s** (`load_duration ≈ 1.57e11 ns`). ⇒ `request_timeout` must be ≥ 600 s and `keep_alive` should hold the model warm.
- **[V] `done_reason`** is `"stop"` on natural end, `"length"` when `num_predict` is hit. `num_predict` caps **total** output (reasoning + answer).
- **[V] Structured thinking DOES work on capable models** (`gpt-oss`, `qwen3.5`, `qwen3.6:35b-a3b`) — so the client keeps a structured path for when a thinking‑capable model is configured.

Consequence: the reasoning feature has **three modes** (`reasoning.think_mode`): `auto` (structured if capable → else inline‑parse), `inline` (always parse tags), `off` (strip thinking entirely).

---

## 3. Architecture

```
                    Telegram (long polling, official API or local Bot API server)
                                   │  Update
                                   ▼
┌───────────────────────────── telegram_bot.py ─────────────────────────────┐
│  access_control decorator → per‑chat task registry (cancel‑and‑supersede)  │
│  handlers: /start /help /reset /stats /think, on_text, on_photo, on_error  │
│  TypingAction (cold‑load gap)                                              │
└───────┬──────────────────────────────────────────────┬────────────────────┘
        │ build messages                                │ Delta stream
        ▼                                               ▼
   history.py                                     streaming.py
   Message / InMemoryStore / trim_history          RenderDriver + MessageEditor
        │                                               │ finalize
        ▼                                               ▼
   ollama_client.py  ── Delta(kind,…) ──►         formatting.py
   AsyncClient wrapper, capability cache,          rich → .txt → MarkdownV2 → plain
   resolve_think, InlineThinkSplitter, sanitize    (pantherabot chain)
        │
        ▼
   Ollama /api/chat (stream), /api/show (capabilities)
```

**Module map** (flat package `abliteratedllmbot/`):

| Module | Responsibility | Milestone |
|---|---|---|
| `config.py` | Frozen `Settings` (pydantic‑settings, yaml+env). **Contract.** | 0 |
| `errors.py` | Typed exceptions + the `Delta` dataclass. **Contract.** | 0 |
| `reasoning.py` | `InlineThinkSplitter`, `sanitize_content` (pure, unit‑tested). | 1 |
| `ollama_client.py` | Capability cache, `resolve_think`, `chat_stream() → AsyncIterator[Delta]`. | 1 |
| `history.py` | `Message`, `InMemoryStore`, `trim_history`. | 2 |
| `streaming.py` | `MessageEditor`, `RenderDriver` (the live‑edit state machine). | 3 |
| `formatting.py` | Finalize chain incl. `sendRichMessage` POST. | 3 |
| `telegram_bot.py` | Handlers, access control, typing, task registry, error handler. | 4 |
| `__main__.py` | Load config → build `Application` → `run_polling`. | 4 |

---

## 4. FROZEN CONTRACT — the stream delta

The **Ollama client owns** capability gating, inline‑think splitting, and sanitisation, and yields exactly one type. The renderer consumes it and knows nothing about Ollama.

```python
# errors.py
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class Delta:
    kind: Literal["thinking", "content", "done", "error"]
    text: str = ""                       # incremental text (thinking|content)
    done_reason: Optional[str] = None    # "stop" | "length" | ...  (kind="done")
    stats: dict = field(default_factory=dict)   # eval_count, *_duration (kind="done")
    error: Optional[str] = None          # human message (kind="error")
```

**Rules**
- Ordering: zero+ `thinking` deltas, then zero+ `content` deltas, then exactly one terminal delta (`done` **or** `error`).
- `thinking` deltas are emitted **only** when reasoning text actually exists (structured `message.thinking`, or a non‑empty inline `<think>` block). An empty `<think></think>` produces **no** thinking delta.
- **Pre‑stream** failures (Ollama unreachable, model missing) → the client **raises** a typed exception (`OllamaUnavailableError`, `ModelNotFoundError`) *before* yielding anything. The handler catches these and replies with a friendly error; no partial message is created.
- **Mid‑stream** failures (connection dropped after tokens) → the client yields `Delta(kind="error", …)` after whatever content it already produced, so the renderer can show *partial answer + error note*.
- All `content` text is already **sanitised** (special tokens stripped) by the time it leaves the client.

```python
# errors.py
class BotError(Exception): ...
class OllamaUnavailableError(BotError): ...   # connect refused / timeout pre-stream
class ModelNotFoundError(BotError): ...       # /api/show 404
class ThinkingUnsupportedError(BotError): ...  # defensive; should be prevented by gating
```

---

## 5. Ollama client (`ollama_client.py`)

Transport: the official **`ollama` AsyncClient (0.5.1, already installed)** — `chat()` accepts `model, messages, stream, think, format, options, keep_alive`. (Rationale for choosing it over raw httpx: it is present, typed, and its streaming yields the same NDJSON chunks we verified; we still wrap it so the rest of the bot depends only on `Delta`.)

```python
class OllamaClient:
    def __init__(self, cfg: OllamaConfig): ...
    async def startup(self) -> None                      # warm capability cache for model + vision_model
    async def close(self) -> None
    async def capabilities(self, model: str) -> set[str] # cached /api/show; {'completion',['thinking'],['vision'],['tools']}
    def resolve_think(self, model: str) -> bool          # cfg think_mode + 'thinking' in caps
    async def chat_stream(
        self, *, messages: list[dict], model: str | None = None,
        images: list[bytes] | None = None,
    ) -> AsyncIterator[Delta]: ...
```

**`chat_stream` behaviour**
1. `model = model or cfg.model`. Look up capabilities (raise `ModelNotFoundError` if `/api/show` 404).
2. `think = resolve_think(caps)`. **Only** `think_mode == "auto"` on a model whose caps include `"thinking"` sends `think:true`; `"inline"` (always parse tags per §2) and `"off"` send **no** `think` param — never `think:true` to an incapable model **[V]**.
3. Build options from `cfg.options` + `stop = cfg.stop`. Attach `images` (base64) only on the vision path.
4. Stream chunks. For each chunk:
   - **Structured path** (`chunk.message.thinking` present): emit `Delta("thinking", text=…)` / `Delta("content", text=…)`.
   - **Inline path** (no `thinking` field): feed `chunk.message.content` through the per‑stream `InlineThinkSplitter`, which classifies text as thinking vs content across chunk boundaries.
   - Sanitise every content span before emitting (carry buffer of `len(longest_token)-1`).
5. On the terminal chunk emit `Delta("done", done_reason=…, stats={eval_count, prompt_eval_count, total_duration, load_duration})`.
6. Wrap transport exceptions: pre‑first‑chunk → raise typed; after first chunk → `yield Delta("error", …)`.

**Options defaults** (`config.example.yaml`): `temperature 0.7, top_p 0.9, top_k 40, num_ctx 8192, num_predict 2048`, `stop: ["<|im_start|>","<|im_end|>","<|endoftext|>"]`, `keep_alive 30m`, `request_timeout 600`, `connect_timeout 10`. `num_ctx` is capped well under the 262144 max for latency/VRAM; expose it.

---

## 6. Reasoning split + sanitise (`reasoning.py`, pure & unit‑tested first)

```python
class InlineThinkSplitter:
    """Stateful, streaming splitter for inline <think>…</think>.
    feed(text) -> list[tuple[kind, span]] where kind in {'thinking','content'}.
    Holds a carry buffer so a tag split across chunks ('<thi' | 'nk>') is handled.
    Suppresses whitespace-only thinking spans (empty <think>\\n\\n</think>)."""
    def feed(self, text: str) -> list[tuple[str, str]]: ...
    def flush(self) -> list[tuple[str, str]]: ...

SPECIAL_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]+\|>")  # <|im_start|>, <|endoftext|>, …

def sanitize_content(text: str, extra: list[str] = ()) -> str: ...
```

The splitter and sanitiser are **the same layer**; the client applies `sanitize_content` to every `content` span (with a trailing carry so a token straddling chunks isn't missed). These two classes are the **first** thing built and are exhaustively unit‑tested against the verified samples (empty think block, straddled tag, trailing `<|endoftext|><|im_start|>` leak).

---

## 7. History (`history.py`)

```python
@dataclass
class Message:
    role: Literal["system","user","assistant"]
    content: str
    # vision turns are STATELESS in v1 (not stored) — see §9

class InMemoryStore:                     # dict[chat_id -> list[Message]]
    def get(self, chat_id: int) -> list[Message]: ...
    def append(self, chat_id: int, msg: Message) -> None: ...
    def reset(self, chat_id: int) -> None: ...

def build_messages(cfg, history, user_text) -> list[dict]: ...  # system + trimmed history + user
def trim_history(history, *, max_turns, max_tokens, chars_per_token) -> list[Message]: ...
```

- Trim by **turns first, then token budget**. Token estimate = `len(text) / chars_per_token`. Default `chars_per_token = 2.5` — deliberately low because the owner's usage is heavily **Cyrillic**, which tokenises denser than the naive `/4` [reviewer‑flagged]. When a prior `done` delta carried `prompt_eval_count`, calibrate the estimate against it.
- Always keep the system prompt; drop oldest user/assistant pairs.
- `/reset` clears the chat's list. v1 is **in‑memory** (lost on restart); v1.1 adds optional SQLite (`memory.persist`, `memory.db_path`).

---

## 8. Streaming render state machine (`streaming.py`) — the crux

Honors **edit‑one‑message‑live + thinking‑separate + rollover + throttle**.

### States
```
IDLE ──first thinking delta──► THINKING ──first content delta──► ANSWER ──done──► DONE
  └────────────────────first content delta───────────────────────►┘
                                                (skip THINKING if no reasoning)
error at any point ──► finalize partial + append error note
```

### Objects
- **`MessageEditor`** — owns exactly one Telegram message; `feed(text)` appends to a buffer; flushes an `edit_message_text` when **`chars_since_edit ≥ batch_size`** *or* **`now-last_edit ≥ min_edit_interval`**; on a trailing timeout does a final flush; **rolls over** to a new message when the buffer would pass `rollover_chars` (3900), splitting on the last newline/space. Keeps `effective_interval` as a **local** (never mutates shared `Settings`) and multiplies it on `RetryAfter`.
- **`RenderDriver`** — consumes `AsyncIterator[Delta]`, owns a thinking `MessageEditor` (lazily created on the first `thinking` delta, prefixed 💭) and an answer `MessageEditor`. On the first `content` delta it **finalizes** the thinking editor and opens the answer editor. On `done`/`error` it finalizes and hands the full answer text to `formatting.finalize()`.

### Flush mechanism
A single `asyncio.Queue` fed by the delta consumer; the editor loop does `asyncio.wait_for(queue.get(), timeout=max_interval_seconds)` to get **trailing‑edge** flushes for free (D3 model), with `batch_size`/`min_edit_interval` as the leading‑edge knobs.

### Telegram‑error handling in the editor
| Error | Handling |
|---|---|
| `RetryAfter` (429) | sleep `min(retry_after, max_retry_after)`; grow local interval ×1.5 |
| `BadRequest "not modified"` | ignore (identical content) |
| `BadRequest "can't parse entities"` | only possible on the rich/markdown finalize — fall down the chain; **never during streaming** (streaming is `parse_mode=None`) |
| `BadRequest "message to edit not found"` | user deleted it → stop editing, keep consuming the stream for history/logging |

### Concurrency — FROZEN: cancel‑and‑supersede
A **per‑chat task registry** `dict[chat_id, asyncio.Task]` in `bot_data`. A new prompt for a chat with a live task **cancels** it (finalize cleanup guarded by `try/except CancelledError` + `asyncio.shield` on the last edit; optionally append `superseded_note`) then starts the new one. Registration is atomic **before** the first await, so two near‑simultaneous messages can't both start (avoids the TOCTOU a lock‑inspection would have). App uses `concurrent_updates(True)` so different chats run in parallel.

### Rate limiting covers sends too
Rollover `send_message`, the thinking bubble, and typing are all to the **same chat**; throttle **new sends** as well as edits and handle `RetryAfter` uniformly on both. Rollover is rare by design (3900‑char messages).

### Typing indicator
A background task sends `ChatAction.TYPING` every `refresh_interval` (4 s) starting at accept; when `typing.stop_after_first_token` (default true) it **stops as soon as the first real message is on screen** — typing is most valuable only during the ~156 s cold‑load gap, and flickering "typing…" over a growing message is noise.

---

## 9. Images (`telegram_bot.on_photo`)

1. Download the largest `PhotoSize`.
2. `caps = await client.capabilities(active_model)`.
3. Branch on `images.mode`:
   - **`reject_note`** (default): reply `images.note`, then process the **caption text** (if any) as a normal text prompt. Image discarded.
   - **`caption_only`**: silently process caption only.
   - **`vision_fallback`**: route image+caption to `ollama.vision_model` (`qwen3-vl:30b-a3b-instruct-q8_0`, `'vision'` capable **[V]**).
4. **History rule (v1):** vision turns are **stateless** — not appended to history — to avoid contaminating the text model's context with turns it can't reconstruct. Documented; revisited in v1.1.

> pantherabot's Telegram file paths carry a `token:`‑prefix that breaks bind mounts on Linux; if this bot ever runs behind the **local Bot API server**, reuse pantherabot's `re.sub(r'^/[^/]+:', '/', path)` cleanup and the `mount --bind` note.

---

## 10. Formatting / finalize (`formatting.py`) — pantherabot rich chain

Streaming stays **plain** (`parse_mode=None`) so intermediate edits can never throw. Exactly **one** finalize pass runs on the whole answer, governed by `formatting.finalize_mode`:

```
finalize(chat_id, full_text, scratch_msg_ids):
  if mode == "plain":       leave scratch as-is.
  if mode == "markdown_v2": edit scratch in place with MarkdownV2 (placeholder-escape);
                            on BadRequest → revert to plain.  (in place, no "swap")
  if mode == "rich":        try sendRichMessage (only if base_url set OR rich_requires_base_url=false)
                              → on ok: delete scratch, message re-appears rich (32768 cap)
                              → on fail/>rich_max_chars:
                                   if len>txt_fallback_over: send .txt document, delete scratch
                                   else: fall to markdown_v2 in place
```

**`send_rich_message`** is ported verbatim from pantherabot: `POST {base_url}/bot{token}/sendRichMessage` with `{"chat_id", "rich_message":{"markdown":…}}`, `ok` → True else False. **MarkdownV2** reuses pantherabot's placeholder‑token trick (`&&&`→`*`, `%%%`→`_`, `@@@`→`__`, `~~~`→`~`, protect ` ``` ` / `||`) around `telegram.helpers.escape_markdown(version=2)`.

Default `finalize_mode: rich` is **safe everywhere**: with `rich_requires_base_url: true` and no `base_url` (official API) it degrades to `markdown_v2` automatically — no wasted round‑trip. Point `telegram.base_url` at your local `http://localhost:8081/bot` server to light up true rich messages.

---

## 11. Access control & commands

- **`access_control(allowed_ids, cfg)`** decorator wraps every handler; unauthorized `effective_user.id` → `on_denied` policy: `reply_once` (default; a `bot_data` set suppresses repeats — resets on restart, acceptable for v1), `reply_always`, or `ignore`.
- **Commands (v1):** `/start` (greeting + who‑am‑I), `/help`, `/reset` (clear history), `/stats` (last‑gen tokens/sec + durations from the `done` stats, in‑memory), `/think` (per‑chat toggle of `show_thinking`). **Model‑admin commands** (gated by `commands.enable_model_admin`, default true): `/status` (which models Ollama holds in VRAM + auto‑unload time, via `/api/ps`), `/load` (warm the model — an empty `generate` with `keep_alive`), `/unload` (evict now, `keep_alive:0`). `/model` switching is v1.1, gated by `commands.allow_model_switch` (default false).

---

## 12. FROZEN CONTRACT — configuration

`.env` (gitignored) holds **only** `TELEGRAM_BOT_TOKEN` (+ optional overrides). Everything else lives in `config.yaml` (**gitignored**; `config.example.yaml` committed). Loaded via **pydantic‑settings 2.13** `YamlConfigSettingsSource` layered under env (**env > yaml**). Validated on startup with clear errors. The canonical schema is the annotated [`config.example.yaml`](./config.example.yaml); every key below is final:

```
telegram:   allowed_user_ids, base_url, drop_pending_updates, long_poll_timeout, poll_interval
access:     on_denied, denial_message
ollama:     host, model, vision_model, keep_alive, connect_timeout, request_timeout,
            options.{temperature,top_p,top_k,num_ctx,num_predict,seed}, stop[]
system_prompt
reasoning:  think_mode(auto|inline|off), show_thinking, max_thinking_chars
memory:     enabled, max_turns, max_tokens, chars_per_token, persist, db_path
streaming:  batch_size, min_edit_interval, max_interval_seconds, rollover_chars,
            supersede_policy(cancel), placeholder_text, empty_response_text,
            error_note, superseded_note, max_retry_after
formatting: finalize_mode(plain|markdown_v2|rich), rich_requires_base_url, rich_max_chars, txt_fallback_over
sanitize:   strip_special_tokens, extra_tokens[]
images:     mode(reject_note|caption_only|vision_fallback), note
typing:     enabled, refresh_interval, stop_after_first_token
commands:   allow_model_switch, enable_stats, enable_model_admin
logging:    level
```

There is exactly **one** name for each concept (this resolves every naming conflict the four sub‑designs had, e.g. `allowed_user_ids` lives under `telegram`, the model under `ollama.model`, the reasoning cap is `reasoning.max_thinking_chars`, the edit gap is `streaming.min_edit_interval`).

---

## 13. Deployment

- **Bare metal / systemd:** `deploy/abliteratedllmbot.service` runs `python -m abliteratedllmbot` as the owner; Ollama reachable at `localhost:11434`.
- **Docker:** `Dockerfile` + `docker-compose.yml`. Container reaches host Ollama via `extra_hosts: ["host.docker.internal:host-gateway"]` and `ollama.host: http://host.docker.internal:11434` (or `network_mode: host`). `config.yaml` and `.env` bind‑mounted, never baked into the image.
- **Rich messages** require the user's local Bot API server (`format37/telegram_bot`) on `:8081`; optional. Without it the bot runs fully on the official API with MarkdownV2 finalize.

---

## 14. Security (public repo)

- **Never commit** `.env` or `config.yaml`. `.gitignore` covers `.env`, `config.yaml`, `*.db`, `__pycache__`, `.venv`.
- `allowed_user_ids` defaults to a single owner — this is an **uncensored** model; the README must warn against opening access.
- No secret in `config.example.yaml`; token only via env.

---

## Appendix A — Verified facts

All checked on this host, 2026‑07‑12, Ollama **0.20.5** at `http://localhost:11434`:

| Fact | How verified |
|---|---|
| `qwen3.6-abliterated:35b-a3b` caps = `['completion']`; `context_length` 262144 (`qwen35moe.context_length`) | `curl /api/show` |
| `think:true` on it → `{"error":"…does not support thinking"}`, 0 stream lines | `curl /api/chat -d '{…,"think":true}'` |
| It emits inline `<think>…</think>` in `content` (often empty); no `message.thinking` field | streamed NDJSON parse |
| It leaks `<\|endoftext\|><\|im_start\|>…` into content even at `done_reason:"stop"` | `curl … 'Reply with exactly: OK'` |
| Cold load ≈ 156.8 s (`load_duration 156778044475`) | `done` chunk of first 35B gen |
| `done_reason` `"stop"` natural / `"length"` at `num_predict` | short vs `num_predict:200` runs |
| Structured thinking works on `gpt-oss`, `qwen3.5`, `qwen3.6:35b-a3b` | `/api/show` caps + `think:true` stream |
| `qwen3-vl:30b-a3b-instruct-q8_0` caps = `['completion','vision','tools']` | `/api/show` |
| `ollama` py pkg 0.5.1 `AsyncClient.chat(model,messages,stream,think,format,options,keep_alive)` | `pip show` + `inspect.signature` |
| `pydantic-settings` 2.13.1 has `YamlConfigSettingsSource`; pydantic 2.12.5, httpx 0.28.1 present | import + `pip show` |
| PTB latest stable **22.8** (async, Py≥3.9); `escape_markdown(text,version=1,…)` (v2 must be explicit); `RetryAfter.retry_after`, `BadRequest`, `Conflict` exist; `ChatAction.TYPING == 'typing'` | PTB docs (WebFetch) |
| Telegram text limit 4096 UTF‑16 code units; identical edit → 400 "not modified"; flood → 429 `retry_after` | Telegram Bot API docs |

## Appendix B — pantherabot lineage

Reused ideas/code from [`../pantherabot`](../pantherabot): `send_rich_message` (raw `sendRichMessage` POST, 32768 cap), the MarkdownV2 placeholder‑escape trick, the `.txt`‑document fallback for >4096, the local Bot API server wiring (`API_URL`/`FILE_URL`, `mount --bind` for `token:` paths), and its `pyTelegramBotAPI` heritage (offered as an alternative framework). This bot **adds** live streaming, capability‑gated reasoning, and the uncensored‑model sanitiser, which pantherabot (whole‑answer, Claude‑backed) never needed.

# abliteratedllmbot — Implementation Plan

Build order is chosen so each layer is **testable before the next depends on it**. Milestone 0 freezes the seams (already done in [SPEC.md](./SPEC.md)); the rest is bottom‑up: pure logic → I/O wrappers → the Telegram shell last. Every milestone lists a **Definition of Done (DoD)** you can check without a running bot until Milestone 3.

> Scaffolding for M0–M1 is already in the repo as stubs with the frozen signatures. Fill them in; don't re‑litigate the contracts.

---

## Milestone 0 — Freeze the contracts ✅ (done in SPEC)

The four parallel designs agreed on the Ollama facts but disagreed on every seam. SPEC.md resolves them: one `Settings` model, one `Delta` type, `cancel‑and‑supersede`, sanitizer owned by the client. **Deliverable already in repo:** `config.example.yaml`, `abliteratedllmbot/config.py`, `abliteratedllmbot/errors.py`.

**DoD:** `python -c "from abliteratedllmbot.config import load_config; load_config('config.example.yaml')"` validates and prints the settings.

---

## Milestone 1 — Ollama client + reasoning (pure logic first)

**1a. `reasoning.py` (pure, no I/O — build & test FIRST).**
- `InlineThinkSplitter.feed()/flush()` with cross‑chunk carry buffer; suppress whitespace‑only `<think>` spans.
- `sanitize_content()` stripping `<|…|>` special tokens (+ `extra_tokens`), with a trailing carry so a token straddling chunks isn't emitted.

**1b. `ollama_client.py`.**
- `capabilities()` — cached `/api/show`; `ModelNotFoundError` on 404.
- `resolve_think()` — `think_mode` × capability gate; **never** `think:true` to an incapable model.
- `chat_stream()` — structured‑thinking path and inline path both normalised to `Delta`; sanitise content spans; terminal `done` with stats; typed exception pre‑stream vs `error` Delta mid‑stream.

**DoD (offline‑testable):**
- Unit tests: empty `<think>\n\n</think>` → no thinking delta; tag split `'<thi'|'nk>'` across two `feed()` calls → correct classification; `'OK<|endoftext|><|im_start|>x'` → `'OK'`.
- Integration test (marked `@pytest.mark.ollama`, skipped in CI): `chat_stream` against the live default model yields `content` deltas then one `done` with `done_reason`; passing a thinking‑capable model yields `thinking` deltas.

---

## Milestone 2 — History store + trimming (pure)

- `Message`, `InMemoryStore.get/append/reset`.
- `build_messages()` = system + trimmed history + current user.
- `trim_history()` — turns‑then‑token budget; `chars_per_token` default 2.5 (Cyrillic‑aware); calibrate against `prompt_eval_count` when available.

**DoD:** unit tests prove oldest pairs drop first, system prompt always kept, `/reset` empties a chat without touching others, and a Cyrillic string trims more aggressively than the naive `/4` would.

---

## Milestone 3 — Streaming render state machine (against a FAKE bot)

- `MessageEditor` — buffer, leading‑edge (`batch_size`/`min_edit_interval`) + trailing‑edge (`asyncio.wait_for` timeout) flush, rollover at `rollover_chars` on a safe boundary, local `effective_interval` (never mutate `Settings`), Telegram‑error table from SPEC §8.
- `RenderDriver` — IDLE→THINKING→ANSWER→DONE; lazy thinking editor; finalize handoff to `formatting.finalize()`.
- `formatting.py` — the `rich → .txt → markdown_v2 → plain` chain + `send_rich_message` POST + MarkdownV2 placeholder‑escape.

**DoD:** a **recording FakeBot** (captures `send_message`/`edit_message_text`/`delete_message` calls) driven by scripted `Delta` streams asserts: throttling (N deltas → far fewer edits), rollover (>3900 chars → 2 messages split on whitespace), thinking‑then‑answer message ordering, empty‑answer path, mid‑stream `error` Delta shows partial+note, `RetryAfter` backs off, "message not found" aborts cleanly. **No real Telegram needed.**

---

## Milestone 4 — Telegram layer (first end‑to‑end run)

- `__main__.py` — `load_config` → `build_application` (`concurrent_updates(True)`, `post_init`/`post_shutdown` for the shared `OllamaClient` + capability warmup) → `run_polling(allowed_updates=['message'], drop_pending_updates=…)`; `Conflict` fatal.
- `telegram_bot.py` — `access_control` decorator (`reply_once`), handlers `/start /help /reset /stats /think` + `on_text` + `on_photo` (SPEC §9) + `on_error`; the **per‑chat cancel‑and‑supersede registry**; `TypingAction` that stops after the first token.

**DoD:** run against a real bot token + live Ollama; verify (a) unauthorized user gets one denial then silence; (b) a prompt streams live into one edited message; (c) `/reset` forgets context; (d) a photo triggers the note + caption handling; (e) sending a second message mid‑generation cancels and supersedes the first cleanly; (f) `/stats` reports tokens/sec.

---

## Milestone 5 — Packaging & deployment

- `pyproject.toml` (console script `abliteratedllmbot = abliteratedllmbot.__main__:main`), pinned `requirements.txt`.
- `config.example.yaml`, `.env.example`, `.gitignore` (incl. **`config.yaml`**), `README.md` with the uncensored‑model + secrets warnings.
- `Dockerfile` + `docker-compose.yml` (`host.docker.internal:host-gateway`), `deploy/abliteratedllmbot.service` systemd unit.

**DoD:** `pip install -e .` then `abliteratedllmbot` boots from a fresh clone given only `.env` + `config.yaml`; `docker compose up` reaches host Ollama.

---

## Milestone 6 — v1.1 (optional, only after core is solid)

SQLite history persistence · vision routing with a real history story · `/model` switching · `/think` persisted per chat · MarkdownV2 vs rich A/B · group‑chat support (reuse pantherabot's group‑membership access model).

---

## Risk register (from the adversarial review)

| Risk | Mitigation (already in SPEC) |
|---|---|
| Stream seam specified 3 ways | ONE `Delta` type; client owns splitter+sanitizer (§4) |
| Renderer designed twice | ONE `MessageEditor`+`RenderDriver`, queue+`wait_for` flush (§8) |
| Token leak not wired into render path | `sanitize_content` in the client, before `content` deltas (§5,§6) |
| Config fragmentation | ONE frozen `Settings` schema (§12) |
| Cold load (156 s) vs timeout | `request_timeout 600`, `keep_alive 30m`, typing during the gap |
| Cyrillic token estimate | `chars_per_token 2.5`, calibrate via `prompt_eval_count` |
| Public repo leaking owner id | `config.yaml` gitignored; example only |
| Same‑chat concurrency corrupting the live edit | per‑chat registry, cancel‑and‑supersede, atomic registration |

---

## Suggested first session kickoff

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml    # then set allowed_user_ids
cp .env.example .env                  # then set TELEGRAM_BOT_TOKEN
# Milestone 1a first — it's pure and needs no bot:
pytest tests/test_reasoning.py -q
```

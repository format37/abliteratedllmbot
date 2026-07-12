"""Milestone 4 gate (offline): access control, the per-chat cancel-and-supersede
registry, and commands — driven by lightweight fake Update/Context objects and the
recording FakeBot. The live end-to-end run (real token + Ollama) is manual (PLAN M4)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from telegram.error import Forbidden

import abliteratedllmbot.telegram_bot as tb
from abliteratedllmbot.errors import Delta, OllamaUnavailableError
from abliteratedllmbot.history import InMemoryStore, Message


class FakeFile:
    def __init__(self, data):
        self._data = data

    async def download_as_bytearray(self):
        return bytearray(self._data)


class FakePhoto:
    def __init__(self, data):
        self._data = data

    async def get_file(self):
        return FakeFile(self._data)

OWNER = 106129214


# --- fake PTB Update / Context ----------------------------------------------

class FakeSentMessage:
    def __init__(self, text):
        self.text = text
        self.edits: list = []

    async def edit_text(self, text, **kw):
        self.edits.append(text)
        self.text = text


class FakeTgMessage:
    def __init__(self, text=None, caption=None, photo=None):
        self.text = text
        self.caption = caption
        self.photo = photo or []
        self.replies: list = []
        self.sent: list = []

    async def reply_text(self, text, **kw):
        self.replies.append(text)
        sent = FakeSentMessage(text)
        self.sent.append(sent)
        return sent


def make_update(user_id=OWNER, chat_id=5, text=None, caption=None, photo=None):
    msg = FakeTgMessage(text=text, caption=caption, photo=photo)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id) if user_id is not None else None,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=msg,
    )


def make_context(bot, cfg, ollama=None, history=None):
    return SimpleNamespace(
        bot=bot,
        bot_data={
            "cfg": cfg,
            "ollama": ollama,
            "history": history if history is not None else InMemoryStore(),
            "tasks": {},
        },
    )


class FakeOllama:
    """Serves scripted async delta streams, one per chat_stream() call."""

    def __init__(self, *gens, ps_result=None):
        self._gens = list(gens)
        self.calls: list = []
        self._ps = ps_result if ps_result is not None else []
        self.load_calls = 0
        self.unload_calls = 0

    def chat_stream(self, *, messages, model=None, images=None):
        self.calls.append({"messages": messages, "model": model, "images": images})
        return self._gens.pop(0)

    async def ps(self):
        return list(self._ps)

    async def load(self, model=None):
        self.load_calls += 1

    async def unload(self, model=None):
        self.unload_calls += 1


async def _astream(*deltas):
    for d in deltas:
        yield d


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


# --- access control ---------------------------------------------------------

async def test_access_allows_owner(make_cfg):
    cfg = make_cfg()
    called = {"hit": False}

    @tb.access_control(cfg)
    async def handler(update, context):
        called["hit"] = True

    upd = make_update(user_id=OWNER)
    await handler(upd, make_context(None, cfg))
    assert called["hit"] is True


async def test_access_denies_stranger_reply_once(make_cfg):
    cfg = make_cfg(telegram={"allowed_user_ids": [OWNER]}, access={"on_denied": "reply_once"})

    @tb.access_control(cfg)
    async def handler(update, context):
        raise AssertionError("handler must not run for a stranger")

    ctx = make_context(None, cfg)
    u1 = make_update(user_id=999)
    await handler(u1, ctx)
    u2 = make_update(user_id=999)
    await handler(u2, ctx)
    assert u1.effective_message.replies == [cfg.access.denial_message]
    assert u2.effective_message.replies == []  # suppressed on the second attempt


async def test_access_denies_reply_always(make_cfg):
    cfg = make_cfg(access={"on_denied": "reply_always"})

    @tb.access_control(cfg)
    async def handler(update, context):
        raise AssertionError

    ctx = make_context(None, cfg)
    u1 = make_update(user_id=1)
    u2 = make_update(user_id=1)
    await handler(u1, ctx)
    await handler(u2, ctx)
    assert u1.effective_message.replies and u2.effective_message.replies


async def test_access_ignore_is_silent(make_cfg):
    cfg = make_cfg(access={"on_denied": "ignore"})

    @tb.access_control(cfg)
    async def handler(update, context):
        raise AssertionError

    u = make_update(user_id=2)
    await handler(u, make_context(None, cfg))
    assert u.effective_message.replies == []


# --- cancel-and-supersede (SPEC §8, frozen policy) --------------------------

async def test_supersede_cancels_previous_and_keeps_latest(make_bot, make_cfg):
    cfg = make_cfg()
    bot = make_bot()
    store = InMemoryStore()

    started = asyncio.Event()

    async def slow_gen():
        yield Delta("content", text="first partial")
        started.set()
        await asyncio.sleep(3600)  # block until cancelled
        yield Delta("done", done_reason="stop")

    fast_gen = _astream(Delta("content", text="second answer"), Delta("done", done_reason="stop"))
    ctx = make_context(bot, cfg, ollama=FakeOllama(slow_gen(), fast_gen), history=store)

    tb._dispatch_generation(make_update(chat_id=5, text="q1"), ctx, user_text="q1")
    await started.wait()  # first task streamed "first partial", is now blocked
    tb._dispatch_generation(make_update(chat_id=5, text="q2"), ctx, user_text="q2")
    await _settle()

    texts = list(bot.messages.values())
    # the superseded first answer carries the note; the second answer is delivered
    assert any("first partial" in t and cfg.streaming.superseded_note in t for t in texts)
    assert any("second answer" in t for t in texts)
    # only the completed (second) turn is stored; the cancelled one is not
    assert [m.content for m in store.get(5)] == ["q2", "second answer"]
    assert 5 not in ctx.bot_data["tasks"]  # registry cleaned up


async def test_different_chats_run_independently(make_bot, make_cfg):
    # Chat 2 must NOT supersede chat 1: a supersede is per-chat only.
    cfg = make_cfg()
    bot = make_bot()
    started1 = asyncio.Event()
    release1 = asyncio.Event()

    async def slow1():
        yield Delta("content", text="A")
        started1.set()
        await release1.wait()
        yield Delta("done", done_reason="stop")

    fast2 = _astream(Delta("content", text="B"), Delta("done", done_reason="stop"))
    ctx = make_context(bot, cfg, ollama=FakeOllama(slow1(), fast2))

    tb._dispatch_generation(make_update(chat_id=1, text="x"), ctx, user_text="x")
    await started1.wait()
    tb._dispatch_generation(make_update(chat_id=2, text="y"), ctx, user_text="y")
    await _settle()
    # chat 2 finished while chat 1 is still streaming — neither blocked nor cancelled the other
    assert any("B" in t for t in bot.messages.values())
    assert 1 in ctx.bot_data["tasks"]      # chat 1 still live (not superseded by chat 2)
    assert 2 not in ctx.bot_data["tasks"]  # chat 2 completed and was cleaned up

    release1.set()
    await _settle()
    assert any("A" in t for t in bot.messages.values())
    assert 1 not in ctx.bot_data["tasks"]  # chat 1 completed and cleaned up


async def test_pre_stream_error_replies_friendly(make_bot, make_cfg):
    cfg = make_cfg()
    bot = make_bot()

    async def boom_gen():
        raise OllamaUnavailableError("Ollama is down")
        yield  # pragma: no cover

    ctx = make_context(bot, cfg, ollama=FakeOllama(boom_gen()))
    tb._dispatch_generation(make_update(chat_id=7, text="hi"), ctx, user_text="hi")
    await _settle()
    assert any("Ollama is down" in t for t in bot.messages.values())
    assert store_empty(ctx)  # nothing stored on a failed generation


def store_empty(ctx):
    return ctx.bot_data["history"].get(7) == []


# --- commands ---------------------------------------------------------------

async def test_cmd_reset_clears_only_this_chat(make_cfg):
    cfg = make_cfg()
    store = InMemoryStore()
    store.append(5, Message("user", "a"))
    store.append(9, Message("user", "b"))
    ctx = make_context(None, cfg, history=store)
    upd = make_update(chat_id=5)
    await tb.cmd_reset(upd, ctx)
    assert store.get(5) == [] and [m.content for m in store.get(9)] == ["b"]
    assert upd.effective_message.replies  # acknowledged


async def test_cmd_think_toggles_per_chat(make_cfg):
    cfg = make_cfg()
    ctx = make_context(None, cfg)
    upd = make_update(chat_id=5)
    await tb.cmd_think(upd, ctx)               # hide
    assert 5 in ctx.bot_data["_think_off"]
    assert tb._chat_cfg(ctx, 5).reasoning.show_thinking is False
    await tb.cmd_think(make_update(chat_id=5), ctx)  # show again
    assert 5 not in ctx.bot_data["_think_off"]
    assert tb._chat_cfg(ctx, 5).reasoning.show_thinking is True
    # the shared cfg is never mutated
    assert cfg.reasoning.show_thinking is True


async def test_cmd_stats_reports_tokens_per_second(make_cfg):
    cfg = make_cfg()
    ctx = make_context(None, cfg)
    ctx.bot_data["_last_stats"] = {
        5: {"stats": {"eval_count": 200, "eval_duration": 2_000_000_000,
                      "prompt_eval_count": 15}, "done_reason": "stop"}
    }
    upd = make_update(chat_id=5)
    await tb.cmd_stats(upd, ctx)
    reply = upd.effective_message.replies[0]
    assert "100.0 tok/s" in reply and "prompt: 15" in reply


async def test_cmd_stats_without_data(make_cfg):
    cfg = make_cfg()
    upd = make_update(chat_id=5)
    await tb.cmd_stats(upd, make_context(None, cfg))
    assert "No generation stats" in upd.effective_message.replies[0]


# --- image handling (SPEC §9) ----------------------------------------------

async def test_on_photo_reject_note_processes_caption(make_bot, make_cfg):
    cfg = make_cfg(images={"mode": "reject_note"})
    bot = make_bot()
    g = _astream(Delta("content", text="answer"), Delta("done", done_reason="stop"))
    ctx = make_context(bot, cfg, ollama=FakeOllama(g))
    upd = make_update(chat_id=3, caption="describe this", photo=[SimpleNamespace()])
    await tb.on_photo(upd, ctx)
    await _settle()
    assert upd.effective_message.replies == [cfg.images.note]  # the reject note
    assert ctx.bot_data["ollama"].calls  # caption was processed as a text prompt
    assert ctx.bot_data["ollama"].calls[0]["images"] is None  # image discarded


async def test_on_photo_reject_note_no_caption_only_notes(make_bot, make_cfg):
    cfg = make_cfg(images={"mode": "reject_note"})
    bot = make_bot()
    ctx = make_context(bot, cfg, ollama=FakeOllama())
    upd = make_update(chat_id=3, caption="", photo=[SimpleNamespace()])
    await tb.on_photo(upd, ctx)
    await _settle()
    assert upd.effective_message.replies == [cfg.images.note]
    assert ctx.bot_data["ollama"].calls == []  # nothing to generate


async def test_on_photo_vision_fallback_is_stateless(make_bot, make_cfg):
    # SPEC §9: route largest photo + caption to the vision model; DO NOT store the turn.
    cfg = make_cfg(images={"mode": "vision_fallback"})
    bot = make_bot()
    store = InMemoryStore()
    g = _astream(Delta("content", text="a cat"), Delta("done", done_reason="stop"))
    ctx = make_context(bot, cfg, ollama=FakeOllama(g), history=store)
    upd = make_update(chat_id=4, caption="what is this",
                      photo=[FakePhoto(b"small"), FakePhoto(b"LARGEST")])
    await tb.on_photo(upd, ctx)
    await _settle()
    call = ctx.bot_data["ollama"].calls[0]
    assert call["model"] == cfg.ollama.vision_model
    assert call["images"] == [b"LARGEST"]      # largest PhotoSize forwarded as bytes
    assert upd.effective_message.replies == []  # no reject note in vision mode
    assert store.get(4) == []                   # STATELESS: vision turn not remembered


async def test_on_photo_caption_only_with_caption(make_bot, make_cfg):
    cfg = make_cfg(images={"mode": "caption_only"})
    bot = make_bot()
    g = _astream(Delta("content", text="ans"), Delta("done", done_reason="stop"))
    ctx = make_context(bot, cfg, ollama=FakeOllama(g))
    upd = make_update(chat_id=4, caption="describe", photo=[FakePhoto(b"x")])
    await tb.on_photo(upd, ctx)
    await _settle()
    assert ctx.bot_data["ollama"].calls[0]["images"] is None  # image silently discarded
    assert upd.effective_message.replies == []                 # caption_only: no note


async def test_on_photo_caption_only_no_caption_is_silent(make_bot, make_cfg):
    cfg = make_cfg(images={"mode": "caption_only"})
    bot = make_bot()
    ctx = make_context(bot, cfg, ollama=FakeOllama())
    upd = make_update(chat_id=4, caption="", photo=[FakePhoto(b"x")])
    await tb.on_photo(upd, ctx)
    await _settle()
    assert ctx.bot_data["ollama"].calls == [] and upd.effective_message.replies == []


# --- render-path error robustness (the MAJOR M4 finding) --------------------

async def test_render_path_telegram_error_persists_turn_and_replies(make_bot, make_cfg):
    # A non-BotError TelegramError during a live edit must NOT silently kill the task:
    # surface a friendly note AND persist the produced answer (screen/memory agree).
    cfg = make_cfg()
    store = InMemoryStore()

    def on_edit(nth, kw):
        raise Forbidden("bot was blocked by the user")

    bot = make_bot(on_edit=on_edit)
    g = _astream(
        Delta("content", text="hello "),
        Delta("content", text="world and more"),  # this triggers an edit -> Forbidden
        Delta("done", done_reason="stop"),
    )
    ctx = make_context(bot, cfg, ollama=FakeOllama(g), history=store)
    tb._dispatch_generation(make_update(chat_id=8, text="q"), ctx, user_text="q")
    await _settle()
    assert any("went wrong" in t.lower() for t in bot.messages.values())  # friendly note
    stored = store.get(8)
    assert stored and stored[0].content == "q" and "hello" in stored[1].content  # turn kept
    assert 8 not in ctx.bot_data["tasks"]  # task cleaned, not leaked


# --- lifecycle --------------------------------------------------------------

async def test_on_startup_warmup_failure_is_non_fatal(make_cfg, monkeypatch):
    cfg = make_cfg()

    class FailClient:
        async def startup(self):
            raise OllamaUnavailableError("server cold")

        async def close(self):
            pass

    monkeypatch.setattr(tb, "OllamaClient", lambda *a, **k: FailClient())
    app = SimpleNamespace(bot_data={"cfg": cfg})
    await tb.on_startup(app)  # must NOT raise even though warmup failed
    assert "ollama" in app.bot_data  # client still stored for per-request use


async def test_on_shutdown_without_client_is_noop():
    await tb.on_shutdown(SimpleNamespace(bot_data={}))  # no 'ollama' key -> no error


# --- entrypoint handlers coverage -------------------------------------------

async def test_on_text_ignores_whitespace(make_bot, make_cfg):
    bot = make_bot()
    ctx = make_context(bot, make_cfg(), ollama=FakeOllama())
    await tb.on_text(make_update(chat_id=1, text="   "), ctx)
    await _settle()
    assert ctx.bot_data["ollama"].calls == []  # whitespace-only -> no generation


async def test_on_text_dispatches_real_text(make_bot, make_cfg):
    bot = make_bot()
    g = _astream(Delta("content", text="a"), Delta("done", done_reason="stop"))
    ctx = make_context(bot, make_cfg(), ollama=FakeOllama(g))
    await tb.on_text(make_update(chat_id=1, text="hello"), ctx)
    await _settle()
    assert ctx.bot_data["ollama"].calls


async def test_cmd_start_and_help_reply(make_cfg):
    for cmd in (tb.cmd_start, tb.cmd_help):
        upd = make_update(chat_id=1)
        await cmd(upd, make_context(None, make_cfg()))
        assert upd.effective_message.replies


async def test_help_omits_stats_when_disabled(make_cfg):
    upd = make_update(chat_id=1)
    await tb.cmd_help(upd, make_context(None, make_cfg(commands={"enable_stats": False})))
    assert "/stats" not in upd.effective_message.replies[0]


async def test_think_is_honest_when_globally_off(make_cfg):
    upd = make_update(chat_id=1)
    await tb.cmd_think(upd, make_context(None, make_cfg(reasoning={"show_thinking": False})))
    assert "turned off" in upd.effective_message.replies[0].lower()
    # no per-chat state was created (it can't force-show past the global setting)


# --- model-admin commands (/status /load /unload) ---------------------------

async def test_cmd_status_reports_loaded_model(make_cfg):
    import datetime

    ps = [{"model": "qwen3.6-abliterated:35b-a3b", "size_vram": 23 * 1024 ** 3,
           "expires_at": datetime.datetime(2026, 7, 12, 22, 46, 28)}]
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(ps_result=ps))
    upd = make_update(chat_id=1)
    await tb.cmd_status(upd, ctx)
    r = upd.effective_message.replies[0]
    assert "qwen3.6-abliterated:35b-a3b" in r and "23.0 GB VRAM" in r and "22:46:28" in r


async def test_cmd_status_none_loaded(make_cfg):
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(ps_result=[]))
    upd = make_update(chat_id=1)
    await tb.cmd_status(upd, ctx)
    assert "No model loaded" in upd.effective_message.replies[0]


async def test_cmd_load_warms_and_edits_placeholder(make_cfg):
    fo = FakeOllama()
    upd = make_update(chat_id=1)
    await tb.cmd_load(upd, make_context(None, make_cfg(), ollama=fo))
    assert fo.load_calls == 1
    assert "Loading" in upd.effective_message.replies[0]  # placeholder sent first
    sent = upd.effective_message.sent[0]
    assert "loaded and warm" in sent.text  # placeholder edited to success


async def test_cmd_unload_frees_model(make_cfg):
    fo = FakeOllama()
    upd = make_update(chat_id=1)
    await tb.cmd_unload(upd, make_context(None, make_cfg(), ollama=fo))
    assert fo.unload_calls == 1 and "unloaded" in upd.effective_message.replies[0]


async def test_help_lists_model_admin_when_enabled(make_cfg):
    upd = make_update(chat_id=1)
    await tb.cmd_help(upd, make_context(None, make_cfg()))
    r = upd.effective_message.replies[0]
    assert "/status" in r and "/load" in r and "/unload" in r


async def test_help_omits_model_admin_when_disabled(make_cfg):
    upd = make_update(chat_id=1)
    cfg = make_cfg(commands={"enable_model_admin": False})
    await tb.cmd_help(upd, make_context(None, cfg))
    r = upd.effective_message.replies[0]
    assert "/status" not in r and "/load" not in r and "/unload" not in r


def test_build_application_gates_model_admin_handlers(monkeypatch):
    import os
    os.environ["TELEGRAM_BOT_TOKEN"] = "1:FAKE"
    from abliteratedllmbot.config import Settings

    cfg = Settings(commands={"enable_model_admin": False})
    app = tb.build_application(cfg)
    names = {getattr(h.callback, "__name__", "") for h in app.handlers[0]}
    assert not ({"cmd_status", "cmd_load", "cmd_unload"} & names)  # not registered

    cfg2 = Settings(commands={"enable_model_admin": True})
    app2 = tb.build_application(cfg2)
    names2 = {getattr(h.callback, "__name__", "") for h in app2.handlers[0]}
    assert {"cmd_status", "cmd_load", "cmd_unload"} <= names2

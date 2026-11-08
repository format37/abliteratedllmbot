"""Milestone 4 gate (offline): access control, the per-chat cancel-and-supersede
registry, and commands — driven by lightweight fake Update/Context objects and the
recording FakeBot. The live end-to-end run (real token + Ollama) is manual (PLAN M4)."""

from __future__ import annotations

import asyncio
import sys
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

    def __init__(self, *gens, ps_result=None, models=None, load_error=None):
        self._gens = list(gens)
        self.calls: list = []
        self._ps = ps_result if ps_result is not None else []
        self._models = list(models) if models is not None else []
        self._load_error = load_error
        self.load_calls = 0
        self.unload_calls = 0
        self.loaded: list = []    # models passed to load(), in order
        self.unloaded: list = []  # models passed to unload(), in order

    def chat_stream(self, *, messages, model=None, images=None):
        self.calls.append({"messages": messages, "model": model, "images": images})
        return self._gens.pop(0)

    async def ps(self):
        return list(self._ps)

    async def chat_models(self):
        return list(self._models)

    async def load(self, model=None):
        self.load_calls += 1
        self.loaded.append(model)
        if self._load_error is not None:
            raise self._load_error

    async def unload(self, model=None):
        self.unload_calls += 1
        self.unloaded.append(model)


class FakeQuery:
    """A callback_query: records answers and in-place edits of the menu message."""

    def __init__(self, data):
        self.data = data
        self.answers: list = []
        self.edits: list = []
        self.markups: list = []

    async def answer(self, text="", show_alert=False):
        self.answers.append({"text": text, "show_alert": show_alert})

    async def edit_message_text(self, text, reply_markup=None, **kw):
        self.edits.append(text)
        self.markups.append(reply_markup)


def make_callback(data, user_id=OWNER, chat_id=5):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=FakeTgMessage(),
        callback_query=FakeQuery(data),
    )


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


async def test_interrupt_cancels_live_generation(make_bot, make_cfg):
    cfg = make_cfg()
    bot = make_bot()
    store = InMemoryStore()

    started = asyncio.Event()

    async def slow_gen():
        yield Delta("content", text="partial answer")
        started.set()
        await asyncio.sleep(3600)  # block until cancelled
        yield Delta("done", done_reason="stop")

    ctx = make_context(bot, cfg, ollama=FakeOllama(slow_gen()), history=store)

    tb._dispatch_generation(make_update(chat_id=5, text="q"), ctx, user_text="q")
    await started.wait()

    u = make_update(chat_id=5, text="/interrupt")
    await tb.cmd_interrupt(u, ctx)
    await _settle()

    # confirmed to the user; partial sealed with the INTERRUPTED note, not the supersede one
    assert any("Interrupted" in r for r in u.effective_message.replies)
    texts = list(bot.messages.values())
    assert any("partial answer" in t and cfg.streaming.interrupted_note in t for t in texts)
    assert not any(cfg.streaming.superseded_note in t for t in texts)
    assert store.get(5) == []              # the interrupted turn is not stored
    assert 5 not in ctx.bot_data["tasks"]  # registry cleaned up
    assert 5 not in ctx.bot_data.get("drivers", {})


async def test_interrupt_when_idle_says_so(make_bot, make_cfg):
    ctx = make_context(make_bot(), make_cfg(), ollama=FakeOllama())
    u = make_update(chat_id=5, text="/interrupt")
    await tb.cmd_interrupt(u, ctx)
    assert u.effective_message.replies == ["Nothing is generating right now."]


async def test_interrupt_only_stops_its_own_chat(make_bot, make_cfg):
    cfg = make_cfg()
    bot = make_bot()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_gen():
        yield Delta("content", text="other chat streaming")
        started.set()
        await release.wait()
        yield Delta("done", done_reason="stop")

    ctx = make_context(bot, cfg, ollama=FakeOllama(slow_gen()))
    tb._dispatch_generation(make_update(chat_id=1, text="x"), ctx, user_text="x")
    await started.wait()

    u = make_update(chat_id=2, text="/interrupt")  # a DIFFERENT chat interrupts
    await tb.cmd_interrupt(u, ctx)
    await _settle()
    assert u.effective_message.replies == ["Nothing is generating right now."]
    assert 1 in ctx.bot_data["tasks"]  # chat 1's generation untouched

    release.set()
    await _settle()
    assert any("other chat streaming" in t for t in bot.messages.values())


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


async def test_cmd_stats_without_data_still_reports_context(make_cfg):
    """A fresh chat has no tokenizer counts, but the window is not empty (system
    prompt) and the limit is known — report an estimate instead of nothing."""
    cfg = make_cfg()
    upd = make_update(chat_id=5)
    await tb.cmd_stats(upd, make_context(None, cfg))
    reply = upd.effective_message.replies[0]
    assert "No generation in this chat yet" in reply
    assert "Context:" in reply                      # the gauge is present ...
    assert f"/ {cfg.ollama.options.num_ctx} tokens" in reply
    assert "~" in reply and "estimated" in reply    # ... and honestly marked inexact


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

    soon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=25)
    ps = [{"model": "qwen3.6-abliterated:35b-a3b", "size_vram": 23 * 1024 ** 3,
           "expires_at": soon}]
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(ps_result=ps))
    upd = make_update(chat_id=1)
    await tb.cmd_status(upd, ctx)
    r = upd.effective_message.replies[0]
    assert "qwen3.6-abliterated:35b-a3b" in r and "23.0 GB VRAM" in r
    assert "auto-unloads in 24m" in r or "auto-unloads in 25m" in r


async def test_cmd_status_pinned_model_says_pinned(make_cfg):
    """The regression that started this: keep_alive=-1 -> year-2318 sentinel, which
    used to print as a time-of-day and read as an imminent unload."""
    import datetime

    ps = [{"model": "m", "size_vram": 1024 ** 3,
           "expires_at": datetime.datetime(2318, 10, 23, 11, 52, 15,
                                           tzinfo=datetime.timezone.utc)}]
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(ps_result=ps))
    upd = make_update(chat_id=1)
    await tb.cmd_status(upd, ctx)
    r = upd.effective_message.replies[0]
    assert "pinned in VRAM" in r and "11:52" not in r


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


# --- /model: the paged picker -----------------------------------------------

DEFAULT = "qwen3.6-abliterated:35b-a3b"   # == Settings().ollama.model
MODELS = sorted([
    DEFAULT,
    "dagbs/deepseek-coder-v2-lite-instruct:latest",
    "gpt-oss:latest",
    "llama3-groq-tool-use:8b",
    "llama3.1:8b",
    "llama3.2:latest",
    "mistral-small3.1:latest",
    "qwen3-vl:30b-a3b-instruct-q8_0",
    "qwen3.5:0.8b",
    "qwen3.5:27b",
    "qwen3.5:latest",
    "qwen3.6:35b-a3b",
])                                          # 12 models -> 3 pages at 5 per page


def rows_of(markup):
    return [[b.text for b in row] for row in markup.inline_keyboard]


def data_of(markup):
    return [[b.callback_data for b in row] for row in markup.inline_keyboard]


def render(models=MODELS, page=0, current=DEFAULT, per_page=5):
    return tb._render_model_page(models, page, current, per_page, {})


def test_single_page_has_no_nav_row():
    # Exactly per_page models: one page, so no navigation buttons at all.
    text, kb = render(models=MODELS[:5])
    assert len(rows_of(kb)) == 5 and all(len(r) == 1 for r in rows_of(kb))
    assert "page" not in text  # nothing to page through


def test_first_page_has_no_left_button():
    text, kb = render(page=0)
    nav = rows_of(kb)[-1]
    assert nav == ["1/3", "▶"]              # counter + right only
    assert len(rows_of(kb)) == 6            # 5 models + nav
    assert "page 1 of 3" in text


def test_middle_page_has_both_arrows():
    _, kb = render(page=1)
    assert rows_of(kb)[-1] == ["◀", "2/3", "▶"]


def test_last_page_has_no_right_button():
    text, kb = render(page=2)
    assert rows_of(kb)[-1] == ["◀", "3/3"]  # counter + left only
    assert len(rows_of(kb)) == 3            # 12 - 10 = 2 models + nav
    assert "page 3 of 3" in text


def test_buttons_carry_the_full_model_name():
    _, kb = render(page=0)
    labels = [r[0] for r in rows_of(kb)[:-1]]
    assert labels == MODELS[:5]             # full names, verbatim, one per row


def test_current_model_is_marked():
    page = MODELS.index(DEFAULT) // 5
    _, kb = render(page=page)
    marked = [lbl for r in rows_of(kb) for lbl in r if lbl.startswith("✅")]
    assert marked == [f"✅ {DEFAULT}"]       # marked, and still the full name


def test_page_is_clamped_to_the_ends():
    # A stale keyboard can ask for any page; it must land on a real one, not crash.
    assert "page 3 of 3" in render(page=99)[0]
    assert "page 1 of 3" in render(page=-4)[0]


def test_callback_data_stays_under_telegram_64_byte_cap():
    _, kb = render(models=MODELS, page=0)
    for row in data_of(kb):
        for d in row:
            assert len(d.encode()) <= 64


def test_token_maps_back_to_the_exact_name():
    tokens: dict = {}
    tb._render_model_page(MODELS, 0, DEFAULT, 5, tokens)
    for name in MODELS[:5]:
        assert tokens[tb._model_token(name)] == name


async def test_cmd_model_opens_on_the_page_holding_the_current_model(make_cfg):
    cfg = make_cfg()
    ctx = make_context(None, cfg, ollama=FakeOllama(models=MODELS))
    upd = make_update(chat_id=5)
    await tb.cmd_model(upd, ctx)
    text = upd.effective_message.replies[0]
    assert "page 3 of 3" in text            # DEFAULT sorts into the last page, not the first
    assert f"Current: {DEFAULT}" in text
    assert ctx.bot_data["_models"] == MODELS  # snapshot kept so paging needn't re-ask Ollama


async def test_cmd_model_reports_when_no_chat_model_exists(make_cfg):
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(models=[]))
    upd = make_update(chat_id=5)
    await tb.cmd_model(upd, ctx)
    assert "No chat-capable models" in upd.effective_message.replies[0]


async def test_page_button_re_renders_in_place(make_cfg):
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(models=MODELS))
    ctx.bot_data["_models"] = MODELS
    cb = make_callback("mdl:p:1", chat_id=5)
    await tb.on_model_button(cb, ctx)
    q = cb.callback_query
    assert q.answers == [{"text": "", "show_alert": False}]  # acked, no alert
    assert "page 2 of 3" in q.edits[0]
    assert rows_of(q.markups[0])[-1] == ["◀", "2/3", "▶"]


async def test_counter_button_is_inert_but_answered(make_cfg):
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(models=MODELS))
    cb = make_callback("mdl:noop", chat_id=5)
    await tb.on_model_button(cb, ctx)
    assert cb.callback_query.answers and cb.callback_query.edits == []


async def test_stale_page_button_after_restart_says_so(make_cfg):
    # bot_data lost its snapshot (process restarted); the old keyboard is still on screen.
    ctx = make_context(None, make_cfg(), ollama=FakeOllama(models=MODELS))
    cb = make_callback("mdl:p:1", chat_id=5)
    await tb.on_model_button(cb, ctx)
    assert "stale" in cb.callback_query.edits[0]


async def test_stale_token_alerts_instead_of_switching(make_cfg):
    fo = FakeOllama(models=MODELS)
    ctx = make_context(None, make_cfg(), ollama=fo)
    cb = make_callback("mdl:s:0123456789abcdef", chat_id=5)
    await tb.on_model_button(cb, ctx)
    assert cb.callback_query.answers[0]["show_alert"] is True
    assert fo.load_calls == 0 and fo.unload_calls == 0


def _pick(name, chat_id=5):
    return make_callback(f"mdl:s:{tb._model_token(name)}", chat_id=chat_id)


def _ctx_with_menu(make_cfg, fo, cfg=None):
    """A context whose token map is populated, as it would be after /model."""
    ctx = make_context(None, cfg or make_cfg(), ollama=fo)
    ctx.bot_data["_models"] = MODELS
    tokens: dict = {}
    for page in range(3):
        tb._render_model_page(MODELS, page, DEFAULT, 5, tokens)
    ctx.bot_data["_model_tokens"] = tokens
    return ctx


async def test_select_unloads_old_then_warms_new_and_switches(make_cfg):
    cfg = make_cfg()
    new = "gpt-oss:latest"
    fo = FakeOllama(models=MODELS, ps_result=[{"model": DEFAULT, "size_vram": 1}])
    ctx = _ctx_with_menu(make_cfg, fo, cfg)

    cb = _pick(new)
    await tb.on_model_button(cb, ctx)

    assert fo.unloaded == [DEFAULT]        # the old model was evicted first ...
    assert fo.loaded == [new]              # ... then the new one warmed
    assert tb.active_model(ctx, 5) == new
    assert tb._chat_cfg(ctx, 5).ollama.model == new   # generations follow the choice
    assert cfg.ollama.model == DEFAULT                # shared Settings never mutated
    assert "Now using" in cb.callback_query.edits[-1]


async def test_select_skips_the_unload_when_old_is_not_resident(make_cfg):
    fo = FakeOllama(models=MODELS, ps_result=[])       # nothing in VRAM
    ctx = _ctx_with_menu(make_cfg, fo)
    await tb.on_model_button(_pick("gpt-oss:latest"), ctx)
    assert fo.unloaded == [] and fo.loaded == ["gpt-oss:latest"]


async def test_select_is_declined_while_a_generation_runs(make_cfg):
    fo = FakeOllama(models=MODELS, ps_result=[{"model": DEFAULT, "size_vram": 1}])
    ctx = _ctx_with_menu(make_cfg, fo)

    async def forever():
        await asyncio.sleep(3600)

    live = asyncio.create_task(forever())
    ctx.bot_data["tasks"][9] = live        # a DIFFERENT chat is mid-stream
    try:
        cb = _pick("gpt-oss:latest")
        await tb.on_model_button(cb, ctx)
        # unloading now would have killed chat 9's answer mid-flight
        assert fo.unload_calls == 0 and fo.load_calls == 0
        assert tb.active_model(ctx, 5) == DEFAULT
        assert "still running" in cb.callback_query.edits[-1]
    finally:
        live.cancel()


async def test_select_keeps_the_old_model_when_the_new_one_fails_to_load(make_cfg):
    from abliteratedllmbot.errors import ModelNotFoundError

    fo = FakeOllama(models=MODELS, ps_result=[{"model": DEFAULT, "size_vram": 1}],
                    load_error=ModelNotFoundError("Model 'gpt-oss:latest' is not available."))
    ctx = _ctx_with_menu(make_cfg, fo)
    cb = _pick("gpt-oss:latest")
    await tb.on_model_button(cb, ctx)

    assert tb.active_model(ctx, 5) == DEFAULT          # a switch that never loaded is no switch
    last = cb.callback_query.edits[-1]
    assert "not available" in last and f"Still using {DEFAULT}" in last


async def test_selecting_the_current_model_is_a_noop(make_cfg):
    fo = FakeOllama(models=MODELS, ps_result=[{"model": DEFAULT, "size_vram": 1}])
    ctx = _ctx_with_menu(make_cfg, fo)
    cb = _pick(DEFAULT)
    await tb.on_model_button(cb, ctx)
    assert fo.load_calls == 0 and fo.unload_calls == 0
    assert "Already using" in cb.callback_query.edits[-1]


async def test_switch_is_per_chat(make_cfg):
    fo = FakeOllama(models=MODELS, ps_result=[])
    ctx = _ctx_with_menu(make_cfg, fo)
    await tb.on_model_button(_pick("gpt-oss:latest", chat_id=5), ctx)
    assert tb.active_model(ctx, 5) == "gpt-oss:latest"
    assert tb.active_model(ctx, 6) == DEFAULT          # another chat keeps the default


async def test_generation_uses_the_chats_selected_model(make_bot, make_cfg):
    bot = make_bot()
    g = _astream(Delta("content", text="hi"), Delta("done", done_reason="stop"))
    fo = FakeOllama(g, models=MODELS)
    ctx = make_context(bot, make_cfg(), ollama=fo)
    ctx.bot_data["_model"] = {5: "gpt-oss:latest"}
    tb._dispatch_generation(make_update(chat_id=5, text="q"), ctx, user_text="q")
    await _settle()
    assert fo.calls[0]["model"] == "gpt-oss:latest"


async def test_vision_override_still_beats_the_chat_choice(make_bot, make_cfg):
    # An explicit per-turn model (the vision path) is deliberate and must still win.
    cfg = make_cfg(images={"mode": "vision_fallback"})
    bot = make_bot()
    g = _astream(Delta("content", text="a cat"), Delta("done", done_reason="stop"))
    fo = FakeOllama(g, models=MODELS)
    ctx = make_context(bot, cfg, ollama=fo)
    ctx.bot_data["_model"] = {4: "gpt-oss:latest"}
    await tb.on_photo(make_update(chat_id=4, caption="what", photo=[FakePhoto(b"x")]), ctx)
    await _settle()
    assert fo.calls[0]["model"] == cfg.ollama.vision_model


async def test_model_admin_commands_follow_the_chat_choice(make_cfg):
    fo = FakeOllama()
    ctx = make_context(None, make_cfg(), ollama=fo)
    ctx.bot_data["_model"] = {5: "gpt-oss:latest"}
    await tb.cmd_load(make_update(chat_id=5), ctx)
    await tb.cmd_unload(make_update(chat_id=5), ctx)
    assert fo.loaded == ["gpt-oss:latest"] and fo.unloaded == ["gpt-oss:latest"]


async def test_a_second_tap_during_a_switch_is_declined(make_cfg):
    # Two taps land while the first is parked in a ~2.5 min load. Without a claim taken
    # before the first await, BOTH would unload+load: two models resident and the chat
    # left on whichever load finished last, not the one tapped.
    loading = asyncio.Event()
    release = asyncio.Event()

    class SlowOllama(FakeOllama):
        async def load(self, model=None):
            self.load_calls += 1
            self.loaded.append(model)
            loading.set()
            await release.wait()

    fo = SlowOllama(models=MODELS, ps_result=[{"model": DEFAULT, "size_vram": 1}])
    ctx = _ctx_with_menu(make_cfg, fo)

    first = asyncio.create_task(tb.on_model_button(_pick("gpt-oss:latest"), ctx))
    await loading.wait()                       # switch #1 is mid-load

    second = _pick("llama3.2:latest")          # user taps another model
    # wait_for, not a bare await: an unguarded second switch would block in load() until
    # the first is released, so the bug must surface as a failure, not a hung test.
    await asyncio.wait_for(tb.on_model_button(second, ctx), timeout=2)
    assert "Already switching to gpt-oss:latest" in second.callback_query.edits[-1]
    assert fo.load_calls == 1                  # the second switch never started

    release.set()
    await first
    assert fo.loaded == ["gpt-oss:latest"] and fo.unloaded == [DEFAULT]
    assert tb.active_model(ctx, 5) == "gpt-oss:latest"   # the model actually tapped
    assert ctx.bot_data["_switching"] == {}              # claim released


async def test_a_message_sent_during_a_switch_targets_the_new_model(make_bot, make_cfg):
    # The user taps a model, then types while staring at "warming …". That turn must go
    # to the model being warmed — resolving it to the old one would cold-load the model
    # we just evicted, right back beside the new one.
    loading = asyncio.Event()
    release = asyncio.Event()

    class SlowOllama(FakeOllama):
        async def load(self, model=None):
            self.load_calls += 1
            self.loaded.append(model)
            loading.set()
            await release.wait()

    g = _astream(Delta("content", text="hi"), Delta("done", done_reason="stop"))
    fo = SlowOllama(g, models=MODELS, ps_result=[{"model": DEFAULT, "size_vram": 1}])
    ctx = _ctx_with_menu(make_cfg, fo)
    ctx.bot = make_bot()

    switch = asyncio.create_task(tb.on_model_button(_pick("gpt-oss:latest"), ctx))
    await loading.wait()

    tb._dispatch_generation(make_update(chat_id=5, text="what's up"), ctx,
                            user_text="what's up")
    await _settle()
    assert fo.calls[0]["model"] == "gpt-oss:latest"   # NOT the just-unloaded DEFAULT

    release.set()
    await switch


async def test_a_failed_load_rolls_the_claim_back(make_cfg):
    from abliteratedllmbot.errors import ModelNotFoundError

    fo = FakeOllama(models=MODELS, ps_result=[{"model": DEFAULT, "size_vram": 1}],
                    load_error=ModelNotFoundError("gone"))
    ctx = _ctx_with_menu(make_cfg, fo)
    ctx.bot_data["_model"] = {5: "qwen3.5:27b"}       # this chat had already switched once
    await tb.on_model_button(_pick("gpt-oss:latest"), ctx)
    assert tb.active_model(ctx, 5) == "qwen3.5:27b"   # rolled back to the PREVIOUS choice
    assert ctx.bot_data["_switching"] == {}           # and the claim is not stuck


# --- a bare config model name must still match what Ollama reports ------------

def test_qualify_tags_a_bare_name_the_way_ollama_reports_it():
    assert tb._qualify("gemma3") == "gemma3:latest"        # /api/tags says gemma3:latest
    assert tb._qualify("qwen3.5:27b") == "qwen3.5:27b"     # already tagged -> untouched
    assert tb._qualify("dagbs/deepseek:latest") == "dagbs/deepseek:latest"
    assert tb._qualify("dagbs/deepseek") == "dagbs/deepseek:latest"  # ':' only in the tail


async def test_bare_config_name_still_marks_and_unloads(make_cfg):
    # config `ollama.model: llama3.2` — legal, and `ollama run llama3.2` works — but
    # /api/tags and /api/ps both say "llama3.2:latest". Compared raw, the ✅ never renders
    # and `old in resident` is false, so the old model is NEVER unloaded and both stay
    # pinned in VRAM: the exact outcome unload-then-load exists to prevent.
    cfg = make_cfg(ollama={"model": "llama3.2"})
    fo = FakeOllama(models=MODELS, ps_result=[{"model": "llama3.2:latest", "size_vram": 1}])
    ctx = make_context(None, cfg, ollama=fo)

    upd = make_update(chat_id=5)
    await tb.cmd_model(upd, ctx)
    text = upd.effective_message.replies[0]
    assert "Current: llama3.2:latest" in text
    page = MODELS.index("llama3.2:latest") // 5
    assert f"page {page + 1} of 3" in text            # opened on the page holding it

    ctx.bot_data["_models"] = MODELS
    await tb.on_model_button(_pick("gpt-oss:latest"), ctx)
    assert fo.unloaded == ["llama3.2:latest"]         # the old model really was evicted
    assert fo.loaded == ["gpt-oss:latest"]


async def test_denied_button_press_is_answered_not_left_spinning(make_cfg):
    cfg = make_cfg(access={"on_denied": "reply_always"})
    guarded = tb.access_control(cfg)(tb.on_model_button)
    cb = make_callback("mdl:p:1", user_id=999)   # a stranger
    await guarded(cb, make_context(None, cfg))
    q = cb.callback_query
    assert q.answers == [{"text": cfg.access.denial_message, "show_alert": True}]
    assert q.edits == []                          # and the menu is untouched


async def test_denied_button_press_stays_silent_under_ignore(make_cfg):
    cfg = make_cfg(access={"on_denied": "ignore"})
    guarded = tb.access_control(cfg)(tb.on_model_button)
    cb = make_callback("mdl:p:1", user_id=999)
    await guarded(cb, make_context(None, cfg))
    # acked (no spinner) but told nothing
    assert cb.callback_query.answers == [{"text": "", "show_alert": False}]


async def test_help_lists_model_only_when_enabled(make_cfg):
    on = make_update(chat_id=1)
    await tb.cmd_help(on, make_context(None, make_cfg()))
    assert "/model" in on.effective_message.replies[0]

    off = make_update(chat_id=1)
    cfg = make_cfg(commands={"allow_model_switch": False})
    await tb.cmd_help(off, make_context(None, cfg))
    assert "/model" not in off.effective_message.replies[0]


def test_polling_subscribes_to_the_updates_our_handlers_need():
    """The bug this pins: run_polling asked for allowed_updates=["message"], so Telegram
    never DELIVERED callback_query at all. The picker rendered, the handler was registered
    — and every button tap silently did nothing, with no error and no log line. Subscribing
    is derived from the handlers so a keyboard can never again go unsubscribed."""
    import os
    os.environ["TELEGRAM_BOT_TOKEN"] = "1:FAKE"
    from telegram import Update
    from abliteratedllmbot.config import Settings

    on = tb.build_application(Settings(commands={"allow_model_switch": True}))
    assert Update.CALLBACK_QUERY in tb.allowed_updates(on)   # buttons are actually delivered
    assert Update.MESSAGE in tb.allowed_updates(on)

    off = tb.build_application(Settings(commands={"allow_model_switch": False}))
    assert tb.allowed_updates(off) == [Update.MESSAGE]       # no keyboards -> nothing to ask for


def test_entrypoint_passes_the_derived_allowed_updates(monkeypatch):
    """…and the entrypoint must actually USE it — asserting on allowed_updates() alone
    would still have passed while __main__ hard-coded ["message"]."""
    import os
    from telegram import Update
    import abliteratedllmbot.__main__ as entry

    os.environ["TELEGRAM_BOT_TOKEN"] = "1:FAKE"
    seen: dict = {}

    class FakeApp:
        handlers = {0: []}

        def run_polling(self, **kw):
            seen.update(kw)

    app = tb.build_application(__import__(
        "abliteratedllmbot.config", fromlist=["Settings"]).Settings(
            commands={"allow_model_switch": True}))
    fake = FakeApp()
    fake.handlers = app.handlers
    monkeypatch.setattr(entry, "build_application", lambda cfg: fake)
    monkeypatch.setattr(sys, "argv", ["prog"])
    entry.main()
    assert Update.CALLBACK_QUERY in seen["allowed_updates"]


def test_build_application_gates_the_model_picker():
    import os
    os.environ["TELEGRAM_BOT_TOKEN"] = "1:FAKE"
    from abliteratedllmbot.config import Settings

    off = tb.build_application(Settings(commands={"allow_model_switch": False}))
    names = {getattr(h.callback, "__name__", "") for h in off.handlers[0]}
    assert not ({"cmd_model", "on_model_button"} & names)

    on = tb.build_application(Settings(commands={"allow_model_switch": True}))
    names = {getattr(h.callback, "__name__", "") for h in on.handlers[0]}
    assert {"cmd_model", "on_model_button"} <= names


# --- /prompt: per-chat system prompt (SPEC §11) -----------------------------

def _gen_ok():
    return _astream(Delta("content", text="ok"), Delta("done", done_reason="stop"))


async def test_prompt_sets_the_system_prompt_for_the_next_turn(make_bot, make_cfg):
    fo = FakeOllama(_gen_ok())
    ctx = make_context(make_bot(), make_cfg(), ollama=fo)
    u = make_update(chat_id=5, text="/prompt you are helpful assistant")
    await tb.cmd_prompt(u, ctx)
    assert "System prompt set" in u.effective_message.replies[0]

    tb._dispatch_generation(make_update(chat_id=5, text="q"), ctx, user_text="q")
    await _settle()
    assert fo.calls[0]["messages"][0] == {
        "role": "system", "content": "you are helpful assistant"}


async def test_prompt_keeps_newlines_and_strips_bot_mention(make_bot, make_cfg):
    ctx = make_context(make_bot(), make_cfg(), ollama=FakeOllama())
    await tb.cmd_prompt(
        make_update(chat_id=5, text="/prompt@MyBot  line one\nline two  "), ctx)
    assert ctx.bot_data["_system_prompt"][5] == "line one\nline two"


async def test_prompt_without_args_shows_default_then_custom(make_cfg):
    cfg = make_cfg()
    ctx = make_context(None, cfg)
    u = make_update(chat_id=5, text="/prompt")
    await tb.cmd_prompt(u, ctx)
    assert "default" in u.effective_message.replies[0]
    assert cfg.system_prompt.strip() in u.effective_message.replies[0]

    await tb.cmd_prompt(make_update(chat_id=5, text="/prompt be terse"), ctx)
    u = make_update(chat_id=5, text="/prompt")
    await tb.cmd_prompt(u, ctx)
    assert "custom" in u.effective_message.replies[0]
    assert "be terse" in u.effective_message.replies[0]


async def test_prompt_reset_restores_the_default(make_bot, make_cfg):
    cfg = make_cfg()
    fo = FakeOllama(_gen_ok())
    ctx = make_context(make_bot(), cfg, ollama=fo)
    await tb.cmd_prompt(make_update(chat_id=5, text="/prompt be a pirate"), ctx)
    u = make_update(chat_id=5, text="/prompt RESET")
    await tb.cmd_prompt(u, ctx)
    assert "restored" in u.effective_message.replies[0]
    assert 5 not in ctx.bot_data["_system_prompt"]

    tb._dispatch_generation(make_update(chat_id=5, text="q"), ctx, user_text="q")
    await _settle()
    assert fo.calls[0]["messages"][0]["content"] == cfg.system_prompt


async def test_prompt_is_per_chat_and_keeps_memory(make_bot, make_cfg):
    cfg = make_cfg()
    store = InMemoryStore()
    store.append(5, Message("user", "earlier"))
    store.append(5, Message("assistant", "reply"))
    fo = FakeOllama(_gen_ok())
    ctx = make_context(make_bot(), cfg, ollama=fo, history=store)
    await tb.cmd_prompt(make_update(chat_id=5, text="/prompt be a pirate"), ctx)
    assert len(store.get(5)) == 2                       # memory kept

    tb._dispatch_generation(make_update(chat_id=6, text="q"), ctx, user_text="q")
    await _settle()
    assert fo.calls[0]["messages"][0]["content"] == cfg.system_prompt  # chat 6 untouched


async def test_prompt_echo_is_capped_below_telegram_limit(make_cfg):
    ctx = make_context(None, make_cfg())
    u = make_update(chat_id=5, text="/prompt " + "x" * 4090)
    await tb.cmd_prompt(u, ctx)
    assert len(ctx.bot_data["_system_prompt"][5]) == 4090  # stored in full
    assert len(u.effective_message.replies[0]) <= 4096     # but the echo fits a message


async def test_stats_estimate_counts_the_chat_prompt(make_cfg):
    cfg = make_cfg()
    ctx = make_context(None, cfg)
    ctx.bot_data["_system_prompt"] = {5: "x" * 4000}
    u = make_update(chat_id=5)
    await tb.cmd_stats(u, ctx)
    default_u = make_update(chat_id=6)
    await tb.cmd_stats(default_u, ctx)
    assert u.effective_message.replies[0] != default_u.effective_message.replies[0]

"""Milestone 3 gate: the live-edit render state machine, driven by scripted Delta
streams against the recording FakeBot. NO real Telegram. (PLAN M3 DoD)"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest, RetryAfter

import abliteratedllmbot.streaming as streaming
from abliteratedllmbot.errors import Delta
from abliteratedllmbot.streaming import MessageEditor, RenderDriver


async def _stream(*deltas):
    for d in deltas:
        yield d


async def _run(bot, cfg, *deltas):
    driver = RenderDriver(bot, chat_id=42, cfg=cfg)
    await driver.run(_stream(*deltas))
    return driver


# --- ordering: thinking then answer -----------------------------------------

async def test_thinking_then_answer_ordering(make_bot, make_cfg):
    bot = make_bot()
    driver = await _run(
        bot, make_cfg(),
        Delta("thinking", text="let me think"),
        Delta("content", text="the answer"),
        Delta("done", done_reason="stop"),
    )
    sends = bot.calls_of("send")
    # first message is the thinking bubble (💭 prefix), then the answer message
    assert sends[0]["text"].startswith("💭")
    thinking_id = min(bot.messages) if bot.messages else None
    # thinking message created before the answer message
    assert any("💭" in t for t in bot.messages.values())
    assert driver.answer_text == "the answer"
    assert driver.answer_message_ids  # answer message tracked
    # the thinking id precedes the answer id (creation order)
    assert min(driver.answer_message_ids) > 1000


async def test_no_thinking_when_show_thinking_false(make_bot, make_cfg):
    bot = make_bot()
    await _run(
        bot, make_cfg(reasoning={"show_thinking": False}),
        Delta("thinking", text="hidden"),
        Delta("content", text="visible"),
        Delta("done", done_reason="stop"),
    )
    assert not any("💭" in kw.get("text", "") for _, kw in bot.calls)
    assert not any("hidden" in kw.get("text", "") for _, kw in bot.calls)


async def test_thinking_cap_truncates(make_bot, make_cfg):
    bot = make_bot()
    await _run(
        bot, make_cfg(reasoning={"max_thinking_chars": 10}),
        Delta("thinking", text="x" * 50),
        Delta("content", text="a"),
        Delta("done", done_reason="stop"),
    )
    think_msgs = [t for t in bot.messages.values() if "💭" in t]
    assert think_msgs
    # capped: prefix + <=10 x's + " …"
    assert think_msgs[0].count("x") == 10 and "…" in think_msgs[0]


# --- throttling -------------------------------------------------------------

async def test_throttling_far_fewer_edits_than_deltas(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(streaming={"min_edit_interval": 1000, "batch_size": 100})
    deltas = [Delta("content", text="x" * 10) for _ in range(50)]  # 500 chars
    await _run(bot, cfg, *deltas, Delta("done", done_reason="stop"))
    telegram_writes = len(bot.calls_of("send")) + len(bot.calls_of("edit"))
    # 50 content deltas -> a handful of writes (first send + ~every 100 chars + final)
    assert telegram_writes < 12
    # but the full text is shown at the end
    assert bot.messages[max(bot.messages)].endswith("x")


# --- rollover ---------------------------------------------------------------

async def test_rollover_splits_on_whitespace(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(streaming={"rollover_chars": 20, "min_edit_interval": 0.0})
    text = "word " * 12  # 60 chars, must roll into multiple <=20-char messages
    await _run(bot, cfg, Delta("content", text=text), Delta("done", done_reason="stop"))
    live = [t for mid, t in sorted(bot.messages.items())]
    assert len(live) >= 2  # rolled over
    for t in live:
        assert len(t) <= 20  # each within the rollover budget
        # split on whitespace: never a truncated 'wor'/'wo' fragment
        for token in t.split():
            assert token == "word"


async def test_rollover_reassembles_all_words(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(streaming={"rollover_chars": 25})
    text = " ".join(f"w{i}" for i in range(30))
    await _run(bot, cfg, Delta("content", text=text), Delta("done", done_reason="stop"))
    seen = " ".join(t for _, t in sorted(bot.messages.items())).split()
    assert seen == text.split()  # no word lost or duplicated across the split


# --- empty answer -----------------------------------------------------------

async def test_empty_answer_shows_note(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg()
    await _run(bot, cfg, Delta("done", done_reason="stop"))
    assert list(bot.messages.values()) == [cfg.streaming.empty_response_text]


# --- mid-stream error: partial + note ---------------------------------------

async def test_mid_stream_error_shows_partial_and_note(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg()
    await _run(
        bot, cfg,
        Delta("content", text="partial answer"),
        Delta("error", error="connection lost"),
    )
    final = bot.messages[max(bot.messages)]
    assert "partial answer" in final
    assert cfg.streaming.error_note.format(error="connection lost") in final


async def test_error_before_content_shows_note_only(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg()
    await _run(bot, cfg, Delta("error", error="boom"))
    assert list(bot.messages.values()) == [cfg.streaming.error_note.format(error="boom")]


# --- MessageEditor telegram-error handling ----------------------------------

async def test_editor_retryafter_backs_off(make_bot, make_cfg, monkeypatch):
    slept: list = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(streaming.asyncio, "sleep", fake_sleep)

    def on_send(nth, kw):
        if nth == 1:
            raise RetryAfter(retry_after=2)

    bot = make_bot(on_send=on_send)
    cfg = make_cfg(streaming={"min_edit_interval": 1.2, "max_retry_after": 30})
    ed = MessageEditor(bot, 1, cfg)
    await ed.feed("hello")       # first send raises RetryAfter -> slept, retried, interval grows
    await ed.feed(" world")      # throttled (within the grown interval)
    await ed.finalize()          # terminal flush shows the buffered tail
    assert slept == [2]          # min(retry_after=2, max_retry_after=30)
    assert ed._effective_interval == pytest.approx(1.2 * 1.5)
    assert bot.messages[max(bot.messages)] == "hello world"  # no text lost to the 429


async def test_retryafter_on_terminal_flush_is_retried_not_dropped(make_bot, make_cfg, monkeypatch):
    # The MAJOR bug: a 429 on a forced/terminal flush must retry, not silently drop
    # the buffered answer tail.
    slept: list = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(streaming.asyncio, "sleep", fake_sleep)

    def on_edit(nth, kw):
        if nth == 1:  # the finalize edit hits a 429 the first time
            raise RetryAfter(retry_after=3)

    bot = make_bot(on_edit=on_edit)
    cfg = make_cfg(streaming={"min_edit_interval": 1000, "batch_size": 1000})
    ed = MessageEditor(bot, 1, cfg)
    await ed.feed("start")       # first send lands "start"
    await ed.feed(" and end")    # buffered (throttled)
    await ed.finalize()          # forced flush: edit 429 -> retry -> success
    assert slept == [3]
    assert bot.messages[max(bot.messages)] == "start and end"  # tail NOT dropped


async def test_editor_message_not_found_aborts_editing(make_bot, make_cfg):
    def on_edit(nth, kw):
        raise BadRequest("Message to edit not found")

    bot = make_bot(on_edit=on_edit)
    cfg = make_cfg(streaming={"min_edit_interval": 0.0})
    ed = MessageEditor(bot, 1, cfg)
    await ed.feed("first")       # send ok -> message created
    await ed.feed("-second")     # edit raises not-found -> editor marks deleted
    await ed.feed("-third")      # no further writes attempted
    assert ed._deleted is True
    edits_after_delete = [c for c in bot.calls if c[0] == "edit"]
    assert len(edits_after_delete) == 1  # only the one that failed; no retries


async def test_editor_not_modified_is_ignored(make_bot, make_cfg):
    def on_edit(nth, kw):
        raise BadRequest("message is not modified")

    bot = make_bot(on_edit=on_edit)
    cfg = make_cfg(streaming={"min_edit_interval": 0.0})
    ed = MessageEditor(bot, 1, cfg)
    await ed.feed("a")
    await ed.feed("b")           # edit -> 'not modified' -> swallowed, not deleted
    assert ed._deleted is False


# --- trailing-edge flush (max_interval_seconds) -----------------------------

async def test_tick_flushes_buffered_trailing_text(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(streaming={"min_edit_interval": 1000, "batch_size": 1000})
    ed = MessageEditor(bot, 1, cfg)
    await ed.feed("hi")        # first feed flushes immediately
    await ed.feed(" more")     # sub-batch + within interval -> stays buffered
    assert bot.messages[max(bot.messages)] == "hi"
    await ed.tick()            # trailing-edge flush
    assert bot.messages[max(bot.messages)] == "hi more"


async def test_heartbeat_flushes_during_stream_pause(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(
        streaming={"min_edit_interval": 1000, "batch_size": 1000, "max_interval_seconds": 0.05}
    )
    driver = RenderDriver(bot, 1, cfg)
    resume = asyncio.Event()

    async def gen():
        yield Delta("content", text="slow")   # sent immediately (first flush)
        yield Delta("content", text=" tail")  # throttled -> buffered
        await resume.wait()                    # model pauses mid-generation
        yield Delta("done", done_reason="stop")

    task = asyncio.create_task(driver.run(gen()))
    await asyncio.sleep(0.15)  # a heartbeat tick (every 0.05s) should flush the buffered tail
    assert bot.messages[max(bot.messages)] == "slow tail"
    resume.set()
    await task


# --- finalize handoff (SPEC §8/§10: once on success, never on error/empty) --

async def test_finalize_called_once_on_success(make_bot, make_cfg, monkeypatch):
    calls: list = []

    async def spy(bot, cfg, chat_id, full_text, ids):
        calls.append((full_text, ids))

    monkeypatch.setattr(streaming.formatting, "finalize", spy)
    bot = make_bot()
    await _run(
        bot, make_cfg(formatting={"finalize_mode": "markdown_v2"}),
        Delta("content", text="the answer"),
        Delta("done", done_reason="stop"),
    )
    assert len(calls) == 1 and calls[0][0] == "the answer"


@pytest.mark.parametrize(
    "deltas",
    [
        [Delta("content", text="partial"), Delta("error", error="x")],  # mid-stream error
        [Delta("error", error="x")],                                    # error before content
        [Delta("done", done_reason="stop")],                           # empty answer
    ],
)
async def test_finalize_not_called_on_error_or_empty(make_bot, make_cfg, monkeypatch, deltas):
    calls: list = []

    async def spy(*a):
        calls.append(a)

    monkeypatch.setattr(streaming.formatting, "finalize", spy)
    bot = make_bot()
    await _run(bot, make_cfg(formatting={"finalize_mode": "markdown_v2"}), *deltas)
    assert calls == []  # error/partial/empty stays plain — no rich finalize


# --- cancel-and-supersede ---------------------------------------------------

async def test_cancel_appends_superseded_note(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg()
    driver = RenderDriver(bot, 1, cfg)
    started = asyncio.Event()

    async def gen():
        yield Delta("content", text="partial")
        started.set()
        await asyncio.sleep(3600)  # block until cancelled
        yield Delta("done", done_reason="stop")

    task = asyncio.create_task(driver.run(gen()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    final = bot.messages[driver.answer_message_ids[0]]
    assert "partial" in final and cfg.streaming.superseded_note in final


async def test_cancel_shield_survives_second_cancellation(make_cfg):
    # The asyncio.shield is load-bearing only under a SECOND cancel arriving while the
    # superseded-note edit is in flight — assert the note still lands and run() still
    # raises CancelledError. (SPEC §8 cancel-and-supersede)
    cfg = make_cfg()
    edit_started = asyncio.Event()
    release_edit = asyncio.Event()

    class BlockingBot:
        def __init__(self):
            self.messages: dict = {}
            self._id = 500

        async def send_message(self, chat_id, text, **kw):
            self._id += 1
            self.messages[self._id] = text
            return SimpleNamespace(message_id=self._id)

        async def edit_message_text(self, text, chat_id, message_id, **kw):
            if not edit_started.is_set():
                edit_started.set()
                await release_edit.wait()  # hold the note edit open for the 2nd cancel
            self.messages[message_id] = text
            return SimpleNamespace(message_id=message_id)

        async def delete_message(self, *a, **k):
            return True

    bot = BlockingBot()
    driver = RenderDriver(bot, 1, cfg)
    started = asyncio.Event()

    async def gen():
        yield Delta("content", text="partial")
        started.set()
        await asyncio.sleep(3600)
        yield Delta("done", done_reason="stop")

    task = asyncio.create_task(driver.run(gen()))
    await started.wait()
    task.cancel()                     # supersede -> _on_cancel -> shielded note edit blocks
    await edit_started.wait()
    task.cancel()                     # SECOND cancel while the shielded edit is in flight
    release_edit.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.02)         # let the shielded edit settle in the background
    assert any(cfg.streaming.superseded_note in t for t in bot.messages.values())

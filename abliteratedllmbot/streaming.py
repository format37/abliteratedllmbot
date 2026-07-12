"""The live-edit streaming state machine — the crux feature. See SPEC.md §8.

Consumes an AsyncIterator[Delta] and drives one or two live-edited Telegram
messages (a lazily-created 'thinking' message, then the answer), throttled and
rolling over near the 4096 limit, handing the final text to formatting.finalize.

Designed to be tested against a recording FakeBot with scripted Delta streams —
NO real Telegram needed for Milestone 3.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Awaitable, Callable, List, Optional, Protocol

from . import formatting
from .config import Settings
from .errors import Delta

logger = logging.getLogger(__name__)

_THINK_PREFIX = "💭 "


class BotLike(Protocol):
    """The subset of telegram.Bot the renderer uses (so a FakeBot can stand in)."""

    async def send_message(self, chat_id: int, text: str, **kw): ...
    async def edit_message_text(self, text: str, chat_id: int, message_id: int, **kw): ...
    async def delete_message(self, chat_id: int, message_id: int) -> bool: ...


def _split_point(text: str, budget: int) -> int:
    """Index to cut ``text`` at so the head fits ``budget`` chars, preferring the
    last newline, then the last space, then a hard cut."""
    if len(text) <= budget:
        return len(text)
    window = text[:budget]
    nl = window.rfind("\n")
    if nl > 0:
        return nl
    sp = window.rfind(" ")
    if sp > 0:
        return sp
    return budget


class MessageEditor:
    """Owns exactly one live Telegram message (spilling into more on rollover);
    buffers text and flushes throttled edits, rolling over near ``rollover_chars``.
    (SPEC §8)

    Flush when: chars_since_edit >= batch_size  OR  now-last_edit >= effective_interval.
    ``_effective_interval`` is a LOCAL that grows ×1.5 on RetryAfter — never mutate Settings.
    """

    def __init__(self, bot: "BotLike", chat_id: int, cfg: Settings, prefix: str = "") -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.cfg = cfg
        self.s = cfg.streaming
        self.prefix = prefix
        self.message_ids: List[int] = []

        self._seg_index = 0          # which rollover segment is active (prefix on seg 0 only)
        self._cur_id: Optional[int] = None
        self._cur_text = ""          # content of the active message (sans prefix)
        self._shown = ""             # text currently displayed in the active message
        self._chars_since_edit = 0
        self._last_edit = 0.0        # monotonic; 0 -> first flush fires immediately
        self._effective_interval = self.s.min_edit_interval
        self._deleted = False        # user deleted the message -> stop editing
        self._lock = asyncio.Lock()  # serialize flushes (feed vs trailing-edge tick)

    # -- public API ----------------------------------------------------------

    async def feed(self, text: str) -> None:
        if not text:
            return
        self._cur_text += text
        self._chars_since_edit += len(text)
        await self._maybe_rollover()
        if self._should_flush():
            await self._flush()

    async def finalize(self) -> None:
        """Final flush of any buffered text (unthrottled)."""
        await self._maybe_rollover()
        await self._flush(force=True)

    async def tick(self) -> None:
        """Trailing-edge flush (SPEC §8): emit buffered sub-batch text that has been
        sitting unsent during a stream pause. Driven by RenderDriver's heartbeat at
        ``max_interval_seconds``."""
        await self._flush(force=True)

    # -- internals -----------------------------------------------------------

    def _active_prefix(self) -> str:
        return self.prefix if self._seg_index == 0 else ""

    def _should_flush(self) -> bool:
        if self._chars_since_edit >= self.s.batch_size:
            return True
        return (time.monotonic() - self._last_edit) >= self._effective_interval

    async def _maybe_rollover(self) -> None:
        budget = self.s.rollover_chars - len(self._active_prefix())
        while len(self._cur_text) > budget:
            cut = _split_point(self._cur_text, budget)
            head, tail = self._cur_text[:cut].rstrip(), self._cur_text[cut:].lstrip(" \n")
            self._cur_text = head
            await self._flush(force=True)          # seal the active message with `head`
            # open a fresh active message for the overflow
            self._seg_index += 1
            self._cur_id = None
            self._cur_text = tail
            self._shown = ""
            self._chars_since_edit = len(tail)
            budget = self.s.rollover_chars - len(self._active_prefix())

    _MAX_FLUSH_ATTEMPTS = 6

    async def _flush(self, force: bool = False) -> None:
        if self._deleted:
            return
        if not force and not self._should_flush():
            return
        async with self._lock:
            for _ in range(self._MAX_FLUSH_ATTEMPTS):
                # Recompute each attempt: more text may have buffered during a
                # RetryAfter sleep, and we always want to send the freshest text.
                text = self._active_prefix() + self._cur_text
                if not text.strip() or text == self._shown:
                    return  # never send whitespace-only (Telegram: 400 "Text must be non-empty")
                try:
                    if self._cur_id is None:
                        msg = await self.bot.send_message(chat_id=self.chat_id, text=text)
                        self._cur_id = msg.message_id
                        self.message_ids.append(self._cur_id)
                    else:
                        await self.bot.edit_message_text(
                            text=text, chat_id=self.chat_id, message_id=self._cur_id
                        )
                    self._shown = text
                    self._last_edit = time.monotonic()
                    self._chars_since_edit = 0
                    return
                except Exception as exc:  # noqa: BLE001 - narrowed in the handler
                    if not await self._handle_flush_error(exc):
                        return  # handled terminally (deleted / not-modified / logged)
            logger.warning("MessageEditor: gave up flush after %d attempts", self._MAX_FLUSH_ATTEMPTS)

    async def _handle_flush_error(self, exc: Exception) -> bool:
        """Returns True to RE-ATTEMPT the send/edit (so a RetryAfter never silently
        drops a terminal/rollover flush), False to stop. Imported lazily so the
        module imports without PTB in pure-logic tests."""
        from telegram.error import BadRequest, RetryAfter

        if isinstance(exc, RetryAfter):
            delay = min(getattr(exc, "retry_after", 1) or 1, self.s.max_retry_after)
            await asyncio.sleep(delay)
            self._effective_interval *= 1.5  # local backoff; Settings untouched
            return True
        if isinstance(exc, BadRequest):
            msg = str(exc).lower()
            if "not modified" in msg:
                self._shown = self._active_prefix() + self._cur_text  # already on screen
                return False
            if "not found" in msg or "can't be edited" in msg or "to edit" in msg:
                self._deleted = True  # user deleted it: stop editing, keep consuming
                return False
            logger.warning("unexpected BadRequest during streaming edit: %s", exc)
            return False
        raise exc


class RenderDriver:
    """IDLE -> THINKING -> ANSWER -> DONE. Lazily creates the thinking editor on the
    first thinking delta; finalizes it and opens the answer editor on the first
    content delta; on done/error finalizes and calls formatting.finalize. (SPEC §8)
    """

    def __init__(self, bot: "BotLike", chat_id: int, cfg: Settings) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.cfg = cfg
        self.answer_text = ""
        self.answer_message_ids: List[int] = []
        self.thinking_editor: Optional[MessageEditor] = None
        self.answer_editor: Optional[MessageEditor] = None
        self.done_reason: Optional[str] = None
        self.stats: dict = {}
        self._thinking_total = 0
        self._thinking_capped = False
        self._saw_content = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        # Optional hook fired once when the first real answer token lands (typing-stop).
        self.on_first_content: Optional[Callable[[], Awaitable[None] | None]] = None

    async def run(self, deltas: AsyncIterator[Delta]) -> None:
        """Drive the stream to completion, then finalize formatting. (SPEC §8, §10)"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        try:
            async for d in deltas:
                if d.kind == "thinking":
                    await self._on_thinking(d.text)
                elif d.kind == "content":
                    await self._on_content(d.text)
                elif d.kind == "done":
                    self.done_reason = d.done_reason
                    self.stats = d.stats or {}
                    self._stop_heartbeat()  # before finalize, so a tick can't clobber it
                    await self._finish(error=None)
                    return
                elif d.kind == "error":
                    self._stop_heartbeat()
                    await self._finish(error=d.error or "unknown error")
                    return
            self._stop_heartbeat()
            await self._finish(error=None)  # stream ended w/o terminal (defensive)
        except asyncio.CancelledError:
            self._stop_heartbeat()
            await self._on_cancel()
            raise
        finally:
            self._stop_heartbeat()

    def _active_editor(self) -> Optional[MessageEditor]:
        return self.answer_editor or self.thinking_editor

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        self._heartbeat_task = None

    async def _heartbeat(self) -> None:
        """Trailing-edge flush loop (SPEC §8): every max_interval_seconds, flush the
        active editor's buffered text so a mid-stream pause can't hide it."""
        interval = self.cfg.streaming.max_interval_seconds
        try:
            while True:
                await asyncio.sleep(interval)
                editor = self._active_editor()
                if editor is not None:
                    await editor.tick()
        except asyncio.CancelledError:
            return

    # -- delta handlers ------------------------------------------------------

    async def _on_thinking(self, text: str) -> None:
        if not self.cfg.reasoning.show_thinking or self._thinking_capped or not text:
            return
        if self.thinking_editor is None:
            self.thinking_editor = MessageEditor(self.bot, self.chat_id, self.cfg, _THINK_PREFIX)
        cap = self.cfg.reasoning.max_thinking_chars
        remaining = cap - self._thinking_total
        if remaining <= 0:
            return
        chunk = text[:remaining]
        self._thinking_total += len(chunk)
        await self.thinking_editor.feed(chunk)
        if self._thinking_total >= cap:
            self._thinking_capped = True
            await self.thinking_editor.feed(" …")

    async def _on_content(self, text: str) -> None:
        if not self._saw_content:
            # Drop leading whitespace before the first real character — the abliterated
            # model emits "\n\n" after the suppressed empty <think>. Opening the answer
            # on whitespace would both look ugly and trip Telegram's non-empty rule.
            text = text.lstrip()
            if not text:
                return  # still only whitespace — wait for the first real content
            self._saw_content = True
            if self.thinking_editor is not None:
                await self.thinking_editor.finalize()  # seal the thinking bubble
            self.answer_editor = MessageEditor(self.bot, self.chat_id, self.cfg)
            await self._fire_first_content()
        self.answer_text += text
        if self.answer_editor is not None:
            await self.answer_editor.feed(text)

    async def _fire_first_content(self) -> None:
        cb = self.on_first_content
        if cb is None:
            return
        try:
            res = cb()
            if asyncio.iscoroutine(res):
                await res
        except Exception as exc:  # a typing-stop failure must never break rendering
            logger.debug("on_first_content hook failed: %s", exc)

    # -- terminal ------------------------------------------------------------

    async def _finish(self, error: Optional[str]) -> None:
        if self.thinking_editor is not None:
            await self.thinking_editor.finalize()

        if error is not None:
            note = self.cfg.streaming.error_note.format(error=error)
            if self.answer_editor is None:
                # error before any content: show the note as the answer message
                self.answer_editor = MessageEditor(self.bot, self.chat_id, self.cfg)
                await self.answer_editor.feed(note)
            else:
                await self.answer_editor.feed("\n\n" + note)
            await self.answer_editor.finalize()
            self.answer_message_ids = self.answer_editor.message_ids
            return  # partial answer + note stays plain — no rich finalize

        if self.answer_editor is None:
            # completed with zero content -> empty-response note (plain, no finalize)
            editor = MessageEditor(self.bot, self.chat_id, self.cfg)
            await editor.feed(self.cfg.streaming.empty_response_text)
            await editor.finalize()
            self.answer_message_ids = editor.message_ids
            return

        await self.answer_editor.finalize()
        self.answer_message_ids = self.answer_editor.message_ids
        # Exactly ONE finalize pass on the whole answer (SPEC §10).
        await formatting.finalize(
            self.bot, self.cfg, self.chat_id, self.answer_text, self.answer_message_ids
        )

    async def _on_cancel(self) -> None:
        """Cancel-and-supersede cleanup: append the superseded note and seal the
        live message, shielded so the cancellation can't abort the final edit.
        (SPEC §8 concurrency)"""
        note = self.cfg.streaming.superseded_note
        try:
            if self.answer_editor is not None:
                if note:
                    await asyncio.shield(self.answer_editor.feed("\n\n" + note))
                await asyncio.shield(self.answer_editor.finalize())
                self.answer_message_ids = self.answer_editor.message_ids
            elif self.thinking_editor is not None:
                await asyncio.shield(self.thinking_editor.finalize())
        except Exception as exc:  # cleanup is best-effort
            logger.debug("supersede cleanup failed: %s", exc)

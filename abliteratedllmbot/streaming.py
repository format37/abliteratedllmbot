"""The live-edit streaming state machine — the crux feature. See SPEC.md §8.

Consumes an AsyncIterator[Delta] and drives one or two live-edited Telegram
messages (a lazily-created 'thinking' message, then the answer), throttled and
rolling over near the 4096 limit, handing the final text to formatting.finalize.

Designed to be tested against a recording FakeBot with scripted Delta streams —
NO real Telegram needed for Milestone 3.

STATUS: stub (Milestone 3).
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional, Protocol

from .config import Settings
from .errors import Delta


class BotLike(Protocol):
    """The subset of telegram.Bot the renderer uses (so a FakeBot can stand in)."""

    async def send_message(self, chat_id: int, text: str, **kw): ...
    async def edit_message_text(self, text: str, chat_id: int, message_id: int, **kw): ...
    async def delete_message(self, chat_id: int, message_id: int) -> bool: ...


class MessageEditor:
    """Owns exactly one Telegram message; buffers text and flushes throttled edits,
    rolling over to a new message near ``rollover_chars``. (SPEC §8)

    Flush when: chars_since_edit >= batch_size  OR  now-last_edit >= min_edit_interval.
    ``effective_interval`` is a LOCAL that grows ×1.5 on RetryAfter — never mutate Settings.
    """

    def __init__(self, bot: "BotLike", chat_id: int, cfg: Settings, prefix: str = "") -> None:
        raise NotImplementedError  # Milestone 3

    async def feed(self, text: str) -> None:
        raise NotImplementedError  # Milestone 3

    async def finalize(self) -> None:
        """Final flush; returns control so the driver can run formatting.finalize."""
        raise NotImplementedError  # Milestone 3


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

    async def run(self, deltas: AsyncIterator[Delta]) -> None:
        """Drive the stream to completion, then finalize formatting. (SPEC §8, §10)"""
        raise NotImplementedError  # Milestone 3

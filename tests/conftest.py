"""Shared test helpers: a recording FakeBot (captures every Telegram call) and a
Settings builder — so the M3 render/format layers are tested with NO real Telegram."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pytest

from abliteratedllmbot.config import Settings


class FakeMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class FakeBot:
    """Records send_message / edit_message_text / delete_message / send_document.

    Optional ``on_send`` / ``on_edit`` hooks — ``callable(nth, kwargs)`` — may raise a
    telegram error to script RetryAfter / BadRequest scenarios. ``messages`` holds the
    current visible text per live message id; ``calls`` is the ordered call log."""

    def __init__(
        self,
        on_send: Optional[Callable[[int, dict], None]] = None,
        on_edit: Optional[Callable[[int, dict], None]] = None,
    ) -> None:
        self.calls: List[Tuple[str, dict]] = []
        self.messages: dict[int, str] = {}
        self.deleted: set[int] = set()
        self.documents: List[dict] = []
        self._next_id = 1000
        self._on_send = on_send
        self._on_edit = on_edit
        self._send_n = 0
        self._edit_n = 0

    async def send_message(self, chat_id: int, text: str, **kw):
        self._send_n += 1
        self.calls.append(("send", {"chat_id": chat_id, "text": text, **kw}))
        if self._on_send is not None:
            self._on_send(self._send_n, {"chat_id": chat_id, "text": text, **kw})
        mid = self._next_id
        self._next_id += 1
        self.messages[mid] = text
        return FakeMessage(mid)

    async def edit_message_text(self, text: str, chat_id: int, message_id: int, **kw):
        self._edit_n += 1
        self.calls.append(
            ("edit", {"text": text, "chat_id": chat_id, "message_id": message_id, **kw})
        )
        if self._on_edit is not None:
            self._on_edit(self._edit_n, {"text": text, "message_id": message_id, **kw})
        self.messages[message_id] = text
        return FakeMessage(message_id)

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        self.calls.append(("delete", {"chat_id": chat_id, "message_id": message_id}))
        self.deleted.add(message_id)
        self.messages.pop(message_id, None)
        return True

    async def send_chat_action(self, chat_id: int, action, **kw):
        self.calls.append(("chat_action", {"chat_id": chat_id, "action": str(action)}))

    async def send_document(self, chat_id: int, document, filename=None, **kw):
        self.calls.append(("document", {"chat_id": chat_id, "filename": filename}))
        self.documents.append({"chat_id": chat_id, "filename": filename})
        mid = self._next_id
        self._next_id += 1
        return FakeMessage(mid)

    # convenience
    def calls_of(self, kind: str) -> List[dict]:
        return [kw for k, kw in self.calls if k == kind]


@pytest.fixture
def make_bot():
    """Factory for a fresh recording FakeBot, optionally with error hooks."""

    def _make(on_send=None, on_edit=None) -> FakeBot:
        return FakeBot(on_send=on_send, on_edit=on_edit)

    return _make


@pytest.fixture
def make_cfg():
    def _make(**overrides) -> Settings:
        base: dict = {
            "reasoning": {"show_thinking": True, "max_thinking_chars": 3500},
            # min_edit_interval 0 -> every feed flushes immediately (deterministic);
            # override per-test to exercise throttling.
            "streaming": {
                "batch_size": 80,
                "min_edit_interval": 0.0,
                "max_interval_seconds": 5.0,
                "rollover_chars": 3900,
                "max_retry_after": 30,
            },
            "formatting": {"finalize_mode": "plain"},
        }
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        return Settings(**base)

    return _make

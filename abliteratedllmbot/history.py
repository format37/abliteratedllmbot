"""Per-chat conversation memory + trimming. Pure logic, no I/O. See SPEC.md §7.

STATUS: stub (Milestone 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

from .config import MemoryConfig

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str
    # v1: vision turns are stateless and NOT stored here (SPEC §9).


class InMemoryStore:
    """dict[chat_id -> list[Message]]. v1.1 adds an optional SQLite-backed store."""

    def __init__(self) -> None:
        self._chats: Dict[int, List[Message]] = {}

    def get(self, chat_id: int) -> List[Message]:
        return list(self._chats.get(chat_id, []))

    def append(self, chat_id: int, msg: Message) -> None:
        self._chats.setdefault(chat_id, []).append(msg)

    def reset(self, chat_id: int) -> None:
        self._chats.pop(chat_id, None)


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(len(text) / chars_per_token) + 1


def trim_history(
    history: List[Message],
    *,
    max_turns: int,
    max_tokens: int,
    chars_per_token: float,
) -> List[Message]:
    """Drop oldest user/assistant turns until within max_turns AND max_tokens.
    A system message (if present at index 0) is always kept. (SPEC §7)"""
    raise NotImplementedError  # Milestone 2


def build_messages(
    system_prompt: str,
    history: List[Message],
    user_text: str,
    memory: MemoryConfig,
) -> List[dict]:
    """system + trim_history(history) + current user, as Ollama chat dicts."""
    raise NotImplementedError  # Milestone 2

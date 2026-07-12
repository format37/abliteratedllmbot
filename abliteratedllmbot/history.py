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


def calibrate_chars_per_token(
    total_chars: int,
    prompt_eval_count: int,
    *,
    floor: float = 1.5,
    ceil: float = 6.0,
) -> float:
    """Observed chars/token from a real ``prompt_eval_count`` (SPEC §7), clamped to a
    sane band so a degenerate stat can't wreck future estimates. The bot feeds this
    back into ``memory.chars_per_token`` after a generation reports its token count."""
    if total_chars <= 0 or prompt_eval_count <= 0:
        return max(floor, min(ceil, (total_chars / prompt_eval_count) if prompt_eval_count else ceil))
    return max(floor, min(ceil, total_chars / prompt_eval_count))


def _split_system(history: List[Message]):
    """Peel off a leading system message (kept verbatim) from the u/a body."""
    if history and history[0].role == "system":
        return [history[0]], list(history[1:])
    return [], list(history)


def _group_turns(body: List[Message]) -> List[List[Message]]:
    """Group a user/assistant sequence into turns — normally ``[user, assistant]``
    pairs. Defensive against a leading assistant or a trailing lone user so trimming
    never splits a pair or loses alignment."""
    turns: List[List[Message]] = []
    i, n = 0, len(body)
    while i < n:
        if body[i].role == "user" and i + 1 < n and body[i + 1].role == "assistant":
            turns.append(body[i : i + 2])
            i += 2
        else:
            turns.append([body[i]])
            i += 1
    return turns


def trim_history(
    history: List[Message],
    *,
    max_turns: int,
    max_tokens: int,
    chars_per_token: float,
) -> List[Message]:
    """Drop oldest user/assistant turns until within max_turns AND max_tokens.
    A system message (if present at index 0) is always kept. (SPEC §7)

    Turns first, then token budget: cap the count to the most recent ``max_turns``
    pairs, then drop whole oldest pairs until the estimated token total (system
    included) fits ``max_tokens``. The system prompt is never dropped even if it
    alone exceeds the budget."""
    if not history:
        return []
    system, body = _split_system(history)
    turns = _group_turns(body)

    # 1) turn cap — keep the most recent max_turns turns.
    if max_turns >= 0 and len(turns) > max_turns:
        turns = turns[-max_turns:] if max_turns > 0 else []

    # 2) token budget — drop oldest whole turns until within max_tokens.
    def toks(msgs: List[Message]) -> int:
        return sum(estimate_tokens(m.content, chars_per_token) for m in msgs)

    sys_toks = toks(system)
    flat = [m for t in turns for m in t]
    while turns and sys_toks + toks(flat) > max_tokens:
        turns.pop(0)
        flat = [m for t in turns for m in t]

    return system + flat


def build_messages(
    system_prompt: str,
    history: List[Message],
    user_text: str,
    memory: MemoryConfig,
) -> List[dict]:
    """system + trim_history(history) + current user, as Ollama chat dicts.

    When memory is disabled the turn is stateless (system + current user only).
    The trim budget reserves room for the system prompt and the incoming user turn,
    so the WHOLE prompt — not just the stored history — stays under ``max_tokens``."""
    if memory.enabled:
        reserve = estimate_tokens(system_prompt, memory.chars_per_token) + estimate_tokens(
            user_text, memory.chars_per_token
        )
        budget = max(0, memory.max_tokens - reserve)
        trimmed = trim_history(
            history,
            max_turns=memory.max_turns,
            max_tokens=budget,
            chars_per_token=memory.chars_per_token,
        )
    else:
        trimmed = []

    msgs: List[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend({"role": m.role, "content": m.content} for m in trimmed)
    msgs.append({"role": "user", "content": user_text})
    return msgs

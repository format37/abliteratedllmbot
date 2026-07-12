"""Frozen contracts shared across layers: the streaming Delta type and the
typed exception hierarchy. Keep this dependency-free (stdlib only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

DeltaKind = Literal["thinking", "content", "done", "error"]


@dataclass
class Delta:
    """The single value type flowing from the Ollama client to the renderer.

    Ordering guarantee per stream: zero+ ``thinking`` deltas, then zero+
    ``content`` deltas, then exactly one terminal delta (``done`` or ``error``).
    ``content`` text is already sanitised (special tokens stripped) by the client.
    """

    kind: DeltaKind
    text: str = ""                          # incremental text (thinking | content)
    done_reason: Optional[str] = None       # "stop" | "length" | ...  (kind="done")
    stats: dict = field(default_factory=dict)  # eval_count, *_duration (kind="done")
    error: Optional[str] = None             # human-readable message (kind="error")


class BotError(Exception):
    """Base class for expected, user-surfaceable errors."""


class OllamaUnavailableError(BotError):
    """Ollama unreachable or timed out BEFORE the first stream chunk."""


class ModelNotFoundError(BotError):
    """/api/show returned 404 for the requested model."""


class ThinkingUnsupportedError(BotError):
    """Defensive: think:true sent to an incapable model. Gating should prevent this."""

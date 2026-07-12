"""Ollama client — the only module that talks to Ollama. It owns capability
gating, inline-think splitting, and sanitising, and exposes the rest of the bot
a single ``AsyncIterator[Delta]``. See SPEC.md §4-§6.

Transport: the official ``ollama`` AsyncClient (0.5.1). STATUS: stub (Milestone 1b).
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional

from .config import OllamaConfig, ReasoningConfig, SanitizeConfig
from .errors import Delta, ModelNotFoundError, OllamaUnavailableError  # noqa: F401
from .reasoning import InlineThinkSplitter, StreamSanitizer  # noqa: F401


class OllamaClient:
    def __init__(
        self,
        ollama: OllamaConfig,
        reasoning: ReasoningConfig,
        sanitize: SanitizeConfig,
    ) -> None:
        self._ollama = ollama
        self._reasoning = reasoning
        self._sanitize = sanitize
        self._caps: dict[str, set[str]] = {}
        # TODO: construct ollama.AsyncClient(host=ollama.host, timeout=...)

    async def startup(self) -> None:
        """Verify reachability and warm the capability cache for model + vision_model."""
        raise NotImplementedError  # Milestone 1b

    async def close(self) -> None:
        raise NotImplementedError  # Milestone 1b

    async def capabilities(self, model: str) -> set[str]:
        """Cached /api/show capabilities. Raises ModelNotFoundError on 404."""
        raise NotImplementedError  # Milestone 1b

    def resolve_think(self, caps: set[str]) -> bool:
        """think_mode × capability gate. NEVER True unless 'thinking' in caps. (SPEC §5)"""
        if self._reasoning.think_mode == "off":
            return False
        return "thinking" in caps  # 'auto' and 'inline' both require capability for think:true

    async def chat_stream(
        self,
        *,
        messages: List[dict],
        model: Optional[str] = None,
        images: Optional[List[bytes]] = None,
    ) -> AsyncIterator[Delta]:
        """Stream a chat completion as canonical Delta events. (SPEC §4, §5)

        Contract:
          * pre-first-chunk failure -> raise OllamaUnavailableError/ModelNotFoundError
          * mid-stream failure       -> yield Delta("error", ...)
          * structured path: chunk.message.thinking present -> thinking/content deltas
          * inline path: feed chunk.message.content through InlineThinkSplitter
          * every content span passes through StreamSanitizer before it is yielded
          * terminal: yield Delta("done", done_reason=..., stats={...})
        """
        raise NotImplementedError  # Milestone 1b
        yield  # pragma: no cover  (marks this as an async generator)

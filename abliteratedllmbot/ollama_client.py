"""Ollama client — the only module that talks to Ollama. It owns capability
gating, inline-think splitting, and sanitising, and exposes the rest of the bot
a single ``AsyncIterator[Delta]``. See SPEC.md §4-§6.

Transport: the official ``ollama`` AsyncClient (0.5.1). It maps ``httpx.ConnectError``
to a builtin ``ConnectionError`` and HTTP/stream errors to ``ollama.ResponseError``;
timeouts propagate as raw ``httpx.TransportError`` subclasses. We wrap all three so
the rest of the bot depends only on ``Delta`` + the typed exceptions in errors.py.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List, Optional

import httpx
from ollama import AsyncClient, ResponseError

from .config import OllamaConfig, ReasoningConfig, SanitizeConfig
from .errors import (
    BotError,
    Delta,
    ModelNotFoundError,
    OllamaUnavailableError,
    ThinkingUnsupportedError,
)
from .reasoning import InlineThinkSplitter, StreamSanitizer

# Transport-layer failures the ollama client surfaces to us. ConnectionError is
# builtin (ollama re-raises httpx.ConnectError as it); httpx.TransportError covers
# connect/read timeouts and mid-stream read/protocol errors ollama does NOT wrap;
# ResponseError carries HTTP status codes and streamed ``{"error": …}`` lines.
_TRANSPORT_ERRORS = (ConnectionError, httpx.TransportError, ResponseError)

# Stats surfaced on the terminal `done` chunk (SPEC §5.5). Superset of the four
# the spec names; `/stats` derives tokens/sec from eval_count / eval_duration.
_STAT_KEYS = (
    "eval_count",
    "prompt_eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)


def _collect_stats(chunk) -> dict:
    return {k: v for k in _STAT_KEYS if (v := getattr(chunk, k, None)) is not None}


def _spans_to_deltas(
    spans, sanitizer: StreamSanitizer, emit_thinking: bool
) -> List[Delta]:
    """Turn (kind, text) spans into Deltas: thinking passes through (dropped when
    ``emit_thinking`` is False), content is streamed through the sanitiser."""
    out: List[Delta] = []
    for kind, text in spans:
        if kind == "thinking":
            if emit_thinking and text:
                out.append(Delta("thinking", text=text))
        else:
            clean = sanitizer.feed(text)
            if clean:
                out.append(Delta("content", text=clean))
    return out


def _attach_images(messages: List[dict], images: List[bytes]) -> List[dict]:
    """Attach image bytes to the last user turn (the ollama Message serialiser
    base64-encodes bytes). Only ever called on the vision path (SPEC §5.3, §9)."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["images"] = list(images)
            return out
    out.append({"role": "user", "content": "", "images": list(images)})
    return out


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
        self._client = AsyncClient(
            host=ollama.host,
            timeout=httpx.Timeout(ollama.request_timeout, connect=ollama.connect_timeout),
        )

    async def startup(self) -> None:
        """Verify reachability and warm the capability cache for model + vision_model.

        Fails fast if the primary model is unreachable/missing. The vision model is
        optional (only images.mode=vision_fallback needs it) so a *missing* vision
        model is tolerated; an unreachable server still propagates."""
        await self.capabilities(self._ollama.model)
        vm = self._ollama.vision_model
        if vm and vm != self._ollama.model:
            try:
                await self.capabilities(vm)
            except ModelNotFoundError:
                pass  # optional; surfaced later only if a vision turn is attempted

    async def close(self) -> None:
        inner = getattr(self._client, "_client", None)
        if inner is not None and hasattr(inner, "aclose"):
            await inner.aclose()

    async def capabilities(self, model: str) -> set[str]:
        """Cached /api/show capabilities. Raises ModelNotFoundError on 404,
        OllamaUnavailableError if the server can't be reached."""
        if model in self._caps:
            return self._caps[model]
        try:
            resp = await self._client.show(model)
        except _TRANSPORT_ERRORS as exc:
            raise self._as_pre_stream_error(exc, model) from exc
        caps = set(resp.capabilities or [])
        self._caps[model] = caps
        return caps

    async def ps(self) -> List[dict]:
        """Currently-loaded models with VRAM + expiry (GET /api/ps). Empty if none."""
        try:
            resp = await self._client.ps()
        except _TRANSPORT_ERRORS as exc:
            raise self._as_pre_stream_error(exc, self._ollama.model) from exc
        return [
            {"model": m.model, "size_vram": m.size_vram, "expires_at": m.expires_at}
            for m in (resp.models or [])
        ]

    async def chat_models(self) -> List[str]:
        """Installed models that can actually hold a conversation, sorted by name.

        ``/api/tags`` lists everything pulled, including embedding-only models
        (`nomic-embed-text`, `bge-m3` expose capabilities ``['embedding']`` and no
        ``completion``). Offering those as a choice would hand the user a button that
        can only fail, so each name is filtered through the cached capability lookup.
        A model whose /api/show fails individually is dropped rather than fataled."""
        try:
            resp = await self._client.list()
        except _TRANSPORT_ERRORS as exc:
            raise self._as_pre_stream_error(exc, self._ollama.model) from exc

        names = sorted({m.model for m in (resp.models or []) if m.model})
        caps = await asyncio.gather(
            *(self.capabilities(n) for n in names), return_exceptions=True
        )
        return [
            name
            for name, cap in zip(names, caps)
            if isinstance(cap, set) and "completion" in cap
        ]

    async def load(self, model: Optional[str] = None) -> None:
        """Warm a model into memory: an empty generate carrying ``keep_alive`` (no
        tokens produced; server responds done_reason='load'). Cold-loads the 35B in
        ~156s. Raises the typed transport errors on failure."""
        model = model or self._ollama.model
        try:
            await self._client.generate(
                model=model, prompt="", keep_alive=self._ollama.keep_alive
            )
        except _TRANSPORT_ERRORS as exc:
            raise self._as_pre_stream_error(exc, model) from exc

    async def unload(self, model: Optional[str] = None) -> None:
        """Evict a model from memory now (keep_alive=0)."""
        model = model or self._ollama.model
        try:
            await self._client.generate(model=model, prompt="", keep_alive=0)
        except _TRANSPORT_ERRORS as exc:
            raise self._as_pre_stream_error(exc, model) from exc

    def resolve_think(self, caps: set[str]) -> bool:
        """think_mode × capability gate. Returns True (⇒ send think:true) ONLY in
        ``auto`` mode against a model whose /api/show capabilities include
        ``thinking``. ``inline`` forces the inline-<think> parse path and ``off``
        strips reasoning — neither ever requests structured thinking. This is the
        single guarantee that think:true is NEVER sent to an incapable model [V].
        (SPEC §2 three-mode semantics; §5.2 gate.)"""
        if self._reasoning.think_mode == "auto":
            return "thinking" in caps
        return False

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
        model = model or self._ollama.model

        # Pre-stream gate: resolve capabilities first (SPEC §5.1). A missing model
        # or unreachable server raises here, BEFORE any Delta is yielded.
        caps = await self.capabilities(model)
        think = self.resolve_think(caps)
        emit_thinking = self._reasoning.think_mode != "off"

        options = self._ollama.options.model_dump(exclude_none=True)
        if self._ollama.stop:
            options["stop"] = list(self._ollama.stop)

        msgs = _attach_images(messages, images) if images else messages

        splitter = InlineThinkSplitter()
        sanitizer = StreamSanitizer(
            extra=self._sanitize.extra_tokens,
            enabled=self._sanitize.strip_special_tokens,
        )

        try:
            stream = await self._client.chat(
                model=model,
                messages=msgs,
                stream=True,
                think=True if think else None,  # None -> omitted; never think:false either [V]
                options=options,
                keep_alive=self._ollama.keep_alive,
            )
        except _TRANSPORT_ERRORS as exc:  # param/build errors (I/O is deferred to iteration)
            raise self._as_pre_stream_error(exc, model) from exc

        received = False
        done_seen = False
        try:
            async for chunk in stream:
                received = True
                msg = getattr(chunk, "message", None)
                thinking_text = getattr(msg, "thinking", None) if msg else None
                content_text = getattr(msg, "content", None) if msg else None

                if think:
                    # Structured path: reasoning in message.thinking, answer in content.
                    if thinking_text and emit_thinking:
                        yield Delta("thinking", text=thinking_text)
                    if content_text:
                        for d in _spans_to_deltas(
                            [("content", content_text)], sanitizer, emit_thinking
                        ):
                            yield d
                else:
                    # Inline path: split <think>…</think> out of the content stream [V].
                    # A stray structured field shouldn't appear, but honour it if it does.
                    if thinking_text and emit_thinking:
                        yield Delta("thinking", text=thinking_text)
                    for d in _spans_to_deltas(
                        splitter.feed(content_text or ""), sanitizer, emit_thinking
                    ):
                        yield d

                if getattr(chunk, "done", False):
                    done_seen = True
                    for d in self._finalize(splitter, sanitizer, emit_thinking):
                        yield d
                    yield Delta(
                        "done",
                        done_reason=getattr(chunk, "done_reason", None),
                        stats=_collect_stats(chunk),
                    )
                    return
        except _TRANSPORT_ERRORS as exc:
            if not received:  # first-chunk failure is still pre-stream -> raise typed
                raise self._as_pre_stream_error(exc, model) from exc
            # Mid-stream drop: surface whatever was already produced + a terminal
            # error Delta. We deliberately do NOT flush the splitter/sanitizer carry
            # here (unlike the clean end-of-stream path below): on an abrupt drop the
            # held fragment is an *incomplete* tag/token, so emitting it would leak a
            # truncated '<|' / '<thi'. A clean close (no exception) means the buffer
            # is settled real content, so that path does flush.
            yield Delta("error", error=_human_error(exc))
            return

        # Stream ended without a terminal `done` chunk (server closed gracefully but
        # sent no done=True). The carry buffer is settled content -> flush it.
        if not done_seen:
            for d in self._finalize(splitter, sanitizer, emit_thinking):
                yield d
            yield Delta("error", error="The response stream ended before completion.")

    @staticmethod
    def _finalize(
        splitter: InlineThinkSplitter, sanitizer: StreamSanitizer, emit_thinking: bool
    ) -> List[Delta]:
        """Drain the splitter's held tag fragment and the sanitiser's carry buffer."""
        out = _spans_to_deltas(splitter.flush(), sanitizer, emit_thinking)
        tail = sanitizer.flush()
        if tail:
            out.append(Delta("content", text=tail))
        return out

    def _as_pre_stream_error(self, exc: Exception, model: str) -> BotError:
        """Map a transport failure to a typed, user-surfaceable exception."""
        if isinstance(exc, ResponseError):
            text = exc.error or str(exc)
            low = text.lower()
            if exc.status_code == 404 or "not found" in low:
                return ModelNotFoundError(
                    f"Model {model!r} is not available on the Ollama server at "
                    f"{self._ollama.host}."
                )
            if "thinking" in low:  # defensive: gating should prevent think:true 400 [V]
                return ThinkingUnsupportedError(text)
            return OllamaUnavailableError(
                f"Ollama returned an error for model {model!r}: {text}"
            )
        return OllamaUnavailableError(
            f"Could not reach the Ollama server at {self._ollama.host} "
            f"({exc.__class__.__name__})."
        )


def _human_error(exc: Exception) -> str:
    if isinstance(exc, ResponseError):
        return exc.error or str(exc)
    return "Lost connection to the model server mid-response."

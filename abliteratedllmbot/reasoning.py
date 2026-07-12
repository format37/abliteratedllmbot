"""Pure, dependency-free reasoning utilities — build and unit-test these FIRST.

Two verified problems with the abliterated GGUF drive this module:

  * It wraps replies in an inline ``<think>…</think>`` block (often empty) inside
    ``message.content`` — there is no structured ``message.thinking`` field. So we
    split reasoning out of the content STREAM. -> InlineThinkSplitter
  * It leaks chat-template special tokens (``<|im_start|>``, ``<|endoftext|>`` …)
    into visible content even at ``done_reason:"stop"``. -> StreamSanitizer

Both are stateful streaming classes (they must survive tags/tokens split across
network chunks) plus a whole-string ``sanitize_content`` convenience.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")  # <|im_start|>, <|endoftext|>, <|im_end|>, …

Span = Tuple[str, str]  # (kind, text) where kind in {"thinking", "content"}


def _partial_suffix_len(buf: str, tag: str) -> int:
    """Longest k (0 < k < len(tag)) such that buf ends with tag[:k]; else 0.

    Used to hold back a trailing fragment that might be the start of ``tag``
    completed by the next chunk (e.g. buf endswith '<thi', tag '<think>')."""
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


class InlineThinkSplitter:
    """Streaming splitter for inline ``<think>…</think>``.

    ``feed(text)`` returns a list of ``(kind, text)`` spans. Whitespace-only
    *leading* thinking (the empty ``<think>\\n\\n</think>`` case) is suppressed so
    the renderer never opens an empty thinking bubble. Call ``flush()`` at end of
    stream to drain any held fragment.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.in_thinking = False
        self._think_has_content = False  # per-block: seen non-whitespace yet?

    def _emit(self, out: List[Span], text: str) -> None:
        if not text:
            return
        if self.in_thinking:
            if not self._think_has_content and text.strip() == "":
                return  # suppress leading whitespace-only thinking
            if text.strip():
                self._think_has_content = True
            out.append(("thinking", text))
        else:
            out.append(("content", text))

    def feed(self, text: str) -> List[Span]:
        out: List[Span] = []
        self.buf += text
        while True:
            tag = self.CLOSE if self.in_thinking else self.OPEN
            idx = self.buf.find(tag)
            if idx != -1:
                self._emit(out, self.buf[:idx])
                self.buf = self.buf[idx + len(tag):]
                self.in_thinking = not self.in_thinking
                if self.in_thinking:
                    self._think_has_content = False
                continue
            hold = _partial_suffix_len(self.buf, tag)
            cut = len(self.buf) - hold
            if cut > 0:
                self._emit(out, self.buf[:cut])
            self.buf = self.buf[cut:]
            break
        return out

    def flush(self) -> List[Span]:
        out: List[Span] = []
        if self.buf:
            self._emit(out, self.buf)  # dangling partial tag -> literal text of current kind
            self.buf = ""
        return out


class StreamSanitizer:
    """Streaming remover of ``<|…|>`` special tokens (+ configured extra tokens).

    Holds back a trailing ``<|`` that hasn't closed yet (and trailing fragments of
    ``extra`` tokens) so a token split across chunks is still removed."""

    def __init__(self, extra: Iterable[str] = (), enabled: bool = True) -> None:
        self.extra = [t for t in extra if t]
        self.enabled = enabled
        self.buf = ""

    def _strip(self, text: str) -> str:
        text = SPECIAL_TOKEN_RE.sub("", text)
        for tok in self.extra:
            text = text.replace(tok, "")
        return text

    def feed(self, text: str) -> str:
        if not self.enabled:
            return text
        self.buf = self._strip(self.buf + text)
        hold_idx = len(self.buf)
        lt = self.buf.rfind("<|")
        if lt != -1 and "|>" not in self.buf[lt:]:
            hold_idx = min(hold_idx, lt)  # unclosed special token — wait for the close
        for tok in self.extra:
            k = _partial_suffix_len(self.buf, tok)
            if k:
                hold_idx = min(hold_idx, len(self.buf) - k)
        emit, self.buf = self.buf[:hold_idx], self.buf[hold_idx:]
        return emit

    def flush(self) -> str:
        out = self._strip(self.buf)  # a dangling '<|' with no close is emitted literally
        self.buf = ""
        return out


def sanitize_content(text: str, extra: Iterable[str] = ()) -> str:
    """Whole-string sanitise (non-streaming convenience)."""
    text = SPECIAL_TOKEN_RE.sub("", text)
    for tok in extra:
        if tok:
            text = text.replace(tok, "")
    return text

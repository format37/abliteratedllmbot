"""Finalize formatting — the pantherabot rich chain. See SPEC.md §10.

Streaming stays plain (parse_mode=None) so intermediate edits never throw. Exactly
ONE finalize pass runs on the whole answer:

    rich  -> sendRichMessage (32768 cap) -> .txt document -> markdown_v2 -> plain
    markdown_v2 -> edit in place with MarkdownV2 (placeholder-escape) -> plain on error
    plain -> leave the streamed messages as-is

``send_rich_message`` (raw POST) and the MarkdownV2 placeholder-escape are ported
from pantherabot (server/server.py) and adapted to async httpx / PTB.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import List, Optional

import httpx
from telegram.error import BadRequest, TelegramError
from telegram.helpers import escape_markdown

from .config import Settings

logger = logging.getLogger(__name__)

# PTB convention: base ends in "/bot", the token is appended directly.
OFFICIAL_API_BASE = "https://api.telegram.org/bot"

# Placeholder sentinels wrap a stash index; they contain no MarkdownV2 special
# chars, so escape_markdown leaves them intact and we restore afterwards.
_PH = "\x00{}\x00"
_PH_RE = re.compile(r"\x00(\d+)\x00")


def _esc(text: str, entity_type: Optional[str] = None) -> str:
    return escape_markdown(text, version=2, entity_type=entity_type)


def to_markdown_v2(text: str) -> str:
    """Best-effort convert standard Markdown to Telegram MarkdownV2.

    Recognised markup (code fences, inline code, links, bold, strthrough, italic,
    spoiler) is protected as placeholders holding its final MarkdownV2 rendering,
    the remainder is escaped, then placeholders are restored. Anything malformed
    still yields *valid* MarkdownV2 (worst case: literal text) — and the finalize
    chain reverts to plain on a parse error, so this never has to be perfect."""
    text = text.replace("\x00", "")  # NUL is our sentinel; strip so real input can't collide
    stash: List[str] = []

    def hold(final: str) -> str:
        stash.append(final)
        return _PH.format(len(stash) - 1)

    # Order matters: code first (its body must not be treated as markup), then
    # links, then the paired inline styles (bold's ** before italic's *).
    text = re.sub(
        r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```",
        lambda m: hold(f"```{m.group('lang')}\n{_esc(m.group('body'), 'pre')}```"),
        text,
        flags=re.S,
    )
    text = re.sub(
        r"`(?P<c>[^`\n]+)`",
        lambda m: hold(f"`{_esc(m.group('c'), 'code')}`"),
        text,
    )
    text = re.sub(
        r"\[(?P<t>[^\]\n]+)\]\((?P<u>[^)\s]+)\)",
        lambda m: hold(f"[{_esc(m.group('t'))}]({_esc(m.group('u'), 'text_link')})"),
        text,
    )
    # bold: **x** or __x__  ->  *x*
    text = re.sub(r"\*\*(?P<b>.+?)\*\*", lambda m: hold("*" + _esc(m.group("b")) + "*"), text, flags=re.S)
    text = re.sub(r"__(?P<b>.+?)__", lambda m: hold("*" + _esc(m.group("b")) + "*"), text, flags=re.S)
    # strikethrough: ~~x~~ -> ~x~
    text = re.sub(r"~~(?P<s>.+?)~~", lambda m: hold("~" + _esc(m.group("s")) + "~"), text, flags=re.S)
    # spoiler: ||x|| -> ||x||
    text = re.sub(r"\|\|(?P<s>.+?)\|\|", lambda m: hold("||" + _esc(m.group("s")) + "||"), text, flags=re.S)
    # italic: single *x* or _x_ -> _x_  (doubles already consumed above)
    text = re.sub(r"(?<!\*)\*(?!\*)(?P<i>[^*\n]+?)\*(?!\*)", lambda m: hold("_" + _esc(m.group("i")) + "_"), text)
    text = re.sub(r"(?<!_)_(?!_)(?P<i>[^_\n]+?)_(?!_)", lambda m: hold("_" + _esc(m.group("i")) + "_"), text)

    text = _esc(text)  # escape everything left (placeholder sentinels survive)

    # Restore iteratively: an emphasis span can WRAP an already-stashed inner span
    # (inline code, fence, link), so its stash entry itself contains a nested
    # sentinel. A single pass would leave that inner sentinel unexpanded — leaking
    # a raw NUL and dropping the inner content. Loop until none remain.
    def _restore(m):
        i = int(m.group(1))
        return stash[i] if 0 <= i < len(stash) else m.group(0)

    for _ in range(len(stash) + 1):
        if not _PH_RE.search(text):
            break
        text = _PH_RE.sub(_restore, text)
    return text


def to_expandable_blockquote(text: str) -> str:
    """Render text as a Telegram MarkdownV2 EXPANDABLE (collapsed-by-default) blockquote:
    every line prefixed with ``>``, the whole block ended with ``||``. Verified accepted
    by the official Bot API (entity type ``expandable_blockquote``)."""
    escaped = escape_markdown(text, version=2)  # escapes all specials, incl. literal > and |
    lines = escaped.split("\n")
    return "\n".join(">" + ln for ln in lines) + "||"


async def send_rich_message(
    cfg: Settings, chat_id: int, markdown_text: str, reply_to: Optional[int] = None
) -> bool:
    """POST {base_url}{token}/sendRichMessage (Bot API 10.1). Returns True on ``ok``,
    False on any failure so the caller can fall down the chain. Ported from
    pantherabot; base_url follows the PTB convention (ends in ``/bot``).

    The OFFICIAL API implements sendRichMessage too — a local Bot API server is
    NOT required for rich/LaTeX output (verified: api.telegram.org answers
    "rich message must be non-empty", not 404). So when no base_url is set we
    still send rich, against api.telegram.org."""
    base_url = cfg.telegram.base_url or OFFICIAL_API_BASE
    token = cfg.telegram_bot_token
    if not token:
        return False
    url = f"{base_url}{token}/sendRichMessage"
    payload: dict = {"chat_id": chat_id, "rich_message": {"markdown": markdown_text}}
    if reply_to is not None:
        payload["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
        data = resp.json()
        if data.get("ok"):
            return True
        logger.warning("sendRichMessage rejected: %s", data)
        return False
    except Exception as exc:  # network / json / anything -> fall back
        logger.warning("sendRichMessage error: %s", exc)
        return False


async def _safe_delete(bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except TelegramError as exc:
        logger.debug("scratch delete failed (%s): %s", message_id, exc)


async def _send_txt(bot, chat_id: int, full_text: str) -> None:
    buffer = BytesIO(full_text.encode("utf-8"))
    buffer.name = "response.txt"
    buffer.seek(0)
    await bot.send_document(chat_id=chat_id, document=buffer, filename="response.txt")


_REASONING_HEADER = "💭 Reasoning"
_TG_TEXT_LIMIT = 4096  # a message's text is capped at 4096 UTF-16 code units


def _tg_len(text: str) -> int:
    """Length in UTF-16 code units — how Telegram counts toward the 4096 limit."""
    return len(text.encode("utf-16-le")) // 2


def _reasoning_block(reasoning: str) -> str:
    """A collapsed (expandable) MarkdownV2 blockquote carrying the reasoning."""
    return to_expandable_blockquote(f"{_REASONING_HEADER}\n{reasoning}")


def _fit_reasoning_block(reasoning: str, budget: int) -> Optional[str]:
    """The collapsed reasoning block, trimmed so it fits ``budget`` UTF-16 units.

    Returns the full block when it already fits; otherwise binary-searches the
    largest reasoning prefix whose block (with a trailing ``…`` truncation marker)
    stays within budget. Returns ``None`` when even a minimal block overflows —
    i.e. the answer alone already consumes almost the whole message."""
    block = _reasoning_block(reasoning)
    if _tg_len(block) <= budget:
        return block
    lo, hi, best = 0, len(reasoning), None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _reasoning_block(reasoning[:mid].rstrip() + " …")
        if _tg_len(candidate) <= budget:
            best, lo = candidate, mid + 1  # fits: try to keep more
        else:
            hi = mid - 1
    return best


async def _markdown_v2_in_place(
    bot,
    cfg: Settings,
    chat_id: int,
    full_text: str,
    scratch_message_ids: List[int],
    reasoning: str = "",
) -> None:
    """Edit the streamed scratch message in place with MarkdownV2 — with the collapsed
    reasoning blockquote above the answer. Escaping (and the reasoning block) can push a
    near-rollover answer past 4096, so render the richest thing that fits: reasoning is
    TRIMMED (not dropped) to whatever budget the answer leaves, keeping the tap-to-expand
    block alive for the long-reasoning case it exists for. If even answer-only markdown
    overflows, the already-streamed plain answer is left as-is. Only the single-message
    case is re-rendered; a rolled-over multi-message answer stays plain."""
    if len(scratch_message_ids) != 1:
        return  # multi-message rollover: leave the plain streamed text as-is
    mid = scratch_message_ids[0]
    try:
        answer_md = to_markdown_v2(full_text)  # inside try: a conversion slip keeps plain
        answer_units = _tg_len(answer_md)
        if answer_units > _TG_TEXT_LIMIT:
            logger.info(
                "MarkdownV2 finalize: answer alone is %d > %d units; keeping the plain answer",
                answer_units, _TG_TEXT_LIMIT,
            )
            return  # even the bare answer overflows -> the plain answer is already shown
        md = answer_md
        if reasoning:
            budget = _TG_TEXT_LIMIT - answer_units - 2  # 2 units for the "\n\n" separator
            block = _fit_reasoning_block(reasoning, budget)
            if block is not None:
                md = block + "\n\n" + answer_md  # reasoning trimmed to fit above the answer
            else:
                logger.info("MarkdownV2 finalize: no room for reasoning; answer-only markdown")
        await bot.edit_message_text(
            text=md, chat_id=chat_id, message_id=mid, parse_mode="MarkdownV2"
        )
    except BadRequest as exc:
        # The failed edit didn't change the message, so the already-streamed plain answer
        # remains correct on screen — nothing to revert.
        logger.info("MarkdownV2 finalize rejected, keeping the plain answer: %s", exc)


async def finalize(
    bot,
    cfg: Settings,
    chat_id: int,
    full_text: str,
    scratch_message_ids: List[int],
    reasoning: str = "",
) -> None:
    """Apply cfg.formatting.finalize_mode to the completed answer, optionally with the
    model's reasoning as a collapsed (tap-to-expand) block above it. (SPEC §8, §10)

    'rich' sends via sendRichMessage, which the official API supports — no local
    Bot API server needed (set rich_requires_base_url to gate it on a base_url);
    on success it deletes the scratch messages
    so the answer re-appears richly formatted; otherwise it degrades gracefully
    (>txt_fallback_over -> .txt document, else markdown_v2 in place). Reasoning
    collapse needs markup, so 'plain' and the .txt fallback show the answer only."""
    fmt = cfg.formatting
    mode = fmt.finalize_mode
    reasoning = (reasoning or "").strip()

    if mode == "plain" or not scratch_message_ids:
        return  # leave the plain streamed text as-is

    if mode == "markdown_v2":
        await _markdown_v2_in_place(bot, cfg, chat_id, full_text, scratch_message_ids, reasoning)
        return

    # mode == "rich"
    may_rich = bool(cfg.telegram.base_url) or not fmt.rich_requires_base_url
    if may_rich and len(full_text) <= fmt.rich_max_chars:
        rich_md = full_text
        if reasoning:  # rich messages take standard Markdown; a '>' blockquote for reasoning
            quoted = "\n".join("> " + ln for ln in f"{_REASONING_HEADER}\n{reasoning}".split("\n"))
            rich_md = quoted + "\n\n" + full_text
        if await send_rich_message(cfg, chat_id, rich_md):
            for mid in scratch_message_ids:
                await _safe_delete(bot, chat_id, mid)
            return

    # rich not attempted / failed / too big -> degrade
    if len(full_text) > fmt.txt_fallback_over:
        await _send_txt(bot, chat_id, full_text)
        for mid in scratch_message_ids:
            await _safe_delete(bot, chat_id, mid)
        return

    await _markdown_v2_in_place(bot, cfg, chat_id, full_text, scratch_message_ids, reasoning)

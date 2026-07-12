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
    pantherabot; base_url follows the PTB convention (ends in ``/bot``)."""
    base_url = cfg.telegram.base_url
    token = cfg.telegram_bot_token
    if not base_url or not token:
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


def _reasoning_block(reasoning: str) -> str:
    """A collapsed (expandable) MarkdownV2 blockquote carrying the reasoning."""
    return to_expandable_blockquote(f"{_REASONING_HEADER}\n{reasoning}")


async def _markdown_v2_in_place(
    bot,
    cfg: Settings,
    chat_id: int,
    full_text: str,
    scratch_message_ids: List[int],
    reasoning: str = "",
) -> None:
    """Edit the streamed scratch message in place with MarkdownV2 — optionally with a
    collapsed reasoning blockquote above the answer. On a parse error revert to plain
    (answer only). Only the single-message case is re-rendered (the common one); a
    rolled-over multi-message answer stays plain — markup can't be re-segmented safely."""
    if len(scratch_message_ids) != 1:
        return  # multi-message rollover: leave the plain streamed text as-is
    mid = scratch_message_ids[0]
    try:
        md = to_markdown_v2(full_text)  # inside try: a conversion slip also reverts to plain
        if reasoning:
            md = _reasoning_block(reasoning) + "\n\n" + md
        await bot.edit_message_text(
            text=md, chat_id=chat_id, message_id=mid, parse_mode="MarkdownV2"
        )
    except BadRequest as exc:
        logger.info("MarkdownV2 finalize rejected, reverting to plain: %s", exc)
        try:  # plain revert: answer only — reasoning needs markup to collapse
            await bot.edit_message_text(text=full_text, chat_id=chat_id, message_id=mid)
        except BadRequest as exc2:
            logger.debug("plain revert edit failed (likely 'not modified'): %s", exc2)


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

    'rich' only attempts sendRichMessage when telegram.base_url is set (unless
    rich_requires_base_url is False); on success it deletes the scratch messages
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

"""Finalize formatting — the pantherabot rich chain. See SPEC.md §10.

Streaming stays plain (parse_mode=None) so intermediate edits never throw. Exactly
ONE finalize pass runs on the whole answer:

    rich  -> sendRichMessage (32768 cap) -> .txt document -> markdown_v2 -> plain
    markdown_v2 -> edit in place with MarkdownV2 (placeholder-escape) -> plain on error
    plain -> leave the streamed messages as-is

``send_rich_message`` and the MarkdownV2 placeholder-escape are ported from
pantherabot (server/server.py). STATUS: stub (Milestone 3).
"""

from __future__ import annotations

from typing import List

from .config import Settings

# pantherabot's placeholder-token trick: protect intended markup from escape_markdown,
# then restore it after escaping the rest.
_MARKUP = {"&&&": "*", "%%%": "_", "@@@": "__", "~~~": "~"}


def to_markdown_v2(text: str) -> str:
    """Escape ``text`` for Telegram MarkdownV2 while preserving intended markup.
    Reuses pantherabot's UUID-placeholder approach around escape_markdown(version=2)."""
    raise NotImplementedError  # Milestone 3


async def send_rich_message(cfg: Settings, chat_id: int, markdown_text: str, reply_to=None) -> bool:
    """POST {base_url}/bot{token}/sendRichMessage. Returns True on ok, else False so
    the caller can fall down the chain. Ported from pantherabot. (SPEC §10)"""
    raise NotImplementedError  # Milestone 3


async def finalize(
    bot,
    cfg: Settings,
    chat_id: int,
    full_text: str,
    scratch_message_ids: List[int],
) -> None:
    """Apply cfg.formatting.finalize_mode to the completed answer. (SPEC §10)

    'rich' only attempts sendRichMessage when telegram.base_url is set (unless
    rich_requires_base_url is False); on success it deletes the scratch messages
    so the answer re-appears richly formatted; otherwise it degrades gracefully.
    """
    raise NotImplementedError  # Milestone 3

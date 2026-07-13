"""Milestone 3 gate: the pantherabot finalize chain — MarkdownV2 conversion,
sendRichMessage POST, and the rich -> .txt -> markdown_v2 -> plain degradation.
FakeBot + monkeypatched transport; NO real Telegram/HTTP. (PLAN M3 DoD, SPEC §10)"""

from __future__ import annotations

import pytest
from telegram.error import BadRequest

import abliteratedllmbot.formatting as fmt
from abliteratedllmbot.formatting import (
    finalize,
    format_model_expiry,
    output_summary,
    rich_enabled,
    send_rich_message,
    to_expandable_blockquote,
    to_markdown_v2,
)


# --- to_markdown_v2 ---------------------------------------------------------

@pytest.mark.parametrize(
    "src, expected",
    [
        ("Hello **world**!", "Hello *world*\\!"),
        ("a *italic* b", "a _italic_ b"),
        ("_under_ too", "_under_ too"),
        ("~~gone~~", "~gone~"),
        ("||secret||", "||secret||"),
        ("dots. and (x)", "dots\\. and \\(x\\)"),
        ("2 * 3 = 6", "2 \\* 3 \\= 6"),  # lone * not treated as italic
    ],
)
def test_to_markdown_v2_cases(src, expected):
    assert to_markdown_v2(src) == expected


def test_to_markdown_v2_inline_code_body_not_escaped_as_markup():
    # inside code only backtick/backslash are escaped; '*' and '_' stay literal
    assert to_markdown_v2("run `a_b*c`") == "run `a_b*c`"


def test_to_markdown_v2_fenced_code_preserved():
    out = to_markdown_v2("```python\nx = a_b * c\n```")
    assert out == "```python\nx = a_b * c\n```"


def test_to_markdown_v2_link_url_dots_not_escaped():
    assert to_markdown_v2("[site](https://x.com/a.b)") == "[site](https://x.com/a.b)"


def test_to_markdown_v2_no_sentinel_leak():
    for s in ["**a** *b* `c` ~~d~~ ||e|| [f](http://g.h)", "plain. text!", "```\ncode\n```"]:
        assert "\x00" not in to_markdown_v2(s)


@pytest.mark.parametrize(
    "src",
    [
        "**See [docs](http://x.com)**",       # bold wraps link
        "**run `pip install x`**",            # bold wraps inline code
        "~~old `v1`~~ new",                    # strike wraps code
        "a **bold [link](http://c.d)** and `code`",
        "\x00 literal nul \x005\x00 injected",  # raw sentinel in real input
    ],
)
def test_to_markdown_v2_nested_no_sentinel_leak(src):
    # Regression for the placeholder-restore bug: emphasis wrapping an already-stashed
    # span must NOT leak a NUL sentinel or drop the inner content.
    assert "\x00" not in to_markdown_v2(src)


def test_to_markdown_v2_nested_preserves_inner_content():
    out = to_markdown_v2("**See [docs](http://x.com)**")
    assert "docs" in out and "http://x.com" in out and out == "*See [docs](http://x.com)*"


# --- to_expandable_blockquote (collapsed reasoning) -------------------------

def test_expandable_blockquote_quotes_lines_and_marks_expandable():
    out = to_expandable_blockquote("line one\nline two")
    assert out == ">line one\n>line two||"  # each line quoted, ends with ||


def test_expandable_blockquote_escapes_special_chars():
    out = to_expandable_blockquote("cost is 3.5 (a-b)")
    # MarkdownV2 specials escaped inside the quote; still one expandable block
    assert out == ">cost is 3\\.5 \\(a\\-b\\)||"
    assert "\x00" not in out


def test_expandable_blockquote_escapes_literal_pipe_and_gt():
    # a literal > or | in reasoning must be escaped so it can't form markers
    out = to_expandable_blockquote("a>b | c")
    assert out.startswith(">a\\>b \\| c") and out.endswith("||")


# --- finalize with collapsed reasoning --------------------------------------

async def test_finalize_markdown_v2_prepends_collapsed_reasoning(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "the answer", [1000], reasoning="because reasons")
    edits = bot.calls_of("edit")
    assert len(edits) == 1 and edits[0]["parse_mode"] == "MarkdownV2"
    text = edits[0]["text"]
    assert text.startswith(">💭 Reasoning\n>because reasons||")
    assert text.endswith("the answer")


async def test_finalize_no_reasoning_is_just_the_answer(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "the answer", [1000], reasoning="")
    assert bot.calls_of("edit")[0]["text"] == "the answer"  # no blockquote


async def test_finalize_markdown_rejected_keeps_streamed_plain(make_bot, make_cfg):
    def on_edit(nth, kw):
        if kw.get("parse_mode") == "MarkdownV2":
            raise BadRequest("can't parse entities")

    bot = make_bot(on_edit=on_edit)
    bot.messages[1000] = "the answer"  # the streamed plain answer already on screen
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "the answer", [1000], reasoning="some reasoning")
    edits = bot.calls_of("edit")
    # only the failed MarkdownV2 attempt — no redundant plain re-edit (would be 'not modified')
    assert len(edits) == 1 and edits[0]["parse_mode"] == "MarkdownV2"
    assert bot.messages[1000] == "the answer"  # streamed plain preserved


async def test_finalize_trims_reasoning_to_fit_when_merged_overflows_4096(make_bot, make_cfg):
    # merged (full reasoning+answer) > 4096 -> reasoning is TRIMMED to fit, not dropped,
    # and the whole message still fits under the 4096 cap.
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    answer = "a" * 3000
    await finalize(bot, cfg, 1, answer, [1000], reasoning="r" * 2000)
    edits = bot.calls_of("edit")
    assert len(edits) == 1 and edits[0]["parse_mode"] == "MarkdownV2"
    text = edits[0]["text"]
    assert text.startswith(">💭 Reasoning")            # collapsed reasoning survives
    assert text.rstrip().endswith(answer)               # full answer preserved below it
    assert "…" in text and "r" in text                  # trimmed with an ellipsis marker
    assert fmt._tg_len(text) <= fmt._TG_TEXT_LIMIT       # and the whole thing fits


async def test_finalize_reasoning_dropped_only_when_answer_nearly_fills_message(make_bot, make_cfg):
    # answer alone fits but leaves no room for even a minimal reasoning block -> answer-only md
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    answer = "a" * 4090  # answer_md ~4090 units; < 4096 but no budget left for reasoning
    await finalize(bot, cfg, 1, answer, [1000], reasoning="r" * 2000)
    edits = bot.calls_of("edit")
    assert len(edits) == 1 and edits[0]["parse_mode"] == "MarkdownV2"
    assert "Reasoning" not in edits[0]["text"] and edits[0]["text"] == answer


async def test_finalize_keeps_plain_when_answer_markdown_overflows_4096(make_bot, make_cfg):
    # escaping doubles length: 3000 dots -> ~6000 md chars > 4096 -> no edit; plain stays
    bot = make_bot()
    bot.messages[1000] = "placeholder"
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "." * 3000, [1000])
    assert bot.calls_of("edit") == []  # no markdown attempt; streamed plain already correct


# --- send_rich_message ------------------------------------------------------

class _Resp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


class _FakeHTTP:
    """Stands in for httpx.AsyncClient used as an async context manager."""

    last_payload = None

    def __init__(self, data=None, exc=None, **kw):
        self._d = data
        self._exc = exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        _FakeHTTP.last_payload = {"url": url, "json": json}
        if self._exc is not None:
            raise self._exc
        return _Resp(self._d)


def _rich_cfg(make_cfg, monkeypatch, token="TESTTOKEN", base="http://localhost:8081/bot"):
    if token is not None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    return make_cfg(telegram={"base_url": base})


async def test_send_rich_no_base_url_uses_official_api(make_cfg, monkeypatch):
    """No local Bot API server needed: api.telegram.org implements sendRichMessage."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TESTTOKEN")
    cfg = make_cfg(telegram={"base_url": None})
    monkeypatch.setattr(fmt.httpx, "AsyncClient", lambda **kw: _FakeHTTP(data={"ok": True}))
    assert await send_rich_message(cfg, 1, "hi") is True
    assert _FakeHTTP.last_payload["url"] == (
        "https://api.telegram.org/botTESTTOKEN/sendRichMessage"
    )


async def test_send_rich_no_token_returns_false(make_cfg, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    cfg = make_cfg(telegram={"base_url": None})
    assert await send_rich_message(cfg, 1, "hi") is False


async def test_send_rich_ok_true_and_payload_shape(make_cfg, monkeypatch):
    cfg = _rich_cfg(make_cfg, monkeypatch)
    monkeypatch.setattr(fmt.httpx, "AsyncClient", lambda **kw: _FakeHTTP(data={"ok": True}))
    assert await send_rich_message(cfg, 99, "**hi**", reply_to=7) is True
    p = _FakeHTTP.last_payload
    assert p["url"].endswith("TESTTOKEN/sendRichMessage")
    assert p["json"]["chat_id"] == 99
    assert p["json"]["rich_message"] == {"markdown": "**hi**"}
    assert p["json"]["reply_parameters"]["message_id"] == 7


async def test_send_rich_rejected_returns_false(make_cfg, monkeypatch):
    cfg = _rich_cfg(make_cfg, monkeypatch)
    monkeypatch.setattr(fmt.httpx, "AsyncClient", lambda **kw: _FakeHTTP(data={"ok": False}))
    assert await send_rich_message(cfg, 1, "x") is False


async def test_send_rich_exception_returns_false(make_cfg, monkeypatch):
    cfg = _rich_cfg(make_cfg, monkeypatch)
    monkeypatch.setattr(fmt.httpx, "AsyncClient", lambda **kw: _FakeHTTP(exc=RuntimeError("net")))
    assert await send_rich_message(cfg, 1, "x") is False


# --- finalize chain ---------------------------------------------------------

async def test_finalize_plain_is_noop(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "plain"})
    await finalize(bot, cfg, 1, "answer", [1000])
    assert bot.calls == []  # scratch left exactly as streamed


async def test_finalize_markdown_v2_edits_single_message(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "hi **there**", [1000])
    edits = bot.calls_of("edit")
    assert len(edits) == 1
    assert edits[0]["parse_mode"] == "MarkdownV2"
    assert edits[0]["text"] == "hi *there*"


async def test_finalize_markdown_v2_keeps_plain_on_badrequest(make_bot, make_cfg):
    def on_edit(nth, kw):
        if kw.get("parse_mode") == "MarkdownV2":
            raise BadRequest("can't parse entities")

    bot = make_bot(on_edit=on_edit)
    bot.messages[1000] = "tricky *text"  # streamed plain answer already on screen
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "tricky *text", [1000])
    edits = bot.calls_of("edit")
    # only the failed MarkdownV2 attempt; the streamed plain answer is left in place
    assert len(edits) == 1 and edits[0]["parse_mode"] == "MarkdownV2"
    assert bot.messages[1000] == "tricky *text"


async def test_finalize_markdown_v2_multi_message_stays_plain(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "long answer", [1000, 1001])
    assert bot.calls == []  # can't re-segment markup across messages -> leave plain


async def test_finalize_rich_success_deletes_scratch(make_bot, make_cfg, monkeypatch):
    async def fake_rich(cfg, chat_id, text, reply_to=None):
        return True

    monkeypatch.setattr(fmt, "send_rich_message", fake_rich)
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "rich"}, telegram={"base_url": "http://x/bot"})
    await finalize(bot, cfg, 1, "hello", [1000, 1001])
    assert bot.deleted == {1000, 1001}  # scratch removed; rich message re-appears


async def test_finalize_rich_without_base_url_still_attempts_rich(make_bot, make_cfg, monkeypatch):
    """Default rich_requires_base_url=False: the official API serves rich, so a
    missing base_url must NOT silently degrade LaTeX/tables to MarkdownV2."""
    called = {"rich": False}

    async def fake_rich(*a, **k):
        called["rich"] = True
        return True

    monkeypatch.setattr(fmt, "send_rich_message", fake_rich)
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "rich"}, telegram={"base_url": None})
    await finalize(bot, cfg, 1, "hi **there**", [1000])
    assert called["rich"] is True
    assert bot.deleted == {1000}


async def test_finalize_rich_gated_by_base_url_degrades_to_markdown(make_bot, make_cfg, monkeypatch):
    """Opt-in gate: rich_requires_base_url=True + no base_url -> never attempt rich."""
    called = {"rich": False}

    async def fake_rich(*a, **k):
        called["rich"] = True
        return True

    monkeypatch.setattr(fmt, "send_rich_message", fake_rich)
    bot = make_bot()
    cfg = make_cfg(
        formatting={"finalize_mode": "rich", "rich_requires_base_url": True},
        telegram={"base_url": None},
    )
    await finalize(bot, cfg, 1, "hi **there**", [1000])
    assert called["rich"] is False
    edits = bot.calls_of("edit")
    assert len(edits) == 1 and edits[0]["parse_mode"] == "MarkdownV2"


async def test_finalize_rich_long_text_falls_back_to_txt(make_bot, make_cfg, monkeypatch):
    async def fake_rich(*a, **k):
        return False  # rich rejected

    monkeypatch.setattr(fmt, "send_rich_message", fake_rich)
    bot = make_bot()
    cfg = make_cfg(
        formatting={"finalize_mode": "rich", "txt_fallback_over": 100},
        telegram={"base_url": "http://x/bot"},
    )
    big = "x" * 500
    await finalize(bot, cfg, 1, big, [1000])
    assert len(bot.documents) == 1 and bot.documents[0]["filename"] == "response.txt"
    assert bot.deleted == {1000}


async def test_finalize_rich_fails_short_degrades_to_markdown(make_bot, make_cfg, monkeypatch):
    async def fake_rich(*a, **k):
        return False

    monkeypatch.setattr(fmt, "send_rich_message", fake_rich)
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "rich"}, telegram={"base_url": "http://x/bot"})
    await finalize(bot, cfg, 1, "short **b**", [1000])
    edits = bot.calls_of("edit")
    assert len(edits) == 1 and edits[0]["parse_mode"] == "MarkdownV2"


async def test_finalize_no_scratch_is_noop(make_bot, make_cfg):
    bot = make_bot()
    cfg = make_cfg(formatting={"finalize_mode": "rich"})
    await finalize(bot, cfg, 1, "x", [])
    assert bot.calls == []


async def test_finalize_rich_over_max_chars_skips_rich(make_bot, make_cfg, monkeypatch):
    called = {"rich": False}

    async def fake_rich(*a, **k):
        called["rich"] = True
        return True

    monkeypatch.setattr(fmt, "send_rich_message", fake_rich)
    bot = make_bot()
    cfg = make_cfg(
        formatting={"finalize_mode": "rich", "rich_max_chars": 50, "txt_fallback_over": 100},
        telegram={"base_url": "http://x/bot"},
    )
    big = "x" * 500  # exceeds rich_max_chars(50) -> rich must be skipped, degrade to .txt
    await finalize(bot, cfg, 1, big, [1000])
    assert called["rich"] is False
    assert len(bot.documents) == 1 and bot.deleted == {1000}


# --- startup banner ---------------------------------------------------------
# The banner exists to make a silent rich->markdown_v2 degradation loud (a stale
# `docker compose restart` shipped old code against new config, and LaTeX quietly
# stopped rendering). If it ever disagrees with finalize(), it is worse than useless.

def test_banner_reports_rich_on_official_api(make_cfg):
    cfg = make_cfg(formatting={"finalize_mode": "rich"}, telegram={"base_url": None})
    assert rich_enabled(cfg) is True
    out = output_summary(cfg)
    assert "rich" in out and "api.telegram.org" in out
    assert "NOT render" not in out


def test_banner_shouts_when_rich_is_gated_off(make_cfg):
    cfg = make_cfg(
        formatting={"finalize_mode": "rich", "rich_requires_base_url": True},
        telegram={"base_url": None},
    )
    assert rich_enabled(cfg) is False
    out = output_summary(cfg)
    assert "DEGRADED" in out and "LaTeX will NOT render" in out


def test_banner_shouts_when_mode_cannot_render_latex(make_cfg):
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    assert "LaTeX will NOT render" in output_summary(cfg)


@pytest.mark.parametrize("base,gated", [(None, False), ("http://x/bot", False), ("http://x/bot", True)])
def test_banner_never_disagrees_with_finalize(make_cfg, base, gated):
    """rich_enabled() is what finalize() branches on — they cannot drift apart."""
    cfg = make_cfg(
        formatting={"finalize_mode": "rich", "rich_requires_base_url": gated},
        telegram={"base_url": base},
    )
    claims_rich = "sendRichMessage" in output_summary(cfg)
    assert claims_rich is rich_enabled(cfg)


# --- /status expiry ---------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

_NOW = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)


def test_pinned_model_is_not_reported_as_an_imminent_unload():
    """keep_alive=-1 -> Ollama emits a sentinel expiry (year 2318). The old code
    printed only %H:%M:%S, turning "never" into a time that read as imminent."""
    sentinel = datetime(2318, 10, 23, 11, 52, 15, tzinfo=timezone(timedelta(hours=4)))
    out = format_model_expiry(sentinel, now=_NOW)
    assert "pinned" in out and "no auto-unload" in out
    assert "11:52" not in out


def test_real_expiry_leads_with_a_timezone_proof_duration():
    out = format_model_expiry(_NOW + timedelta(minutes=30), now=_NOW)
    assert "auto-unloads in 30m" in out


def test_real_expiry_over_an_hour_is_humanised():
    assert "1h 30m" in format_model_expiry(_NOW + timedelta(minutes=90), now=_NOW)


def test_absolute_time_is_rendered_in_OUR_zone_not_the_ollama_hosts():
    """Ollama stamps expires_at with ITS host's offset; ours may differ. The same
    instant must print as our local wall clock."""
    same_instant_odd_offset = (_NOW + timedelta(minutes=30)).astimezone(
        timezone(timedelta(hours=9))  # pretend the Ollama host is in Tokyo
    )
    ours = format_model_expiry(_NOW + timedelta(minutes=30), now=_NOW)
    theirs = format_model_expiry(same_instant_odd_offset, now=_NOW)
    assert ours == theirs  # identical instant -> identical rendering


def test_expired_and_missing_are_not_crashes():
    assert "expiring now" in format_model_expiry(_NOW - timedelta(minutes=1), now=_NOW)
    assert format_model_expiry(None) == "unload time unknown"
    assert format_model_expiry("garbage") == "unload time unknown"

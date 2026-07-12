"""Milestone 3 gate: the pantherabot finalize chain — MarkdownV2 conversion,
sendRichMessage POST, and the rich -> .txt -> markdown_v2 -> plain degradation.
FakeBot + monkeypatched transport; NO real Telegram/HTTP. (PLAN M3 DoD, SPEC §10)"""

from __future__ import annotations

import pytest
from telegram.error import BadRequest

import abliteratedllmbot.formatting as fmt
from abliteratedllmbot.formatting import finalize, send_rich_message, to_markdown_v2


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


async def test_send_rich_no_base_url_returns_false(make_cfg):
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


async def test_finalize_markdown_v2_reverts_to_plain_on_badrequest(make_bot, make_cfg):
    def on_edit(nth, kw):
        if kw.get("parse_mode") == "MarkdownV2":
            raise BadRequest("can't parse entities")

    bot = make_bot(on_edit=on_edit)
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    await finalize(bot, cfg, 1, "tricky *text", [1000])
    edits = bot.calls_of("edit")
    assert len(edits) == 2  # MarkdownV2 attempt, then plain revert
    assert "parse_mode" not in edits[1] and edits[1]["text"] == "tricky *text"


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


async def test_finalize_rich_no_base_url_degrades_to_markdown(make_bot, make_cfg, monkeypatch):
    called = {"rich": False}

    async def fake_rich(*a, **k):
        called["rich"] = True
        return True

    monkeypatch.setattr(fmt, "send_rich_message", fake_rich)
    bot = make_bot()
    # default: rich_requires_base_url=True, base_url=None -> never attempt rich
    cfg = make_cfg(formatting={"finalize_mode": "rich"}, telegram={"base_url": None})
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

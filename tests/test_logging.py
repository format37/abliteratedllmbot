"""The bot token must never reach the log: httpx is quieted, and the formatter masks
the token anywhere it appears, tracebacks included."""

from __future__ import annotations

import logging

from abliteratedllmbot.__main__ import LOG_FORMAT, RedactingFormatter, configure_logging
from abliteratedllmbot.config import Settings

TOKEN = "123456:SECRET-token_value"


def _record(msg, *args, exc_info=None):
    return logging.LogRecord("httpx", logging.INFO, __file__, 1, msg, args, exc_info)


def test_formatter_masks_the_token_in_the_message():
    fmt = RedactingFormatter(LOG_FORMAT, TOKEN)
    line = fmt.format(_record('HTTP Request: POST %s "200 OK"',
                              f"https://api.telegram.org/bot{TOKEN}/getUpdates"))
    assert TOKEN not in line
    assert "bot<redacted>/getUpdates" in line


def test_formatter_masks_the_token_in_a_traceback():
    fmt = RedactingFormatter(LOG_FORMAT, TOKEN)
    try:
        raise RuntimeError(f"failed: https://api.telegram.org/bot{TOKEN}/getMe")
    except RuntimeError:
        import sys
        line = fmt.format(_record("boom", exc_info=sys.exc_info()))
    assert TOKEN not in line and "<redacted>" in line


def test_formatter_without_a_token_leaves_text_alone():
    assert RedactingFormatter(LOG_FORMAT, "").format(_record("plain")).endswith("plain")


def test_configure_logging_quiets_httpx(monkeypatch):
    httpx_logger = logging.getLogger("httpx")
    monkeypatch.setattr(httpx_logger, "level", logging.NOTSET)
    configure_logging(Settings(TELEGRAM_BOT_TOKEN=TOKEN))
    assert httpx_logger.getEffectiveLevel() == logging.WARNING

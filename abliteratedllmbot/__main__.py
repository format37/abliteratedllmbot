"""Entrypoint: load config, build the Application, run long polling. (SPEC §13)

    python -m abliteratedllmbot            # uses ./config.yaml
    python -m abliteratedllmbot my.yaml    # explicit path
"""

from __future__ import annotations

import logging
import os
import sys

from .config import Settings, load_config
from .telegram_bot import allowed_updates, build_application

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RedactingFormatter(logging.Formatter):
    """Masks the bot token wherever it lands in a formatted line, tracebacks included.
    The Bot API puts the token in every request path (/bot<token>/getUpdates), so any
    library that logs a URL leaks it."""

    def __init__(self, fmt: str, secret: str) -> None:
        super().__init__(fmt)
        self._secret = secret

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text.replace(self._secret, "<redacted>") if self._secret else text


def configure_logging(cfg: Settings) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter(LOG_FORMAT, cfg.telegram_bot_token))
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        handlers=[handler],
    )
    # httpx logs every request URL at INFO: the token, plus a getUpdates line on every
    # long poll. Its warnings and errors still come through.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "ABLITERATEDLLMBOT_CONFIG", "config.yaml"
    )
    cfg = load_config(config_path)
    configure_logging(cfg)
    if not cfg.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (put it in .env).")

    app = build_application(cfg)
    app.run_polling(
        allowed_updates=allowed_updates(app),  # "message" alone never delivers button taps
        drop_pending_updates=cfg.telegram.drop_pending_updates,
        poll_interval=cfg.telegram.poll_interval,
        timeout=cfg.telegram.long_poll_timeout,
    )


if __name__ == "__main__":
    main()

"""Entrypoint: load config, build the Application, run long polling. (SPEC §13)

    python -m abliteratedllmbot            # uses ./config.yaml
    python -m abliteratedllmbot my.yaml    # explicit path
"""

from __future__ import annotations

import logging
import os
import sys

from .config import load_config
from .telegram_bot import build_application


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "ABLITERATEDLLMBOT_CONFIG", "config.yaml"
    )
    cfg = load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not cfg.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (put it in .env).")

    app = build_application(cfg)
    app.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=cfg.telegram.drop_pending_updates,
        poll_interval=cfg.telegram.poll_interval,
        timeout=cfg.telegram.long_poll_timeout,
    )


if __name__ == "__main__":
    main()

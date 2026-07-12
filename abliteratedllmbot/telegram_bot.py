"""Telegram layer: PTB handlers, access control, typing, per-chat task registry.
See SPEC.md §8 (concurrency), §9 (images), §11 (access/commands).

STATUS: stub (Milestone 4).
"""

from __future__ import annotations

import functools
from typing import Callable

from telegram import Update
from telegram.ext import Application, ApplicationBuilder, ContextTypes

from .config import Settings
from .ollama_client import OllamaClient
from .history import InMemoryStore


def access_control(cfg: Settings) -> Callable:
    """Decorator: allow only cfg.telegram.allowed_user_ids; else apply
    cfg.access.on_denied (reply_once | reply_always | ignore). (SPEC §11)"""
    allowed = set(cfg.telegram.allowed_user_ids)

    def decorator(handler: Callable) -> Callable:
        @functools.wraps(handler)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user = update.effective_user
            if user and user.id in allowed:
                return await handler(update, context)
            # TODO: on_denied policy using a bot_data notified-set (SPEC §11)
            return None
        return wrapper

    return decorator


async def on_startup(app: Application) -> None:
    """post_init: build the shared OllamaClient, warm capabilities. (SPEC §5)"""
    raise NotImplementedError  # Milestone 4


async def on_shutdown(app: Application) -> None:
    raise NotImplementedError  # Milestone 4


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main flow: cancel-and-supersede any live task for this chat, register a new
    one, start typing, build messages, run RenderDriver over chat_stream. (SPEC §8)"""
    raise NotImplementedError  # Milestone 4


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """images.mode: reject_note | caption_only | vision_fallback. (SPEC §9)"""
    raise NotImplementedError  # Milestone 4


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    raise NotImplementedError  # Milestone 4


def register_handlers(app: Application, cfg: Settings) -> None:
    """Register /start /help /reset /stats /think + text + photo + error handlers,
    each wrapped by access_control(cfg). (SPEC §11)"""
    raise NotImplementedError  # Milestone 4


def build_application(cfg: Settings) -> Application:
    """ApplicationBuilder with token, optional base_url, concurrent_updates(True),
    post_init/post_shutdown hooks; then register_handlers. (SPEC §13, PLAN M4)"""
    builder = ApplicationBuilder().token(cfg.telegram_bot_token).concurrent_updates(True)
    if cfg.telegram.base_url:
        builder = builder.base_url(cfg.telegram.base_url)
    app = builder.post_init(on_startup).post_shutdown(on_shutdown).build()
    app.bot_data["cfg"] = cfg
    app.bot_data["history"] = InMemoryStore()
    register_handlers(app, cfg)
    return app

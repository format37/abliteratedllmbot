"""Telegram layer: PTB handlers, access control, typing, per-chat task registry.
See SPEC.md §8 (concurrency), §9 (images), §11 (access/commands)."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Callable, Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import Conflict, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Settings
from .errors import BotError
from .formatting import output_summary
from .history import ContextUsage, InMemoryStore, Message, build_messages
from .ollama_client import OllamaClient
from .streaming import RenderDriver

logger = logging.getLogger(__name__)


# --- access control (SPEC §11) ---------------------------------------------

def access_control(cfg: Settings) -> Callable:
    """Decorator: allow only cfg.telegram.allowed_user_ids; else apply
    cfg.access.on_denied (reply_once | reply_always | ignore). (SPEC §11)"""
    allowed = set(cfg.telegram.allowed_user_ids)
    policy = cfg.access.on_denied

    def decorator(handler: Callable) -> Callable:
        @functools.wraps(handler)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user = update.effective_user
            if user and user.id in allowed:
                return await handler(update, context)
            if policy == "ignore":
                return None
            notified: set = context.bot_data.setdefault("_denied_notified", set())
            if policy == "reply_once" and user and user.id in notified:
                return None
            if user:
                notified.add(user.id)
            message = update.effective_message
            if message is not None:
                try:
                    await message.reply_text(cfg.access.denial_message)
                except TelegramError as exc:
                    logger.debug("denial reply failed: %s", exc)
            return None

        return wrapper

    return decorator


# --- typing indicator (SPEC §8) --------------------------------------------

class TypingIndicator:
    """Sends ChatAction.TYPING every refresh_interval until stopped — most valuable
    during the ~156s cold-load gap. ``stop`` is sync + idempotent so it is safe to
    call from a delta callback or a cancellation finally-block."""

    def __init__(self, bot, chat_id: int, cfg: Settings) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.interval = cfg.typing.refresh_interval
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        try:
            while True:
                try:
                    await self.bot.send_chat_action(self.chat_id, ChatAction.TYPING)
                except TelegramError as exc:
                    logger.debug("send_chat_action failed: %s", exc)
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            return

    def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None


# --- lifecycle (SPEC §5, §13) ----------------------------------------------

async def on_startup(app: Application) -> None:
    """post_init: build the shared OllamaClient, warm capabilities. (SPEC §5)"""
    cfg: Settings = app.bot_data["cfg"]
    client = OllamaClient(cfg.ollama, cfg.reasoning, cfg.sanitize)
    app.bot_data["ollama"] = client

    # Say what the running code will actually DO, not what the config asks for. A stale
    # `docker compose restart` (no --build) keeps old code against new config, and the
    # resulting degradation is otherwise silent.
    logger.info("Output: %s", output_summary(cfg))
    logger.info("Images: mode=%s (photos are %s)", cfg.images.mode,
                "DROPPED, only the caption is sent" if cfg.images.mode == "reject_note"
                else "sent to " + cfg.ollama.vision_model)

    try:
        await client.startup()  # warm caps for model (+ vision_model); tolerant of vision
        logger.info("Ollama reachable; capability cache warm.")
    except BotError as exc:
        # Non-fatal: the server may be starting / cold. Per-request errors stay friendly.
        logger.warning("Ollama warmup failed (continuing): %s", exc)


async def on_shutdown(app: Application) -> None:
    client: Optional[OllamaClient] = app.bot_data.get("ollama")
    if client is not None:
        await client.close()


# --- the generation flow (SPEC §8) -----------------------------------------

def _chat_cfg(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> Settings:
    """Apply the per-chat /think toggle (SPEC §11) without mutating shared Settings."""
    cfg: Settings = context.bot_data["cfg"]
    hidden: set = context.bot_data.get("_think_off", set())
    if chat_id in hidden and cfg.reasoning.show_thinking:
        return cfg.model_copy(
            update={"reasoning": cfg.reasoning.model_copy(update={"show_thinking": False})}
        )
    return cfg


def _dispatch_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    user_text: str,
    images: Optional[list] = None,
    model: Optional[str] = None,
    store_turn: bool = True,
) -> None:
    """Cancel-and-supersede: atomically register a new per-chat task BEFORE the first
    await, then cancel any prior live task for this chat. (SPEC §8 — frozen policy)"""
    chat_id = update.effective_chat.id
    registry: dict = context.bot_data["tasks"]

    task = asyncio.create_task(
        _run_generation(
            update, context, chat_id,
            user_text=user_text, images=images, model=model, store_turn=store_turn,
        )
    )
    old = registry.get(chat_id)
    registry[chat_id] = task  # atomic: set before any await so two messages can't both start
    if old is not None and not old.done():
        old.cancel()  # supersede -> RenderDriver._on_cancel appends the superseded note

    def _cleanup(finished: asyncio.Task, cid: int = chat_id) -> None:
        if registry.get(cid) is finished:
            registry.pop(cid, None)
        # Retrieve any exception so asyncio never logs "Task exception was never
        # retrieved" (the task is detached; PTB's error handler doesn't see it).
        if not finished.cancelled():
            exc = finished.exception()
            if exc is not None:
                logger.error("generation task for chat %s crashed", cid, exc_info=exc)

    task.add_done_callback(_cleanup)


async def _run_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    *,
    user_text: str,
    images: Optional[list],
    model: Optional[str],
    store_turn: bool,
) -> None:
    cfg = _chat_cfg(context, chat_id)
    client: OllamaClient = context.bot_data["ollama"]
    store: InMemoryStore = context.bot_data["history"]
    bot = context.bot

    typing = TypingIndicator(bot, chat_id, cfg) if cfg.typing.enabled else None
    if typing is not None:
        typing.start()

    use_history = store_turn and cfg.memory.enabled
    driver: Optional[RenderDriver] = None
    try:
        history = store.get(chat_id) if use_history else []
        messages = build_messages(cfg.system_prompt, history, user_text, cfg.memory)

        driver = RenderDriver(bot, chat_id, cfg)
        if typing is not None and cfg.typing.stop_after_first_token:
            driver.on_first_content = typing.stop

        stream = client.chat_stream(messages=messages, model=model, images=images)
        await driver.run(stream)
    except asyncio.CancelledError:
        raise  # superseded: RenderDriver appended the note; do NOT store the partial
    except BotError as exc:
        # Pre-stream failure on the first chunk (Ollama unreachable / model missing /
        # thinking-unsupported). No partial produced -> friendly note, nothing stored.
        await _reply_error(bot, chat_id, str(exc))
        return
    except Exception:
        # Render-path failure AFTER tokens (e.g. a Telegram TimedOut/Forbidden on an
        # edit, or the finalize edit). The answer may be complete on screen — surface
        # the error but still persist what was produced (below) so memory/screen agree.
        logger.exception("generation failed mid/post-stream for chat %s", chat_id)
        await _reply_error(bot, chat_id, "Something went wrong finishing this response.")
    finally:
        if typing is not None:
            typing.stop()

    # Persist the produced turn on success AND on a post-content failure (not on a
    # pre-stream BotError, which returned above, nor on cancel, which re-raised).
    if driver is not None and use_history and driver.answer_text:
        store.append(chat_id, Message("user", user_text))
        store.append(chat_id, Message("assistant", driver.answer_text))
    if driver is not None and driver.stats:
        context.bot_data.setdefault("_last_stats", {})[chat_id] = {
            "stats": driver.stats,
            "done_reason": driver.done_reason,
        }


async def _reply_error(bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=f"⚠️ {text}")
    except TelegramError as exc:
        logger.debug("error reply failed: %s", exc)


# --- handlers (SPEC §9, §11) -----------------------------------------------

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.effective_message.text or "").strip()
    if not text:
        return
    _dispatch_generation(update, context, user_text=text)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """images.mode: reject_note | caption_only | vision_fallback. (SPEC §9)"""
    cfg: Settings = context.bot_data["cfg"]
    message = update.effective_message
    caption = (message.caption or "").strip()
    mode = cfg.images.mode

    if mode == "reject_note":
        try:
            await message.reply_text(cfg.images.note)
        except TelegramError as exc:
            logger.debug("image note failed: %s", exc)
        if caption:
            _dispatch_generation(update, context, user_text=caption)
        return

    if mode == "caption_only":
        if caption:
            _dispatch_generation(update, context, user_text=caption)
        return

    # vision_fallback: route image+caption to the vision model; STATELESS (SPEC §9).
    photo = message.photo[-1]  # largest PhotoSize
    tg_file = await photo.get_file()
    data = await tg_file.download_as_bytearray()
    _dispatch_generation(
        update, context,
        user_text=caption,
        images=[bytes(data)],
        model=cfg.ollama.vision_model,
        store_turn=False,
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(context.error, Conflict):
        # Another instance is polling the same token — fatal (PLAN M4). Stop loudly.
        logger.critical("Conflict: another bot instance is polling this token. Shutting down.")
        context.application.stop_running()
        return
    logger.error("Unhandled handler error", exc_info=context.error)


# --- commands (SPEC §11) ----------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Settings = context.bot_data["cfg"]
    user = update.effective_user
    who = f" You are user `{user.id}`." if user else ""
    await update.effective_message.reply_text(
        f"👋 I'm a local uncensored LLM bot backed by `{cfg.ollama.model}`.{who}\n"
        "Send me a message and I'll stream the answer. /help for commands."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Settings = context.bot_data["cfg"]
    lines = [
        "Commands:",
        "/start — who am I",
        "/help — this help",
        "/reset — clear this chat's memory",
    ]
    if cfg.commands.enable_stats:  # only advertise /stats if it's actually registered
        lines.append("/stats — last generation's tokens/sec")
    lines.append("/think — toggle showing the model's reasoning")
    if cfg.commands.enable_model_admin:
        lines += [
            "/status — which models Ollama has loaded",
            "/load — warm the model into memory",
            "/unload — free the model from memory",
        ]
    await update.effective_message.reply_text("\n".join(lines))


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store: InMemoryStore = context.bot_data["history"]
    store.reset(update.effective_chat.id)
    await update.effective_message.reply_text("🧹 Memory cleared for this chat.")


async def cmd_think(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Settings = context.bot_data["cfg"]
    if not cfg.reasoning.show_thinking:
        # /think only ever HIDES relative to the global setting; be honest when the
        # bot is configured to never show reasoning in the first place.
        await update.effective_message.reply_text(
            "Reasoning display is turned off in this bot's configuration."
        )
        return
    chat_id = update.effective_chat.id
    hidden: set = context.bot_data.setdefault("_think_off", set())
    if chat_id in hidden:
        hidden.discard(chat_id)
        await update.effective_message.reply_text("💭 Reasoning will now be shown.")
    else:
        hidden.add(chat_id)
        await update.effective_message.reply_text("🙈 Reasoning will now be hidden.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Report which models Ollama currently holds in memory (GET /api/ps)."""
    cfg: Settings = context.bot_data["cfg"]
    client: Optional[OllamaClient] = context.bot_data.get("ollama")
    if client is None:
        await update.effective_message.reply_text("Model backend is not ready yet.")
        return
    try:
        loaded = await client.ps()
    except BotError as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")
        return
    if not loaded:
        await update.effective_message.reply_text(
            f"💤 No model loaded. Default is {cfg.ollama.model} — /load to warm it."
        )
        return
    lines = ["🧠 Loaded models:"]
    for m in loaded:
        vram = (m.get("size_vram") or 0) / (1024 ** 3)
        expires = m.get("expires_at")
        when = expires.strftime("%H:%M:%S") if hasattr(expires, "strftime") else str(expires)
        lines.append(f"• {m['model']} — {vram:.1f} GB VRAM, auto-unloads at {when}")
    await update.effective_message.reply_text("\n".join(lines))


async def cmd_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Settings = context.bot_data["cfg"]
    client: Optional[OllamaClient] = context.bot_data.get("ollama")
    if client is None:
        await update.effective_message.reply_text("Model backend is not ready yet.")
        return
    notice = await update.effective_message.reply_text(
        f"⏳ Loading {cfg.ollama.model} … first cold load takes ~2.5 min."
    )
    try:
        await client.load()
    except BotError as exc:
        await _edit_or_reply(notice, update, f"⚠️ {exc}")
        return
    await _edit_or_reply(notice, update, f"✅ {cfg.ollama.model} is loaded and warm.")


async def cmd_unload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Settings = context.bot_data["cfg"]
    client: Optional[OllamaClient] = context.bot_data.get("ollama")
    if client is None:
        await update.effective_message.reply_text("Model backend is not ready yet.")
        return
    try:
        await client.unload()
    except BotError as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")
        return
    await update.effective_message.reply_text(f"💤 {cfg.ollama.model} unloaded from memory.")


async def _edit_or_reply(sent, update: Update, text: str) -> None:
    """Edit the placeholder message if we can, else send a fresh reply."""
    edit = getattr(sent, "edit_text", None)
    try:
        if callable(edit):
            await edit(text)
            return
    except TelegramError as exc:
        logger.debug("edit placeholder failed: %s", exc)
    await update.effective_message.reply_text(text)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    last = context.bot_data.get("_last_stats", {}).get(chat_id)
    if not last:
        await update.effective_message.reply_text("No generation stats yet.")
        return
    cfg: Settings = context.bot_data["cfg"]
    s = last["stats"]
    lines = [f"📊 Last generation ({last.get('done_reason')}):"]
    ec, ed = s.get("eval_count"), s.get("eval_duration")
    if ec and ed:
        lines.append(f"• {ec} tokens in {ed / 1e9:.1f}s = {ec / (ed / 1e9):.1f} tok/s")
    if s.get("prompt_eval_count"):
        lines.append(f"• prompt: {s['prompt_eval_count']} tokens")
    if s.get("load_duration"):
        lines.append(f"• model load: {s['load_duration'] / 1e9:.1f}s")

    # Context occupancy from the model's OWN tokenizer (prompt_eval_count is the full
    # prompt, not just post-cache tokens), so this is exact rather than a guess.
    pec = s.get("prompt_eval_count")
    if pec:
        u = ContextUsage(prompt=pec, answer=ec or 0, limit=cfg.ollama.options.num_ctx)
        lines += [
            "",
            f"🧠 Context: {u.bar()} {u.percent:.0f}%",
            f"• {u.used} / {u.limit} tokens used ({u.headroom} free)",
        ]
        if u.is_tight(cfg.ollama.options.num_predict):
            lines.append(
                f"⚠️ Less than num_predict ({cfg.ollama.options.num_predict}) left — the next "
                "answer may be cut short and the oldest turns evicted. /reset to clear."
            )
        turns = len(context.bot_data["history"].get(chat_id)) // 2
        lines.append(f"• memory: {turns}/{cfg.memory.max_turns} turns kept")
    await update.effective_message.reply_text("\n".join(lines))


# --- wiring (SPEC §13, PLAN M4) --------------------------------------------

def register_handlers(app: Application, cfg: Settings) -> None:
    """Register /start /help /reset /stats /think + text + photo + error handlers,
    each wrapped by access_control(cfg). (SPEC §11)"""
    guard = access_control(cfg)
    app.add_handler(CommandHandler("start", guard(cmd_start)))
    app.add_handler(CommandHandler("help", guard(cmd_help)))
    app.add_handler(CommandHandler("reset", guard(cmd_reset)))
    app.add_handler(CommandHandler("think", guard(cmd_think)))
    if cfg.commands.enable_stats:
        app.add_handler(CommandHandler("stats", guard(cmd_stats)))
    if cfg.commands.enable_model_admin:
        app.add_handler(CommandHandler("status", guard(cmd_status)))
        app.add_handler(CommandHandler("load", guard(cmd_load)))
        app.add_handler(CommandHandler("unload", guard(cmd_unload)))
    app.add_handler(MessageHandler(filters.PHOTO, guard(on_photo)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, guard(on_text)))
    app.add_error_handler(on_error)


def build_application(cfg: Settings) -> Application:
    """ApplicationBuilder with token, optional base_url, concurrent_updates(True),
    post_init/post_shutdown hooks; then register_handlers. (SPEC §13, PLAN M4)"""
    builder = ApplicationBuilder().token(cfg.telegram_bot_token).concurrent_updates(True)
    if cfg.telegram.base_url:
        builder = builder.base_url(cfg.telegram.base_url)
    app = builder.post_init(on_startup).post_shutdown(on_shutdown).build()
    app.bot_data["cfg"] = cfg
    app.bot_data["history"] = InMemoryStore()
    app.bot_data["tasks"] = {}
    register_handlers(app, cfg)
    return app

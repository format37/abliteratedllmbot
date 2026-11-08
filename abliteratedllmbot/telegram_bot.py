"""Telegram layer: PTB handlers, access control, typing, per-chat task registry.
See SPEC.md §8 (concurrency), §9 (images), §11 (access/commands)."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
from typing import Callable, List, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, Conflict, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Settings
from .errors import BotError
from .formatting import format_model_expiry, output_summary
from .history import (
    ContextUsage,
    InMemoryStore,
    Message,
    build_messages,
    estimate_context_usage,
)
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

            # A denied *button* press is refused in the callback answer, not as a chat
            # message: an unanswered callback query spins in the client forever, and
            # under a silent policy the ack must stay silent too.
            query = getattr(update, "callback_query", None)

            if policy == "ignore":
                await _answer_query(query)
                return None
            notified: set = context.bot_data.setdefault("_denied_notified", set())
            if policy == "reply_once" and user and user.id in notified:
                await _answer_query(query)
                return None
            if user:
                notified.add(user.id)
            if query is not None:
                await _answer_query(query, cfg.access.denial_message)
                return None
            message = update.effective_message
            if message is not None:
                try:
                    await message.reply_text(cfg.access.denial_message)
                except TelegramError as exc:
                    logger.debug("denial reply failed: %s", exc)
            return None

        return wrapper

    return decorator


async def _answer_query(query, text: str = "") -> None:
    """Acknowledge a callback query (no-op for a non-callback update)."""
    if query is None:
        return
    try:
        await query.answer(text, show_alert=bool(text))
    except TelegramError as exc:
        logger.debug("callback answer failed: %s", exc)


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

def _qualify(model: str) -> str:
    """Tag a model name the way Ollama reports it. ``/api/tags`` and ``/api/ps`` always
    answer fully-qualified (a config of ``gemma3`` comes back as ``gemma3:latest``), so a
    bare name would match nothing we compare it against: no ✅ in the picker, and — worse —
    ``old in resident`` false, so the old model is silently never unloaded and both stay
    pinned in VRAM. A ':' may only appear in the last path segment (registry/name:tag)."""
    _, _, tail = model.rpartition("/")
    return model if ":" in tail else f"{model}:latest"


def active_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> str:
    """The model this chat talks to: its /model choice, else the configured default."""
    cfg: Settings = context.bot_data["cfg"]
    return _qualify(context.bot_data.get("_model", {}).get(chat_id) or cfg.ollama.model)


def _chat_cfg(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> Settings:
    """Fold this chat's overrides — /think and /model — into a copy of the shared
    Settings, which is never mutated. (SPEC §11)"""
    cfg: Settings = context.bot_data["cfg"]
    overrides: dict = {}
    hidden: set = context.bot_data.get("_think_off", set())
    if chat_id in hidden and cfg.reasoning.show_thinking:
        overrides["reasoning"] = cfg.reasoning.model_copy(update={"show_thinking": False})
    chosen = active_model(context, chat_id)
    if chosen != cfg.ollama.model:
        overrides["ollama"] = cfg.ollama.model_copy(update={"model": chosen})
    return cfg.model_copy(update=overrides) if overrides else cfg


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

    # None -> this chat's active model (/model). An explicit model (the vision path)
    # is a deliberate one-turn override and still wins.
    model = model or cfg.ollama.model

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
        # Expose the live driver so /interrupt can retarget its cancel note. Keyed
        # per chat like `tasks`; a superseding turn simply overwrites the entry.
        context.bot_data.setdefault("drivers", {})[chat_id] = driver

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
        drivers: dict = context.bot_data.get("drivers", {})
        if driver is not None and drivers.get(chat_id) is driver:
            drivers.pop(chat_id, None)

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
    model = active_model(context, update.effective_chat.id)
    user = update.effective_user
    who = f" You are user `{user.id}`." if user else ""
    await update.effective_message.reply_text(
        f"👋 I'm a local uncensored LLM bot backed by `{model}`.{who}\n"
        "Send me a message and I'll stream the answer. /help for commands."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cfg: Settings = context.bot_data["cfg"]
    lines = [
        "Commands:",
        "/start — who am I",
        "/help — this help",
        "/reset — clear this chat's memory",
        "/interrupt — stop the answer being generated",
    ]
    if cfg.commands.enable_stats:  # only advertise /stats if it's actually registered
        lines.append("/stats — last generation's tokens/sec")
    lines.append("/think — toggle showing the model's reasoning")
    if cfg.commands.allow_model_switch:
        lines.append("/model — pick the model this chat talks to")
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


async def cmd_interrupt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop this chat's in-flight generation. Cancelling the task closes the HTTP
    stream, which makes Ollama abort the generation server-side; the partial answer
    is sealed with the interrupted note and NOT stored (same path as supersede)."""
    chat_id = update.effective_chat.id
    cfg: Settings = context.bot_data["cfg"]
    task = context.bot_data["tasks"].get(chat_id)
    if task is None or task.done():
        await update.effective_message.reply_text("Nothing is generating right now.")
        return
    driver = context.bot_data.get("drivers", {}).get(chat_id)
    if driver is not None:
        driver.cancel_note = cfg.streaming.interrupted_note
    task.cancel()
    await update.effective_message.reply_text("⏹️ Interrupted the current generation.")


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
    model = active_model(context, update.effective_chat.id)
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
            f"💤 No model loaded. This chat uses {model} — /load to warm it."
        )
        return
    lines = ["🧠 Loaded models:"]
    for m in loaded:
        vram = (m.get("size_vram") or 0) / (1024 ** 3)
        # Several models can be resident at once now that /model exists — say which
        # one this chat is actually talking to.
        mine = "  ← this chat" if m["model"] == model else ""
        lines.append(
            f"• {m['model']} — {vram:.1f} GB VRAM, "
            f"{format_model_expiry(m.get('expires_at'))}{mine}"
        )
    await update.effective_message.reply_text("\n".join(lines))


async def cmd_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model = active_model(context, update.effective_chat.id)
    client: Optional[OllamaClient] = context.bot_data.get("ollama")
    if client is None:
        await update.effective_message.reply_text("Model backend is not ready yet.")
        return
    notice = await update.effective_message.reply_text(
        f"⏳ Loading {model} … first cold load takes ~2.5 min."
    )
    try:
        await client.load(model)
    except BotError as exc:
        await _edit_or_reply(notice, update, f"⚠️ {exc}")
        return
    await _edit_or_reply(notice, update, f"✅ {model} is loaded and warm.")


async def cmd_unload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model = active_model(context, update.effective_chat.id)
    client: Optional[OllamaClient] = context.bot_data.get("ollama")
    if client is None:
        await update.effective_message.reply_text("Model backend is not ready yet.")
        return
    try:
        await client.unload(model)
    except BotError as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")
        return
    await update.effective_message.reply_text(f"💤 {model} unloaded from memory.")


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


# --- /model: the paged picker (SPEC §11) ------------------------------------

_MODEL_PAGE = "mdl:p:"     # page navigation
_MODEL_PICK = "mdl:s:"     # a model button
_MODEL_NOOP = "mdl:noop"   # the page counter — inert, but must still be answered


def _model_token(name: str) -> str:
    """A short id for a model name. ``callback_data`` is capped at 64 BYTES and model
    names run long (`dagbs/deepseek-coder-v2-lite-instruct:latest`), so buttons carry a
    digest and ``bot_data['_model_tokens']`` maps it back to the exact name shown."""
    return hashlib.blake2s(name.encode(), digest_size=8).hexdigest()


def _render_model_page(
    models: List[str],
    page: int,
    current: str,
    per_page: int,
    tokens: dict,
) -> Tuple[str, InlineKeyboardMarkup]:
    """One page of the picker: a row per full model name, then the nav row. The page
    is clamped, so a stale callback can never scroll past either end."""
    pages = max(1, -(-len(models) // per_page))  # ceil-div
    page = min(max(page, 0), pages - 1)

    rows: List[List[InlineKeyboardButton]] = []
    for name in models[page * per_page : (page + 1) * per_page]:
        token = _model_token(name)
        tokens[token] = name
        mark = "✅ " if name == current else ""
        rows.append(
            [InlineKeyboardButton(f"{mark}{name}", callback_data=_MODEL_PICK + token)]
        )

    if pages > 1:  # one page needs no navigation at all
        nav = []
        if page > 0:  # no left arrow on the first page
            nav.append(InlineKeyboardButton("◀", callback_data=f"{_MODEL_PAGE}{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{pages}", callback_data=_MODEL_NOOP))
        if page < pages - 1:  # no right arrow on the last page
            nav.append(InlineKeyboardButton("▶", callback_data=f"{_MODEL_PAGE}{page + 1}"))
        rows.append(nav)

    header = f"🧠 {len(models)} model{'' if len(models) == 1 else 's'} available"
    if pages > 1:
        header += f" — page {page + 1} of {pages}"
    return f"{header}\nCurrent: {current}", InlineKeyboardMarkup(rows)


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Open the picker on the page that holds the chat's current model."""
    cfg: Settings = context.bot_data["cfg"]
    client: Optional[OllamaClient] = context.bot_data.get("ollama")
    if client is None:
        await update.effective_message.reply_text("Model backend is not ready yet.")
        return
    try:
        models = await client.chat_models()
    except BotError as exc:
        await update.effective_message.reply_text(f"⚠️ {exc}")
        return
    if not models:
        await update.effective_message.reply_text(
            f"No chat-capable models are installed on {cfg.ollama.host}."
        )
        return

    context.bot_data["_models"] = models  # snapshot: paging must not re-hit Ollama
    chat_id = update.effective_chat.id
    current = active_model(context, chat_id)
    per_page = cfg.commands.models_per_page
    page = models.index(current) // per_page if current in models else 0
    text, keyboard = _render_model_page(
        models, page, current, per_page,
        context.bot_data.setdefault("_model_tokens", {}),
    )
    await update.effective_message.reply_text(text, reply_markup=keyboard)


async def on_model_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Callback router for the picker's keyboard: page nav, counter, selection."""
    query = update.callback_query
    data = query.data or ""
    chat_id = update.effective_chat.id
    cfg: Settings = context.bot_data["cfg"]

    if data == _MODEL_NOOP:  # answer it or the client spins on the button forever
        await query.answer()
        return

    if data.startswith(_MODEL_PAGE):
        await query.answer()
        try:
            page = int(data[len(_MODEL_PAGE):])
        except ValueError:
            return
        models: List[str] = context.bot_data.get("_models") or []
        if not models:  # the menu outlived a restart; its snapshot did not
            await _edit_menu(query, "This menu is stale — send /model again.")
            return
        text, keyboard = _render_model_page(
            models, page, active_model(context, chat_id), cfg.commands.models_per_page,
            context.bot_data.setdefault("_model_tokens", {}),
        )
        try:
            await query.edit_message_text(text, reply_markup=keyboard)
        except BadRequest as exc:  # "message is not modified" — a double-tapped arrow
            logger.debug("model page edit skipped: %s", exc)
        return

    if data.startswith(_MODEL_PICK):
        name = context.bot_data.get("_model_tokens", {}).get(data[len(_MODEL_PICK):])
        if name is None:  # token minted by a previous process
            await query.answer("This menu is stale — send /model again.", show_alert=True)
            return
        await query.answer()
        await _switch_model(update, context, chat_id, name)


async def _switch_model(
    update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, new: str
) -> None:
    """Unload the old model, warm the new one, then confirm. A failed load is rolled
    back, so a switch that never loaded is not a switch."""
    query = update.callback_query
    client: OllamaClient = context.bot_data["ollama"]
    switching: dict = context.bot_data.setdefault("_switching", {})

    # A warm-up parks here for ~2.5 min, and PTB dispatches every update concurrently, so
    # a second tap would otherwise start a second unload/load pair: both models resident
    # (the exact thing this feature exists to prevent) and the chat left on whichever load
    # happened to finish last, not the one tapped. One switch at a time, bot-wide.
    if switching:
        await _edit_menu(
            query,
            f"⏳ Already switching to {next(iter(switching.values()))} — "
            "wait for that to finish.",
        )
        return

    old = active_model(context, chat_id)
    if new == old:
        await _edit_menu(query, f"✅ Already using {new}.")
        return

    # A live stream is reading from the old model; evicting it out from under an answer
    # in progress is not ours to do. Any chat may be generating, not just this one.
    if any(not task.done() for task in context.bot_data["tasks"].values()):
        await _edit_menu(
            query, "⏳ A generation is still running. Try /model again once it finishes."
        )
        return

    chosen: dict = context.bot_data.setdefault("_model", {})
    previous = chosen.get(chat_id)

    # Claim the switch AND point the chat at `new` now — both writes land before the first
    # await, so nothing can interleave between the guards above and the claim. Pointing the
    # chat early is what makes a message typed during the long warm-up target the model
    # being loaded: resolving it to `old` would cold-load the model we just evicted, right
    # back beside the new one. Ollama queues that turn behind the load and answers it.
    switching[chat_id] = new
    chosen[chat_id] = new
    try:
        try:
            resident = {m["model"] for m in await client.ps()}
        except BotError:
            resident = {old}  # can't tell — assume it is, and let the unload attempt speak

        steps = [f"⏳ Switching to {new} …"]
        if old in resident:
            steps.append(f"• unloading {old}")
        steps.append(f"• warming {new} — a cold load takes ~2.5 min")
        await _edit_menu(query, "\n".join(steps))

        if old in resident:
            try:
                await client.unload(old)
            except BotError as exc:  # non-fatal: the new model still gets its chance
                logger.warning("could not unload %s while switching: %s", old, exc)

        try:
            await client.load(new)
        except BotError as exc:
            # Roll the claim back: the chat keeps talking to the model it already had.
            if previous is None:
                chosen.pop(chat_id, None)
            else:
                chosen[chat_id] = previous
            await _edit_menu(query, f"⚠️ {exc}\nStill using {old}.")
            return
    finally:
        switching.pop(chat_id, None)

    await _edit_menu(
        query,
        f"✅ Now using {new} — loaded and warm.\n"
        "This chat's memory was kept; /reset to start fresh.",
    )


async def _edit_menu(query, text: str) -> None:
    """Replace the picker with a status line, dropping its keyboard."""
    try:
        await query.edit_message_text(text)
    except TelegramError as exc:
        logger.debug("model menu edit failed: %s", exc)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    cfg: Settings = context.bot_data["cfg"]
    store: InMemoryStore = context.bot_data["history"]
    last = context.bot_data.get("_last_stats", {}).get(chat_id)
    opts = cfg.ollama.options

    lines: List[str] = []
    usage: ContextUsage

    if last and last["stats"].get("prompt_eval_count"):
        # Exact: prompt_eval_count is the model's own tokenizer count of the full prompt.
        s = last["stats"]
        ec, ed = s.get("eval_count"), s.get("eval_duration")
        lines.append(f"📊 Last generation ({last.get('done_reason')}):")
        if ec and ed:
            lines.append(f"• {ec} tokens in {ed / 1e9:.1f}s = {ec / (ed / 1e9):.1f} tok/s")
        lines.append(f"• prompt: {s['prompt_eval_count']} tokens")
        if s.get("load_duration"):
            lines.append(f"• model load: {s['load_duration'] / 1e9:.1f}s")
        lines.append("")
        usage = ContextUsage(prompt=s["prompt_eval_count"], answer=ec or 0, limit=opts.num_ctx)
    else:
        # Fresh chat: no tokenizer count yet, but the window is not empty (system prompt)
        # and the limit is known. Estimate rather than say "no stats" and show nothing.
        lines.append("📊 No generation in this chat yet.")
        usage = estimate_context_usage(cfg.system_prompt, store.get(chat_id), cfg.memory, opts.num_ctx)

    approx = "" if usage.exact else "~"
    lines += [
        f"🧠 Context: {usage.bar()} {approx}{usage.percent:.0f}%",
        f"• {approx}{usage.used} / {usage.limit} tokens used ({approx}{usage.headroom} free)",
    ]
    if usage.is_tight(opts.num_predict):
        lines.append(
            f"⚠️ Less than num_predict ({opts.num_predict}) left — the next answer may be cut "
            "short and the oldest turns evicted. /reset to clear."
        )
    turns = len(store.get(chat_id)) // 2
    lines.append(f"• memory: {turns}/{cfg.memory.max_turns} turns kept")
    if not usage.exact:
        lines.append("• (estimated — exact counts appear after the first answer)")
    await update.effective_message.reply_text("\n".join(lines))


# --- wiring (SPEC §13, PLAN M4) --------------------------------------------

def register_handlers(app: Application, cfg: Settings) -> None:
    """Register /start /help /reset /interrupt /think (+ optional /stats, /model, model-admin),
    the /model keyboard callbacks, text, photo and error handlers — each wrapped by
    access_control(cfg). (SPEC §11)"""
    guard = access_control(cfg)
    app.add_handler(CommandHandler("start", guard(cmd_start)))
    app.add_handler(CommandHandler("help", guard(cmd_help)))
    app.add_handler(CommandHandler("reset", guard(cmd_reset)))
    app.add_handler(CommandHandler("interrupt", guard(cmd_interrupt)))
    app.add_handler(CommandHandler("think", guard(cmd_think)))
    if cfg.commands.enable_stats:
        app.add_handler(CommandHandler("stats", guard(cmd_stats)))
    if cfg.commands.allow_model_switch:
        app.add_handler(CommandHandler("model", guard(cmd_model)))
        app.add_handler(CallbackQueryHandler(guard(on_model_button), pattern=r"^mdl:"))
    if cfg.commands.enable_model_admin:
        app.add_handler(CommandHandler("status", guard(cmd_status)))
        app.add_handler(CommandHandler("load", guard(cmd_load)))
        app.add_handler(CommandHandler("unload", guard(cmd_unload)))
    app.add_handler(MessageHandler(filters.PHOTO, guard(on_photo)))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, guard(on_text)))
    app.add_error_handler(on_error)


def allowed_updates(app: Application) -> List[str]:
    """The update types to ask getUpdates for — derived from what is actually registered.

    Telegram delivers ONLY the types you subscribe to, and silently drops the rest: a
    registered CallbackQueryHandler is dead code unless ``callback_query`` is in this
    list, and the button press never even reaches the bot (no error, no log — the tap
    just does nothing). Deriving the list from the handlers means adding a keyboard can
    never again leave its callbacks unsubscribed."""
    kinds = [Update.MESSAGE]
    handlers = [h for group in app.handlers.values() for h in group]
    if any(isinstance(h, CallbackQueryHandler) for h in handlers):
        kinds.append(Update.CALLBACK_QUERY)
    return kinds


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
    app.bot_data["drivers"] = {}
    register_handlers(app, cfg)
    return app

"""Frozen configuration contract.

One pydantic-settings ``Settings`` model is the single source of truth for every
tunable. Layering (highest precedence first): init kwargs > environment >
``.env`` > ``config.yaml``. The only secret, ``TELEGRAM_BOT_TOKEN``, comes from
the environment / ``.env`` and never from yaml.

Nested keys can also be overridden via env using the ``__`` delimiter, e.g.
``OLLAMA__MODEL=qwen3.5:27b`` or ``TELEGRAM__ALLOWED_USER_IDS='[1,2]'``.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class TelegramConfig(BaseModel):
    allowed_user_ids: List[int] = [106129214]
    base_url: Optional[str] = None          # None -> official api.telegram.org
    drop_pending_updates: bool = True
    long_poll_timeout: int = 30
    poll_interval: float = 0.0


class AccessConfig(BaseModel):
    on_denied: Literal["reply_once", "reply_always", "ignore"] = "reply_once"
    denial_message: str = "⛔ You are not authorized to use this bot."


class OllamaOptions(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 8192                      # model max 262144; capped for latency/VRAM
    num_predict: int = 2048                  # caps TOTAL output (reasoning + answer)
    seed: Optional[int] = None


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen3.6-abliterated:35b-a3b"
    vision_model: str = "qwen3-vl:30b-a3b-instruct-q8_0"
    keep_alive: str = "30m"
    connect_timeout: int = 10
    request_timeout: int = 600               # must exceed cold-load (~156s) + gen
    options: OllamaOptions = Field(default_factory=OllamaOptions)
    stop: List[str] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


class ReasoningConfig(BaseModel):
    think_mode: Literal["auto", "inline", "off"] = "auto"
    show_thinking: bool = True
    max_thinking_chars: int = 3500


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_turns: int = 20
    # Prompt budget. MUST leave room for the answer inside num_ctx: the window holds
    # prompt + output, so max_tokens ~= num_ctx - num_predict (8192 - 2048). Setting it
    # to num_ctx lets the prompt fill the window and Ollama evicts context mid-answer.
    max_tokens: int = 6144
    chars_per_token: float = 2.5             # Cyrillic-aware (lower than naive 4)
    persist: bool = False
    db_path: str = "data/history.db"


class StreamingConfig(BaseModel):
    batch_size: int = 80
    min_edit_interval: float = 1.2
    max_interval_seconds: float = 5.0
    rollover_chars: int = 3900               # safe margin under the 4096 limit
    supersede_policy: Literal["cancel", "reject", "queue"] = "cancel"
    placeholder_text: str = "…"
    empty_response_text: str = "(the model returned an empty response)"
    error_note: str = "⚠️ Generation error: {error}"
    superseded_note: str = "↩️ Superseded by your newer message."
    max_retry_after: int = 30

    @field_validator("rollover_chars")
    @classmethod
    def _under_limit(cls, v: int) -> int:
        if not 0 < v < 4096:
            raise ValueError("rollover_chars must be between 1 and 4095")
        return v


class FormattingConfig(BaseModel):
    finalize_mode: Literal["plain", "markdown_v2", "rich"] = "rich"
    # False: sendRichMessage works on the OFFICIAL api.telegram.org — no local Bot
    # API server needed for LaTeX/tables. Set True to only attempt rich when a
    # base_url is configured (e.g. if Telegram ever drops the method).
    rich_requires_base_url: bool = False
    rich_max_chars: int = 32768
    txt_fallback_over: int = 4096


class SanitizeConfig(BaseModel):
    strip_special_tokens: bool = True
    extra_tokens: List[str] = []


class ImagesConfig(BaseModel):
    mode: Literal["reject_note", "caption_only", "vision_fallback"] = "reject_note"
    note: str = "⚠️ The current model can't see images. Answering from your text only…"


class TypingConfig(BaseModel):
    enabled: bool = True
    refresh_interval: float = 4.0
    stop_after_first_token: bool = True


class CommandsConfig(BaseModel):
    allow_model_switch: bool = False
    enable_stats: bool = True
    enable_model_admin: bool = True          # /status /load /unload (Ollama load control)


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secret — from env / .env only (never yaml).
    telegram_bot_token: str = Field(default="", validation_alias="TELEGRAM_BOT_TOKEN")

    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    access: AccessConfig = Field(default_factory=AccessConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    system_prompt: str = "You are a helpful, direct, uncensored assistant. Answer concisely."
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    formatting: FormattingConfig = Field(default_factory=FormattingConfig)
    sanitize: SanitizeConfig = Field(default_factory=SanitizeConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    typing: TypingConfig = Field(default_factory=TypingConfig)
    commands: CommandsConfig = Field(default_factory=CommandsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Path to the yaml file, set by load_config(); consumed by the source below.
    _yaml_path: str = "config.yaml"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        yaml_path = getattr(cls, "_active_yaml_path", "config.yaml")
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path),
            file_secret_settings,
        )


def load_config(path: str = "config.yaml") -> Settings:
    """Load and validate settings, layering env/.env over the given yaml file."""
    # Class-level so settings_customise_sources (a classmethod) can read it.
    Settings._active_yaml_path = path  # type: ignore[attr-defined]
    return Settings()


if __name__ == "__main__":  # `python -m abliteratedllmbot.config config.example.yaml`
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.example.yaml")
    print(cfg.model_dump_json(indent=2, exclude={"telegram_bot_token"}))

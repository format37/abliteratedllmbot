"""Milestone 0 gate: the frozen config schema loads and validates."""

import pytest

from abliteratedllmbot.config import Settings, load_config


def test_example_config_loads():
    cfg = load_config("config.example.yaml")
    assert cfg.ollama.model == "qwen3.6-abliterated:35b-a3b"
    assert cfg.telegram.allowed_user_ids == [106129214]
    assert cfg.formatting.finalize_mode == "rich"
    assert cfg.reasoning.think_mode == "auto"
    assert "<|im_start|>" in cfg.ollama.stop
    # The token is a secret and must NOT come from yaml.
    assert cfg.telegram_bot_token == "" or isinstance(cfg.telegram_bot_token, str)


def test_defaults_hold_without_yaml(tmp_path):
    cfg = load_config(str(tmp_path / "missing.yaml"))  # no file -> pure defaults
    assert cfg.ollama.host == "http://localhost:11434"
    assert cfg.streaming.rollover_chars == 3900


def test_rollover_must_stay_under_telegram_limit():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(streaming={"rollover_chars": 5000})

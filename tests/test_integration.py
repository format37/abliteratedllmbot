"""End-to-end integration against a LIVE Ollama (marked `ollama`, opt-in / CI-skipped).

Exercises the FULL assembled pipeline with real streaming data — build_messages ->
OllamaClient.chat_stream (real inline <think> + token sanitising) -> RenderDriver
(throttle/heartbeat/finalize) -> recording FakeBot -> formatting.finalize — which the
offline fakes cannot cover. Run with:  pytest -m ollama"""

from __future__ import annotations

import pytest

from abliteratedllmbot.history import Message, build_messages
from abliteratedllmbot.ollama_client import OllamaClient
from abliteratedllmbot.streaming import RenderDriver


@pytest.mark.ollama
async def test_full_pipeline_against_live_model(make_bot, make_cfg):
    cfg = make_cfg(formatting={"finalize_mode": "markdown_v2"})
    client = OllamaClient(cfg.ollama, cfg.reasoning, cfg.sanitize)
    bot = make_bot()
    try:
        messages = build_messages(cfg.system_prompt, [], "Reply with exactly: OK", cfg.memory)
        driver = RenderDriver(bot, chat_id=1, cfg=cfg)
        await driver.run(client.chat_stream(messages=messages))
    finally:
        await client.close()

    assert driver.answer_text.strip(), "the model produced a rendered answer"
    assert "<|" not in driver.answer_text, "content is sanitised end-to-end"
    assert driver.done_reason in ("stop", "length")
    assert bot.messages, "at least one Telegram message was produced"
    # the finalize pass edited the single scratch message (markdown_v2 or plain revert)
    assert any(m[0] == "edit" for m in bot.calls) or any(m[0] == "send" for m in bot.calls)


@pytest.mark.ollama
async def test_multi_turn_memory_against_live_model(make_bot, make_cfg):
    """Second turn must see the first (history threaded through build_messages)."""
    cfg = make_cfg()
    client = OllamaClient(cfg.ollama, cfg.reasoning, cfg.sanitize)
    bot = make_bot()
    history = [Message("user", "My favourite number is 7. Remember it."),
               Message("assistant", "Got it, your favourite number is 7.")]
    try:
        messages = build_messages(cfg.system_prompt, history, "What is my favourite number?", cfg.memory)
        driver = RenderDriver(bot, chat_id=1, cfg=cfg)
        await driver.run(client.chat_stream(messages=messages))
    finally:
        await client.close()
    assert "7" in driver.answer_text, f"expected the recalled number, got: {driver.answer_text!r}"

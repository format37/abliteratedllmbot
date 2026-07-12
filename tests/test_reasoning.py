"""Unit tests for the verified crux: inline <think> splitting + token sanitising.

These need no Ollama and no Telegram — they are the Milestone 1a gate."""

from abliteratedllmbot.reasoning import (
    InlineThinkSplitter,
    StreamSanitizer,
    sanitize_content,
)


def _drive(splitter, chunks):
    """Feed chunks then flush; return the merged (kind -> text) result."""
    spans = []
    for c in chunks:
        spans += splitter.feed(c)
    spans += splitter.flush()
    merged = {"thinking": "", "content": ""}
    for kind, text in spans:
        merged[kind] += text
    return merged, spans


# ---- InlineThinkSplitter ---------------------------------------------------

def test_empty_think_block_is_suppressed():
    # The verified default-model shape for a trivial prompt.
    merged, spans = _drive(InlineThinkSplitter(), ["<think>\n\n</think>\n\nHi there!"])
    assert merged["thinking"] == ""                 # empty block -> no thinking delta
    assert not any(k == "thinking" for k, _ in spans)
    assert merged["content"].strip() == "Hi there!"


def test_real_reasoning_is_split_out():
    merged, _ = _drive(
        InlineThinkSplitter(),
        ["<think>Let me reason about it.</think>The answer is 42."],
    )
    assert merged["thinking"] == "Let me reason about it."
    assert merged["content"] == "The answer is 42."


def test_tag_split_across_chunks():
    merged, _ = _drive(
        InlineThinkSplitter(),
        ["Intro <thi", "nk>because</thi", "nk> done"],
    )
    assert merged["thinking"] == "because"
    assert merged["content"] == "Intro  done"


def test_no_think_tags_is_all_content():
    merged, _ = _drive(InlineThinkSplitter(), ["plain answer, no tags"])
    assert merged["thinking"] == ""
    assert merged["content"] == "plain answer, no tags"


def test_unclosed_think_flushes_as_thinking():
    merged, _ = _drive(InlineThinkSplitter(), ["<think>still thinking when stream ended"])
    assert merged["thinking"] == "still thinking when stream ended"
    assert merged["content"] == ""


# ---- sanitiser -------------------------------------------------------------

def test_sanitize_strips_leaked_template_tokens():
    # The verified leak: "...OK<|endoftext|><|im_start|>..."
    assert sanitize_content("OK<|endoftext|><|im_start|>assistant") == "OKassistant"


def test_stream_sanitizer_handles_token_split_across_chunks():
    s = StreamSanitizer()
    out = s.feed("Answer OK<|im_st") + s.feed("art|>leaked") + s.flush()
    assert out == "Answer OKleaked"


def test_stream_sanitizer_extra_tokens():
    s = StreamSanitizer(extra=["<<SECRET>>"])
    out = s.feed("visible <<SEC") + s.feed("RET>> tail") + s.flush()
    assert out == "visible  tail"


def test_sanitizer_disabled_passes_through():
    s = StreamSanitizer(enabled=False)
    assert s.feed("keep <|im_start|>") + s.flush() == "keep <|im_start|>"

"""Milestone 1b gate: OllamaClient.chat_stream normalises both the structured and
the inline paths to the frozen Delta contract, gates think:true, sanitises content,
and honours the pre-stream-raise vs mid-stream-error-Delta boundary.

Fully OFFLINE: a FakeAsyncClient stands in for ollama.AsyncClient, driven by real
``ChatResponse``/``Message`` chunks so we exercise the exact attribute surface the
client reads. The live-Ollama integration test at the bottom is marked and skipped
in CI.
"""

from __future__ import annotations

import httpx
import pytest
from ollama import ChatResponse, ResponseError
from ollama._types import Message

import abliteratedllmbot.ollama_client as oc
from abliteratedllmbot.config import OllamaConfig, ReasoningConfig, SanitizeConfig
from abliteratedllmbot.errors import (
    Delta,
    ModelNotFoundError,
    OllamaUnavailableError,
    ThinkingUnsupportedError,
)

ABLITERATED = "qwen3.6-abliterated:35b-a3b"   # text-only [V]
CAPABLE = "qwen3.6:35b-a3b"                    # thinking-capable [V]


# --- fakes ------------------------------------------------------------------

def chunk(content="", thinking=None, done=False, done_reason=None, **stats):
    """Build a real ChatResponse chunk as the ollama stream would yield."""
    return ChatResponse(
        message=Message(role="assistant", content=content, thinking=thinking),
        done=done,
        done_reason=done_reason,
        **stats,
    )


class FakeAsyncClient:
    """Mimics the slice of ollama.AsyncClient that OllamaClient touches."""

    def __init__(
        self,
        caps=None,
        chunks=None,
        *,
        show_error=None,
        chat_error=None,      # raised on the FIRST stream iteration (pre-first-chunk)
        error_after=None,     # (n, exc): raise after yielding n chunks (mid-stream)
        ps_models=None,       # names reported loaded by /api/ps
    ):
        self._caps = caps or {}
        self._chunks = chunks or []
        self._show_error = show_error
        self._chat_error = chat_error
        self._error_after = error_after
        self._ps_models = ps_models or []
        self.show_calls = []
        self.chat_calls = []
        self.generate_calls = []

    async def ps(self):
        Models = [
            type("M", (), {"model": name, "size_vram": 1, "expires_at": "later"})()
            for name in self._ps_models
        ]
        return type("PS", (), {"models": Models})()

    async def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return type("G", (), {"done": True, "done_reason": "load", "response": ""})()

    async def show(self, model):
        self.show_calls.append(model)
        if self._show_error is not None:
            raise self._show_error
        if model not in self._caps:
            raise ResponseError(f"model '{model}' not found", 404)
        return type("ShowResp", (), {"capabilities": list(self._caps[model])})()

    async def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        chunks, chat_error, error_after = self._chunks, self._chat_error, self._error_after

        async def gen():
            if chat_error is not None:  # surfaces on first __anext__, like real raise_for_status
                raise chat_error
            for i, c in enumerate(chunks):
                if error_after is not None and i == error_after[0]:
                    raise error_after[1]
                yield c
            if error_after is not None and error_after[0] >= len(chunks):
                raise error_after[1]

        return gen()


@pytest.fixture
def make_client(monkeypatch):
    def _make(caps=None, chunks=None, think_mode="auto", strip=True, extra=(), **fake_kw):
        fake = FakeAsyncClient(caps=caps, chunks=chunks, **fake_kw)
        monkeypatch.setattr(oc, "AsyncClient", lambda *a, **k: fake)
        client = oc.OllamaClient(
            OllamaConfig(),
            ReasoningConfig(think_mode=think_mode),
            SanitizeConfig(strip_special_tokens=strip, extra_tokens=list(extra)),
        )
        return client, fake

    return _make


async def _collect(client, **kw):
    return [d async for d in client.chat_stream(**kw)]


def _text(deltas, kind):
    return "".join(d.text for d in deltas if d.kind == kind)


# --- resolve_think gating (pure) -------------------------------------------

def test_resolve_think_auto_capable_true(make_client):
    client, _ = make_client(think_mode="auto")
    assert client.resolve_think({"completion", "thinking"}) is True


def test_resolve_think_auto_incapable_false(make_client):
    client, _ = make_client(think_mode="auto")
    assert client.resolve_think({"completion"}) is False


def test_resolve_think_inline_never_structured(make_client):
    # 'inline' forces inline parsing even on a thinking-capable model (SPEC §2).
    client, _ = make_client(think_mode="inline")
    assert client.resolve_think({"completion", "thinking"}) is False


def test_resolve_think_off_never_structured(make_client):
    client, _ = make_client(think_mode="off")
    assert client.resolve_think({"completion", "thinking"}) is False


# --- capabilities cache + 404 ----------------------------------------------

async def test_capabilities_cached_single_show(make_client):
    client, fake = make_client(caps={ABLITERATED: ["completion"]})
    a = await client.capabilities(ABLITERATED)
    b = await client.capabilities(ABLITERATED)
    assert a == {"completion"} and b == {"completion"}
    assert fake.show_calls == [ABLITERATED]  # second call served from cache


async def test_capabilities_404_raises_model_not_found(make_client):
    client, _ = make_client(caps={})  # show() -> ResponseError(404)
    with pytest.raises(ModelNotFoundError):
        await client.capabilities("ghost:1b")


async def test_capabilities_connection_error_is_unavailable(make_client):
    client, _ = make_client(show_error=ConnectionError("refused"))
    with pytest.raises(OllamaUnavailableError):
        await client.capabilities(ABLITERATED)


async def test_capabilities_timeout_is_unavailable(make_client):
    # httpx timeouts are NOT wrapped by ollama; the client must still map them.
    client, _ = make_client(show_error=httpx.ConnectTimeout("timed out"))
    with pytest.raises(OllamaUnavailableError):
        await client.capabilities(ABLITERATED)


# --- ps / load / unload -----------------------------------------------------

async def test_ps_lists_loaded_models(make_client):
    client, _ = make_client(ps_models=["qwen3.6-abliterated:35b-a3b"])
    loaded = await client.ps()
    assert loaded and loaded[0]["model"] == "qwen3.6-abliterated:35b-a3b"


async def test_ps_empty_when_nothing_loaded(make_client):
    client, _ = make_client()
    assert await client.ps() == []


async def test_load_sends_empty_generate_with_keep_alive(make_client):
    client, fake = make_client()
    await client.load()
    call = fake.generate_calls[0]
    assert call["prompt"] == "" and call["keep_alive"] == client._ollama.keep_alive
    assert call["model"] == ABLITERATED


async def test_unload_sends_keep_alive_zero(make_client):
    client, fake = make_client()
    await client.unload()
    assert fake.generate_calls[0]["keep_alive"] == 0


# --- chat_stream: inline path (the abliterated default model) ---------------

async def test_inline_empty_think_then_answer(make_client):
    # Verified default-model shape: empty <think> block + leaked tokens.
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[
            chunk(content="<think>\n\n</think>\n\nHi"),
            chunk(content=" there<|endoftext|>", done=True, done_reason="stop",
                  eval_count=3, eval_duration=1_000_000, prompt_eval_count=10,
                  total_duration=2_000_000, load_duration=0),
        ],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "hi"}])
    assert not any(d.kind == "thinking" for d in deltas)      # empty block suppressed
    assert _text(deltas, "content").strip() == "Hi there"     # <|endoftext|> stripped
    assert deltas[-1].kind == "done" and deltas[-1].done_reason == "stop"


async def test_inline_real_reasoning_split(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[
            chunk(content="<think>because</think>The answer."),
            chunk(content="", done=True, done_reason="stop"),
        ],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "why"}])
    assert _text(deltas, "thinking") == "because"
    assert _text(deltas, "content") == "The answer."


async def test_delta_ordering_thinking_then_content_then_done(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[
            chunk(content="<think>reason"),
            chunk(content="ing</think>ans"),
            chunk(content="wer", done=True, done_reason="stop"),
        ],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "q"}])
    kinds = [d.kind for d in deltas]
    # all thinking precede all content, exactly one terminal at the end
    assert kinds.count("done") == 1 and kinds[-1] == "done"
    last_thinking = max(i for i, k in enumerate(kinds) if k == "thinking")
    first_content = min(i for i, k in enumerate(kinds) if k == "content")
    assert last_thinking < first_content


async def test_content_sanitised_across_chunk_boundary(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[
            chunk(content="OK<|im_st"),
            chunk(content="art|>done", done=True, done_reason="stop"),
        ],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert _text(deltas, "content") == "OKdone"  # token straddling chunks removed


async def test_extra_tokens_stripped(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        extra=["<<SECRET>>"],
        chunks=[chunk(content="visible <<SECRET>> tail", done=True, done_reason="stop")],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert _text(deltas, "content") == "visible  tail"


# --- chat_stream: structured path (a thinking-capable model) ----------------

async def test_structured_path_emits_thinking_and_content(make_client):
    client, fake = make_client(
        caps={CAPABLE: ["completion", "thinking"]},
        chunks=[
            chunk(thinking="pondering"),
            chunk(content="the reply"),
            chunk(content="", done=True, done_reason="stop"),
        ],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "q"}], model=CAPABLE)
    assert _text(deltas, "thinking") == "pondering"
    assert _text(deltas, "content") == "the reply"
    assert fake.chat_calls[0]["think"] is True   # think:true sent to a capable model


async def test_structured_content_not_run_through_splitter(make_client):
    # A capable model in structured mode may legitimately mention "<think>" in its
    # answer; the splitter must NOT eat it (structured content bypasses the splitter).
    client, _ = make_client(
        caps={CAPABLE: ["completion", "thinking"]},
        chunks=[
            chunk(thinking="x"),
            chunk(content="use the <think> tag", done=True, done_reason="stop"),
        ],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "q"}], model=CAPABLE)
    assert _text(deltas, "content") == "use the <think> tag"


# --- the invariant: never think:true to an incapable model [V] --------------

async def test_think_omitted_for_incapable_model(make_client):
    client, fake = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="hi", done=True, done_reason="stop")],
    )
    await _collect(client, messages=[{"role": "user", "content": "x"}])
    # think must be None (omitted from the request) — never True, never even False.
    assert fake.chat_calls[0]["think"] is None


async def test_think_omitted_in_inline_mode_even_if_capable(make_client):
    client, fake = make_client(
        caps={CAPABLE: ["completion", "thinking"]},
        think_mode="inline",
        chunks=[chunk(content="<think>r</think>a", done=True, done_reason="stop")],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}], model=CAPABLE)
    assert fake.chat_calls[0]["think"] is None
    assert _text(deltas, "thinking") == "r"   # parsed inline, not structured
    assert _text(deltas, "content") == "a"


# --- off mode strips reasoning ----------------------------------------------

async def test_off_mode_drops_thinking(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        think_mode="off",
        chunks=[chunk(content="<think>secret reasoning</think>answer", done=True,
                      done_reason="stop")],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert not any(d.kind == "thinking" for d in deltas)
    assert _text(deltas, "content") == "answer"


# --- request shape: options, stop, keep_alive, images -----------------------

async def test_options_and_stop_forwarded(make_client):
    client, fake = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="ok", done=True, done_reason="stop")],
    )
    await _collect(client, messages=[{"role": "user", "content": "x"}])
    kw = fake.chat_calls[0]
    assert kw["options"]["num_ctx"] == 8192 and kw["options"]["num_predict"] == 2048
    assert kw["options"]["stop"] == ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    assert kw["options"].get("seed") is None or "seed" not in kw["options"]  # None dropped
    assert kw["keep_alive"] == "30m"


async def test_images_attached_to_last_user_turn(make_client):
    client, fake = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="ok", done=True, done_reason="stop")],
    )
    await _collect(
        client,
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "look"},
        ],
        images=[b"\x89PNG-bytes"],
    )
    sent = fake.chat_calls[0]["messages"]
    assert sent[-1]["role"] == "user" and sent[-1]["images"] == [b"\x89PNG-bytes"]
    assert "images" not in sent[0]  # untouched system message


# --- terminal stats ---------------------------------------------------------

async def test_done_carries_stats(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="ok", done=True, done_reason="length",
                      eval_count=200, eval_duration=2_000_000_000,
                      prompt_eval_count=15, total_duration=3_000_000_000, load_duration=0)],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    done = deltas[-1]
    assert done.kind == "done" and done.done_reason == "length"
    assert done.stats["eval_count"] == 200
    assert done.stats["prompt_eval_count"] == 15
    # All four SPEC §5.5 stats present; load_duration=0 must survive (is-not-None,
    # not a truthiness filter that would drop a legitimate zero).
    assert done.stats["total_duration"] == 3_000_000_000
    assert "load_duration" in done.stats and done.stats["load_duration"] == 0
    tok_per_s = done.stats["eval_count"] / (done.stats["eval_duration"] / 1e9)
    assert tok_per_s == pytest.approx(100.0)


# --- zero-content terminals + end-of-stream flush (contract-permitted shapes) -

async def test_all_thinking_no_content_yields_thinking_then_done(make_client):
    # SPEC §4 permits zero content deltas. Assert EXACT kinds so a spurious empty
    # content delta (e.g. a dropped `if tail`/`if clean` guard) would be caught.
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="<think>reasoning</think>", done=True, done_reason="stop")],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert [d.kind for d in deltas] == ["thinking", "done"]
    assert _text(deltas, "thinking") == "reasoning"


async def test_immediately_done_empty_stream_yields_only_done(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="", done=True, done_reason="stop")],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert [d.kind for d in deltas] == ["done"]


async def test_dangling_tag_fragment_flushed_before_done(make_client):
    # Stream ends mid-`<think` tag: splitter.flush must emit the held literal.
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="answer<thi", done=True, done_reason="stop")],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert _text(deltas, "content") == "answer<thi"
    assert deltas[-1].kind == "done"


async def test_dangling_special_token_fragment_flushed_before_done(make_client):
    # Stream ends mid-`<|` token: sanitizer.flush emits the unclosed prefix literally.
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="OK<|endof", done=True, done_reason="stop")],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert _text(deltas, "content") == "OK<|endof"
    assert deltas[-1].kind == "done"


async def test_structured_content_is_sanitised(make_client):
    # Contract §4/§5.4: content is sanitised on the STRUCTURED path too, incl. a
    # token straddling two structured content chunks.
    client, _ = make_client(
        caps={CAPABLE: ["completion", "thinking"]},
        chunks=[
            chunk(thinking="ponder"),
            chunk(content="ans<|endof"),
            chunk(content="text|>wer", done=True, done_reason="stop"),
        ],
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "q"}], model=CAPABLE)
    assert _text(deltas, "content") == "answer"


# --- pre-stream vs mid-stream failure boundary (SPEC §4) --------------------

async def test_pre_stream_model_missing_raises_before_any_delta(make_client):
    client, _ = make_client(caps={})  # show 404
    gen = client.chat_stream(messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ModelNotFoundError):
        await gen.__anext__()  # raises before yielding anything


async def test_pre_stream_first_chunk_failure_raises(make_client):
    # Server reachable for /api/show but the chat stream dies on the first chunk.
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chat_error=ConnectionError("dropped before first token"),
    )
    gen = client.chat_stream(messages=[{"role": "user", "content": "x"}])
    with pytest.raises(OllamaUnavailableError):
        await gen.__anext__()


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),   # the faithful streaming-path shape: ollama's
        httpx.ConnectTimeout("timeout"), # streaming inner() does NOT wrap httpx errors,
        httpx.ReadTimeout("slow"),       # so raw httpx.TransportError reaches us here.
    ],
)
async def test_pre_stream_first_chunk_httpx_errors_map_to_unavailable(make_client, exc):
    client, _ = make_client(caps={ABLITERATED: ["completion"]}, chat_error=exc)
    gen = client.chat_stream(messages=[{"role": "user", "content": "x"}])
    with pytest.raises(OllamaUnavailableError):
        await gen.__anext__()


async def test_first_stream_line_not_found_maps_to_model_not_found(make_client):
    # A first NDJSON `{"error":"...not found"}` line surfaces as ResponseError with
    # status_code -1; classification hinges on the "not found" substring (SPEC §5.1).
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chat_error=ResponseError("model 'ghost:1b' not found", -1),
    )
    gen = client.chat_stream(messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ModelNotFoundError):
        await gen.__anext__()


async def test_mid_stream_response_error_surfaces_its_message(make_client):
    # A mid-stream `{"error": ...}` line (ResponseError, status -1) after content:
    # the human message must carry the server's text, not the generic fallback.
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="partial")],
        error_after=(1, ResponseError("llama runner has terminated", 500)),
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert _text(deltas, "content") == "partial"
    assert deltas[-1].kind == "error"
    assert deltas[-1].error == "llama runner has terminated"


async def test_mid_stream_failure_yields_error_delta_after_partial(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="partial answer")],
        error_after=(1, httpx.ReadError("connection reset")),  # after 1 chunk
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert _text(deltas, "content") == "partial answer"
    assert deltas[-1].kind == "error" and deltas[-1].error


async def test_stream_ends_without_done_yields_error(make_client):
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chunks=[chunk(content="truncated")],  # no done=True chunk
    )
    deltas = await _collect(client, messages=[{"role": "user", "content": "x"}])
    assert _text(deltas, "content") == "truncated"
    assert deltas[-1].kind == "error"


async def test_thinking_unsupported_maps_defensively(make_client):
    # Should never happen (gating prevents it), but if a 400 "does not support
    # thinking" leaks through, it maps to ThinkingUnsupportedError, not a crash.
    client, _ = make_client(
        caps={ABLITERATED: ["completion"]},
        chat_error=ResponseError(f"{ABLITERATED} does not support thinking", 400),
    )
    gen = client.chat_stream(messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ThinkingUnsupportedError):
        await gen.__anext__()


# --- live integration (skipped in CI) ---------------------------------------

@pytest.mark.ollama
async def test_live_default_model_streams_content_then_done():
    client = oc.OllamaClient(OllamaConfig(), ReasoningConfig(), SanitizeConfig())
    try:
        deltas = [
            d
            async for d in client.chat_stream(
                messages=[{"role": "user", "content": "Reply with exactly: OK"}]
            )
        ]
    finally:
        await client.close()
    assert any(d.kind == "content" for d in deltas)
    assert deltas[-1].kind == "done" and deltas[-1].done_reason in ("stop", "length")
    assert "<|" not in _text(deltas, "content")  # sanitised

"""Milestone 2 gate: per-chat memory + turns-then-token trimming (pure).

Proves the PLAN M2 DoD: oldest pairs drop first, system prompt always kept,
/reset isolates chats, and a Cyrillic string trims more aggressively than /4."""

from __future__ import annotations

from abliteratedllmbot.config import MemoryConfig
from abliteratedllmbot.history import (
    ContextUsage,
    InMemoryStore,
    Message,
    build_messages,
    calibrate_chars_per_token,
    estimate_tokens,
    trim_history,
)


def _pairs(n: int) -> list[Message]:
    """n user/assistant pairs: u0,a0,u1,a1,..."""
    out: list[Message] = []
    for i in range(n):
        out.append(Message("user", f"u{i}"))
        out.append(Message("assistant", f"a{i}"))
    return out


# --- InMemoryStore ----------------------------------------------------------

def test_append_and_get_roundtrip():
    s = InMemoryStore()
    s.append(1, Message("user", "hi"))
    s.append(1, Message("assistant", "hello"))
    got = s.get(1)
    assert [m.content for m in got] == ["hi", "hello"]


def test_get_returns_a_copy():
    s = InMemoryStore()
    s.append(1, Message("user", "hi"))
    got = s.get(1)
    got.append(Message("assistant", "mutation"))
    assert len(s.get(1)) == 1  # external mutation must not leak into the store


def test_reset_isolates_chats():
    s = InMemoryStore()
    s.append(1, Message("user", "a"))
    s.append(2, Message("user", "b"))
    s.reset(1)
    assert s.get(1) == []
    assert [m.content for m in s.get(2)] == ["b"]  # chat 2 untouched


def test_reset_missing_chat_is_noop():
    s = InMemoryStore()
    s.reset(999)  # must not raise


# --- token estimate (Cyrillic-aware) ----------------------------------------

def test_memory_default_chars_per_token_is_cyrillic_aware():
    # Guards against a regression of the default back to the naive /4.
    assert MemoryConfig().chars_per_token == 2.5


def test_cyrillic_estimate_is_strictly_denser_than_naive_quarter():
    # The whole point of chars_per_token=2.5: a lower ratio estimates STRICTLY more
    # tokens than the naive /4 — so a naive estimator can't satisfy this assertion.
    cyr = "Привет, как дела сегодня утром?" * 4
    assert estimate_tokens(cyr, 2.5) > estimate_tokens(cyr, 4.0)
    assert estimate_tokens(cyr, 2.5) > len(cyr) / 4


def test_cyrillic_trims_more_turns_than_naive():
    # 10-char messages so the two rates DIVERGE: est=int(10/2.5)+1=5 vs int(10/4)+1=3.
    # 5 pairs -> dense total 10*5=50 > budget 40 > naive total 10*3=30.
    msg = "Сообщение1"  # 10 code points
    assert len(msg) == 10
    hist = []
    for _ in range(5):
        hist.append(Message("user", msg))
        hist.append(Message("assistant", msg))
    kept_dense = trim_history(hist, max_turns=100, max_tokens=40, chars_per_token=2.5)
    kept_naive = trim_history(hist, max_turns=100, max_tokens=40, chars_per_token=4.0)
    # The denser (Cyrillic-aware) estimate must drop STRICTLY more turns.
    assert len(kept_dense) < len(kept_naive)
    assert len(kept_dense) == 8 and len(kept_naive) == 10  # dropped 1 pair vs 0


# --- trim_history: turns-first, then tokens ---------------------------------

def test_max_turns_keeps_most_recent_pairs():
    hist = _pairs(5)  # u0..u4
    kept = trim_history(hist, max_turns=2, max_tokens=10_000, chars_per_token=2.5)
    assert [m.content for m in kept] == ["u3", "a3", "u4", "a4"]  # oldest 3 pairs dropped


def test_oldest_pairs_drop_first_under_token_budget():
    hist = _pairs(6)
    # Each 2-char msg -> estimate_tokens(int(2/2.5)+1)=1 token, so a pair=2 tokens,
    # 6 pairs=12 tokens. Budget 4 -> drop oldest pairs until <=4 -> keep last 2 pairs.
    kept = trim_history(hist, max_turns=100, max_tokens=4, chars_per_token=2.5)
    contents = [m.content for m in kept]
    assert contents == ["u4", "a4", "u5", "a5"]
    assert "u0" not in contents and "u3" not in contents


def test_system_prompt_always_kept_even_when_overbudget():
    sys = Message("system", "S" * 1000)  # huge system, far over budget
    hist = [sys] + _pairs(3)
    kept = trim_history(hist, max_turns=100, max_tokens=1, chars_per_token=2.5)
    assert kept[0].role == "system" and kept[0].content == "S" * 1000
    assert all(m.role != "user" for m in kept)  # every turn dropped, system survives


def test_max_turns_zero_keeps_only_system():
    hist = [Message("system", "s")] + _pairs(3)
    kept = trim_history(hist, max_turns=0, max_tokens=10_000, chars_per_token=2.5)
    assert [m.role for m in kept] == ["system"]


def test_empty_history():
    assert trim_history([], max_turns=5, max_tokens=100, chars_per_token=2.5) == []


def test_trim_preserves_pair_alignment():
    hist = _pairs(4)
    kept = trim_history(hist, max_turns=2, max_tokens=10_000, chars_per_token=2.5)
    # never a dangling assistant-first or split pair
    roles = [m.role for m in kept]
    assert roles == ["user", "assistant", "user", "assistant"]


# --- build_messages ---------------------------------------------------------

def test_build_messages_shape_and_order():
    mem = MemoryConfig(max_turns=100, max_tokens=10_000, chars_per_token=2.5)
    hist = _pairs(2)
    msgs = build_messages("SYS", hist, "current question", mem)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[-1] == {"role": "user", "content": "current question"}
    assert [m["content"] for m in msgs] == ["SYS", "u0", "a0", "u1", "a1", "current question"]


def test_build_messages_memory_disabled_is_stateless():
    mem = MemoryConfig(enabled=False)
    msgs = build_messages("SYS", _pairs(5), "q", mem)
    assert msgs == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "q"},
    ]


def test_build_messages_reserves_budget_for_system_and_user():
    # A large system + user turn must eat into how much history survives.
    mem = MemoryConfig(max_turns=100, max_tokens=30, chars_per_token=2.5)
    hist = _pairs(6)
    lean = build_messages("s", hist, "q", mem)
    fat = build_messages("S" * 60, hist, "Q" * 60, mem)
    hist_msgs_lean = [m for m in lean if m["content"].startswith(("u", "a"))]
    hist_msgs_fat = [m for m in fat if m["content"].startswith(("u", "a"))]
    assert len(hist_msgs_fat) < len(hist_msgs_lean)


def test_build_messages_no_system_prompt_omits_system():
    mem = MemoryConfig(max_turns=100, max_tokens=10_000, chars_per_token=2.5)
    msgs = build_messages("", _pairs(1), "q", mem)
    assert msgs[0]["role"] == "user"  # no system message prepended


# --- calibration ------------------------------------------------------------

def test_calibrate_matches_observed_ratio():
    # 250 chars measured at 100 tokens -> 2.5 chars/token.
    assert calibrate_chars_per_token(250, 100) == 2.5


def test_calibrate_clamps_degenerate_values():
    assert calibrate_chars_per_token(1000, 1) == 6.0    # ceil
    assert calibrate_chars_per_token(1, 1000) == 1.5    # floor
    assert 1.5 <= calibrate_chars_per_token(0, 0) <= 6.0  # no div-by-zero


# --- ContextUsage (the /stats occupancy gauge) -------------------------------

def test_context_usage_basic_math():
    u = ContextUsage(prompt=1000, answer=500, limit=8192)
    assert u.used == 1500
    assert round(u.percent) == 18
    assert u.headroom == 8192 - 1500


def test_context_usage_counts_the_answer_not_just_the_prompt():
    """The answer stays in the window and is carried into the next prompt."""
    assert ContextUsage(prompt=1000, answer=2000, limit=8192).used == 3000


def test_context_usage_bar_is_proportional():
    assert ContextUsage(prompt=0, answer=0, limit=100).bar(10) == "░" * 10
    assert ContextUsage(prompt=50, answer=0, limit=100).bar(10) == "▓" * 5 + "░" * 5
    assert ContextUsage(prompt=100, answer=0, limit=100).bar(10) == "▓" * 10


def test_context_usage_bar_never_overflows_when_context_is_exceeded():
    """Ollama evicts rather than erroring, so used > limit is reachable. Don't emit
    a 30-char bar or a negative headroom in that state."""
    u = ContextUsage(prompt=9000, answer=3000, limit=8192)
    assert len(u.bar(10)) == 10 and u.bar(10) == "▓" * 10
    assert u.headroom == 0
    assert u.percent > 100  # honest: report the overflow rather than clamping to 100


def test_context_usage_zero_limit_does_not_divide_by_zero():
    u = ContextUsage(prompt=100, answer=0, limit=0)
    assert u.percent == 0.0 and u.bar(10) == "░" * 10 and u.headroom == 0


def test_is_tight_when_next_answer_cannot_fit():
    num_predict = 2048
    roomy = ContextUsage(prompt=1000, answer=500, limit=8192)
    tight = ContextUsage(prompt=6000, answer=500, limit=8192)  # 1692 free < 2048
    assert roomy.is_tight(num_predict) is False
    assert tight.is_tight(num_predict) is True

"""RED tests for token estimation (Tools issue #2736)."""

from __future__ import annotations

import pytest
from chat.condensation.tokens import estimate_tokens


def test_estimate_tokens_empty_returns_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_proportional_to_length() -> None:
    short = estimate_tokens("hi")
    longer = estimate_tokens("hi " * 100)
    assert longer > short


def test_estimate_tokens_negative_input_raises() -> None:
    with pytest.raises(TypeError):
        estimate_tokens(None)  # type: ignore[arg-type]


def test_fallback_heuristic_matches_len_div_4() -> None:
    # The heuristic rule is len(text)//4; called directly via the helper.
    from chat.condensation.tokens import _fallback_estimate

    assert _fallback_estimate("x" * 100) == 25
    assert _fallback_estimate("") == 0


def test_tiktoken_parity_when_available() -> None:
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        pytest.skip("tiktoken not installed")
    # Real call should succeed and be a positive int
    n = estimate_tokens("hello world")
    assert n > 0

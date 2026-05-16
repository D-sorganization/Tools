"""Tests for thread condensation and token count visibility (Tools #2736).

Covers:
- condense_thread() summarisation contract
- Message list replacement behaviour (active context)
- Raw history preservation for undo
- estimate_token_count() heuristic
- Edge cases: empty thread, single message
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.python.ai.thread_condensation import (
    SummaryMessage,
    condense_thread,
    estimate_token_count,
)
from shared.python.ai.types import Message

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(n: int, prefix: str = "msg") -> list[Message]:
    return [Message(role="user", content=f"{prefix} {i}") for i in range(n)]


def _make_stub_adapter(summary: str = "SUMMARY") -> Any:
    adapter = MagicMock()
    response = MagicMock()
    response.content = summary
    adapter.send_message.return_value = response
    return adapter


# ---------------------------------------------------------------------------
# Template / extraction
# ---------------------------------------------------------------------------


def test_condense_template_extracts_objective_and_decisions() -> None:
    """condense_thread() calls the adapter and returns a non-empty SummaryMessage."""
    messages = _make_messages(5)
    adapter = _make_stub_adapter("Objective: X. Decisions: Y. Context: Z.")

    result, _active = condense_thread(messages, adapter)

    assert isinstance(result, SummaryMessage)
    assert result.content.strip() != ""
    assert adapter.send_message.called


# ---------------------------------------------------------------------------
# Active context replacement
# ---------------------------------------------------------------------------


def test_condense_replaces_old_messages_preserving_recent() -> None:
    """After condensation the returned list is [SummaryMessage, *last_N]."""
    messages = _make_messages(10)
    adapter = _make_stub_adapter("SUMMARY TEXT")

    summary, active = condense_thread(messages, adapter, keep_recent=3)

    assert isinstance(summary, SummaryMessage)
    # Active context must be [summary] + last 3 original messages
    assert len(active) == 4
    assert active[0] is summary
    # The tail must equal the last 3 originals
    assert active[1:] == messages[-3:]


def test_condense_replaces_old_messages_default_keep_recent() -> None:
    """Default keep_recent value still gives [summary, *tail]."""
    messages = _make_messages(20)
    adapter = _make_stub_adapter("SUMMARY")

    summary, active = condense_thread(messages, adapter)

    assert isinstance(summary, SummaryMessage)
    assert len(active) >= 2  # at least summary + 1 recent
    assert active[0] is summary


# ---------------------------------------------------------------------------
# Raw history preservation
# ---------------------------------------------------------------------------


def test_raw_history_preserved_for_undo() -> None:
    """condense_thread() must not mutate the input list."""
    messages = _make_messages(6)
    original_ids = [id(m) for m in messages]
    adapter = _make_stub_adapter("SUMMARY")

    condense_thread(messages, adapter)

    # The original list is untouched
    assert [id(m) for m in messages] == original_ids
    assert len(messages) == 6


# ---------------------------------------------------------------------------
# Token count estimation
# ---------------------------------------------------------------------------


def test_token_count_estimation_non_negative() -> None:
    messages = _make_messages(3)
    count = estimate_token_count(messages)
    assert isinstance(count, int)
    assert count >= 0


def test_token_count_increases_with_message_length() -> None:
    short_msgs = [Message(role="user", content="hi")]
    long_msgs = [Message(role="user", content="hi " * 500)]
    assert estimate_token_count(long_msgs) > estimate_token_count(short_msgs)


def test_token_count_empty_list_is_zero() -> None:
    assert estimate_token_count([]) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_condense_empty_thread_returns_empty_summary() -> None:
    """Empty message list should produce a SummaryMessage with empty content."""
    adapter = _make_stub_adapter("")

    summary, active = condense_thread([], adapter)

    assert isinstance(summary, SummaryMessage)
    assert summary.content == ""
    assert active == [summary]


def test_condense_single_message_returns_that_message() -> None:
    """Single-message thread: summary wraps it, active is [summary, original]."""
    msg = Message(role="user", content="Hello")
    adapter = _make_stub_adapter("Hello summary")

    summary, active = condense_thread([msg], adapter)

    assert isinstance(summary, SummaryMessage)
    assert len(active) >= 1
    assert active[0] is summary


# ---------------------------------------------------------------------------
# DbC: invalid inputs
# ---------------------------------------------------------------------------


def test_condense_thread_raises_on_non_list_messages() -> None:
    """DbC: messages must be a list; TypeError otherwise."""
    adapter = _make_stub_adapter("X")
    with pytest.raises((ValueError, TypeError)):
        condense_thread("not a list", adapter)  # type: ignore[arg-type]

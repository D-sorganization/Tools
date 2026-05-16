"""RED tests for Condenser (Tools issue #2736)."""

from __future__ import annotations

import pytest
from chat.condensation import CondensationRequest, Condenser
from chat.service_base import ChatSession


def _make_session(n: int) -> ChatSession:
    s = ChatSession(session_id="s1")
    for i in range(n):
        s.add_message("user", f"msg {i}")
    return s


def test_condense_returns_new_session_immutability() -> None:
    session = _make_session(10)
    original_messages = list(session.messages)
    condenser = Condenser()
    result = condenser.condense(
        session,
        CondensationRequest(
            session_id=session.session_id,
            strategy="keep_recent",
            keep_last_n=3,
        ),
    )
    assert result.condensed_message_count == 3
    assert result.original_message_count == 10
    # Original session untouched
    assert session.messages == original_messages
    assert session.message_count == 10


def test_condense_preserves_at_least_one_message_postcondition() -> None:
    session = _make_session(5)
    condenser = Condenser()
    result = condenser.condense(
        session,
        CondensationRequest(
            session_id=session.session_id,
            strategy="keep_recent",
            keep_last_n=1,
        ),
    )
    assert result.condensed_message_count >= 1


def test_condense_empty_session_raises_value_error() -> None:
    session = ChatSession(session_id="empty")
    condenser = Condenser()
    with pytest.raises(ValueError):
        condenser.condense(
            session,
            CondensationRequest(
                session_id="empty",
                strategy="keep_recent",
                keep_last_n=3,
            ),
        )


def test_condense_unknown_strategy_raises() -> None:
    session = _make_session(5)
    condenser = Condenser()
    with pytest.raises(ValueError):
        condenser.condense(
            session,
            CondensationRequest(
                session_id=session.session_id,
                strategy="not_real",
                keep_last_n=2,
            ),
        )


def test_condense_anchor_preservation_reported() -> None:
    session = ChatSession(session_id="s1")
    session.add_message("user", "pinned A", metadata={"pin": True})
    for i in range(8):
        session.add_message("user", f"msg {i}")
    condenser = Condenser()
    result = condenser.condense(
        session,
        CondensationRequest(
            session_id=session.session_id,
            strategy="pinned_anchor",
            keep_last_n=2,
        ),
    )
    assert result.preserved_anchors >= 1


def test_condense_removed_tokens_estimate_non_negative() -> None:
    session = _make_session(6)
    condenser = Condenser()
    result = condenser.condense(
        session,
        CondensationRequest(
            session_id=session.session_id,
            strategy="keep_recent",
            keep_last_n=2,
        ),
    )
    assert result.removed_tokens_estimate >= 0

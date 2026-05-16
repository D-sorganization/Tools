"""RED tests for CondensationStrategy implementations (Tools issue #2736)."""

from __future__ import annotations

import pytest
from chat.condensation import (
    CondensationRequest,
    KeepRecentStrategy,
    PinnedAnchorStrategy,
    SemanticSummaryStrategy,
)
from chat.service_base import ChatSession


def _make_session(n: int) -> ChatSession:
    s = ChatSession(session_id="s1")
    for i in range(n):
        s.add_message("user" if i % 2 == 0 else "assistant", f"msg {i}")
    return s


def test_keep_recent_drops_old_messages() -> None:
    session = _make_session(10)
    strategy = KeepRecentStrategy()
    req = CondensationRequest(
        session_id=session.session_id, strategy="keep_recent", keep_last_n=3
    )
    new_session = strategy.apply(session, req)
    assert new_session.message_count == 3
    # Most recent 3 messages preserved
    assert new_session.messages[-1].content == "msg 9"
    assert new_session.messages[0].content == "msg 7"


def test_keep_recent_does_not_mutate_original() -> None:
    session = _make_session(5)
    original_count = session.message_count
    strategy = KeepRecentStrategy()
    req = CondensationRequest(
        session_id=session.session_id, strategy="keep_recent", keep_last_n=2
    )
    strategy.apply(session, req)
    assert session.message_count == original_count


def test_pinned_anchor_preserves_pinned_messages() -> None:
    session = ChatSession(session_id="s1")
    session.add_message("user", "old pinned", metadata={"pin": True})
    for i in range(5):
        session.add_message("user", f"old {i}")
    session.add_message("user", "recent")
    strategy = PinnedAnchorStrategy()
    req = CondensationRequest(
        session_id=session.session_id, strategy="pinned_anchor", keep_last_n=2
    )
    new_session = strategy.apply(session, req)
    contents = [m.content for m in new_session.messages]
    assert "old pinned" in contents
    assert "recent" in contents


def test_semantic_summary_collapses_old_into_anchor() -> None:
    session = _make_session(10)
    strategy = SemanticSummaryStrategy()
    req = CondensationRequest(
        session_id=session.session_id,
        strategy="semantic_summary",
        keep_last_n=2,
    )
    new_session = strategy.apply(session, req)
    contents = [m.content for m in new_session.messages]
    # One summary anchor + last 2 messages
    assert any("Earlier in conversation" in c for c in contents)
    assert "msg 9" in contents
    assert "msg 8" in contents


def test_semantic_summary_no_collapse_when_short() -> None:
    session = _make_session(2)
    strategy = SemanticSummaryStrategy()
    req = CondensationRequest(
        session_id=session.session_id,
        strategy="semantic_summary",
        keep_last_n=5,
    )
    new_session = strategy.apply(session, req)
    # Nothing to collapse; pass-through with original 2 messages
    assert new_session.message_count == 2


def test_keep_recent_invalid_n_raises() -> None:
    session = _make_session(3)
    strategy = KeepRecentStrategy()
    with pytest.raises(ValueError):
        strategy.apply(
            session,
            CondensationRequest(
                session_id=session.session_id,
                strategy="keep_recent",
                keep_last_n=0,
            ),
        )

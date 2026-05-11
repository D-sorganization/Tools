"""Tests for shared chat Pydantic models."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from chat.models import (  # noqa: E402
    ChatChunkResponse,
    ChatHistoryResponse,
    ChatIndexStatusResponse,
    ChatMessageRequest,
    ChatSessionInfo,
)


class TestChatMessageRequest:
    """Tests for ChatMessageRequest validation."""

    def test_valid_message(self):
        req = ChatMessageRequest(message="Hello")
        assert req.message == "Hello"
        assert req.app_context is None
        assert req.expertise_level == "beginner"

    def test_with_app_context(self):
        req = ChatMessageRequest(message="Help", app_context="gasification")
        assert req.app_context == "gasification"

    def test_message_min_length(self):
        with pytest.raises(ValueError):
            ChatMessageRequest(message="")

    def test_message_max_length(self):
        with pytest.raises(ValueError):
            ChatMessageRequest(message="x" * 10001)

    def test_custom_expertise_level(self):
        req = ChatMessageRequest(message="Hi", expertise_level="expert")
        assert req.expertise_level == "expert"


class TestChatChunkResponse:
    """Tests for ChatChunkResponse defaults."""

    def test_defaults(self):
        chunk = ChatChunkResponse(content="hello")
        assert chunk.content == "hello"
        assert chunk.is_final is False
        assert chunk.index == 0

    def test_final_chunk(self):
        chunk = ChatChunkResponse(content="done", is_final=True, index=5)
        assert chunk.is_final is True
        assert chunk.index == 5


class TestChatSessionInfo:
    """Tests for ChatSessionInfo construction."""

    def test_construction(self):
        info = ChatSessionInfo(
            session_id="abc",
            message_count=5,
            created_at="2026-01-01",
            last_active="2026-01-01",
        )
        assert info.session_id == "abc"
        assert info.app_contexts == []

    def test_with_contexts(self):
        info = ChatSessionInfo(
            session_id="abc",
            message_count=1,
            created_at="2026-01-01",
            last_active="2026-01-01",
            app_contexts=["mujoco", "drake"],
        )
        assert info.app_contexts == ["mujoco", "drake"]


class TestChatHistoryResponse:
    """Tests for ChatHistoryResponse."""

    def test_empty_history(self):
        resp = ChatHistoryResponse(session_id="abc", messages=[])
        assert resp.messages == []

    def test_with_messages(self):
        resp = ChatHistoryResponse(
            session_id="abc",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert len(resp.messages) == 1


class TestChatIndexStatusResponse:
    """Tests for ChatIndexStatusResponse (added in #2549)."""

    def test_minimal_running(self):
        resp = ChatIndexStatusResponse(state="running")
        assert resp.state == "running"
        assert resp.files_parsed == 0
        assert resp.symbols_inserted == 0
        assert resp.duration_seconds is None
        assert resp.error is None

    def test_complete(self):
        resp = ChatIndexStatusResponse(
            state="complete",
            files_parsed=120,
            symbols_inserted=4500,
            duration_seconds=2.7,
        )
        assert resp.state == "complete"
        assert resp.files_parsed == 120
        assert resp.symbols_inserted == 4500
        assert resp.duration_seconds == pytest.approx(2.7)

    def test_error_state(self):
        resp = ChatIndexStatusResponse(state="error", error="git not available")
        assert resp.error == "git not available"

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError):
            ChatIndexStatusResponse(state="running", files_parsed=-1)
        with pytest.raises(ValueError):
            ChatIndexStatusResponse(state="running", symbols_inserted=-3)
        with pytest.raises(ValueError):
            ChatIndexStatusResponse(state="complete", duration_seconds=-0.5)

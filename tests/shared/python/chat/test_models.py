"""Tests for shared chat Pydantic models."""

from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

from chat.models import (  # noqa: E402
    ChatChunkResponse,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatModelInfo,
    ChatModelListResponse,
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


class TestChatModelInfo:
    """Tests for ChatModelInfo (added in #2547)."""

    def test_defaults(self):
        info = ChatModelInfo(id="ollama:llama3", name="Llama 3", provider="ollama")
        assert info.id == "ollama:llama3"
        assert info.name == "Llama 3"
        assert info.provider == "ollama"
        assert info.available is True

    def test_unavailable(self):
        info = ChatModelInfo(
            id="openai:gpt-4o",
            name="GPT-4o",
            provider="openai",
            available=False,
        )
        assert info.available is False


class TestChatModelListResponse:
    """Tests for ChatModelListResponse (added in #2547)."""

    def test_empty(self):
        resp = ChatModelListResponse(refreshed_at="2026-05-11T00:00:00Z")
        assert resp.models == []
        assert resp.refreshed_at == "2026-05-11T00:00:00Z"

    def test_with_models(self):
        resp = ChatModelListResponse(
            refreshed_at="2026-05-11T00:00:00Z",
            models=[
                ChatModelInfo(id="ollama:llama3", name="Llama 3", provider="ollama"),
                ChatModelInfo(
                    id="openai:gpt-4o",
                    name="GPT-4o",
                    provider="openai",
                    available=False,
                ),
            ],
        )
        assert len(resp.models) == 2
        assert resp.models[0].provider == "ollama"
        assert resp.models[1].available is False

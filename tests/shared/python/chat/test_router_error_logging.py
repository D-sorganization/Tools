"""Tests for structured two-layer error logging in router_factory.

Verifies that:
- Expected exception types (AIProviderError, ValueError, ...) produce a
  warning log entry and send the exception detail to the client.
- Unexpected exceptions produce a logger.exception call (full traceback)
  and send a sanitised "Internal server error" to the client.
- No internal details leak to the client on unexpected errors.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from chat.router_factory import AIProviderError, create_chat_router

from chat import ChatServiceBase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeChatService(ChatServiceBase):
    """Minimal concrete chat service for router tests."""

    def __init__(self) -> None:
        super().__init__()
        # Injected via test; callable so tests can control what happens.
        self._condense_fn: Any = AsyncMock(return_value=None)
        self._skill_fn: Any = AsyncMock(return_value=None)
        self._review_fn: Any = AsyncMock(return_value="review_session_123")
        self._refresh_fn: Any = AsyncMock(return_value=[])
        self._index_fn: Any = AsyncMock(
            return_value={
                "state": "complete",
                "files_parsed": 10,
                "symbols_inserted": 50,
            }
        )

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        yield "ok"

    async def condense_session(self, session_id: str) -> None:
        await self._condense_fn(session_id)

    async def execute_skill(self, session_id: str, skill_id: str) -> None:
        await self._skill_fn(session_id, skill_id)

    async def request_review(self, session_id: str, provider: str) -> str:
        return await self._review_fn(session_id, provider)

    async def refresh_models(self) -> list[dict[str, Any]]:
        return await self._refresh_fn()

    async def index_codebase(self, root_path: str) -> dict[str, Any]:
        return await self._index_fn(root_path)


def _make_client(service: _FakeChatService) -> Any:
    app = fastapi.FastAPI()
    app.state.chat_service = service
    app.include_router(create_chat_router(), prefix="/api")
    return testclient.TestClient(app)


def _open_ws(client: Any) -> Any:
    """Open a WS connection, consume the initial session_info frame."""
    cm = client.websocket_connect("/api/ws/chat/new")
    ws = cm.__enter__()
    ws.receive_json()  # session_info
    return cm, ws


# ---------------------------------------------------------------------------
# condense action
# ---------------------------------------------------------------------------


class TestCondenseErrorLogging:
    def test_expected_exception_warns_and_sends_detail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ValueError from condense_session → warning + detail in client msg."""
        service = _FakeChatService()
        service._condense_fn = AsyncMock(side_effect=ValueError("bad session state"))
        client = _make_client(service)

        with caplog.at_level(logging.WARNING, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()  # session_info
                ws.send_json({"action": "condense"})
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "bad session state"}
        assert any("Condense failed" in r.message for r in caplog.records)
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_expected_ai_provider_error_warns_and_sends_detail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AIProviderError from condense_session → warning + provider msg."""
        service = _FakeChatService()
        service._condense_fn = AsyncMock(
            side_effect=AIProviderError("rate limit hit", provider="openai")
        )
        client = _make_client(service)

        with caplog.at_level(logging.WARNING, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "condense"})
                payload = ws.receive_json()

        assert payload["type"] == "error"
        assert "rate limit hit" in payload["detail"]
        assert any("Condense failed" in r.message for r in caplog.records)

    def test_unexpected_exception_calls_logger_exception_and_sanitises_reply(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RuntimeError from condense_session → logger.exception + generic reply."""
        service = _FakeChatService()
        service._condense_fn = AsyncMock(
            side_effect=RuntimeError("boom — internal secret")
        )
        client = _make_client(service)

        with caplog.at_level(logging.ERROR, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "condense"})
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "Internal server error"}
        # Must NOT leak the internal error message to the client
        assert "internal secret" not in payload["detail"]
        # logger.exception records show exc_info=True (they are ERROR level)
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, "Expected at least one ERROR-level log record"
        assert any("Unexpected error" in r.message for r in error_records)

    def test_unexpected_exception_traceback_is_captured(self) -> None:
        """logger.exception is called (not logger.error) so traceback is logged."""
        service = _FakeChatService()
        service._condense_fn = AsyncMock(side_effect=RuntimeError("boom"))
        client = _make_client(service)

        with patch("chat.router_factory.logger") as mock_logger:
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "condense"})
                ws.receive_json()

        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args
        assert "Unexpected error" in call_args.args[0]


# ---------------------------------------------------------------------------
# skill_invoke action
# ---------------------------------------------------------------------------


class TestSkillInvokeErrorLogging:
    def test_expected_exception_warns_and_sends_detail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ValueError from execute_skill → warning + detail forwarded."""
        service = _FakeChatService()
        service._skill_fn = AsyncMock(side_effect=ValueError("unknown skill"))
        client = _make_client(service)

        with caplog.at_level(logging.WARNING, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "skill_invoke", "skill_id": "my_skill"})
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "unknown skill"}
        assert any("Skill invoke failed" in r.message for r in caplog.records)

    def test_unexpected_exception_sanitises_reply(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RuntimeError from execute_skill → generic reply + exception log."""
        service = _FakeChatService()
        service._skill_fn = AsyncMock(side_effect=RuntimeError("boom"))
        client = _make_client(service)

        with caplog.at_level(logging.ERROR, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "skill_invoke", "skill_id": "my_skill"})
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "Internal server error"}
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records
        assert any("Unexpected error" in r.message for r in error_records)


# ---------------------------------------------------------------------------
# request_review action
# ---------------------------------------------------------------------------


class TestRequestReviewErrorLogging:
    def test_expected_exception_warns_and_sends_detail(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ValueError from request_review → warning + detail forwarded."""
        service = _FakeChatService()
        service._review_fn = AsyncMock(side_effect=ValueError("unsupported provider"))
        client = _make_client(service)

        with caplog.at_level(logging.WARNING, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "request_review", "provider": "gpt4"})
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "unsupported provider"}
        assert any("Review request failed" in r.message for r in caplog.records)

    def test_unexpected_exception_sanitises_reply(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """RuntimeError from request_review → generic reply + exception log."""
        service = _FakeChatService()
        service._review_fn = AsyncMock(side_effect=RuntimeError("boom"))
        client = _make_client(service)

        with caplog.at_level(logging.ERROR, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "request_review", "provider": "gpt4"})
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "Internal server error"}
        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records
        assert any("Unexpected error" in r.message for r in error_records)


class TestNewActionsErrorLogging:
    def test_refresh_models_error_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AIProviderError from refresh_models → warning."""
        service = _FakeChatService()
        service._refresh_fn = AsyncMock(side_effect=AIProviderError("failed to poll"))
        client = _make_client(service)

        with caplog.at_level(logging.WARNING, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "refresh_models"})
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "failed to poll"}
        assert any("Refresh models failed" in r.message for r in caplog.records)

    def test_index_codebase_error_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AIProviderError from index_codebase → warning."""
        service = _FakeChatService()
        service._index_fn = AsyncMock(side_effect=AIProviderError("disk full"))
        client = _make_client(service)

        with caplog.at_level(logging.WARNING, logger="chat.router_factory"):
            with client.websocket_connect("/api/ws/chat/new") as ws:
                ws.receive_json()
                ws.send_json({"action": "index_codebase", "root_path": "/tmp"})  # nosec B108
                payload = ws.receive_json()

        assert payload == {"type": "error", "detail": "disk full"}
        assert any("Indexing failed" in r.message for r in caplog.records)

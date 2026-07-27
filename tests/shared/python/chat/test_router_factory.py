"""Tests for the shared WebSocket chat router factory.

Issue #2751: verify that refresh_models and index_codebase WebSocket actions
are handled correctly and do not fall through to the Unknown action error path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from chat.router_factory import create_chat_router  # noqa: E402
from chat.service_base import ChatServiceBase  # noqa: E402

# ── Minimal fake service ─────────────────────────────────────────────────────


class _FakeChatService(ChatServiceBase):
    """Minimal concrete ChatServiceBase stub for router unit tests.

    All abstract methods are implemented so the class can be instantiated.
    Individual tests override specific methods via AsyncMock to control
    what the router sees.
    """

    def __init__(self) -> None:
        super().__init__()
        self._refresh_fn: Any = AsyncMock(return_value=[])
        self._index_fn: Any = AsyncMock(
            return_value={
                "state": "complete",
                "files_parsed": 0,
                "symbols_inserted": 0,
            }
        )

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        yield {"type": "chunk", "content": "ok"}

    async def refresh_models(self) -> list[dict[str, Any]]:
        return await self._refresh_fn()

    async def index_codebase(self, root_path: str) -> dict[str, Any]:
        return await self._index_fn(root_path)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(service: _FakeChatService | None = None) -> Any:
    """Build a FastAPI TestClient that exposes the chat router."""
    app = fastapi.FastAPI()
    app.state.chat_service = service or _FakeChatService()
    app.include_router(create_chat_router())
    return testclient.TestClient(app)


# ── refresh_models ────────────────────────────────────────────────────────────


class TestRefreshModels:
    """Issue #2751: router must handle refresh_models (was: Unknown action)."""

    def test_returns_model_list_event(self) -> None:
        """Router sends type=model_list with the model list returned by the service."""
        service = _FakeChatService()
        service._refresh_fn = AsyncMock(
            return_value=[
                {
                    "id": "gpt-4o",
                    "name": "GPT-4o",
                    "provider": "openai",
                    "available": True,
                }
            ]
        )
        client = _make_client(service)

        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()  # session_info handshake
            ws.send_json({"action": "refresh_models"})
            reply = ws.receive_json()

        assert reply["type"] == "model_list", f"Expected model_list, got: {reply}"
        assert isinstance(reply["models"], list)
        assert reply["models"][0]["id"] == "gpt-4o"
        assert "refreshed_at" in reply

    def test_value_error_returns_error_not_unknown_action(self) -> None:
        """Service ValueError produces type=error with detail (not Unknown action)."""
        service = _FakeChatService()
        service._refresh_fn = AsyncMock(side_effect=ValueError("provider unavailable"))
        client = _make_client(service)

        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "refresh_models"})
            reply = ws.receive_json()

        assert reply["type"] == "error"
        assert "provider unavailable" in reply["detail"]
        assert "Unknown action" not in reply["detail"]

    def test_empty_model_list_is_valid(self) -> None:
        """Service returning an empty list should still produce model_list event."""
        client = _make_client()  # default stub returns []

        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "refresh_models"})
            reply = ws.receive_json()

        assert reply["type"] == "model_list"
        assert reply["models"] == []


# ── index_codebase ────────────────────────────────────────────────────────────


class TestIndexCodebase:
    """Issue #2751: router must handle index_codebase (was: Unknown action)."""

    def test_returns_index_status_event(self) -> None:
        """Router sends type=index_status with the status dict from the service."""
        service = _FakeChatService()
        service._index_fn = AsyncMock(
            return_value={
                "state": "complete",
                "files_parsed": 42,
                "symbols_inserted": 1024,
                "duration_seconds": 3.7,
            }
        )
        client = _make_client(service)

        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()  # session_info handshake
            ws.send_json({"action": "index_codebase", "root_path": "/repo"})
            reply = ws.receive_json()

        assert reply["type"] == "index_status", f"Expected index_status, got: {reply}"
        assert reply["state"] == "complete"
        assert reply["files_parsed"] == 42
        assert reply["symbols_inserted"] == 1024

    def test_missing_root_path_uses_process_cwd(self, monkeypatch: Any) -> None:
        """Omitting root_path falls back to cwd so dock requests work."""
        service = _FakeChatService()
        client = _make_client(service)
        monkeypatch.setattr("os.getcwd", lambda: "/repo")

        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "index_codebase"})
            reply = ws.receive_json()

        assert reply["type"] == "index_status"
        service._index_fn.assert_awaited_once_with("/repo")

    def test_value_error_returns_error_not_unknown_action(self) -> None:
        """Service ValueError produces type=error with detail (not Unknown action)."""
        service = _FakeChatService()
        service._index_fn = AsyncMock(
            side_effect=ValueError("root path does not exist")
        )
        client = _make_client(service)

        with client.websocket_connect("/ws/chat/new") as ws:
            ws.receive_json()
            ws.send_json({"action": "index_codebase", "root_path": "/nonexistent"})
            reply = ws.receive_json()

        assert reply["type"] == "error"
        assert "Unknown action" not in reply["detail"]


# ── catch-all unknown action ─────────────────────────────────────────────────


def test_unknown_action_still_returns_error_detail() -> None:
    """Sanity check: truly unknown actions still produce an Unknown action error."""
    client = _make_client()

    with client.websocket_connect("/ws/chat/new") as ws:
        ws.receive_json()
        ws.send_json({"action": "definitely_not_a_real_action"})
        reply = ws.receive_json()

    assert reply["type"] == "error"
    assert "Unknown action" in reply["detail"]


def test_session_handshake_keeps_terminal_capabilities() -> None:
    """Moving the loop must not drop the existing router capability frame."""
    client = _make_client()

    with client.websocket_connect("/ws/chat/new") as ws:
        reply = ws.receive_json()

    assert reply["type"] == "session_info"
    assert reply["capabilities"] == {"terminal_runtime": False}

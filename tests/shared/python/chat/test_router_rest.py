"""Tests for router_factory REST endpoints.

Covers the HTTP (non-WebSocket) surface of ``create_chat_router``:
  - GET  /chat/sessions          → list sessions
  - GET  /chat/sessions/{id}/history → per-session history

These paths were previously untested (only the WebSocket path had coverage).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from chat.router_factory import create_chat_router

from chat import ChatServiceBase

# ---------------------------------------------------------------------------
# Minimal concrete service
# ---------------------------------------------------------------------------


class _MinimalService(ChatServiceBase):
    """Concrete service stub with controllable state."""

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:  # type: ignore[override]
        yield "ok"

    async def refresh_models(self) -> list[dict[str, Any]]:
        return []

    async def index_codebase(self, root_path: str) -> dict[str, Any]:
        return {"state": "complete", "files_parsed": 0, "symbols_inserted": 0}


def _make_app(service: ChatServiceBase | None = None) -> Any:
    """Create a FastAPI test app with the chat router mounted."""
    app = fastapi.FastAPI()
    app.state.chat_service = service or _MinimalService()
    app.include_router(create_chat_router(), prefix="/api")
    return app


# ---------------------------------------------------------------------------
# GET /api/chat/sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_returns_empty_list_when_no_sessions_exist(self) -> None:
        """Fresh service has no sessions → endpoint returns []."""
        client = testclient.TestClient(_make_app())
        resp = client.get("/api/chat/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_list_after_session_is_created(self) -> None:
        """Creating a session via the service layer shows up in GET."""
        service = _MinimalService()
        session = service.get_or_create_session(None)
        client = testclient.TestClient(_make_app(service))

        resp = client.get("/api/chat/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["session_id"] == session.session_id

    def test_returns_all_sessions_when_multiple_exist(self) -> None:
        """Multiple sessions are all represented in the list."""
        service = _MinimalService()
        s1 = service.get_or_create_session(None)
        s2 = service.get_or_create_session(None)
        s3 = service.get_or_create_session(None)
        client = testclient.TestClient(_make_app(service))

        resp = client.get("/api/chat/sessions")

        assert resp.status_code == 200
        data = resp.json()
        ids = {item["session_id"] for item in data}
        assert ids == {s1.session_id, s2.session_id, s3.session_id}

    def test_each_session_has_expected_fields(self) -> None:
        """Each entry in the list has the standard session summary shape."""
        service = _MinimalService()
        service.get_or_create_session(None)
        client = testclient.TestClient(_make_app(service))

        resp = client.get("/api/chat/sessions")
        data = resp.json()

        entry = data[0]
        for key in ("session_id", "message_count", "created_at", "last_active"):
            assert key in entry, f"Missing field: {key}"

    def test_message_count_increments_after_adding_messages(self) -> None:
        """message_count reflects actual message history."""
        service = _MinimalService()
        session = service.get_or_create_session(None)
        service.add_user_message(session.session_id, "hello")
        service.add_user_message(session.session_id, "world")
        client = testclient.TestClient(_make_app(service))

        resp = client.get("/api/chat/sessions")
        data = resp.json()

        assert data[0]["message_count"] == 2


# ---------------------------------------------------------------------------
# GET /api/chat/sessions/{id}/history
# ---------------------------------------------------------------------------


class TestGetSessionHistory:
    def test_returns_empty_messages_for_fresh_session(self) -> None:
        """A session with no messages returns an empty list."""
        service = _MinimalService()
        session = service.get_or_create_session(None)
        client = testclient.TestClient(_make_app(service))

        resp = client.get(f"/api/chat/sessions/{session.session_id}/history")

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == session.session_id
        assert body["messages"] == []

    def test_returns_messages_in_order(self) -> None:
        """Messages appear in insertion order."""
        service = _MinimalService()
        session = service.get_or_create_session(None)
        service.add_user_message(session.session_id, "first")
        service.add_user_message(session.session_id, "second")
        client = testclient.TestClient(_make_app(service))

        resp = client.get(f"/api/chat/sessions/{session.session_id}/history")

        assert resp.status_code == 200
        messages = resp.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "first"
        assert messages[1]["content"] == "second"

    def test_each_message_has_role_content_timestamp(self) -> None:
        """Each message dict has role, content, and timestamp keys."""
        service = _MinimalService()
        session = service.get_or_create_session(None)
        service.add_user_message(session.session_id, "ping")
        client = testclient.TestClient(_make_app(service))

        resp = client.get(f"/api/chat/sessions/{session.session_id}/history")

        msg = resp.json()["messages"][0]
        for key in ("role", "content", "timestamp"):
            assert key in msg, f"Missing key: {key}"
        assert msg["role"] == "user"
        assert msg["content"] == "ping"

    def test_unknown_session_returns_empty_messages(self) -> None:
        """Non-existent session_id returns an empty messages list (not 404).

        The service layer returns [] for unknown sessions, so the REST
        endpoint forwards that to the caller.
        """
        client = testclient.TestClient(_make_app())

        resp = client.get("/api/chat/sessions/nonexistent-session-id/history")

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "nonexistent-session-id"
        assert body["messages"] == []

    def test_whitespace_only_session_id_returns_empty_messages(self) -> None:
        """Whitespace-only session_id is treated as empty and returns []."""
        client = testclient.TestClient(_make_app())
        # Path parameter must be non-empty for the route to match; use spaces
        # as a URL-encoded string to exercise the strip guard in the handler.
        resp = client.get("/api/chat/sessions/%20/history")

        assert resp.status_code == 200
        body = resp.json()
        assert body["messages"] == []

    def test_session_id_is_echoed_in_response(self) -> None:
        """The session_id from the path is always reflected in the body."""
        service = _MinimalService()
        session = service.get_or_create_session(None)
        client = testclient.TestClient(_make_app(service))

        resp = client.get(f"/api/chat/sessions/{session.session_id}/history")

        assert resp.json()["session_id"] == session.session_id


# ---------------------------------------------------------------------------
# Integration: REST endpoints share same service state as WebSocket
# ---------------------------------------------------------------------------


class TestRestAndWebSocketShareState:
    def test_session_created_via_ws_appears_in_rest_list(self) -> None:
        """A session opened via WebSocket is visible in GET /chat/sessions."""
        service = _MinimalService()
        client = testclient.TestClient(_make_app(service))

        with client.websocket_connect("/api/ws/chat/new") as ws:
            info = ws.receive_json()
            assert info["type"] == "session_info"
            ws_session_id = info["session_id"]

        resp = client.get("/api/chat/sessions")
        assert resp.status_code == 200
        ids = {item["session_id"] for item in resp.json()}
        assert ws_session_id in ids

    def test_message_sent_via_ws_appears_in_rest_history(self) -> None:
        """A message sent over WebSocket shows up in GET history endpoint."""
        service = _MinimalService()
        client = testclient.TestClient(_make_app(service))

        with client.websocket_connect("/api/ws/chat/new") as ws:
            info = ws.receive_json()
            ws_session_id = info["session_id"]
            ws.send_json({"action": "send", "message": "hello from ws"})
            # Drain all responses until "complete"
            for _ in range(10):
                reply = ws.receive_json()
                if reply.get("type") == "complete":
                    break

        resp = client.get(f"/api/chat/sessions/{ws_session_id}/history")
        assert resp.status_code == 200
        contents = [m["content"] for m in resp.json()["messages"]]
        assert "hello from ws" in contents

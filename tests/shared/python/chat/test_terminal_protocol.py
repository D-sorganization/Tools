"""WebSocket protocol tests for shared chat terminal actions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from chat import ChatServiceBase, TerminalAgentEvent, TerminalAgentSessionInfo
from chat.router_factory import create_chat_router


class FakeChatService(ChatServiceBase):
    """Minimal chat service required by the shared chat router."""

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        yield "ok"


class FakeTerminalRuntime:
    """Runtime double for terminal protocol tests."""

    def __init__(self) -> None:
        self.started: list[Any] = []
        self.writes: list[tuple[str, str]] = []
        self.resizes: list[tuple[str, int, int]] = []
        self.stopped: list[str] = []

    def start(self, request: Any) -> TerminalAgentSessionInfo:
        self.started.append(request)
        return TerminalAgentSessionInfo(
            session_id="terminal_123",
            resolved_cwd=request.project_root,
            shell_id=request.shell_id,
            provider_id=request.provider_id,
            state="running",
        )

    def write(self, session_id: str, text: str) -> None:
        self.writes.append((session_id, text))

    def resize(self, session_id: str, *, columns: int, rows: int) -> None:
        self.resizes.append((session_id, columns, rows))

    def stop(self, session_id: str) -> TerminalAgentSessionInfo:
        self.stopped.append(session_id)
        return TerminalAgentSessionInfo(
            session_id=session_id,
            resolved_cwd=Path.cwd(),
            shell_id="powershell",
            provider_id="codex",
            state="stopped",
        )

    def drain_events(self, session_id: str) -> list[TerminalAgentEvent]:
        return [
            TerminalAgentEvent(
                session_id=session_id,
                event_type="stdout",
                data="hello",
            )
        ]


def _client(runtime: FakeTerminalRuntime | None = None) -> Any:
    app = fastapi.FastAPI()
    app.state.chat_service = FakeChatService()
    if runtime is not None:
        app.state.terminal_runtime = runtime
    app.include_router(create_chat_router(), prefix="/api")
    return testclient.TestClient(app)


def test_terminal_start_returns_session_payload(tmp_path: Path) -> None:
    """Terminal start action validates and forwards session requests."""
    runtime = FakeTerminalRuntime()
    client = _client(runtime)

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "action": "terminal_start",
                "project_root": str(tmp_path),
                "shell_id": "powershell",
                "provider_id": "codex",
                "app_context": "tools",
            }
        )
        payload = websocket.receive_json()

    assert payload["type"] == "terminal_session"
    assert payload["session"]["session_id"] == "terminal_123"
    assert runtime.started[0].project_root == tmp_path.resolve()


def test_terminal_input_resize_stop_and_events_are_forwarded() -> None:
    """Lifecycle protocol actions call the runtime and return typed payloads."""
    runtime = FakeTerminalRuntime()
    client = _client(runtime)

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "action": "terminal_input",
                "terminal_session_id": "terminal_123",
                "text": "hello\n",
            }
        )
        assert websocket.receive_json()["type"] == "terminal_ack"

        websocket.send_json(
            {
                "action": "terminal_resize",
                "terminal_session_id": "terminal_123",
                "columns": 120,
                "rows": 30,
            }
        )
        assert websocket.receive_json()["type"] == "terminal_ack"

        websocket.send_json(
            {"action": "terminal_events", "terminal_session_id": "terminal_123"}
        )
        events = websocket.receive_json()
        assert events["events"][0]["data"] == "hello"

        websocket.send_json(
            {"action": "terminal_stop", "terminal_session_id": "terminal_123"}
        )
        stopped = websocket.receive_json()

    assert stopped["session"]["state"] == "stopped"
    assert runtime.writes == [("terminal_123", "hello\n")]
    assert runtime.resizes == [("terminal_123", 120, 30)]
    assert runtime.stopped == ["terminal_123"]


def test_terminal_action_without_runtime_returns_error(tmp_path: Path) -> None:
    """Apps that do not opt into terminal mode keep a structured error."""
    client = _client()

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "action": "terminal_start",
                "project_root": str(tmp_path),
                "shell_id": "powershell",
                "provider_id": "codex",
            }
        )
        payload = websocket.receive_json()

    assert payload == {
        "type": "error",
        "detail": "Terminal runtime is not configured",
    }


def test_terminal_start_validation_error_is_structured() -> None:
    """Bad terminal requests return error payloads instead of crashing."""
    client = _client(FakeTerminalRuntime())

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "action": "terminal_start",
                "project_root": "missing",
                "shell_id": "powershell",
                "provider_id": "codex",
            }
        )
        payload = websocket.receive_json()

    assert payload["type"] == "error"
    assert "project_root" in payload["detail"]


# ── refresh_models / index_codebase (Tools issue #2751) ──────────────


class _ModelListService(ChatServiceBase):
    """Service double that returns a fixed model list."""

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        yield "ok"

    async def refresh_models(self) -> list[dict[str, Any]]:
        return [
            {"id": "model-a", "name": "model-a", "provider": "fake"},
            {"id": "model-b", "name": "model-b", "provider": "fake"},
        ]


class _IndexService(ChatServiceBase):
    """Service double that records the index_codebase root_path it received."""

    def __init__(self) -> None:
        super().__init__()
        self.received_root: str | None = None

    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:
        yield "ok"

    async def index_codebase(self, root_path: str) -> dict[str, Any]:
        self.received_root = root_path
        return {
            "state": "complete",
            "files_parsed": 42,
            "symbols_inserted": 100,
            "duration_seconds": 1.5,
        }


def _client_with(service: ChatServiceBase) -> Any:
    app = fastapi.FastAPI()
    app.state.chat_service = service
    app.include_router(create_chat_router(), prefix="/api")
    return testclient.TestClient(app)


def test_refresh_models_returns_model_list_payload() -> None:
    """refresh_models action returns a model_list response with models."""
    client = _client_with(_ModelListService())

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json({"action": "refresh_models"})
        payload = websocket.receive_json()

    assert payload["type"] == "model_list"
    assert payload["models"] == [
        {"id": "model-a", "name": "model-a", "provider": "fake"},
        {"id": "model-b", "name": "model-b", "provider": "fake"},
    ]
    assert "refreshed_at" in payload


def test_index_codebase_returns_index_status_payload(tmp_path: Path) -> None:
    """index_codebase action returns an index_status response with totals."""
    service = _IndexService()
    client = _client_with(service)

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json({"action": "index_codebase", "root_path": str(tmp_path)})
        payload = websocket.receive_json()

    assert payload["type"] == "index_status"
    assert payload["state"] == "complete"
    assert payload["files_parsed"] == 42
    assert payload["symbols_inserted"] == 100
    assert service.received_root == str(tmp_path)


def test_index_codebase_defaults_root_path_when_missing() -> None:
    """index_codebase falls back to cwd when no root_path is supplied.

    The dock widget sends ``{"action": "index_codebase"}`` with no
    ``root_path`` (Tools issue #2751).
    """
    service = _IndexService()
    client = _client_with(service)

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json({"action": "index_codebase"})
        payload = websocket.receive_json()

    assert payload["type"] == "index_status"
    assert service.received_root is not None
    assert service.received_root != ""


def test_refresh_models_not_implemented_returns_clear_error() -> None:
    """A service that does not override refresh_models gets a clear error
    message — NOT a generic 'Unknown action' reply (Tools issue #2751)."""
    client = _client_with(FakeChatService())

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json({"action": "refresh_models"})
        payload = websocket.receive_json()

    assert payload["type"] == "error"
    assert "not supported" in payload["detail"]
    assert "Unknown action" not in payload["detail"]


def test_index_codebase_not_implemented_returns_clear_error(tmp_path: Path) -> None:
    """A service that does not override index_codebase gets a clear error
    message — NOT a generic 'Unknown action' reply (Tools issue #2751)."""
    client = _client_with(FakeChatService())

    with client.websocket_connect("/api/ws/chat/new") as websocket:
        websocket.receive_json()
        websocket.send_json({"action": "index_codebase", "root_path": str(tmp_path)})
        payload = websocket.receive_json()

    assert payload["type"] == "error"
    assert "not supported" in payload["detail"]
    assert "Unknown action" not in payload["detail"]

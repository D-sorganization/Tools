"""Focused contract tests for the canonical chat WebSocket protocol."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from chat.websocket_protocol import (
    ChatWebSocketState,
    DisconnectLogConfig,
    run_chat_websocket_protocol,
)
from starlette.websockets import WebSocketDisconnect

pytestmark = [pytest.mark.anyio, pytest.mark.unit]


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


class _Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id


class _ChatService:
    def __init__(self) -> None:
        self.sessions: list[str | None] = []
        self.added: list[tuple[str, str, str | None]] = []

    def get_or_create_session(self, session_id: str | None) -> _Session:
        self.sessions.append(session_id)
        return _Session(session_id or f"new-{len(self.sessions)}")

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": session_id}]

    def add_user_message(
        self,
        session_id: str,
        message: str,
        app_context: str | None,
    ) -> None:
        self.added.append((session_id, message, app_context))

    async def stream_response(self, _session_id: str) -> AsyncIterator[str]:
        yield "reply"


class _FakeWebSocket:
    def __init__(
        self,
        service: _ChatService,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.app = SimpleNamespace(
            state=SimpleNamespace(chat_service=service, terminal_runtime=object())
        )
        self.messages = list(messages or [])
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> dict[str, Any]:
        if not self.messages:
            raise WebSocketDisconnect(code=1000)
        return self.messages.pop(0)

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


async def test_authorization_hook_can_reject_before_accept() -> None:
    service = _ChatService()
    websocket = _FakeWebSocket(service)

    await run_chat_websocket_protocol(
        websocket,
        authorize_fn=lambda _websocket: False,
    )

    assert websocket.accepted is False
    assert websocket.sent == []
    assert service.sessions == []


async def test_before_send_and_context_hooks_preserve_host_behavior() -> None:
    service = _ChatService()
    websocket = _FakeWebSocket(
        service,
        [
            {
                "action": "send",
                "message": " hello ",
                "app_context": "portable",
                "engine_context": "host-engine",
            }
        ],
    )
    observed: list[str] = []

    async def before_send(session: _Session) -> None:
        observed.append(session.session_id)

    await run_chat_websocket_protocol(
        websocket,
        "session-1",
        before_send=before_send,
    )

    assert observed == ["session-1"]
    assert service.added == [("session-1", "hello", "host-engine")]
    assert websocket.sent[-2:] == [
        {"type": "chunk", "content": "reply"},
        {"type": "complete", "session_id": "session-1"},
    ]


async def test_custom_action_handler_receives_mutable_session_state() -> None:
    service = _ChatService()
    websocket = _FakeWebSocket(service, [{"action": "custom"}])
    observed: list[tuple[str, str]] = []

    async def handle_custom(
        target: _FakeWebSocket,
        message: dict[str, Any],
        state: ChatWebSocketState,
    ) -> None:
        observed.append((message["action"], state.session_id))
        await target.send_json({"type": "custom_complete"})

    await run_chat_websocket_protocol(
        websocket,
        "session-1",
        action_handlers={"custom": handle_custom},
    )

    assert observed == [("custom", "session-1")]
    assert websocket.sent[-1] == {"type": "custom_complete"}


async def test_session_info_extra_preserves_router_capabilities() -> None:
    service = _ChatService()
    websocket = _FakeWebSocket(service)

    await run_chat_websocket_protocol(
        websocket,
        "session-1",
        session_info_extra=lambda target, _state: {
            "capabilities": {
                "terminal_runtime": target.app.state.terminal_runtime is not None
            }
        },
    )

    assert websocket.sent[0] == {
        "type": "session_info",
        "session_id": "session-1",
        "capabilities": {"terminal_runtime": True},
    }


async def test_disconnect_logging_can_hash_sensitive_session_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_session_id = "customer-session-raw-secret"
    service = _ChatService()
    websocket = _FakeWebSocket(service)
    protocol_logger = logging.getLogger("test.chat.websocket_protocol")
    caplog.set_level(logging.DEBUG, logger=protocol_logger.name)

    await run_chat_websocket_protocol(
        websocket,
        sensitive_session_id,
        log=protocol_logger,
        disconnect_log=DisconnectLogConfig(
            message="Disconnected: session_token=%s",
            args_fn=lambda _session_id: ("stable-hash-token",),
        ),
    )

    assert "stable-hash-token" in caplog.text
    assert sensitive_session_id not in caplog.text

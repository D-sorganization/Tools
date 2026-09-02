"""Regression tests for the native Sidekick WebSocket startup contract.

Tools issue #3936: the PyQt6 client must authenticate its local WebSocket
handshake with the same ephemeral capability that the host launcher passes to
the API process.  These tests are intentionally owned by Tools because the
chat widget is shared source; downstream copies must only be synchronized.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

from src.shared.python.chat import chat_dock_widget
from src.shared.python.chat._chat_dock_widget_qt import (
    ChatConnectionConfig,
    ChatDockWidget,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Return the single Qt application required to construct chat widgets."""
    return QApplication.instance() or QApplication([])


def test_native_websocket_url_includes_encoded_launcher_capability() -> None:
    """The capability must be query encoded rather than concatenated raw."""
    result = chat_dock_widget._build_native_websocket_url(
        "ws://127.0.0.1:8123",
        "/api/ws/chat/new",
        "token/+ with spaces",
    )

    assert result == (
        "ws://127.0.0.1:8123/api/ws/chat/new?launcher_token=token%2F%2B+with+spaces"
    )


def test_native_websocket_url_omits_absent_launcher_capability() -> None:
    """Standalone remote clients remain compatible when no token is supplied."""
    result = chat_dock_widget._build_native_websocket_url(
        "wss://chat.example",
        "/api/ws/chat/existing",
        None,
    )

    assert result == "wss://chat.example/api/ws/chat/existing"


def test_native_websocket_url_never_forwards_launcher_capability_remotely() -> None:
    """The local launcher secret must not cross the loopback trust boundary."""
    result = chat_dock_widget._build_native_websocket_url(
        "wss://chat.example",
        "/api/ws/chat/existing",
        "local-launcher-secret",
    )

    assert result == "wss://chat.example/api/ws/chat/existing"
    assert "local-launcher-secret" not in result


@pytest.mark.parametrize(
    "server_url",
    [
        "ws://localhost:8123",
        "ws://localhost.:8123",
        "ws://127.0.0.2:8123",
        "ws://[::1]:8123",
    ],
)
def test_native_websocket_url_forwards_capability_only_to_loopback(
    server_url: str,
) -> None:
    """Verified localhost and loopback IP endpoints retain local authentication."""
    result = chat_dock_widget._build_native_websocket_url(
        server_url,
        "/api/ws/chat/new",
        "runtime-secret",
    )

    assert result.endswith("/api/ws/chat/new?launcher_token=runtime-secret")


@pytest.mark.parametrize(
    ("server_url", "expected"),
    [
        ("ws://127.0.0.1:8123", "http://127.0.0.1:8123"),
        ("wss://localhost:9443", "https://localhost:9443"),
    ],
)
def test_native_websocket_origin_matches_server(
    server_url: str,
    expected: str,
) -> None:
    """The native client must send a loopback origin accepted by the API guard."""
    assert chat_dock_widget._native_websocket_origin(server_url) == expected


def test_native_websocket_contract_rejects_invalid_server_scheme() -> None:
    """DbC: only WebSocket server URLs are valid inputs."""
    with pytest.raises(ValueError, match="ws:// or wss://"):
        chat_dock_widget._native_websocket_origin("http://127.0.0.1:8000")


@pytest.mark.parametrize(
    "server_url",
    [
        "ws://127.0.0.1:8000/custom-base",
        "ws://127.0.0.1:8000?transport=native",
        "ws://127.0.0.1:8000#sidekick",
    ],
)
def test_native_websocket_contract_rejects_non_authority_server_url(
    server_url: str,
) -> None:
    """DbC: the server setting cannot corrupt the separately owned WS path."""
    with pytest.raises(ValueError, match="scheme and authority"):
        chat_dock_widget._build_native_websocket_url(
            server_url,
            "/api/ws/chat/new",
            None,
        )


def test_chat_connect_uses_origin_and_runtime_capability(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Qt socket receives both pieces of the local handshake contract."""
    del qapp  # QApplication lifetime is the fixture's purpose.
    monkeypatch.setenv("UD_LAUNCHER_CSRF_TOKEN", "runtime-secret")
    qt_module = sys.modules[ChatDockWidget.__module__]
    previous_session = ChatDockWidget._get_shared_session_id()
    ChatDockWidget._set_shared_session_id(None)

    try:
        with (
            patch.object(ChatDockWidget, "_setup_ui"),
            patch.object(qt_module, "_session_file_path", return_value=Path("unused")),
            patch.object(qt_module, "_read_shared_session_id", return_value=None),
            patch.object(qt_module, "QWebSocket") as socket_class,
        ):
            socket = MagicMock()
            socket_class.return_value = socket
            widget = ChatDockWidget(
                connection=ChatConnectionConfig(
                    app_context="test",
                    app_name="test-sidekick",
                    server_url="ws://127.0.0.1:8123",
                )
            )
            widget._status_label = MagicMock()

            widget._connect()
    finally:
        ChatDockWidget._set_shared_session_id(previous_session)

    socket_class.assert_called_once_with("http://127.0.0.1:8123")
    opened_url = socket.open.call_args.args[0]
    assert opened_url.toString() == (
        "ws://127.0.0.1:8123/api/ws/chat/new?launcher_token=runtime-secret"
    )
    diagnostics = widget.connection_diagnostics()
    assert "runtime-secret" not in repr(diagnostics)

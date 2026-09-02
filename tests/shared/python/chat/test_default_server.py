"""TDD coverage for ``chat_dock_widget._resolve_default_server`` in Tools.

The Tools mirror of the chat dock widget exposes the same env-driven
default-server logic as UpstreamDrift; the launcher probes a free port
and exports ``GOLF_API_PORT`` before spawning the API server, so this
helper must follow that contract or chat connects to the wrong port.
"""

from __future__ import annotations

import pytest
from PyQt6.QtWidgets import QApplication


def test_default_server_honours_golf_api_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLF_API_PORT", "8137")
    monkeypatch.delenv("UD_CHAT_WS_URL", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("GOLF_PORT", raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:8137"


def test_default_server_explicit_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UD_CHAT_WS_URL", "wss://chat.example/ws")
    monkeypatch.setenv("GOLF_API_PORT", "9999")

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "wss://chat.example/ws"


def test_default_server_falls_back_to_8000(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("UD_CHAT_WS_URL", "GOLF_API_PORT", "API_PORT", "GOLF_PORT"):
        monkeypatch.delenv(var, raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:8000"


def test_default_server_rejects_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLF_API_PORT", "not-a-port")
    monkeypatch.delenv("UD_CHAT_WS_URL", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("GOLF_PORT", raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:8000"


def test_default_server_rejects_out_of_range_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOLF_API_PORT", "0")
    monkeypatch.delenv("UD_CHAT_WS_URL", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("GOLF_PORT", raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:8000"


def test_default_server_falls_through_to_next_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid GOLF_API_PORT should fall through to API_PORT."""
    monkeypatch.setenv("GOLF_API_PORT", "bad")
    monkeypatch.setenv("API_PORT", "9001")
    monkeypatch.delenv("UD_CHAT_WS_URL", raising=False)
    monkeypatch.delenv("GOLF_PORT", raising=False)

    from src.shared.python.chat import chat_dock_widget

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:9001"


def test_chat_dock_widget_resolves_default_server_at_instantiation_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("UD_CHAT_WS_URL", "GOLF_API_PORT", "API_PORT", "GOLF_PORT"):
        monkeypatch.delenv(var, raising=False)

    from src.shared.python.chat import chat_dock_widget
    from src.shared.python.chat._chat_dock_widget_qt import (
        ChatConnectionConfig,
        ChatDockWidget,
    )

    monkeypatch.setenv("GOLF_API_PORT", "9012")
    _ = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))

    assert chat_dock_widget._resolve_default_server() == "ws://127.0.0.1:9012"
    assert dock._server_url == "ws://127.0.0.1:9012"

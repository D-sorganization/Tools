"""Tests for Tools issue #3018 chat dock widget close behavior.

Verifies reconnect behavior on intentional close.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("PyQt6.QtWebSockets")

from typing import cast

from chat._chat_dock_widget_qt import (  # noqa: E402
    ChatConnectionConfig,
    ChatDockWidget,
)
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast(QApplication, app)


def test_close_event_disables_reconnect(qapp: QApplication) -> None:
    # Setup mock/patched dependencies to avoid hitting real websockets or disk path code
    with (
        patch("chat._chat_dock_widget_qt.ChatDockWidget._setup_ui"),
        patch(
            "chat._chat_dock_widget_qt._read_shared_session_id",
            return_value="test-session",
        ),
        patch(
            "chat._chat_dock_widget_qt._session_file_path",
            return_value=Path("dummy_session_file"),
        ),
    ):
        widget = ChatDockWidget(
            connection=ChatConnectionConfig(app_context="test", app_name="test_app")
        )
        widget._status_label = MagicMock()
        widget._send_btn = MagicMock()

        # Initially, the timer should not be active
        assert not widget._reconnect_timer.isActive()
        assert not getattr(widget, "_intentional_disconnect", False)

        # Trigger _on_disconnected directly without closeEvent (unexpected disconnect)
        widget._on_disconnected()

        # The reconnect timer should be started
        assert widget._reconnect_timer.isActive()
        assert not getattr(widget, "_intentional_disconnect", False)
        widget._status_label.setText.assert_called_once()
        status_text = widget._status_label.setText.call_args.args[0]
        assert "Sidekick API unavailable" in status_text
        assert "UD_CHAT_WS_URL" in status_text

        # Stop the timer and reset state
        widget._reconnect_timer.stop()

        # Trigger closeEvent
        from PyQt6.QtGui import QCloseEvent

        mock_event = QCloseEvent()
        widget.closeEvent(mock_event)

        # Check that it set _intentional_disconnect
        assert getattr(widget, "_intentional_disconnect", False)

        # Now trigger _on_disconnected (as socket.close() would)
        widget._on_disconnected()

        # The reconnect timer should NOT be active
        assert not widget._reconnect_timer.isActive()

        # Calling _connect should reset the intentional disconnect flag
        widget._intentional_disconnect = True
        with patch("chat._chat_dock_widget_qt.QWebSocket") as mock_ws_cls:
            mock_ws_cls.return_value = MagicMock()
            widget._connect()
            assert not getattr(widget, "_intentional_disconnect", False)

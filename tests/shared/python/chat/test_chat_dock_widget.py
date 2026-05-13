"""Tests for shared ChatDockWidget widget classes.

Widget tests require PyQt6.QtWidgets (display server) and pytest-qt.
The entire file is skipped on headless CI where libEGL is unavailable.

Session-file helper tests (no display needed) have been moved to
test_chat_session_helpers.py, which uses only PyQt6 (import-level)
and the standard library.
"""

from __future__ import annotations

import pytest

# chat_dock_widget.py imports PyQt6.QtWidgets at module level, which
# requires libEGL.so.1 on Linux. Skip the entire file when unavailable.
pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6.QtWidgets requires display server")
pytest.importorskip("PyQt6.QtWebSockets", reason="PyQt6.QtWebSockets DLL load failed")
pytest.importorskip("pytestqt", reason="pytest-qt required for widget tests")


def _track_widget(qtbot, widget) -> None:
    """Register Qt widgets when pytest-qt accepts the platform wrapper."""
    try:
        qtbot.addWidget(widget)
    except TypeError:
        from PyQt6.QtCore import Qt

        widget.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)


class TestChatMessageBubble:
    """Tests for ChatMessageBubble widget."""

    def test_user_bubble(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("user", "Hello")
        _track_widget(qtbot, bubble)
        assert bubble._role == "user"
        assert bubble._content == "Hello"

    def test_assistant_bubble(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("assistant", "Hi there")
        _track_widget(qtbot, bubble)
        assert bubble._role == "assistant"

    def test_append_content(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("assistant", "")
        _track_widget(qtbot, bubble)
        bubble.append_content("Hello ")
        bubble.append_content("world")
        assert bubble._content == "Hello world"

    def test_set_content(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("user", "old")
        _track_widget(qtbot, bubble)
        bubble.set_content("new")
        assert bubble._content == "new"

    def test_custom_accent_color(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("user", "test", accent_color="#3498db")
        _track_widget(qtbot, bubble)
        assert bubble._role == "user"


class TestChatDockWidget:
    """Tests for ChatDockWidget construction."""

    def test_construction_defaults(self, qtbot):
        from chat.chat_dock_widget import ChatDockWidget

        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app")
        _track_widget(qtbot, widget)
        assert widget._app_context == "unknown"
        assert widget._app_name == "test_app"
        assert widget._server_url == "ws://127.0.0.1:8000"
        widget.close()

    def test_custom_parameters(self, qtbot):
        from chat.chat_dock_widget import ChatDockWidget

        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(
            app_context="gasification",
            app_name="ips",
            server_url="ws://localhost:9000",
            accent_color="#3498db",
            placeholder_text="Ask about gasification...",
        )
        _track_widget(qtbot, widget)
        assert widget._app_context == "gasification"
        assert widget._accent_color == "#3498db"
        widget.close()

    def test_explicit_session_id(self, qtbot):
        from chat.chat_dock_widget import ChatDockWidget

        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app", session_id="explicit-123")
        _track_widget(qtbot, widget)
        assert ChatDockWidget._shared_session_id == "explicit-123"
        widget.close()

    def test_terminal_mode_controls_are_available(self, qtbot, tmp_path):
        from chat.chat_dock_widget import ChatDockWidget

        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app", project_root=tmp_path)
        _track_widget(qtbot, widget)

        widget._mode_combo.setCurrentIndex(1)

        assert widget._current_mode() == "terminal"
        assert widget._shell_combo.currentData() == "powershell"
        assert widget._provider_combo.currentData() == "claude-code"
        assert widget._content_stack.currentWidget() is widget._terminal_output
        widget.close()

    def test_terminal_dropdowns_are_registry_backed(self, qtbot, tmp_path):
        from chat.chat_dock_widget import ChatDockWidget
        from chat.terminal_providers import build_default_terminal_provider_registry

        registry = build_default_terminal_provider_registry()
        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(
            app_name="test_app",
            project_root=tmp_path,
            terminal_registry=registry,
        )
        _track_widget(qtbot, widget)

        shell_ids = [
            widget._shell_combo.itemData(i) for i in range(widget._shell_combo.count())
        ]
        assert shell_ids == [shell.id for shell in registry.shells()]
        assert [
            widget._provider_combo.itemData(i)
            for i in range(widget._provider_combo.count())
        ] == [provider.id for provider in registry.providers_for_shell("powershell")]
        widget.close()

    def test_terminal_start_sends_selected_shell_provider(
        self,
        qtbot,
        tmp_path,
        monkeypatch,
    ):
        from chat.chat_dock_widget import ChatDockWidget

        sent = []
        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app", project_root=tmp_path)
        _track_widget(qtbot, widget)
        monkeypatch.setattr(widget, "_send_ws", sent.append)

        widget._mode_combo.setCurrentIndex(1)
        widget._provider_combo.setCurrentIndex(1)
        widget._on_terminal_start()

        assert sent == [
            {
                "action": "terminal_start",
                "project_root": str(tmp_path.resolve()),
                "shell_id": "powershell",
                "provider_id": "codex",
                "app_context": "unknown",
            }
        ]
        widget.close()

    def test_terminal_lifecycle_buttons_track_session_state(
        self,
        qtbot,
        tmp_path,
        monkeypatch,
    ):
        from chat.chat_dock_widget import ChatDockWidget

        sent = []
        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app", project_root=tmp_path)
        _track_widget(qtbot, widget)
        monkeypatch.setattr(widget, "_send_ws", sent.append)

        widget._mode_combo.setCurrentIndex(1)
        assert widget._terminal_start_btn.isEnabled()
        assert not widget._terminal_stop_btn.isEnabled()

        widget._on_terminal_start()
        widget._on_terminal_start()

        assert len(sent) == 1
        assert not widget._terminal_start_btn.isEnabled()
        assert not widget._terminal_stop_btn.isEnabled()
        assert not widget._shell_combo.isEnabled()
        assert not widget._provider_combo.isEnabled()
        assert "session already active" in widget._terminal_output.toPlainText()

        widget._on_message(
            '{"type":"terminal_session","session":'
            '{"session_id":"terminal_123","state":"running"}}'
        )

        assert not widget._terminal_start_btn.isEnabled()
        assert widget._terminal_stop_btn.isEnabled()
        assert not widget._shell_combo.isEnabled()
        assert not widget._provider_combo.isEnabled()

        widget._on_message(
            '{"type":"terminal_session","session":'
            '{"session_id":"terminal_123","state":"stopped"}}'
        )

        assert widget._terminal_start_btn.isEnabled()
        assert not widget._terminal_stop_btn.isEnabled()
        assert widget._shell_combo.isEnabled()
        assert widget._provider_combo.isEnabled()
        widget.close()

    def test_terminal_input_requires_active_session(self, qtbot, monkeypatch):
        from chat.chat_dock_widget import ChatDockWidget

        sent = []
        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app")
        _track_widget(qtbot, widget)
        monkeypatch.setattr(widget, "_send_ws", sent.append)

        widget._mode_combo.setCurrentIndex(1)
        widget._input_edit.setPlainText("status")
        widget._on_send()

        assert sent == []
        assert "start a session first" in widget._terminal_output.toPlainText()
        widget.close()

    def test_terminal_input_sends_to_active_session(self, qtbot, monkeypatch):
        from chat.chat_dock_widget import ChatDockWidget

        sent = []
        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app")
        _track_widget(qtbot, widget)
        monkeypatch.setattr(widget, "_send_ws", sent.append)

        widget._mode_combo.setCurrentIndex(1)
        widget._terminal_session_id = "terminal_123"
        widget._input_edit.setPlainText("pwd")
        widget._on_send()

        assert sent == [
            {
                "action": "terminal_input",
                "terminal_session_id": "terminal_123",
                "text": "pwd\n",
            }
        ]
        widget.close()

    def test_terminal_stop_sends_active_session(self, qtbot, monkeypatch):
        from chat.chat_dock_widget import ChatDockWidget

        sent = []
        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app")
        _track_widget(qtbot, widget)
        monkeypatch.setattr(widget, "_send_ws", sent.append)

        widget._mode_combo.setCurrentIndex(1)
        widget._terminal_session_id = "terminal_123"
        widget._on_terminal_stop()

        assert sent == [
            {
                "action": "terminal_stop",
                "terminal_session_id": "terminal_123",
            }
        ]
        widget.close()

    def test_close_button_closes_dock(self, qtbot):
        from chat.chat_dock_widget import ChatDockWidget

        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app")
        _track_widget(qtbot, widget)
        widget.show()
        assert widget.isVisible()

        widget._close_btn.click()

        assert not widget.isVisible()

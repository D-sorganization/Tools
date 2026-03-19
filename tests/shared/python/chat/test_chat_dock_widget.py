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
pytest.importorskip("pytestqt", reason="pytest-qt required for widget tests")


class TestChatMessageBubble:
    """Tests for ChatMessageBubble widget."""

    def test_user_bubble(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("user", "Hello")
        qtbot.addWidget(bubble)
        assert bubble._role == "user"
        assert bubble._content == "Hello"

    def test_assistant_bubble(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("assistant", "Hi there")
        qtbot.addWidget(bubble)
        assert bubble._role == "assistant"

    def test_append_content(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("assistant", "")
        qtbot.addWidget(bubble)
        bubble.append_content("Hello ")
        bubble.append_content("world")
        assert bubble._content == "Hello world"

    def test_set_content(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("user", "old")
        qtbot.addWidget(bubble)
        bubble.set_content("new")
        assert bubble._content == "new"

    def test_custom_accent_color(self, qtbot):
        from chat.chat_dock_widget import ChatMessageBubble

        bubble = ChatMessageBubble("user", "test", accent_color="#3498db")
        qtbot.addWidget(bubble)
        assert bubble._role == "user"


class TestChatDockWidget:
    """Tests for ChatDockWidget construction."""

    def test_construction_defaults(self, qtbot):
        from chat.chat_dock_widget import ChatDockWidget

        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app")
        qtbot.addWidget(widget)
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
        qtbot.addWidget(widget)
        assert widget._app_context == "gasification"
        assert widget._accent_color == "#3498db"
        widget.close()

    def test_explicit_session_id(self, qtbot):
        from chat.chat_dock_widget import ChatDockWidget

        ChatDockWidget._shared_session_id = None
        widget = ChatDockWidget(app_name="test_app", session_id="explicit-123")
        qtbot.addWidget(widget)
        assert ChatDockWidget._shared_session_id == "explicit-123"
        widget.close()

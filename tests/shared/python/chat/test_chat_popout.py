"""Tests for the chat dock pop-out window (Tools issue: pop-out blank fix).

These tests exercise three things:
- The pop-out window correctly hosts the chat dock's *inner* content
  widget rather than the ``QDockWidget`` itself (the original bug).
- New Chat and Clear Chat buttons live in the dock header chrome and
  behave correctly: New Chat clears the bubble list AND emits a
  ``new_session`` WS payload; Clear Chat only clears the bubble list and
  preserves the server-side ``session_id``.
- The dock exposes a ``pop_out`` method that produces a working
  :class:`ChatPopoutWindow` whose ``content_widget`` is the dock's inner
  widget (which guarantees the message list / input / combos remain
  visible after popping out).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── path bootstrap (matches test_chat_dock_widget.py) ─────────────────
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", _src_pkg)

for _ns in (
    "src.shared",
    "src.shared.python",
    "src.shared.python.chat",
    "src.shared.python.ai",
    "src.shared.python.ai.gui",
):
    _parts = _ns.split(".")
    _mod = types.ModuleType(_ns)
    _mod.__path__ = [str(ROOT.joinpath(*_parts))]
    sys.modules.setdefault(_ns, _mod)

import logging  # noqa: E402

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QDockWidget,
    QLabel,
    QWidget,
)

from src.shared.python.chat._chat_dock_widget_qt import ChatDockWidget  # noqa: E402
from src.shared.python.chat.chat_popout_window import (  # noqa: E402
    ChatPopoutWindow,
    make_chat_popout_window,
)

# ─── ChatPopoutWindow direct tests ───────────────────────────────────


def test_popout_window_with_plain_widget_uses_it_as_central() -> None:
    """Sanity: passing a plain QWidget still works (back-compat)."""
    _app = QApplication.instance() or QApplication([])
    inner = QLabel("hello")
    popout = ChatPopoutWindow(
        inner,
        session_id="sess-1",
        redock_callback=lambda: None,
    )
    assert popout.centralWidget() is inner
    assert popout.content_widget is inner


def test_popout_window_extracts_inner_widget_from_dock() -> None:
    """Passing a QDockWidget should extract its inner content widget.

    This is the regression: previously the bare QDockWidget was passed to
    setCentralWidget, producing a blank popout.
    """
    _app = QApplication.instance() or QApplication([])
    dock = QDockWidget("Dock")
    inner = QLabel("inner content")
    dock.setWidget(inner)

    callback = MagicMock()
    popout = ChatPopoutWindow(
        dock,
        session_id="sess-2",
        redock_callback=callback,
    )

    # The popout should host the inner widget, NOT the QDockWidget.
    assert popout.centralWidget() is inner
    assert popout.content_widget is inner
    # Dock should have released its inner widget.
    assert dock.widget() is None


def test_popout_redock_returns_widget_to_source_dock() -> None:
    """Redocking should hand the inner widget back to the source dock."""
    _app = QApplication.instance() or QApplication([])
    dock = QDockWidget("Dock")
    inner = QLabel("inner content")
    dock.setWidget(inner)

    callback = MagicMock()
    popout = ChatPopoutWindow(
        dock,
        session_id="sess-3",
        redock_callback=callback,
    )

    popout.redock()
    assert callback.called
    assert dock.widget() is inner


def test_popout_factory_wraps_constructor() -> None:
    """``make_chat_popout_window`` is a thin convenience factory."""
    _app = QApplication.instance() or QApplication([])
    inner = QLabel("hi")
    popout = make_chat_popout_window(
        inner,
        session_id="s",
        redock_callback=lambda: None,
    )
    assert isinstance(popout, ChatPopoutWindow)


# ─── ChatDockWidget.pop_out integration tests ────────────────────────


def test_chat_dock_pop_out_shows_inner_widgets() -> None:
    """Popping out the chat dock should yield a popout whose content
    widget contains the message scroll area and input widget."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")

    popout = dock.pop_out()
    assert popout is not None
    assert isinstance(popout, ChatPopoutWindow)
    # The popout's content widget must be the inner container, not the dock.
    assert popout.content_widget is not dock
    assert isinstance(popout.content_widget, QWidget)
    # Critical: the inner widgets we depend on must still be reachable.
    assert dock._scroll_area.isAncestorOf(dock._message_container)
    assert dock._input_edit is not None


def test_chat_dock_pop_out_preserves_session_id() -> None:
    """The popout window's ``session_id`` should mirror the dock's."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    ChatDockWidget._set_shared_session_id("sess-pop-1")

    popout = dock.pop_out()
    assert popout is not None
    assert popout.session_id == "sess-pop-1"


def test_chat_dock_pop_out_then_redock_restores_inner_widget() -> None:
    """Redock should return the inner widget back into the dock."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")

    popout = dock.pop_out()
    assert popout is not None
    # After pop-out the dock's inner widget is detached.
    assert dock.widget() is None
    popout.redock()
    # After redock the dock owns its inner widget again.
    assert dock.widget() is not None


# ─── New Chat + Clear Chat button tests ──────────────────────────────


def _drain_layout(layout: object) -> int:
    """Count the number of message bubbles in the dock's message layout."""
    return layout.count() - 1  # -1 for the trailing addStretch sentinel


def test_new_chat_button_clears_bubbles_and_sends_new_session() -> None:
    """New Chat must clear the bubble list and send a ``new_session`` WS payload."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    # Seed a couple of bubbles
    dock._add_bubble("user", "hello")
    dock._add_bubble("assistant", "world")
    assert _drain_layout(dock._message_layout) == 2

    with patch.object(dock, "_send_ws") as send_ws:
        dock.new_chat()

    assert _drain_layout(dock._message_layout) == 0
    # Should have sent a new-session request
    assert send_ws.called
    payload = send_ws.call_args.args[0]
    assert payload.get("action") == "new_session"


def test_new_chat_button_clears_queue() -> None:
    """New Chat must reset the busy-state queue too."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    # Force a queued message into existence
    from src.shared.python.chat._qt.queue_panel import QueuedMessage

    dock._queued_messages.append(QueuedMessage(text="queued"))
    with patch.object(dock, "_send_ws"):
        dock.new_chat()
    assert dock._queued_messages == []


def test_new_chat_button_resets_streaming_state() -> None:
    """New Chat must exit the thinking state cleanly."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    dock._enter_thinking_state()
    assert dock._is_streaming is True
    with patch.object(dock, "_send_ws"):
        dock.new_chat()
    assert dock._is_streaming is False


def test_clear_chat_clears_bubbles_without_ws_call() -> None:
    """Clear Chat wipes the visible bubble list but never touches the WS."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    dock._add_bubble("user", "one")
    dock._add_bubble("assistant", "two")

    with patch.object(dock, "_send_ws") as send_ws:
        # Auto-accept the confirmation dialog
        with patch.object(dock, "_confirm_clear_chat", return_value=True):
            dock.clear_chat()

    assert _drain_layout(dock._message_layout) == 0
    send_ws.assert_not_called()


def test_clear_chat_preserves_session_id() -> None:
    """Clear Chat must not mutate the shared session id."""
    _app = QApplication.instance() or QApplication([])
    ChatDockWidget._set_shared_session_id("preserve-this-sid")
    dock = ChatDockWidget(app_context="test")
    dock._add_bubble("user", "x")

    with patch.object(dock, "_confirm_clear_chat", return_value=True):
        dock.clear_chat()
    assert ChatDockWidget._get_shared_session_id() == "preserve-this-sid"


def test_clear_chat_cancelled_keeps_bubbles() -> None:
    """If the user declines the confirmation dialog, the bubbles stay."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    dock._add_bubble("user", "still here")
    assert _drain_layout(dock._message_layout) == 1

    with patch.object(dock, "_confirm_clear_chat", return_value=False):
        dock.clear_chat()
    assert _drain_layout(dock._message_layout) == 1


# ─── Model refresh spinner / placeholder tests ───────────────────────


def test_provider_change_triggers_model_refresh_placeholder() -> None:
    """Changing provider should put the model combo into a loading state."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    refreshed: list[str] = []

    # Patch refresh_ai_model_combo so we can observe it being called.
    from src.shared.python.chat._qt import ai_dropdowns as _ai_mod

    original = _ai_mod.refresh_ai_model_combo

    def spy(d):  # noqa: ANN001
        refreshed.append(d._current_provider)
        return original(d)

    with patch.object(_ai_mod, "refresh_ai_model_combo", side_effect=spy):
        dock._apply_settings_change("provider", "openai")

    assert "openai" in refreshed


def test_model_loading_placeholder_helper_sets_combo() -> None:
    """The loading-placeholder helper installs an italic "Loading..." item."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")

    dock._set_model_combo_loading()
    assert dock._ai_model_combo.count() == 1
    assert "Loading" in dock._ai_model_combo.itemText(0)
    assert dock._ai_model_combo.isEnabled() is False


# ─── Terminal-as-chat send routing tests ─────────────────────────────


def test_send_with_cli_provider_routes_to_terminal_path() -> None:
    """If the AI provider is a CLI/terminal one (e.g. ``claude-code``),
    sending a chat message should NOT use the WS ``send`` action but
    rather start (or use) a terminal subprocess.

    Specifically: when ``_current_provider`` is one of the CLI provider
    IDs, ``_submit_or_queue`` should delegate to a terminal-aware sender
    rather than calling ``_send_ws`` with ``action=send``.
    """
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    dock._current_provider = "claude-code"
    with patch.object(dock, "_send_ws") as send_ws:
        with patch.object(dock, "_send_via_terminal_provider") as send_terminal:
            dock._submit_or_queue("hi from chat")

    # WS send action should NOT fire when a CLI provider is active.
    assert not any(
        call.args
        and isinstance(call.args[0], dict)
        and call.args[0].get("action") == "send"
        for call in send_ws.call_args_list
    )
    send_terminal.assert_called_once()
    args = send_terminal.call_args.args
    assert args[0] == "hi from chat"


def test_send_with_api_provider_uses_ws_send() -> None:
    """Sanity: API providers still take the original WS path."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    dock._current_provider = "openai"
    with patch.object(dock, "_send_ws") as send_ws:
        dock._submit_or_queue("api hello")

    sent = [c.args[0] for c in send_ws.call_args_list]
    assert any(p.get("action") == "send" for p in sent)


def test_is_cli_provider_helper_recognizes_known_ids() -> None:
    """The CLI-provider detection helper must recognise installed providers."""
    _app = QApplication.instance() or QApplication([])
    dock = ChatDockWidget(app_context="test")
    assert dock._is_cli_provider("claude-code") is True
    assert dock._is_cli_provider("codex") is True
    assert dock._is_cli_provider("github-cli") is True
    assert dock._is_cli_provider("openai") is False
    assert dock._is_cli_provider("ollama") is False

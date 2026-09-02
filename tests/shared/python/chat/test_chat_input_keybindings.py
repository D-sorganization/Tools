"""Tests for ChatDockWidget input keybindings + busy-state message queue.

Covers:
    1. Enter alone submits the message.
    2. Shift+Enter inserts a newline.
    3. Sending while the agent is busy queues the message.
    4. Pressing Enter while busy queues steering messages.
    5. Queued messages flush in order on ``complete``.

These tests guard the new ``_submit_or_queue`` DRY pathway used by both the
Send button click and the input widget's Enter keypress.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

# Register src namespace packages so dotted imports resolve correctly
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

import logging

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

from src.shared.python.chat._chat_dock_widget_qt import (  # noqa: E402
    ChatConnectionConfig,
    ChatDockWidget,
)

_APP: QApplication | None = None


def _make_widget() -> ChatDockWidget:
    """Construct a fresh ChatDockWidget with WS stubbed.

    Inlined in each test (rather than wrapped in a fixture) because PyQt6's
    parent-tracking around ``QDockWidget.setWidget`` was reaping the dock's
    inner container the moment a fixture's local scope unwound, leaving
    the ``QPlainTextEdit`` with a deleted C++ side. Constructing in the
    test body keeps the dock alive for the duration of the test.
    """
    global _APP
    _APP = QApplication.instance() or QApplication([])
    w = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))
    w._send_ws = MagicMock()  # type: ignore[method-assign]
    return w


def _press_enter(widget: ChatDockWidget, *, shift: bool = False) -> None:
    """Synthesize a Return keypress on the input widget."""
    mods = (
        Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    )
    ev = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        int(Qt.Key.Key_Return),
        mods,
        "\r",
    )
    widget._input_edit.keyPressEvent(ev)


# ── 1. Enter alone submits ───────────────────────────────────────────


def test_enter_alone_submits_message() -> None:
    widget = _make_widget()
    widget._input_edit.setPlainText("hello world")
    _press_enter(widget)

    # Bubble added + WS payload sent + input cleared.
    assert widget._send_ws.called
    payload = widget._send_ws.call_args.args[0]
    assert payload["action"] == "send"
    assert payload["message"] == "hello world"
    assert widget._input_edit.toPlainText() == ""
    assert widget._is_streaming is True


# ── 2. Shift+Enter inserts a newline ─────────────────────────────────


def test_shift_enter_inserts_newline() -> None:
    widget = _make_widget()
    widget._input_edit.setPlainText("line1")
    # Place cursor at end
    cursor = widget._input_edit.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    widget._input_edit.setTextCursor(cursor)

    _press_enter(widget, shift=True)

    assert "\n" in widget._input_edit.toPlainText()
    assert not widget._send_ws.called  # nothing submitted


# ── 3. Sending while busy queues the message ─────────────────────────


def test_send_while_busy_queues() -> None:
    widget = _make_widget()
    # First message starts streaming.
    widget._input_edit.setPlainText("first")
    _press_enter(widget)
    assert widget._is_streaming is True
    widget._send_ws.reset_mock()

    # Second message enters while busy -> queued, NOT sent.
    widget._input_edit.setPlainText("second")
    _press_enter(widget)

    assert not widget._send_ws.called
    assert widget.queued_messages() == ["second"]
    assert widget._input_edit.toPlainText() == ""


# ── 4. Enter-while-busy queues a steering message ────────────────────


def test_multiple_enters_while_busy_accumulate() -> None:
    widget = _make_widget()
    widget._input_edit.setPlainText("first")
    _press_enter(widget)
    widget._send_ws.reset_mock()

    widget._input_edit.setPlainText("steer-a")
    _press_enter(widget)
    widget._input_edit.setPlainText("steer-b")
    _press_enter(widget)

    assert not widget._send_ws.called
    assert widget.queued_messages() == ["steer-a", "steer-b"]


def test_steer_button_stays_queue_only_while_idle() -> None:
    widget = _make_widget()
    widget._input_edit.setPlainText("steer-only")

    widget._on_steer()

    assert not widget._send_ws.called
    assert widget.queued_messages() == ["steer-only"]
    assert widget._input_edit.toPlainText() == ""


# ── 5. Queue flushes in order on 'complete' ──────────────────────────


def test_complete_flushes_queue_in_order() -> None:
    widget = _make_widget()
    widget._input_edit.setPlainText("first")
    _press_enter(widget)
    widget._send_ws.reset_mock()

    widget._input_edit.setPlainText("steer-a")
    _press_enter(widget)
    widget._input_edit.setPlainText("steer-b")
    _press_enter(widget)

    # Simulate server "complete" arrival.
    widget._on_message(json.dumps({"type": "complete", "session_id": "abc"}))

    # First queued message should be sent (becomes the next user turn).
    assert widget._send_ws.called
    sent_msgs = [call.args[0]["message"] for call in widget._send_ws.call_args_list]
    assert sent_msgs[0] == "steer-a"
    # Now busy again on steer-a; steer-b still queued.
    assert widget.queued_messages() == ["steer-b"]
    assert widget._is_streaming is True

    # Second complete drains steer-b.
    widget._send_ws.reset_mock()
    widget._on_message(json.dumps({"type": "complete", "session_id": "abc"}))
    sent_msgs2 = [call.args[0]["message"] for call in widget._send_ws.call_args_list]
    assert sent_msgs2 == ["steer-b"]
    assert widget.queued_messages() == []

    # Third complete: nothing to flush; remain idle.
    widget._send_ws.reset_mock()
    widget._on_message(json.dumps({"type": "complete", "session_id": "abc"}))
    assert not widget._send_ws.called
    assert widget._is_streaming is False


# ── DbC preconditions on _submit_or_queue ────────────────────────────


def test_submit_or_queue_rejects_empty() -> None:
    widget = _make_widget()
    with pytest.raises(ValueError):
        widget._submit_or_queue("")
    with pytest.raises(ValueError):
        widget._submit_or_queue("   ")


def test_input_state_property() -> None:
    widget = _make_widget()
    assert widget.input_state == "idle"
    widget._input_edit.setPlainText("hi")
    _press_enter(widget)
    assert widget.input_state == "sending"

    widget._input_edit.setPlainText("queued one")
    _press_enter(widget)
    assert widget.input_state == "awaiting"

"""Tests for the inline busy-state queue preview + Send-button state machine.

Covers:
    1. ``QueuePanel.set_messages`` renders one row per QueuedMessage.
    2. ``set_messages([])`` hides the panel.
    3. Clicking a row's Steer button emits ``steer_requested`` with that id.
    4. ``ChatDockWidget.steer_to_front`` moves a queued message by id.
    5. Send-button state transitions: idle → awaiting → stop → idle.
    6. Stop-timer fires after 10 s without a chunk.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication

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
from src.shared.python.chat._qt.queue_panel import (  # noqa: E402
    QueuedMessage,
    QueuePanel,
)

_APP: QApplication | None = None


def _qapp() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _make_widget() -> ChatDockWidget:
    _qapp()
    w = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))
    w._send_ws = MagicMock()  # type: ignore[method-assign]
    return w


# ── QueuePanel direct tests ─────────────────────────────────────────


def test_queue_panel_starts_hidden_with_zero_rows() -> None:
    _qapp()
    p = QueuePanel()
    assert p.row_count == 0
    assert p.isHidden()


def test_queue_panel_set_messages_renders_rows_and_shows() -> None:
    _qapp()
    p = QueuePanel()
    p.show()  # parent normally shows it; we test the inner state directly
    msgs = [QueuedMessage(text="first"), QueuedMessage(text="second")]
    p.set_messages(msgs)
    assert p.row_count == 2
    assert p.isVisible()


def test_queue_panel_set_messages_empty_hides() -> None:
    _qapp()
    p = QueuePanel()
    p.set_messages([QueuedMessage(text="hi")])
    assert p.isVisible()
    p.set_messages([])
    assert p.row_count == 0
    assert p.isHidden()


def test_queue_panel_clear_is_equivalent_to_set_empty() -> None:
    _qapp()
    p = QueuePanel()
    p.set_messages([QueuedMessage(text="hi")])
    p.clear()
    assert p.row_count == 0
    assert p.isHidden()


def test_queue_panel_rejects_non_list() -> None:
    _qapp()
    p = QueuePanel()
    with pytest.raises(ValueError):
        p.set_messages(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        p.set_messages("not-a-list")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        p.set_messages(["bare string"])  # type: ignore[list-item]


def test_queue_panel_steer_signal_emits_message_id() -> None:
    _qapp()
    p = QueuePanel()
    received: list[str] = []
    p.steer_requested.connect(received.append)
    msg = QueuedMessage(text="hello")
    p.set_messages([msg])
    # Find the row's Steer button and click it.
    from PyQt6.QtWidgets import QPushButton

    buttons = p.findChildren(QPushButton, "ChatQueueSteerBtn")
    assert len(buttons) == 1
    buttons[0].click()
    assert received == [msg.id]


# ── ChatDockWidget.steer_to_front + state machine ───────────────────


def test_steer_to_front_moves_message_by_id() -> None:
    w = _make_widget()
    w._is_streaming = True
    w._queued_messages.append(QueuedMessage(text="a"))
    w._queued_messages.append(QueuedMessage(text="b"))
    target_id = w._queued_messages[1].id
    w.steer_to_front(target_id)
    assert w.queued_messages() == ["b", "a"]


def test_steer_to_front_unknown_id_raises() -> None:
    w = _make_widget()
    with pytest.raises(ValueError):
        w.steer_to_front("not-a-real-id")


def test_steer_to_front_no_op_when_already_front() -> None:
    w = _make_widget()
    w._is_streaming = True
    w._queued_messages.append(QueuedMessage(text="a"))
    w._queued_messages.append(QueuedMessage(text="b"))
    target_id = w._queued_messages[0].id
    w.steer_to_front(target_id)  # should not raise, list unchanged
    assert w.queued_messages() == ["a", "b"]


def test_send_button_state_idle_on_construction() -> None:
    w = _make_widget()
    assert w._send_button_state == "idle"
    assert w._send_btn.text() == "Send"


def test_send_button_state_transitions_to_awaiting_when_streaming() -> None:
    w = _make_widget()
    w._enter_thinking_state()
    assert w._send_button_state == "awaiting"
    assert "Steer" in w._send_btn.text()


def test_send_button_state_transitions_back_to_idle_on_complete() -> None:
    w = _make_widget()
    w._enter_thinking_state()
    w._exit_thinking_state()
    assert w._send_button_state == "idle"
    assert w._send_btn.text() == "Send"


def test_send_button_state_promoted_to_stop_after_no_chunk_timeout() -> None:
    w = _make_widget()
    w._enter_thinking_state()
    # Manually fire the timeout to avoid waiting 10s in tests.
    w._on_stop_state_timeout()
    assert w._send_button_state == "stop"
    assert w._send_btn.text() == "Stop"


def test_set_send_button_state_rejects_unknown_state() -> None:
    w = _make_widget()
    with pytest.raises(ValueError):
        w.set_send_button_state("bogus")


# ── Queue panel ↔ dock integration ──────────────────────────────────


def test_dock_queue_panel_mirrors_queue_depth() -> None:
    w = _make_widget()
    w._is_streaming = True
    w._submit_or_queue("alpha")
    w._submit_or_queue("beta")
    assert w._queue_panel.row_count == 2
    # ``isVisible`` requires an ancestor to be shown; ``isHidden`` returns
    # False once we've called ``setVisible(True)`` even without a window.
    assert not w._queue_panel.isHidden()


def test_dock_queue_panel_clears_on_flush_completion() -> None:
    w = _make_widget()
    w._is_streaming = True
    w._submit_or_queue("alpha")
    # Simulate complete arrival: flushes ``alpha`` as a fresh user turn
    # then state returns to idle since no further chunks are arriving.
    w._is_streaming = False
    w._flush_queued_messages()  # pops & re-submits ``alpha``
    # After the re-submit ``_is_streaming`` is True again and the queue
    # is empty, so the panel should be hidden.
    assert w._queued_messages == []
    assert w._queue_panel.row_count == 0

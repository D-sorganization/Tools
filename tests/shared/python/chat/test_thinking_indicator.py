"""Tests for the "AI is thinking" indicator and its dock wiring.

Covers the requirements specified in the chat thinking-indicator task:

1. Indicator hidden at idle.
2. Indicator visible while sending / awaiting (``_is_streaming = True``).
3. Indicator hidden again after a ``complete`` chunk.
4. Indicator hidden again after an ``error`` chunk.
5. Indicator stays visible across a queue flush (multiple turns).
6. Animation timer cleans up on dock close (no leaked ``QTimer``).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
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

import logging  # noqa: E402

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
from src.shared.python.chat._thinking_indicator import (  # noqa: E402
    ThinkingIndicator,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


# ── ThinkingIndicator widget unit tests ─────────────────────────────


def test_indicator_hidden_by_default(qapp: QApplication) -> None:
    indicator = ThinkingIndicator()
    assert indicator.isHidden()
    assert indicator.is_active is False


def test_indicator_start_shows_and_activates(qapp: QApplication) -> None:
    indicator = ThinkingIndicator()
    indicator.start()
    assert indicator.isVisible() or not indicator.isHidden()
    assert indicator.is_active is True
    assert "thinking" in indicator.text().lower()


def test_indicator_stop_hides_and_deactivates(qapp: QApplication) -> None:
    indicator = ThinkingIndicator()
    indicator.start()
    indicator.stop()
    assert indicator.isHidden()
    assert indicator.is_active is False


def test_indicator_start_is_idempotent(qapp: QApplication) -> None:
    """Repeated start() while already active is a no-op (queue flush)."""
    indicator = ThinkingIndicator()
    indicator.start()
    text_after_first = indicator.text()
    indicator.start()  # should not crash, should not toggle off
    assert indicator.is_active is True
    assert "thinking" in indicator.text().lower()
    # text is allowed to differ (a tick may have fired) but state must hold
    assert isinstance(text_after_first, str)


def test_indicator_stop_is_idempotent(qapp: QApplication) -> None:
    indicator = ThinkingIndicator()
    indicator.stop()  # no start() yet
    assert indicator.is_active is False
    indicator.start()
    indicator.stop()
    indicator.stop()  # second stop
    assert indicator.is_active is False


def test_indicator_has_accessible_name(qapp: QApplication) -> None:
    indicator = ThinkingIndicator()
    assert indicator.accessibleName() == "AI is thinking"


def test_indicator_timer_is_child_widget(qapp: QApplication) -> None:
    """Timer must be parented to the widget so Qt reaps it."""
    indicator = ThinkingIndicator()
    indicator.start()
    timer = indicator._timer  # type: ignore[attr-defined]
    assert timer.parent() is indicator
    assert timer.isActive() is True
    indicator.stop()
    assert timer.isActive() is False


def test_indicator_tick_cycles_frames(qapp: QApplication) -> None:
    indicator = ThinkingIndicator()
    indicator.start()
    frames = set()
    for _ in range(6):
        indicator._on_tick()  # type: ignore[attr-defined]
        frames.add(indicator.text())
    indicator.stop()
    # Should have seen at least 2 distinct frames over 6 ticks
    assert len(frames) >= 2


# ── ChatDockWidget integration tests ─────────────────────────────────


def _make_dock() -> ChatDockWidget:
    return ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))


def test_dock_exposes_thinking_indicator(qapp: QApplication) -> None:
    dock = _make_dock()
    assert hasattr(dock, "thinking_indicator")
    assert isinstance(dock.thinking_indicator, ThinkingIndicator)


def test_dock_indicator_hidden_at_idle(qapp: QApplication) -> None:
    dock = _make_dock()
    assert dock.thinking_indicator.is_active is False
    assert dock.input_state == "idle"


def test_dock_input_state_property_values(qapp: QApplication) -> None:
    """input_state transitions: idle → sending/awaiting → idle."""
    dock = _make_dock()
    assert dock.input_state == "idle"
    dock._is_streaming = True
    assert dock.input_state in {"sending", "awaiting"}
    dock._is_streaming = False
    assert dock.input_state == "idle"


def test_dock_indicator_shows_on_send(qapp: QApplication) -> None:
    dock = _make_dock()
    # Simulate the send path setting streaming + notifying indicator
    dock._enter_thinking_state()
    assert dock.thinking_indicator.is_active is True
    assert dock.input_state in {"sending", "awaiting"}


def test_dock_indicator_hides_on_complete_chunk(qapp: QApplication) -> None:
    dock = _make_dock()
    dock._enter_thinking_state()
    assert dock.thinking_indicator.is_active is True
    # Feed a complete chunk
    dock._on_message(json.dumps({"type": "complete"}))
    assert dock.thinking_indicator.is_active is False
    assert dock.input_state == "idle"


def test_dock_indicator_hides_on_error_chunk(qapp: QApplication) -> None:
    dock = _make_dock()
    dock._enter_thinking_state()
    assert dock.thinking_indicator.is_active is True
    dock._on_message(json.dumps({"type": "error", "detail": "boom"}))
    assert dock.thinking_indicator.is_active is False
    assert dock.input_state == "idle"


def test_dock_indicator_hides_on_disconnect(qapp: QApplication) -> None:
    dock = _make_dock()
    dock._enter_thinking_state()
    assert dock.thinking_indicator.is_active is True
    dock._on_disconnected()
    assert dock.thinking_indicator.is_active is False


def test_dock_indicator_stays_visible_across_queue_flush(
    qapp: QApplication,
) -> None:
    """A queued message starting its own turn keeps the indicator on."""
    dock = _make_dock()
    dock._enter_thinking_state()
    assert dock.thinking_indicator.is_active is True
    # Simulate complete of first turn, but queue not empty — dock flushes
    # the next queued turn, which calls _enter_thinking_state again.
    dock._on_message(json.dumps({"type": "complete"}))
    # First turn done; if the queue had something, dock starts next turn:
    dock._enter_thinking_state()
    assert dock.thinking_indicator.is_active is True
    # Now the second turn completes
    dock._on_message(json.dumps({"type": "complete"}))
    assert dock.thinking_indicator.is_active is False


def test_dock_indicator_timer_stops_on_close(qapp: QApplication) -> None:
    dock = _make_dock()
    dock._enter_thinking_state()
    timer = dock.thinking_indicator._timer  # type: ignore[attr-defined]
    assert timer.isActive() is True
    dock.close()
    assert timer.isActive() is False

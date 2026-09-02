"""Tests for streaming chunk batching in ``ChatDockWidget``.

Covers:
    1. A single chunk is buffered, not rendered immediately.
    2. Multiple chunks coalesce into one ``append_content`` call.
    3. ``complete`` drains any pending buffer before tearing down.
    4. The flush timer interval is the documented 50 ms.
    5. ``_flush_chunk_buffer`` is idempotent on an empty buffer.

These guard the perf-critical hot path that turns per-network-frame Qt
repaints into a coalesced 20 Hz repaint.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

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

_APP: QApplication | None = None


def _make_widget() -> ChatDockWidget:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    w = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))
    w._send_ws = MagicMock()  # type: ignore[method-assign]
    w._current_bubble = MagicMock()
    return w


def test_chunk_is_buffered_not_immediately_rendered() -> None:
    w = _make_widget()
    w._on_message(json.dumps({"type": "chunk", "content": "hi"}))
    # The bubble has NOT been updated synchronously.
    w._current_bubble.append_content.assert_not_called()
    assert w._chunk_buffer == ["hi"]
    assert w._chunk_flush_timer.isActive()


def test_multiple_chunks_coalesce_into_one_render() -> None:
    w = _make_widget()
    for token in ("he", "ll", "o ", "wo", "rld"):
        w._on_message(json.dumps({"type": "chunk", "content": token}))
    # Manually flush instead of waiting for the timer.
    w._flush_chunk_buffer()
    w._current_bubble.append_content.assert_called_once_with("hello world")


def test_complete_flushes_pending_buffer() -> None:
    w = _make_widget()
    bubble_mock = w._current_bubble
    w._on_message(json.dumps({"type": "chunk", "content": "trailing"}))
    w._on_message(json.dumps({"type": "complete"}))
    # complete flushes the buffer (one ``append_content`` call), THEN
    # clears ``_current_bubble`` to ``None``.
    bubble_mock.append_content.assert_called_once_with("trailing")
    assert w._current_bubble is None


def test_flush_chunk_buffer_is_idempotent_on_empty() -> None:
    w = _make_widget()
    # No chunk arrived yet.
    w._flush_chunk_buffer()  # must not raise
    w._current_bubble.append_content.assert_not_called()


def test_flush_timer_interval_is_50ms() -> None:
    w = _make_widget()
    assert w._chunk_flush_timer.interval() == 50


def test_chunk_buffer_drops_when_no_current_bubble() -> None:
    w = _make_widget()
    w._current_bubble = None
    w._on_message(json.dumps({"type": "chunk", "content": "ignored"}))
    # No bubble means the chunk is buffered but nowhere to land — flush
    # is a safe drop, not a crash.
    w._flush_chunk_buffer()

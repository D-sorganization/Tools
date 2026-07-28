"""Tests for the AutoCompleteLineEdit widget."""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent
except (ImportError, OSError) as exc:
    pytest.skip(
        f"PyQt6 autocomplete dependencies not loadable: {exc}",
        allow_module_level=True,
    )

from src.shared.python.ui.auto_complete import AutoCompleteLineEdit


def _make_widget(qtbot: Any, words: list[str] | None = None) -> AutoCompleteLineEdit:
    """Create an autocomplete widget owned by pytest-qt."""
    widget = AutoCompleteLineEdit(words=words)
    qtbot.addWidget(widget)
    return widget


def test_auto_complete_line_edit_initialization(qtbot: Any) -> None:
    """Test that AutoCompleteLineEdit initializes correctly with words."""
    words = ["gravity", "velocity", "acceleration"]
    widget = _make_widget(qtbot, words)
    assert widget.completer_words == words
    assert widget.completer() is not None


def test_auto_complete_set_completion_words(qtbot: Any) -> None:
    """Test dynamically updating the completion words."""
    widget = _make_widget(qtbot)
    assert widget.completer_words == []

    widget.set_completion_words(["mass", "force"])
    assert widget.completer_words == ["mass", "force"]


def test_auto_complete_add_completion_words(qtbot: Any) -> None:
    """Test adding words to the completion dictionary."""
    widget = _make_widget(qtbot, ["gravity"])
    widget.add_completion_words(["mass"])

    assert "gravity" in widget.completer_words
    assert "mass" in widget.completer_words
    assert len(widget.completer_words) == 2


def test_auto_complete_tab_key(qtbot: Any) -> None:
    """Test that the Tab key accepts the current completion."""
    widget = _make_widget(qtbot, ["acceleration"])
    widget.setText("acc")

    # Simulate completer state
    widget.auto_completer.setCompletionPrefix("acc")
    assert widget.auto_completer.currentCompletion() == "acceleration"

    # Send Tab key event
    event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.keyPressEvent(event)

    # Check that text was updated
    assert widget.text() == "acceleration"

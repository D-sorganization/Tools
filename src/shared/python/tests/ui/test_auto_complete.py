"""Tests for the AutoCompleteLineEdit widget."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent

from src.shared.python.ui.auto_complete import AutoCompleteLineEdit


def test_auto_complete_line_edit_initialization(qtbot: Any) -> None:
    """Test that AutoCompleteLineEdit initializes correctly with words."""
    words = ["gravity", "velocity", "acceleration"]
    widget = AutoCompleteLineEdit(words=words)
    qtbot.addWidget(widget)
    assert widget.completer_words == words
    assert widget.completer() is not None


def test_auto_complete_set_completion_words(qtbot: Any) -> None:
    """Test dynamically updating the completion words."""
    widget = AutoCompleteLineEdit()
    qtbot.addWidget(widget)
    assert widget.completer_words == []

    widget.set_completion_words(["mass", "force"])
    assert widget.completer_words == ["mass", "force"]


def test_auto_complete_add_completion_words(qtbot: Any) -> None:
    """Test adding words to the completion dictionary."""
    widget = AutoCompleteLineEdit(words=["gravity"])
    qtbot.addWidget(widget)
    widget.add_completion_words(["mass"])

    assert "gravity" in widget.completer_words
    assert "mass" in widget.completer_words
    assert len(widget.completer_words) == 2


def test_auto_complete_tab_key(qtbot: Any) -> None:
    """Test that the Tab key accepts the current completion."""
    widget = AutoCompleteLineEdit(words=["acceleration"])
    qtbot.addWidget(widget)
    widget.setText("acc")

    # Simulate completer state
    widget.auto_completer.setCompletionPrefix("acc")
    assert widget.auto_completer.currentCompletion() == "acceleration"

    # Send Tab key event
    event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
        "\t",
    )
    widget.keyPressEvent(event)

    # Check that text was updated
    assert widget.text() == "acceleration"

"""Tests for shared NotesDockWidget.

This test file is skipped when PyQt6 widgets are unavailable in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6.QtWidgets requires display server")
if os.environ.get("RUN_QT_TESTS") != "1":
    pytest.skip("Set RUN_QT_TESTS=1 to run PyQt widget tests", allow_module_level=True)

from notes.notes_dock_widget import NotesDockWidget  # noqa: E402


class TestNotesDockWidget:
    def test_construction_defaults(self, qtbot, tmp_path: Path):
        widget = NotesDockWidget(project_dir=tmp_path)
        qtbot.addWidget(widget)
        assert widget.windowTitle() == "Notes"

    def test_save_persists_text(self, qtbot, tmp_path: Path):
        widget = NotesDockWidget(project_dir=tmp_path)
        qtbot.addWidget(widget)

        widget._text_edit.setPlainText("memo")
        widget.save_notes()

        assert widget.storage.load_text() == "memo"

    def test_clear_button_clears_editor(self, qtbot, tmp_path: Path):
        widget = NotesDockWidget(project_dir=tmp_path)
        qtbot.addWidget(widget)

        widget._text_edit.setPlainText("erase me")
        widget.clear_editor()

        assert widget._text_edit.toPlainText() == ""

    def test_delete_and_restore_cycle(self, qtbot, tmp_path: Path):
        widget = NotesDockWidget(project_dir=tmp_path)
        qtbot.addWidget(widget)

        widget._text_edit.setPlainText("recoverable")
        widget.save_notes()
        widget.delete_to_recycle_bin()
        assert widget.storage.notes_path.exists() is False

        restored = widget.restore_latest_deleted()
        assert restored is True
        assert widget._text_edit.toPlainText() == "recoverable"

    def test_popout_and_embed(self, qtbot, tmp_path: Path):
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QMainWindow

        host = QMainWindow()
        qtbot.addWidget(host)

        widget = NotesDockWidget(project_dir=tmp_path, parent=host)
        qtbot.addWidget(widget)
        host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, widget)

        widget.pop_out()
        assert widget.isFloating() is True

        widget.embed_in(host, Qt.DockWidgetArea.LeftDockWidgetArea)
        assert widget.isFloating() is False

    def test_restore_latest_deleted_none(self, qtbot, tmp_path: Path):
        widget = NotesDockWidget(project_dir=tmp_path)
        qtbot.addWidget(widget)
        assert widget.restore_latest_deleted() is False

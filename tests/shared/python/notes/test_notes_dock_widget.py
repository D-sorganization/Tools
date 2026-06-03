from pathlib import Path
from typing import Any

import pytest
from notes.notes_dock_widget import NotesDockWidget
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QPlainTextEdit


def _editor(widget: NotesDockWidget) -> QPlainTextEdit:
    return widget._require_editor()


def test_notes_dock_requires_project_dir() -> None:
    with pytest.raises(ValueError, match="project_dir must be provided"):
        NotesDockWidget(project_dir=None)  # type: ignore[arg-type]


def test_notes_dock_save_reload_and_clear(
    tmp_path: Path,
    qtbot: Any,
) -> None:
    widget = NotesDockWidget(tmp_path)
    qtbot.addWidget(widget)

    _editor(widget).setPlainText("field note")
    widget.save_notes()

    assert widget.storage.load_text() == "field note"
    assert widget._require_status_label().text() == "Saved"

    _editor(widget).setPlainText("")
    widget.reload_from_disk()
    assert _editor(widget).toPlainText() == "field note"

    widget.clear_editor()
    assert _editor(widget).toPlainText() == ""
    assert widget.storage.load_text() == ""
    assert widget._require_status_label().text() == "Cleared"


def test_notes_dock_delete_and_restore_latest(
    tmp_path: Path,
    qtbot: Any,
) -> None:
    widget = NotesDockWidget(tmp_path)
    qtbot.addWidget(widget)

    _editor(widget).setPlainText("recoverable")

    assert widget.delete_to_recycle_bin() is True
    assert _editor(widget).toPlainText() == ""
    assert widget._require_status_label().text() == "Moved to recycle bin"
    assert not widget.storage.notes_path.exists()

    assert widget.restore_latest_deleted() is True
    assert _editor(widget).toPlainText() == "recoverable"
    assert widget._require_status_label().text() == "Restored"


def test_notes_dock_restore_handles_empty_and_missing_recycle_item(
    tmp_path: Path,
    qtbot: Any,
) -> None:
    widget = NotesDockWidget(tmp_path)
    qtbot.addWidget(widget)

    assert widget.restore_latest_deleted() is False
    assert widget._require_status_label().text() == "Recycle bin empty"

    _editor(widget).setPlainText("lost source")
    assert widget.delete_to_recycle_bin() is True
    item_id = widget.storage.latest_recycled_id()
    assert item_id is not None
    for item in widget.storage.list_recycled():
        Path(item.path).unlink()

    assert widget.restore_latest_deleted() is False
    assert widget._require_status_label().text() == "Restore failed"


def test_notes_dock_pop_out_and_embed(
    tmp_path: Path,
    qtbot: Any,
) -> None:
    widget = NotesDockWidget(tmp_path)
    main_window = QMainWindow()
    qtbot.addWidget(widget)
    qtbot.addWidget(main_window)

    widget.pop_out()
    assert widget.isFloating() is True

    widget.embed_in(main_window, Qt.DockWidgetArea.LeftDockWidgetArea)
    assert widget.isFloating() is False

    with pytest.raises(ValueError, match="main_window must support addDockWidget"):
        widget.embed_in(object(), Qt.DockWidgetArea.RightDockWidgetArea)  # type: ignore[arg-type]


def test_notes_dock_private_accessors_fail_before_ui_initialization(
    tmp_path: Path,
    qtbot: Any,
) -> None:
    widget = NotesDockWidget(tmp_path)
    qtbot.addWidget(widget)

    widget._text_edit = None
    with pytest.raises(RuntimeError, match="notes editor not initialized"):
        widget._require_editor()

    widget._status_label = None
    with pytest.raises(RuntimeError, match="status label not initialized"):
        widget._require_status_label()

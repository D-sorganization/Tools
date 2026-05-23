from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from notes.integration import attach_notes_dock
from notes.notes_dock_widget import NotesDockWidget
from notes.storage import NotesStorage
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow, QWidget


def test_notes_storage_crud_and_recycle(tmp_path: Path) -> None:
    # 1. Constructor preconditions
    with pytest.raises(ValueError, match="project_dir must exist and be a directory"):
        NotesStorage(tmp_path / "nonexistent_dir")

    with pytest.raises(ValueError, match="notes_filename cannot be empty"):
        NotesStorage(tmp_path, notes_filename="")

    storage = NotesStorage(tmp_path)

    # 2. load_text and save_text
    assert storage.load_text() == ""

    with pytest.raises(ValueError, match="text cannot be None"):
        storage.save_text(None)

    storage.save_text("Hello, notes!")
    assert storage.load_text() == "Hello, notes!"

    # 3. clear
    storage.clear()
    assert storage.load_text() == ""

    # Unlink the file to test FileNotFoundError
    storage.notes_path.unlink(missing_ok=True)

    # 4. move_to_recycle preconditions and execution
    with pytest.raises(FileNotFoundError, match="notes file does not exist"):
        storage.move_to_recycle()

    storage.save_text("Note to recycle")
    assert storage.notes_path.exists()

    item = storage.move_to_recycle(reason="test_delete")
    assert not storage.notes_path.exists()
    assert item.reason == "test_delete"
    assert item.deleted_at is not None
    assert item.item_id is not None

    # 5. list_recycled & latest_recycled_id
    recycled_items = storage.list_recycled()
    assert len(recycled_items) == 1
    assert recycled_items[0].item_id == item.item_id
    assert storage.latest_recycled_id() == item.item_id

    # 6. restore preconditions & execution
    with pytest.raises(ValueError, match="item_id must be provided"):
        storage.restore(None)

    # restore nonexistent
    assert storage.restore("nonexistent_id") is None

    # restore valid
    restored_path = storage.restore(item.item_id)
    assert restored_path == storage.notes_path
    assert storage.notes_path.exists()
    assert storage.load_text() == "Note to recycle"
    assert len(storage.list_recycled()) == 0

    # 7. purge preconditions & execution
    # Recycle again
    item = storage.move_to_recycle()
    assert len(storage.list_recycled()) == 1

    with pytest.raises(ValueError, match="item_id must be provided"):
        storage.purge(None)

    assert not storage.purge("nonexistent_id")

    # purge valid
    assert storage.purge(item.item_id)
    assert len(storage.list_recycled()) == 0
    assert not Path(item.path).exists()

    # 8. _write_index preconditions
    with pytest.raises(ValueError, match="items must be provided"):
        storage._write_index(None)


@pytest.mark.gui
def test_attach_notes_dock(tmp_path: Path, qtbot: Any) -> None:
    win = QMainWindow()
    qtbot.addWidget(win)

    dock = attach_notes_dock(win, tmp_path, title="My Custom Notes")
    qtbot.addWidget(dock)
    assert dock is not None
    assert dock.windowTitle() == "My Custom Notes"

    with pytest.raises(ValueError, match="main_window must support addDockWidget"):
        w = QWidget()
        attach_notes_dock(w, tmp_path)


@pytest.mark.gui
def test_notes_dock_widget_interactions(tmp_path: Path, qtbot: Any) -> None:
    win = QMainWindow()
    qtbot.addWidget(win)

    # 1. Constructor checks
    with pytest.raises(ValueError, match="project_dir must be provided"):
        NotesDockWidget(None)

    dock = NotesDockWidget(project_dir=tmp_path, parent=win)
    qtbot.addWidget(dock)

    # Check default UI states
    assert dock.windowTitle() == "Notes"
    assert dock._text_edit is not None
    assert dock._text_edit.toPlainText() == ""
    assert dock._status_label is not None
    assert dock._status_label.text() == "Ready"

    # 2. Write, save, and reload
    dock._text_edit.setPlainText("Workspace notes")
    dock.save_notes()
    assert dock._status_label.text() == "Saved"
    assert dock.storage.load_text() == "Workspace notes"

    dock._text_edit.setPlainText("")
    dock.reload_from_disk()
    assert dock._text_edit.toPlainText() == "Workspace notes"

    # 3. Clear editor
    dock.clear_editor()
    assert dock._text_edit.toPlainText() == ""
    assert dock._status_label.text() == "Cleared"
    assert dock.storage.load_text() == ""

    # 4. Move to recycle bin (failure when file absent / error raised)
    from unittest.mock import patch

    with patch.object(
        dock.storage,
        "move_to_recycle",
        side_effect=FileNotFoundError("notes file does not exist"),
    ):
        assert not dock.delete_to_recycle_bin()
        assert dock._status_label.text() == "No notes file to delete"

    # Move to recycle bin (success when file present)
    dock._text_edit.setPlainText("Trash notes")
    dock.save_notes()
    assert dock.delete_to_recycle_bin()
    assert dock._text_edit.toPlainText() == ""
    assert dock._status_label.text() == "Moved to recycle bin"
    assert dock.storage.latest_recycled_id() is not None

    # 5. Restore latest deleted
    assert dock.restore_latest_deleted()
    assert dock._text_edit.toPlainText() == "Trash notes"
    assert dock._status_label.text() == "Restored"

    # Restore when recycle bin is empty
    dock.storage.move_to_recycle()
    dock.storage.purge(dock.storage.latest_recycled_id())
    assert not dock.restore_latest_deleted()
    assert dock._status_label.text() == "Recycle bin empty"

    # 6. Pop out and embed in
    dock.pop_out()
    assert dock.isFloating()

    dock.embed_in(win, Qt.DockWidgetArea.LeftDockWidgetArea)
    assert not dock.isFloating()

    with pytest.raises(ValueError, match="main_window must support addDockWidget"):
        w = QWidget()
        dock.embed_in(w, Qt.DockWidgetArea.LeftDockWidgetArea)

    # 7. Uninitialized widgets check (defensive)
    dock._text_edit = None
    with pytest.raises(RuntimeError, match="notes editor not initialized"):
        dock._require_editor()

    dock._status_label = None
    with pytest.raises(RuntimeError, match="status label not initialized"):
        dock._require_status_label()

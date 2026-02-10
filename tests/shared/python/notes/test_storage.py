from __future__ import annotations

from pathlib import Path

import pytest
from notes.storage import NotesStorage


class TestNotesStorage:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("alpha\nbeta")
        assert storage.notes_path.exists()
        assert storage.load_text() == "alpha\nbeta"

    def test_clear_keeps_notes_file(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("content")
        storage.clear()
        assert storage.notes_path.exists()
        assert storage.load_text() == ""

    def test_safe_delete_moves_notes_to_recycle_bin(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("to delete")

        recycled = storage.move_to_recycle(reason="user_delete")

        assert not storage.notes_path.exists()
        assert recycled.reason == "user_delete"
        assert Path(recycled.path).exists()

    def test_restore_recycled_note(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("recover me")
        recycled = storage.move_to_recycle(reason="restore_test")

        restored_path = storage.restore(recycled.item_id)

        assert restored_path == storage.notes_path
        assert storage.load_text() == "recover me"

    def test_delete_is_reversible_until_purged(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("keep copy")
        recycled = storage.move_to_recycle(reason="manual")

        assert storage.purge(recycled.item_id) is True
        assert storage.restore(recycled.item_id) is None

    def test_requires_existing_project_directory(self, tmp_path: Path):
        with pytest.raises(ValueError):
            NotesStorage(project_dir=tmp_path / "missing")

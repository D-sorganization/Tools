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

    def test_init_raises_on_empty_filename(self, tmp_path: Path):
        with pytest.raises(ValueError, match="notes_filename cannot be empty"):
            NotesStorage(project_dir=tmp_path, notes_filename="   ")

    def test_load_text_returns_empty_when_absent(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        assert storage.load_text() == ""

    def test_save_text_raises_on_none(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        with pytest.raises(ValueError, match="text cannot be None"):
            storage.save_text(None)

    def test_move_to_recycle_raises_when_file_absent(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="notes file does not exist"):
            storage.move_to_recycle()

    def test_restore_raises_on_none_id(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        with pytest.raises(ValueError, match="item_id must be provided"):
            storage.restore(None)

    def test_restore_returns_none_when_item_not_found(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        assert storage.restore("missing_id") is None

    def test_restore_returns_none_when_file_absent(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("content")
        item = storage.move_to_recycle()
        Path(item.path).unlink()
        assert storage.restore(item.item_id) is None

    def test_purge_raises_on_none_id(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        with pytest.raises(ValueError, match="item_id must be provided"):
            storage.purge(None)

    def test_purge_returns_false_when_item_not_found(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        assert storage.purge("missing_id") is False

    def test_purge_removes_file_if_exists(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("content")
        item = storage.move_to_recycle()
        assert storage.purge(item.item_id) is True
        assert not Path(item.path).exists()
        assert storage._find_item(item.item_id) is None

    def test_list_recycled_returns_sorted(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)

        storage.save_text("first")
        item1 = storage.move_to_recycle()

        storage.save_text("second")
        item2 = storage.move_to_recycle()

        items = storage.list_recycled()
        assert len(items) == 2
        # sorted by deleted_at desc
        assert items[0].item_id == item2.item_id
        assert items[1].item_id == item1.item_id

    def test_latest_recycled_id_returns_none_when_empty(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        assert storage.latest_recycled_id() is None

    def test_latest_recycled_id_returns_newest(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        storage.save_text("content")
        item = storage.move_to_recycle()
        assert storage.latest_recycled_id() == item.item_id

    def test_write_index_raises_on_none(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        with pytest.raises(ValueError, match="items must be provided"):
            storage._write_index(None)

    def test_append_index_raises_on_none(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        with pytest.raises(ValueError, match="item must be provided"):
            storage._append_index(None)

    def test_find_item_raises_on_empty(self, tmp_path: Path):
        storage = NotesStorage(project_dir=tmp_path)
        with pytest.raises(ValueError, match="item_id cannot be empty"):
            storage._find_item("   ")

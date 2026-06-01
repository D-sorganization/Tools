import json
from pathlib import Path

import pytest
from notes.models import RecycledNoteItem
from notes.storage import NotesStorage


def test_storage_requires_existing_project_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project_dir must exist"):
        NotesStorage(tmp_path / "missing")


def test_storage_rejects_blank_notes_filename(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="notes_filename cannot be empty"):
        NotesStorage(tmp_path, notes_filename="  ")


def test_load_save_and_clear_notes_text(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)

    assert storage.load_text() == ""
    assert storage.save_text("alpha\nbeta") == tmp_path / "project.notes.txt"
    assert storage.load_text() == "alpha\nbeta"
    assert storage.clear() == tmp_path / "project.notes.txt"
    assert storage.load_text() == ""


def test_save_text_rejects_none(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)

    with pytest.raises(ValueError, match="text cannot be None"):
        storage.save_text(None)  # type: ignore[arg-type]


def test_move_to_recycle_tracks_item_and_latest_id(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)
    storage.save_text("important notes")

    item = storage.move_to_recycle(reason="cleanup")

    assert not storage.notes_path.exists()
    assert Path(item.path).read_text(encoding="utf-8") == "important notes"
    assert item.reason == "cleanup"
    assert item.original_path == str(tmp_path / "project.notes.txt")
    assert storage.latest_recycled_id() == item.item_id
    assert storage.list_recycled() == [item]


def test_move_to_recycle_requires_existing_notes_file(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)

    with pytest.raises(FileNotFoundError, match="notes file does not exist"):
        storage.move_to_recycle()


def test_restore_moves_recycled_item_back_and_removes_index_entry(
    tmp_path: Path,
) -> None:
    storage = NotesStorage(tmp_path)
    storage.save_text("restore me")
    item = storage.move_to_recycle()

    restored_path = storage.restore(item.item_id)

    assert restored_path == tmp_path / "project.notes.txt"
    assert storage.load_text() == "restore me"
    assert storage.list_recycled() == []
    assert storage.restore(item.item_id) is None


def test_restore_returns_none_when_index_or_file_is_missing(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)
    storage.save_text("lost")
    item = storage.move_to_recycle()
    Path(item.path).unlink()

    assert storage.restore("unknown") is None
    assert storage.restore(item.item_id) is None


def test_restore_rejects_missing_or_blank_item_id(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)

    with pytest.raises(ValueError, match="item_id must be provided"):
        storage.restore(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="item_id cannot be empty"):
        storage.restore(" ")


def test_purge_deletes_recycled_item_and_index_entry(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)
    storage.save_text("delete forever")
    item = storage.move_to_recycle()

    assert storage.purge(item.item_id) is True
    assert not Path(item.path).exists()
    assert storage.list_recycled() == []
    assert storage.purge(item.item_id) is False


def test_purge_succeeds_when_recycled_file_already_missing(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)
    storage.save_text("delete forever")
    item = storage.move_to_recycle()
    Path(item.path).unlink()

    assert storage.purge(item.item_id) is True
    assert storage.list_recycled() == []


def test_purge_rejects_missing_or_blank_item_id(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)

    with pytest.raises(ValueError, match="item_id must be provided"):
        storage.purge(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="item_id cannot be empty"):
        storage.purge(" ")


def test_list_recycled_sorts_newest_first(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)
    older = RecycledNoteItem(
        item_id="old",
        reason="manual",
        path=str(tmp_path / "old.txt"),
        original_path=str(storage.notes_path),
        deleted_at="20250101T000000Z",
    )
    newer = RecycledNoteItem(
        item_id="new",
        reason="manual",
        path=str(tmp_path / "new.txt"),
        original_path=str(storage.notes_path),
        deleted_at="20260101T000000Z",
    )
    storage._write_index([older, newer])

    assert [item.item_id for item in storage.list_recycled()] == ["new", "old"]


def test_latest_recycled_id_returns_none_when_empty(tmp_path: Path) -> None:
    assert NotesStorage(tmp_path).latest_recycled_id() is None


def test_write_index_rejects_none(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)

    with pytest.raises(ValueError, match="items must be provided"):
        storage._write_index(None)  # type: ignore[arg-type]


def test_append_index_rejects_none(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)

    with pytest.raises(ValueError, match="item must be provided"):
        storage._append_index(None)  # type: ignore[arg-type]


def test_read_index_rehydrates_items_from_json(tmp_path: Path) -> None:
    storage = NotesStorage(tmp_path)
    storage.recycle_bin_dir.mkdir()
    storage.recycle_index_path.write_text(
        json.dumps(
            [
                {
                    "item_id": "abc",
                    "reason": "manual",
                    "path": "recycle/abc.txt",
                    "original_path": "project.notes.txt",
                    "deleted_at": "20260101T000000Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert storage._read_index() == [
        RecycledNoteItem(
            item_id="abc",
            reason="manual",
            path="recycle/abc.txt",
            original_path="project.notes.txt",
            deleted_at="20260101T000000Z",
        )
    ]

from __future__ import annotations

from pathlib import Path

import pytest
from notes.card_store import NoteCardStore
from notes.models import NoteCard, NotesBoardSettings


def test_markdown_note_card_roundtrip_preserves_metadata_and_body(
    tmp_path: Path,
) -> None:
    store = NoteCardStore(project_dir=tmp_path)
    card = store.create_note(
        title="Pump notes",
        markdown_body="# Pump\n\n- inspect seal",
        color="#FFE8A3",
        tags=("maintenance", "field"),
    )

    loaded = store.load_note(card.note_id)

    assert loaded == NoteCard(
        note_id=card.note_id,
        title="Pump notes",
        markdown_body="# Pump\n\n- inspect seal",
        color="#ffe8a3",
        created_at=card.created_at,
        updated_at=card.updated_at,
        tags=("maintenance", "field"),
    )
    note_path = store.notes_dir / f"{card.note_id}.md"
    assert note_path.read_text(encoding="utf-8").endswith("# Pump\n\n- inspect seal")


def test_note_color_and_board_background_persist(tmp_path: Path) -> None:
    store = NoteCardStore(project_dir=tmp_path)
    store.save_settings(NotesBoardSettings(background_color="#EEE4FF"))
    card = store.create_note("Color", "body", color="#C7F9CC")

    reloaded = NoteCardStore(project_dir=tmp_path)

    assert reloaded.load_settings().background_color == "#eee4ff"
    assert reloaded.load_note(card.note_id).color == "#c7f9cc"


def test_invalid_note_and_board_colors_are_rejected(tmp_path: Path) -> None:
    store = NoteCardStore(project_dir=tmp_path)

    with pytest.raises(ValueError, match="color"):
        store.create_note("Bad", "body", color="red")

    with pytest.raises(ValueError, match="background_color"):
        store.save_settings(NotesBoardSettings(background_color="red"))


def test_delete_moves_note_to_recycle_and_restore_returns_card(tmp_path: Path) -> None:
    store = NoteCardStore(project_dir=tmp_path)
    card = store.create_note("Recover", "safe body", color="#fff7cc")

    recycled = store.delete_note(card.note_id, reason="user_delete")

    assert store.load_note(card.note_id) is None
    assert Path(recycled.path).exists()
    assert recycled.reason == "user_delete"

    restored = store.restore_note(recycled.item_id)

    assert restored == card
    assert store.load_note(card.note_id) == card


def test_migrates_legacy_project_notes_txt_without_data_loss(tmp_path: Path) -> None:
    (tmp_path / "project.notes.txt").write_text("legacy **markdown**\nline 2", "utf-8")

    store = NoteCardStore(project_dir=tmp_path)
    migrated = store.migrate_legacy_text_note()

    assert migrated is not None
    assert migrated.title == "Project Notes"
    assert migrated.markdown_body == "legacy **markdown**\nline 2"
    assert store.load_note(migrated.note_id) == migrated


def test_backend_card_store_imports_without_qt() -> None:
    import notes.card_store as card_store

    assert card_store.NoteCardStore is NoteCardStore

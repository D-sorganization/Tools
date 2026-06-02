import json
import re
from pathlib import Path

import pytest
from notes.card_store import (
    META_END,
    META_START,
    NoteCardStore,
    _card_from_markdown,
    _card_to_markdown,
    _new_note_id,
    _timestamp,
)
from notes.models import (
    DEFAULT_NOTE_COLOR,
    NoteCard,
    NotesBoardSettings,
    RecycledNoteItem,
)


def make_card(
    note_id: str = "note_123",
    *,
    title: str = "Design note",
    body: str = "## Body\n\nDetails",
    updated_at: str = "2026-01-02T03:04:05Z",
) -> NoteCard:
    return NoteCard(
        note_id=note_id,
        title=title,
        markdown_body=body,
        color="#ABCDEF",
        created_at="2026-01-01T00:00:00Z",
        updated_at=updated_at,
        tags=("ops", "review"),
    )


def test_store_requires_existing_project_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project_dir must exist"):
        NoteCardStore(tmp_path / "missing")


def test_card_markdown_round_trips_metadata_and_body() -> None:
    card = make_card()

    text = _card_to_markdown(card)
    loaded = _card_from_markdown(text)

    assert text.startswith(f"{META_START}\n")
    assert f"\n{META_END}\n\n" in text
    assert loaded == NoteCard(
        note_id="note_123",
        title="Design note",
        markdown_body="## Body\n\nDetails",
        color="#abcdef",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-02T03:04:05Z",
        tags=("ops", "review"),
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("plain markdown", "note markdown is missing metadata"),
        (f"{META_START}\n{{}}\n", "note markdown metadata is not closed"),
    ],
)
def test_card_from_markdown_rejects_invalid_metadata(
    text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _card_from_markdown(text)


def test_card_from_markdown_uses_defaults_for_optional_metadata() -> None:
    text = f"{META_START}\n{json.dumps({'id': 'note_defaults'})}\n{META_END}\n\nbody"

    card = _card_from_markdown(text)

    assert card.title == ""
    assert card.markdown_body == "body"
    assert card.color == DEFAULT_NOTE_COLOR
    assert card.created_at == ""
    assert card.updated_at == ""
    assert card.tags == ()


def test_save_load_and_list_notes_sorted_by_updated_at(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)
    older = make_card("note_old", title="Old", updated_at="2026-01-01T00:00:00Z")
    newer = make_card("note_new", title="New", updated_at="2026-01-03T00:00:00Z")

    assert store.list_notes() == []
    assert store.save_note(older) == older
    assert store.save_note(newer) == newer

    assert store.load_note("note_old") == older
    assert store.load_note("missing") is None
    assert [card.note_id for card in store.list_notes()] == ["note_new", "note_old"]


def test_save_note_rejects_none(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)

    with pytest.raises(ValueError, match="card must be provided"):
        store.save_note(None)  # type: ignore[arg-type]


def test_create_note_generates_stable_id_and_timestamps(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)

    card = store.create_note("Title", "Body", color="#aabbcc", tags=("one",))

    assert re.fullmatch(r"note_[0-9a-f]{16}", card.note_id)
    assert card.created_at.endswith("Z")
    assert card.updated_at == card.created_at
    assert card.color == "#aabbcc"
    assert card.tags == ("one",)
    assert store.load_note(card.note_id) == card


def test_update_note_preserves_id_and_created_timestamp(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)
    original = store.save_note(make_card("note_keep", title="Original"))

    updated = store.update_note(
        "note_keep",
        title="Updated",
        markdown_body="New body",
        color="#112233",
        tags=("done",),
    )

    assert updated.note_id == "note_keep"
    assert updated.created_at == original.created_at
    assert updated.title == "Updated"
    assert updated.markdown_body == "New body"
    assert updated.color == "#112233"
    assert updated.tags == ("done",)
    assert updated.updated_at.endswith("Z")
    assert store.load_note("note_keep") == updated


def test_update_note_requires_existing_card(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)

    with pytest.raises(FileNotFoundError, match="note card does not exist"):
        store.update_note("missing", title="Nope", markdown_body="Nope")


def test_delete_note_moves_markdown_to_recycle_and_tracks_latest(
    tmp_path: Path,
) -> None:
    store = NoteCardStore(tmp_path)
    card = store.save_note(make_card("note_delete"))

    item = store.delete_note("note_delete", reason="cleanup")

    assert store.load_note("note_delete") is None
    assert item.reason == "cleanup"
    assert item.item_id.endswith("_note_delete")
    assert Path(item.path).read_text("utf-8") == _card_to_markdown(card)
    assert item.original_path == str(store.notes_dir / "note_delete.md")
    assert store.latest_recycled_id() == item.item_id


def test_delete_note_requires_existing_card(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)

    with pytest.raises(FileNotFoundError, match="note card does not exist"):
        store.delete_note("missing")


def test_restore_note_moves_recycled_markdown_back(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)
    card = store.save_note(make_card("note_restore"))
    item = store.delete_note("note_restore")

    restored = store.restore_note(item.item_id)

    assert restored == card
    assert store.load_note("note_restore") == card
    assert not Path(item.path).exists()
    assert store.latest_recycled_id() is None


def test_restore_note_returns_none_for_unknown_or_missing_source(
    tmp_path: Path,
) -> None:
    store = NoteCardStore(tmp_path)
    store.save_note(make_card("note_missing_source"))
    item = store.delete_note("note_missing_source")
    Path(item.path).unlink()

    assert store.restore_note("unknown") is None
    assert store.restore_note(item.item_id) is None


def test_restore_note_rejects_blank_item_id(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)

    with pytest.raises(ValueError, match="item_id cannot be empty"):
        store.restore_note(" ")


def test_latest_recycled_id_filters_non_markdown_items_and_sorts(
    tmp_path: Path,
) -> None:
    store = NoteCardStore(tmp_path)
    txt_item = RecycledNoteItem(
        item_id="txt",
        reason="legacy",
        path=str(tmp_path / "legacy.txt"),
        original_path="legacy",
        deleted_at="2027-01-01T00:00:00Z",
    )
    old_md = RecycledNoteItem(
        item_id="old_md",
        reason="manual",
        path=str(tmp_path / "old.md"),
        original_path="old",
        deleted_at="2025-01-01T00:00:00Z",
    )
    new_md = RecycledNoteItem(
        item_id="new_md",
        reason="manual",
        path=str(tmp_path / "new.md"),
        original_path="new",
        deleted_at="2026-01-01T00:00:00Z",
    )
    store._write_index([txt_item, old_md, new_md])

    assert store.latest_recycled_id() == "new_md"


def test_settings_round_trip_and_validation(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)

    assert store.load_settings() == NotesBoardSettings()
    settings = NotesBoardSettings(background_color="#123ABC")

    assert store.save_settings(settings) == settings
    assert store.load_settings() == NotesBoardSettings(background_color="#123abc")

    with pytest.raises(ValueError, match="settings must be provided"):
        store.save_settings(None)  # type: ignore[arg-type]


def test_migrate_legacy_text_note_creates_one_card_when_needed(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)
    store.legacy_notes_path.write_text("legacy body", "utf-8")

    migrated = store.migrate_legacy_text_note()

    assert migrated is not None
    assert migrated.title == "Project Notes"
    assert migrated.markdown_body == "legacy body"
    assert store.migrate_legacy_text_note() is None


def test_migrate_legacy_text_note_skips_missing_empty_or_existing_notes(
    tmp_path: Path,
) -> None:
    store = NoteCardStore(tmp_path)

    assert store.migrate_legacy_text_note() is None

    store.legacy_notes_path.write_text("", "utf-8")
    assert store.migrate_legacy_text_note() is None

    store.legacy_notes_path.write_text("legacy", "utf-8")
    store.save_note(make_card("note_existing"))
    assert store.migrate_legacy_text_note() is None


def test_index_helpers_round_trip_recycled_items(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)
    item = RecycledNoteItem(
        item_id="item",
        reason="manual",
        path=str(tmp_path / "item.md"),
        original_path=str(tmp_path / "project.notes" / "item.md"),
        deleted_at="2026-01-01T00:00:00Z",
    )

    assert store._read_index() == []
    store._append_index(item)
    assert store._find_item("item") == item
    store._remove_index("item")
    assert store._find_item("item") is None


def test_note_path_validates_note_id(tmp_path: Path) -> None:
    store = NoteCardStore(tmp_path)

    with pytest.raises(ValueError, match="note_id must be stable and path-safe"):
        store._note_path("../bad")


def test_timestamp_and_note_id_helpers_return_expected_shapes() -> None:
    assert re.fullmatch(r"\d{8}T\d{6}Z", _timestamp(compact=True))
    assert _timestamp().endswith("Z")
    assert re.fullmatch(r"note_[0-9a-f]{16}", _new_note_id())

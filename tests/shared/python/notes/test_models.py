import pytest
from notes.models import (
    DEFAULT_BOARD_BACKGROUND_COLOR,
    DEFAULT_NOTE_COLOR,
    NoteCard,
    NotesBoardSettings,
    RecycledNoteItem,
    normalize_color,
)


def test_normalize_color_accepts_and_canonicalizes_hex_values() -> None:
    assert normalize_color("  #AbC123  ") == "#abc123"
    assert normalize_color("#000000", field_name="accent") == "#000000"


@pytest.mark.parametrize("value", ["123456", "#12345", "#1234567", "#xyz123", None])
def test_normalize_color_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="color must be a #RRGGBB color"):
        normalize_color(value)  # type: ignore[arg-type]


def test_note_card_normalizes_mutable_ui_fields() -> None:
    card = NoteCard(
        note_id="note_1",
        title=123,
        markdown_body=456,
        color="#ABCDEF",
        tags=["ops", 7],
    )

    assert card.title == "123"
    assert card.markdown_body == "456"
    assert card.color == "#abcdef"
    assert card.tags == ("ops", "7")


@pytest.mark.parametrize("note_id", ["", "_leading", "contains space", "a" * 81])
def test_note_card_rejects_unstable_note_ids(note_id: str) -> None:
    with pytest.raises(ValueError, match="note_id must be stable and path-safe"):
        NoteCard(note_id=note_id, title="Title", markdown_body="Body")


def test_note_card_rejects_none_markdown_body() -> None:
    with pytest.raises(ValueError, match="markdown_body cannot be None"):
        NoteCard(note_id="note-1", title="Title", markdown_body=None)  # type: ignore[arg-type]


def test_note_card_uses_default_color() -> None:
    card = NoteCard(note_id="n1", title="", markdown_body="")

    assert card.color == DEFAULT_NOTE_COLOR


def test_board_settings_normalizes_background_color() -> None:
    settings = NotesBoardSettings(background_color="#AABBCC")

    assert settings.background_color == "#aabbcc"


def test_board_settings_uses_default_background_color() -> None:
    assert NotesBoardSettings().background_color == DEFAULT_BOARD_BACKGROUND_COLOR


def test_recycled_note_item_is_plain_snapshot_contract() -> None:
    item = RecycledNoteItem(
        item_id="20260101T000000Z_project",
        reason="manual_delete",
        path="recycle/project.txt",
        original_path="project.notes.txt",
        deleted_at="20260101T000000Z",
    )

    assert item.item_id.startswith("20260101")
    assert item.reason == "manual_delete"

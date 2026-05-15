"""Tests for the notes models module.

Covers RecycledNoteItem dataclass construction, field access,
immutability (frozen), and equality semantics.
"""

from __future__ import annotations

import dataclasses

import pytest
from notes.models import NoteCard, NotesBoardSettings, RecycledNoteItem


class TestNoteCard:
    def test_note_card_normalizes_color_and_tags(self) -> None:
        card = NoteCard(
            note_id="daily_note",
            title="Daily Note",
            markdown_body="# Heading",
            color="#ABCDEF",
            created_at="2026-05-14T12:00:00Z",
            updated_at="2026-05-14T12:00:00Z",
            tags=("work", "review"),
        )

        assert card.note_id == "daily_note"
        assert card.color == "#abcdef"
        assert card.tags == ("work", "review")

    def test_note_card_rejects_unsafe_id(self) -> None:
        with pytest.raises(ValueError, match="path-safe"):
            NoteCard(
                note_id="../escape",
                title="Bad",
                markdown_body="body",
                color="#ffffff",
                created_at="2026-05-14T12:00:00Z",
                updated_at="2026-05-14T12:00:00Z",
            )

    def test_note_card_rejects_none_markdown_body(self) -> None:
        with pytest.raises(ValueError, match="markdown_body"):
            NoteCard(
                note_id="note",
                title="Bad",
                markdown_body=None,  # type: ignore[arg-type]
                color="#ffffff",
                created_at="2026-05-14T12:00:00Z",
                updated_at="2026-05-14T12:00:00Z",
            )

    def test_note_card_rejects_invalid_color(self) -> None:
        with pytest.raises(ValueError, match="color"):
            NoteCard(
                note_id="note",
                title="Bad",
                markdown_body="body",
                color="alert(1)",
                created_at="2026-05-14T12:00:00Z",
                updated_at="2026-05-14T12:00:00Z",
            )


class TestNotesBoardSettings:
    def test_background_color_is_validated_and_normalized(self) -> None:
        settings = NotesBoardSettings(background_color="#FAFAFA")

        assert settings.background_color == "#fafafa"

    def test_invalid_background_color_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="background_color"):
            NotesBoardSettings(background_color="transparent url(bad)")


class TestRecycledNoteItem:
    def test_construction_with_all_fields(self) -> None:
        item = RecycledNoteItem(
            item_id="20240101T120000Z_project.notes",
            reason="user_delete",
            path="/tmp/.notes_recycle_bin/20240101T120000Z_project.notes.txt",
            original_path="/tmp/project.notes.txt",
            deleted_at="20240101T120000Z",
        )
        assert item.item_id == "20240101T120000Z_project.notes"
        assert item.reason == "user_delete"
        assert item.path == "/tmp/.notes_recycle_bin/20240101T120000Z_project.notes.txt"
        assert item.original_path == "/tmp/project.notes.txt"
        assert item.deleted_at == "20240101T120000Z"

    def test_frozen_dataclass_rejects_attribute_mutation(self) -> None:
        item = RecycledNoteItem(
            item_id="id1",
            reason="test",
            path="/p",
            original_path="/o",
            deleted_at="20240101T000000Z",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.reason = "changed"  # type: ignore[misc]

    def test_equality_based_on_all_fields(self) -> None:
        a = RecycledNoteItem(
            item_id="x",
            reason="r",
            path="/p",
            original_path="/o",
            deleted_at="t",
        )
        b = RecycledNoteItem(
            item_id="x",
            reason="r",
            path="/p",
            original_path="/o",
            deleted_at="t",
        )
        assert a == b

    def test_inequality_when_field_differs(self) -> None:
        a = RecycledNoteItem(
            item_id="x",
            reason="r",
            path="/p",
            original_path="/o",
            deleted_at="t1",
        )
        b = RecycledNoteItem(
            item_id="x",
            reason="r",
            path="/p",
            original_path="/o",
            deleted_at="t2",
        )
        assert a != b

    def test_hashable_because_frozen(self) -> None:
        item = RecycledNoteItem(
            item_id="h",
            reason="r",
            path="/p",
            original_path="/o",
            deleted_at="t",
        )
        result = {item}
        assert len(result) == 1

    def test_repr_contains_field_values(self) -> None:
        item = RecycledNoteItem(
            item_id="repr_test",
            reason="audit",
            path="/path",
            original_path="/orig",
            deleted_at="20240202T000000Z",
        )
        r = repr(item)
        assert "repr_test" in r
        assert "audit" in r

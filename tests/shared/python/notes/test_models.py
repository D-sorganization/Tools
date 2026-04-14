"""Tests for the notes models module.

Covers RecycledNoteItem dataclass construction, field access,
immutability (frozen), and equality semantics.
"""

from __future__ import annotations

import dataclasses

import pytest
from notes.models import RecycledNoteItem


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

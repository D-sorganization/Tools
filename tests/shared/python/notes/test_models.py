from __future__ import annotations

from notes.models import RecycledNoteItem

class TestRecycledNoteItem:
    def test_init_sets_attributes(self):
        item = RecycledNoteItem(
            item_id="id1",
            reason="reason1",
            path="path1",
            original_path="orig1",
            deleted_at="time1",
        )
        assert item.item_id == "id1"
        assert item.reason == "reason1"
        assert item.path == "path1"
        assert item.original_path == "orig1"
        assert item.deleted_at == "time1"

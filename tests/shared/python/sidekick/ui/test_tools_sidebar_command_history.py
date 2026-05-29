"""Unit tests for ``tools_sidebar.command_history``.

``CommandHistoryController`` is a Qt-free dataclass managing a bounded,
optionally-persisted command ring buffer with non-executing preview
navigation. Tests cover submission/dedup/bounding, preview navigation with
draft restoration, validation guards, and JSON persistence round-trips.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sidekick.ui.tools_sidebar.command_history import (
    DEFAULT_COMMAND_HISTORY_LIMIT,
    CommandHistoryController,
)


def test_default_limit_constant() -> None:
    assert DEFAULT_COMMAND_HISTORY_LIMIT == 50


def test_submit_appends_and_normalizes() -> None:
    ctrl = CommandHistoryController()
    assert ctrl.submit("  a = 1  ") == "a = 1"
    assert ctrl.commands == ("a = 1",)


def test_submit_dedupes_consecutive_duplicates() -> None:
    ctrl = CommandHistoryController()
    ctrl.submit("x")
    ctrl.submit("x")
    ctrl.submit("y")
    ctrl.submit("x")
    assert ctrl.commands == ("x", "y", "x")


def test_submit_blank_raises() -> None:
    ctrl = CommandHistoryController()
    with pytest.raises(ValueError, match="must not be blank"):
        ctrl.submit("   ")


def test_submit_non_string_raises_type_error() -> None:
    ctrl = CommandHistoryController()
    with pytest.raises(TypeError, match="command must be a string"):
        ctrl.submit(123)  # type: ignore[arg-type]


def test_max_entries_bounds_history() -> None:
    ctrl = CommandHistoryController(max_entries=3)
    for cmd in ("a", "b", "c", "d", "e"):
        ctrl.submit(cmd)
    assert ctrl.commands == ("c", "d", "e")


def test_max_entries_below_one_raises() -> None:
    with pytest.raises(ValueError, match="max_entries must be at least 1"):
        CommandHistoryController(max_entries=0)


def test_previous_preview_walks_back_then_clamps() -> None:
    ctrl = CommandHistoryController()
    for cmd in ("a", "b", "c"):
        ctrl.submit(cmd)
    assert ctrl.previous_preview("draft") == "c"
    assert ctrl.previous_preview() == "b"
    assert ctrl.previous_preview() == "a"
    # Clamped at the oldest entry.
    assert ctrl.previous_preview() == "a"


def test_previous_preview_empty_history_returns_none() -> None:
    assert CommandHistoryController().previous_preview("draft") is None


def test_next_preview_restores_draft_at_end() -> None:
    ctrl = CommandHistoryController()
    ctrl.submit("a")
    ctrl.submit("b")
    assert ctrl.previous_preview("typed-draft") == "b"
    # Moving forward past the newest entry restores the saved draft.
    assert ctrl.next_preview() == "typed-draft"


def test_next_preview_without_navigation_returns_none() -> None:
    ctrl = CommandHistoryController()
    ctrl.submit("a")
    assert ctrl.next_preview() is None


def test_next_preview_advances_through_history() -> None:
    ctrl = CommandHistoryController()
    for cmd in ("a", "b", "c"):
        ctrl.submit(cmd)
    ctrl.previous_preview("d")  # cursor -> c
    ctrl.previous_preview()  # cursor -> b
    ctrl.previous_preview()  # cursor -> a
    assert ctrl.next_preview() == "b"


def test_reset_navigation_clears_cursor() -> None:
    ctrl = CommandHistoryController()
    ctrl.submit("a")
    ctrl.previous_preview("draft")
    ctrl.reset_navigation()
    assert ctrl.next_preview() is None


def test_replace_dedupes_and_bounds() -> None:
    ctrl = CommandHistoryController(max_entries=2)
    ctrl.replace(["a", "a", "b", "c"])
    assert ctrl.commands == ("b", "c")


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_persistence_round_trip(tmp_path: Path) -> None:
    storage = tmp_path / "history.json"
    ctrl = CommandHistoryController(persist_history=True, storage_path=storage)
    ctrl.submit("first")
    ctrl.submit("second")
    assert storage.exists()

    reloaded = CommandHistoryController(persist_history=True, storage_path=storage)
    assert reloaded.commands == ("first", "second")


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    ctrl = CommandHistoryController(
        persist_history=True, storage_path=tmp_path / "nope.json"
    )
    assert ctrl.commands == ()


def test_load_ignores_non_dict_payload(tmp_path: Path) -> None:
    storage = tmp_path / "history.json"
    storage.write_text("[1, 2, 3]", encoding="utf-8")
    ctrl = CommandHistoryController(persist_history=True, storage_path=storage)
    assert ctrl.commands == ()


def test_load_ignores_non_list_commands(tmp_path: Path) -> None:
    storage = tmp_path / "history.json"
    storage.write_text('{"commands": "nope"}', encoding="utf-8")
    ctrl = CommandHistoryController(persist_history=True, storage_path=storage)
    assert ctrl.commands == ()


def test_load_filters_blank_and_non_string_entries(tmp_path: Path) -> None:
    storage = tmp_path / "history.json"
    storage.write_text('{"commands": ["a", "  ", 5, "b"]}', encoding="utf-8")
    ctrl = CommandHistoryController(persist_history=True, storage_path=storage)
    assert ctrl.commands == ("a", "b")

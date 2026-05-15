"""Tests for Sidekick calculator command history."""

from __future__ import annotations

import pytest
from upstream_drift_tools.ui.tools_sidebar import CommandHistoryController


def test_submit_appends_commands_and_suppresses_consecutive_duplicates() -> None:
    history = CommandHistoryController(max_entries=5)

    assert history.submit("  2 + 2  ") == "2 + 2"
    history.submit("sin(pi / 2)")
    history.submit("sin(pi / 2)")
    history.submit("3 * 7")

    assert history.commands == ("2 + 2", "sin(pi / 2)", "3 * 7")


def test_navigation_previews_without_submitting_or_executing() -> None:
    history = CommandHistoryController()
    history.submit("first")
    history.submit("second")

    assert history.previous_preview("draft") == "second"
    assert history.commands == ("first", "second")
    assert history.previous_preview("ignored while navigating") == "first"
    assert history.previous_preview() == "first"
    assert history.next_preview() == "second"
    assert history.next_preview() == "draft"
    assert history.next_preview() is None


def test_edited_recalled_command_is_submitted_as_new_command() -> None:
    history = CommandHistoryController()
    history.submit("a + b")

    assert history.previous_preview() == "a + b"
    history.submit("a + b + c")

    assert history.commands == ("a + b", "a + b + c")
    assert history.previous_preview() == "a + b + c"


def test_history_length_limit_keeps_most_recent_commands() -> None:
    history = CommandHistoryController(max_entries=3)

    for command in ("one", "two", "three", "four"):
        history.submit(command)

    assert history.commands == ("two", "three", "four")
    assert history.previous_preview() == "four"
    assert history.previous_preview() == "three"
    assert history.previous_preview() == "two"


def test_history_persists_only_when_enabled(tmp_path) -> None:
    persistent_path = tmp_path / "calculator-history.json"
    ephemeral = CommandHistoryController(
        persist_history=False,
        storage_path=persistent_path,
    )
    ephemeral.submit("not persisted")
    assert not persistent_path.exists()

    persistent = CommandHistoryController(
        max_entries=2,
        persist_history=True,
        storage_path=persistent_path,
    )
    persistent.submit("alpha")
    persistent.submit("beta")
    persistent.submit("gamma")

    loaded = CommandHistoryController(
        max_entries=2,
        persist_history=True,
        storage_path=persistent_path,
    )

    assert loaded.commands == ("beta", "gamma")


@pytest.mark.parametrize("command", [None, "", "   "])
def test_submit_rejects_invalid_commands(command) -> None:
    history = CommandHistoryController()

    with pytest.raises((TypeError, ValueError)):
        history.submit(command)

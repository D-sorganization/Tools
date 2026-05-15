"""Tests for Sidekick named state profile persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from upstream_drift_tools.ui.tools_sidebar import (
    CLEAR_SIDEKICK_DATA_CONFIRMATION,
    CLEAR_SIDEKICK_DATA_WARNING,
    SidebarState,
    SidekickStateProfileStore,
)


def test_state_profile_save_load_round_trip(tmp_path: Path) -> None:
    store = SidekickStateProfileStore(tmp_path / "sidekick")
    state = SidebarState(
        dock_area="left",
        floating=True,
        minimized=True,
        width=420,
        height=640,
        active_tab="notes",
        tab_order=["files", "notes", "calculator"],
        hidden_tabs=["chat"],
        popped_out_tabs=["terminal"],
        tab_display_names={"notes": "Run notes"},
        calculator_predictive_text_enabled=True,
    )

    saved = store.save_profile("Run 01", state)
    loaded = store.load_profile("Run 01")

    assert saved.ok is True
    assert saved.path == tmp_path / "sidekick" / "profiles" / "Run 01.json"
    assert loaded.ok is True
    assert loaded.state == state


def test_missing_state_profile_returns_result_without_current_state_change(
    tmp_path: Path,
) -> None:
    store = SidekickStateProfileStore(tmp_path)
    current = SidebarState(active_tab="calculator", width=500)

    result = store.load_profile("missing")

    assert result.ok is False
    assert result.state is None
    assert "not found" in result.message
    assert current.active_tab == "calculator"
    assert current.width == 500


def test_malformed_state_profile_is_rejected_without_crashing(tmp_path: Path) -> None:
    store = SidekickStateProfileStore(tmp_path)
    store.profiles_dir.mkdir(parents=True)
    (store.profiles_dir / "bad.json").write_text(
        '{"width": {}, "active_tab": "notes"}',
        encoding="utf-8",
    )

    result = store.load_profile("bad")

    assert result.ok is False
    assert result.state is None
    assert "Invalid profile payload" in result.message


@pytest.mark.parametrize("name", ["", "../escape", "bad/name", "bad\\name"])
def test_state_profile_names_must_be_path_safe(tmp_path: Path, name: str) -> None:
    store = SidekickStateProfileStore(tmp_path)

    with pytest.raises(ValueError, match="path-safe"):
        store.save_profile(name, SidebarState())


def test_clear_sidekick_data_requires_confirmation(tmp_path: Path) -> None:
    store = SidekickStateProfileStore(tmp_path / "sidekick")
    store.save_profile("default", SidebarState(active_tab="notes"))

    denied = store.clear_data()
    assert denied.ok is False
    assert denied.warning == CLEAR_SIDEKICK_DATA_WARNING
    assert (store.profiles_dir / "default.json").exists()

    cleared = store.clear_data(confirmation=CLEAR_SIDEKICK_DATA_CONFIRMATION)
    assert cleared.ok is True
    assert cleared.warning == CLEAR_SIDEKICK_DATA_WARNING
    assert not store.storage_root.exists()


def test_old_sidebar_state_load_json_contract_stays_compatible(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    assert SidebarState.load_json(missing_path) == SidebarState()

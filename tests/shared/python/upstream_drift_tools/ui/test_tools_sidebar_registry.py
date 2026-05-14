"""Tests for the unified tools sidebar backend contract."""

from __future__ import annotations

import json
from pathlib import Path

from upstream_drift_tools.ui.tools_sidebar import SidebarState, WorkspaceRegistry


class NotJson:
    def __repr__(self) -> str:
        return "<NotJson demo>"


def test_workspace_registry_crud_and_summaries() -> None:
    registry = WorkspaceRegistry()

    registry.set("temperature", 293.15)
    registry.set("matrix", [[1, 2], [3, 4]])
    registry.set("config", {"unit": "K"})

    assert registry.get("temperature") == 293.15
    assert registry.list() == ["config", "matrix", "temperature"]
    assert registry.describe("temperature").summary == "scalar"
    assert registry.describe("matrix").summary == "2x2"
    assert registry.describe("config").summary == "keys=1"

    assert registry.remove("temperature") is True
    assert registry.remove("missing") is False
    registry.clear()
    assert registry.list() == []


def test_workspace_registry_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "workspace.json"
    registry = WorkspaceRegistry({"name": "case-a", "values": [1, 2, 3]})

    registry.save_json(path)
    loaded = WorkspaceRegistry.load_json(path)

    assert loaded.get("name") == "case-a"
    assert loaded.get("values") == [1, 2, 3]


def test_workspace_registry_non_json_repr_metadata(tmp_path: Path) -> None:
    path = tmp_path / "workspace.json"
    registry = WorkspaceRegistry()
    registry.set("object", NotJson())

    registry.save_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    [entry] = payload["variables"]
    assert entry["json_safe"] is False
    assert entry["repr"] == "<NotJson demo>"

    loaded = WorkspaceRegistry.load_json(path)
    assert loaded.get("object") == "<NotJson demo>"
    assert loaded.describe("object").json_safe is False


def test_workspace_registry_environment_export() -> None:
    registry = WorkspaceRegistry({"alpha value": 3, "items": ["a", "b"]})

    exported = registry.export_environment(prefix="TEST_")

    assert exported["TEST_ALPHA_VALUE"] == "3"
    assert exported["TEST_ITEMS"] == '["a", "b"]'


def test_sidebar_state_round_trip_and_sanitizes_values(tmp_path: Path) -> None:
    path = tmp_path / "sidebar.json"
    state = SidebarState(
        dock_area="invalid",
        floating=True,
        minimized=True,
        width=1,
        height=2,
        active_tab="terminal",
        tab_order=["notes", "terminal", "notes", ""],
        hidden_tabs=["chat", "chat"],
        popped_out_tabs=["calculator"],
        tab_display_names={" notes ": " Project notes ", "terminal": "  "},
    )

    state.save_json(path)
    loaded = SidebarState.load_json(path)

    assert loaded.dock_area == "right"
    assert loaded.floating is True
    assert loaded.width == 240
    assert loaded.height == 240
    assert loaded.active_tab == "terminal"
    assert loaded.minimized is True
    assert loaded.tab_order == ["notes", "terminal"]
    assert loaded.hidden_tabs == ["chat"]
    assert loaded.popped_out_tabs == ["calculator"]
    assert loaded.tab_display_names == {"notes": "Project notes"}


def test_sidebar_state_sanitizes_malformed_custom_tab_names() -> None:
    loaded = SidebarState.from_dict(
        {
            "tab_display_names": {
                "files": " Project files ",
                "blank": "",
                "  ": "Missing id",
            }
        }
    )

    assert loaded.tab_display_names == {"files": "Project files"}


def test_sidebar_state_rejects_non_mapping_custom_tab_names() -> None:
    loaded = SidebarState.from_dict({"tab_display_names": ["files"]})

    assert loaded.tab_display_names == {}

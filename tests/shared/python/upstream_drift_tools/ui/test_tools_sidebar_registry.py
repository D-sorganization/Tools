"""Tests for the unified tools sidebar backend contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def test_workspace_registry_matrix_metadata_and_preview() -> None:
    registry = WorkspaceRegistry()

    variable = registry.set("matrix", [[1, 2, 3], [4, 5, 6]])
    metadata = variable.to_metadata()

    assert variable.summary == "2x3"
    assert variable.shape == (2, 3)
    assert variable.dtype == "int"
    assert variable.size == 6
    assert variable.preview == "[[1, 2, 3], [4, 5, 6]]"
    assert metadata["shape"] == [2, 3]
    assert metadata["dtype"] == "int"
    assert metadata["size"] == 6
    assert metadata["preview"] == "[[1, 2, 3], [4, 5, 6]]"


def test_workspace_registry_large_matrix_preview_is_bounded() -> None:
    matrix = [[row * 10 + column for column in range(8)] for row in range(6)]
    registry = WorkspaceRegistry({"large": matrix})

    variable = registry.describe("large")

    assert variable.summary == "6x8"
    assert variable.shape == (6, 8)
    assert variable.size == 48
    assert "..." in variable.preview
    assert len(variable.preview) <= 120
    assert "57" not in variable.preview


def test_workspace_registry_rejects_ragged_matrix_values() -> None:
    registry = WorkspaceRegistry()

    try:
        registry.set("ragged", [[1, 2], [3]])
    except ValueError as exc:
        assert "ragged" in str(exc)
    else:
        raise AssertionError("ragged matrix should be rejected")


def test_workspace_registry_numpy_array_metadata_when_available() -> None:
    np = pytest.importorskip("numpy")
    registry = WorkspaceRegistry({"array": np.array([[1.5, 2.5], [3.5, 4.5]])})

    variable = registry.describe("array")
    metadata = variable.to_metadata()

    assert variable.shape == (2, 2)
    assert variable.dtype == "float64"
    assert variable.size == 4
    assert variable.json_safe is False
    assert metadata["shape"] == [2, 2]
    assert metadata["dtype"] == "float64"
    assert metadata["preview"] == "[[1.5, 2.5], [3.5, 4.5]]"


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


def test_workspace_registry_matrix_json_round_trip_preserves_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "workspace.json"
    registry = WorkspaceRegistry({"matrix": [[1, 2], [3, 4]]})

    registry.save_json(path)
    loaded = WorkspaceRegistry.load_json(path)

    assert loaded.get("matrix") == [[1, 2], [3, 4]]
    assert loaded.describe("matrix").shape == (2, 2)
    assert loaded.describe("matrix").preview == "[[1, 2], [3, 4]]"


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

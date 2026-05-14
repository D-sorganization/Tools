"""Tests for Sidekick calculator-local workspace persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from upstream_drift_tools.ui.tools_sidebar import (
    CALCULATOR_WORKSPACE_SCOPE,
    CalculatorWorkspaceController,
    CalculatorWorkspaceSettings,
    WorkspaceRegistry,
    validate_calculator_workspace_path,
)


def _controller(
    registry: WorkspaceRegistry,
    tmp_path: Path,
) -> CalculatorWorkspaceController:
    return CalculatorWorkspaceController(
        registry,
        settings=CalculatorWorkspaceSettings(default_directory=tmp_path),
    )


def test_save_calculator_workspace_writes_calculator_scope_only(tmp_path: Path) -> None:
    local = WorkspaceRegistry({"local_value": 4})
    global_workspace = WorkspaceRegistry({"global_value": 8})
    path = tmp_path / "calculator.json"

    saved = _controller(local, tmp_path).save(path)

    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["scope"] == CALCULATOR_WORKSPACE_SCOPE
    assert [entry["name"] for entry in payload["variables"]] == ["local_value"]
    assert global_workspace.list_names() == ["global_value"]


def test_load_calculator_workspace_merges_without_mutating_global(
    tmp_path: Path,
) -> None:
    source = WorkspaceRegistry({"imported": 7})
    path = _controller(source, tmp_path).save(tmp_path / "calculator.json")
    local = WorkspaceRegistry({"existing": 1})
    global_workspace = WorkspaceRegistry({"existing": 99})

    result = _controller(local, tmp_path).load(path)

    assert result.summary == "Loaded 1 variables: imported"
    assert local.get("existing") == 1
    assert local.get("imported") == 7
    assert global_workspace.get("existing") == 99
    assert global_workspace.get("imported", None) is None


def test_replace_load_requires_explicit_confirmation(tmp_path: Path) -> None:
    path = _controller(WorkspaceRegistry({"incoming": 2}), tmp_path).save(
        tmp_path / "calculator.json",
    )
    local = WorkspaceRegistry({"existing": 1})
    controller = _controller(local, tmp_path)

    with pytest.raises(PermissionError, match="explicit confirmation"):
        controller.load(path, replace=True)

    assert local.list_names() == ["existing"]
    result = controller.load(path, replace=True, confirm_replace=True)
    assert result.replaced is True
    assert local.list_names() == ["incoming"]


def test_malformed_workspace_leaves_current_variables_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "calculator.json"
    path.write_text("{not json", encoding="utf-8")
    local = WorkspaceRegistry({"existing": 1})

    with pytest.raises(ValueError, match="valid JSON"):
        _controller(local, tmp_path).load(path)

    assert local.get("existing") == 1


def test_path_validation_rejects_non_workspace_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".json"):
        validate_calculator_workspace_path(tmp_path / "calculator.txt")

    with pytest.raises(ValueError, match="file"):
        validate_calculator_workspace_path(tmp_path)

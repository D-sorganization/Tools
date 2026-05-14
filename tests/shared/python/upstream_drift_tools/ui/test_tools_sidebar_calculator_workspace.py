"""Tests for Sidekick calculator-local workspace persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from upstream_drift_tools.ui.tools_sidebar import (
    CALCULATOR_WORKSPACE_SCOPE,
    CalculatorWorkspaceController,
    CalculatorWorkspaceFacade,
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


def test_calculator_facade_shadows_global_without_overwrite() -> None:
    local = WorkspaceRegistry()
    global_workspace = WorkspaceRegistry({"temperature": 900, "pressure": 14.7})
    workspace = CalculatorWorkspaceFacade(
        local_registry=local,
        global_registry=global_workspace,
        calculator_scope_id="calculator-tab-a",
    )

    workspace.set_local("temperature", 300)

    assert workspace.calculator_scope_id == "calculator-tab-a"
    assert workspace.get("temperature", include_global=True) == 300
    assert workspace.get("pressure", include_global=True) == 14.7
    assert workspace.get("pressure") is None
    assert global_workspace.get("temperature") == 900
    assert workspace.export_variables() == {"pressure": 14.7, "temperature": 300}


def test_deleting_local_variable_leaves_global_value_intact() -> None:
    local = WorkspaceRegistry({"answer": 10})
    global_workspace = WorkspaceRegistry({"answer": 42})
    workspace = CalculatorWorkspaceFacade(
        local_registry=local,
        global_registry=global_workspace,
    )

    assert workspace.remove_local("answer") is True

    assert workspace.get("answer", include_global=True) == 42
    assert global_workspace.get("answer") == 42


def test_duplicate_calculator_facades_keep_local_workspaces_separate() -> None:
    global_workspace = WorkspaceRegistry({"shared": 1})
    first_local = WorkspaceRegistry({"scratch": "first"})
    second_local = WorkspaceRegistry({"scratch": "second"})
    first = CalculatorWorkspaceFacade(
        local_registry=first_local,
        global_registry=global_workspace,
        calculator_scope_id="calculator-tab-1",
    )
    second = CalculatorWorkspaceFacade(
        local_registry=second_local,
        global_registry=global_workspace,
        calculator_scope_id="calculator-tab-2",
    )

    first.set_local("shared", 100)

    assert first.export_variables() == {"scratch": "first", "shared": 100}
    assert second.export_variables() == {"scratch": "second", "shared": 1}
    assert global_workspace.get("shared") == 1


def test_promote_local_variable_requires_explicit_overwrite() -> None:
    local = WorkspaceRegistry({"result": 7})
    global_workspace = WorkspaceRegistry({"result": 3})
    workspace = CalculatorWorkspaceFacade(
        local_registry=local,
        global_registry=global_workspace,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        workspace.promote_to_global("result")

    workspace.promote_to_global("result", overwrite=True)

    assert global_workspace.get("result") == 7

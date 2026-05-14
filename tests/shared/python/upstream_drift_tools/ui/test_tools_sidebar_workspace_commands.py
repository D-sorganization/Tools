"""Tests for the Sidekick workspace command line service."""

from __future__ import annotations

from pathlib import Path

import pytest
from upstream_drift_tools.ui.tools_sidebar.calculator_workspace import (
    CalculatorWorkspaceController,
    CalculatorWorkspaceFacade,
    CalculatorWorkspaceSettings,
)
from upstream_drift_tools.ui.tools_sidebar.registry import WorkspaceRegistry
from upstream_drift_tools.ui.tools_sidebar.workspace_commands import (
    WorkspaceCommandExecutor,
)


def _build_executor(
    tmp_path: Path,
) -> tuple[WorkspaceCommandExecutor, WorkspaceRegistry, WorkspaceRegistry]:
    local_registry = WorkspaceRegistry({"seed": 1})
    global_registry = WorkspaceRegistry({"shared": 2})
    local_controller = CalculatorWorkspaceController(
        local_registry,
        settings=CalculatorWorkspaceSettings(
            default_directory=tmp_path,
            default_filename="local-workspace.json",
        ),
    )
    executor = WorkspaceCommandExecutor(
        workspace=CalculatorWorkspaceFacade(
            local_registry=local_registry,
            global_registry=global_registry,
        ),
        local_controller=local_controller,
        global_registry=global_registry,
        global_storage_path=tmp_path / "global-workspace.json",
    )
    return executor, local_registry, global_registry


def test_workspace_command_assignment_and_inspection_for_local_and_global(
    tmp_path: Path,
) -> None:
    executor, local_registry, global_registry = _build_executor(tmp_path)

    local_result = executor.execute("local alpha = [1, 2, 3]")
    global_result = executor.execute("global beta = {'value': 4}")
    local_show = executor.execute("show local alpha")
    global_show = executor.execute("show global beta")

    assert local_result.mutated is True
    assert local_registry.get("alpha") == [1, 2, 3]
    assert "alpha" in local_result.message
    assert global_result.mutated is True
    assert global_registry.get("beta") == {"value": 4}
    assert "beta" in global_result.message
    assert "length=3" in local_show.message
    assert "[1, 2, 3]" in local_show.message
    assert "keys=1" in global_show.message
    assert "{'value': 4}" in global_show.message


def test_workspace_command_delete_and_clear_require_confirmation(
    tmp_path: Path,
) -> None:
    executor, local_registry, global_registry = _build_executor(tmp_path)

    with pytest.raises(PermissionError, match="confirm"):
        executor.execute("delete local seed")
    with pytest.raises(PermissionError, match="confirm"):
        executor.execute("clear global")

    delete_result = executor.execute("delete local seed confirm")
    clear_result = executor.execute("clear global confirm")

    assert delete_result.mutated is True
    assert local_registry.get("seed") is None
    assert clear_result.mutated is True
    assert global_registry.list() == []


def test_workspace_command_load_and_save_use_scope_persistence(
    tmp_path: Path,
) -> None:
    executor, local_registry, global_registry = _build_executor(tmp_path)

    executor.execute("local alpha = [1, 2, 3]")
    executor.execute("global beta = {'value': 4}")
    local_save = executor.execute("save local")
    global_save = executor.execute("save global")

    local_registry.clear()
    global_registry.clear()

    local_load = executor.execute("load local")
    global_load = executor.execute("load global")

    assert "local-workspace.json" in local_save.message
    assert "global-workspace.json" in global_save.message
    assert local_registry.get("alpha") == [1, 2, 3]
    assert global_registry.get("beta") == {"value": 4}
    assert "Loaded" in local_load.message
    assert "Loaded" in global_load.message


def test_workspace_command_invalid_input_leaves_workspace_unchanged(
    tmp_path: Path,
) -> None:
    executor, local_registry, global_registry = _build_executor(tmp_path)
    before_local = local_registry.to_dict()
    before_global = global_registry.to_dict()

    with pytest.raises(ValueError, match="Unsupported"):
        executor.execute("terminal print(42)")
    with pytest.raises(ValueError, match="literal"):
        executor.execute("global bad = numpy.arange(3)")

    assert local_registry.to_dict() == before_local
    assert global_registry.to_dict() == before_global

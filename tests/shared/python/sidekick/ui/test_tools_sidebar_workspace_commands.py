"""Unit tests for ``tools_sidebar.workspace_commands``.

``WorkspaceCommandExecutor`` runs a small, explicit, bounded command language
(assignment + show/delete/clear/save/load) over local/global workspace
registries — deliberately avoiding arbitrary ``eval``. Tests build the executor
from Qt-free facade/controller objects and exercise every command, scope, and
guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sidekick.ui.tools_sidebar.calculator_workspace import (
    CalculatorWorkspaceController,
    CalculatorWorkspaceFacade,
    CalculatorWorkspaceSettings,
)
from sidekick.ui.tools_sidebar.registry import WorkspaceRegistry
from sidekick.ui.tools_sidebar.workspace_commands import (
    WorkspaceCommandExecutor,
    WorkspaceCommandResult,
)


@pytest.fixture
def executor(tmp_path: Path) -> WorkspaceCommandExecutor:
    local_reg = WorkspaceRegistry()
    global_reg = WorkspaceRegistry()
    facade = CalculatorWorkspaceFacade(
        local_registry=local_reg, global_registry=global_reg
    )
    local_ctrl = CalculatorWorkspaceController(
        local_reg, settings=CalculatorWorkspaceSettings(default_directory=tmp_path)
    )
    return WorkspaceCommandExecutor(
        workspace=facade,
        local_controller=local_ctrl,
        global_registry=global_reg,
        global_storage_path=tmp_path / "global_workspace.json",
    )


# ---------------------------------------------------------------------------
# constructor guards
# ---------------------------------------------------------------------------


def test_constructor_requires_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workspace must be provided"):
        WorkspaceCommandExecutor(
            workspace=None,  # type: ignore[arg-type]
            local_controller=object(),  # type: ignore[arg-type]
            global_registry=WorkspaceRegistry(),
            global_storage_path=tmp_path / "g.json",
        )


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------


def test_local_assignment_sets_value(executor: WorkspaceCommandExecutor) -> None:
    result = executor.execute("local answer = 42")
    assert isinstance(result, WorkspaceCommandResult)
    assert result.mutated is True
    assert result.scope == "calculator"
    assert executor._workspace.local_registry.get("answer") == 42


def test_global_assignment_sets_value(executor: WorkspaceCommandExecutor) -> None:
    result = executor.execute("global data = [1, 2, 3]")
    assert result.scope == "global"
    assert executor._workspace.global_registry.get("data") == [1, 2, 3]


def test_assignment_non_literal_rejected(executor: WorkspaceCommandExecutor) -> None:
    with pytest.raises(ValueError, match="Python literal"):
        executor.execute("local x = open('secrets')")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_describes_variable(executor: WorkspaceCommandExecutor) -> None:
    executor.execute("local v = [1, 2, 3]")
    result = executor.execute("show local v")
    # The "local" alias normalizes to the "calculator" scope in the message.
    assert "calculator v" in result.message
    assert result.scope == "calculator"
    assert result.mutated is False


def test_show_wrong_arity_raises(executor: WorkspaceCommandExecutor) -> None:
    with pytest.raises(ValueError, match="show command must be"):
        executor.execute("show local")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_with_confirm(executor: WorkspaceCommandExecutor) -> None:
    executor.execute("local gone = 1")
    result = executor.execute("delete local gone confirm")
    assert result.mutated is True
    assert executor._workspace.local_registry.get("gone") is None


def test_delete_without_confirm_raises(executor: WorkspaceCommandExecutor) -> None:
    executor.execute("local x = 1")
    with pytest.raises(PermissionError, match="confirm"):
        executor.execute("delete local x")


def test_delete_wrong_confirm_word_raises(executor: WorkspaceCommandExecutor) -> None:
    executor.execute("local x = 1")
    with pytest.raises(PermissionError, match="confirm"):
        executor.execute("delete local x yes")


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_local_with_confirm(executor: WorkspaceCommandExecutor) -> None:
    executor.execute("local a = 1")
    result = executor.execute("clear local confirm")
    assert result.mutated is True
    assert executor._workspace.local_registry.list_names() == []


def test_clear_without_confirm_raises(executor: WorkspaceCommandExecutor) -> None:
    with pytest.raises(PermissionError, match="confirm"):
        executor.execute("clear local")


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_then_load_round_trip(
    executor: WorkspaceCommandExecutor, tmp_path: Path
) -> None:
    executor.execute("local persisted = 99")
    target = tmp_path / "ws.json"
    # Use a POSIX-style path string: the executor tokenizes with shlex.split,
    # which would otherwise treat Windows backslashes as escape characters.
    target_arg = target.as_posix()
    saved = executor.execute(f"save local to {target_arg}")
    assert "saved" in saved.message.lower()
    assert target.exists()

    executor.execute("clear local confirm")
    loaded = executor.execute(f"load local from {target_arg} replace confirm")
    assert loaded.scope == "calculator"
    assert executor._workspace.local_registry.get("persisted") == 99


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_unknown_action_raises(executor: WorkspaceCommandExecutor) -> None:
    with pytest.raises(ValueError, match="Unsupported workspace command"):
        executor.execute("frobnicate local x")


def test_blank_command_raises(executor: WorkspaceCommandExecutor) -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        executor.execute("   ")


def test_unsupported_scope_raises(executor: WorkspaceCommandExecutor) -> None:
    with pytest.raises(ValueError, match="Unsupported workspace scope"):
        executor.execute("show sideways v")


def test_invalid_variable_name_raises(executor: WorkspaceCommandExecutor) -> None:
    with pytest.raises(ValueError, match="identifier-like"):
        executor.execute("show local 9bad")

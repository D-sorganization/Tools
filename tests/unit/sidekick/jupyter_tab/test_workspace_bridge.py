# ruff: noqa: E501
"""Tests for ``WorkspaceBridge`` — Phase 3 (Tools #2877).

WorkspaceBridge filters Sidekick workspace variables and injects them
into a Jupyter kernel environment via the session model interface.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sidekick.ui.tools_sidebar.jupyter_tab.notebook_session import (
    NotebookSessionModel,
)
from sidekick.ui.tools_sidebar.jupyter_tab.workspace_bridge import WorkspaceBridge

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_model(tmp_path: Path) -> NotebookSessionModel:
    """A valid session model for use in bridge tests."""
    nb = tmp_path / "notebooks" / "test.ipynb"
    nb.parent.mkdir(parents=True)
    nb.touch()
    return NotebookSessionModel(
        notebook_path=nb,
        workspace_root=tmp_path,
        kernel_env=None,
    )


# ---------------------------------------------------------------------------
# Construction & DbC
# ---------------------------------------------------------------------------


def test_bridge_requires_session_model() -> None:
    """WorkspaceBridge(None) must raise ValueError (DbC)."""
    with pytest.raises(ValueError, match="session_model"):
        WorkspaceBridge(None)  # type: ignore[arg-type]


def test_bridge_constructs_with_valid_session_model(
    session_model: NotebookSessionModel,
) -> None:
    """WorkspaceBridge accepts a valid NotebookSessionModel."""
    bridge = WorkspaceBridge(session_model)
    assert bridge is not None


# ---------------------------------------------------------------------------
# export_variables
# ---------------------------------------------------------------------------


def test_export_variables_returns_dict(session_model: NotebookSessionModel) -> None:
    """export_variables returns a dict."""
    bridge = WorkspaceBridge(session_model)
    result = bridge.export_variables({"x": 1, "y": [1, 2, 3]})
    assert isinstance(result, dict)


def test_export_variables_keeps_primitives(
    session_model: NotebookSessionModel,
) -> None:
    """int, float, str, bool, list, dict all pass through."""
    bridge = WorkspaceBridge(session_model)
    workspace = {
        "an_int": 42,
        "a_float": 3.14,
        "a_str": "hello",
        "a_bool": True,
        "a_list": [1, 2, 3],
        "a_dict": {"key": "value"},
        "none_val": None,
    }
    result = bridge.export_variables(workspace)
    assert result["an_int"] == 42
    assert result["a_float"] == 3.14
    assert result["a_str"] == "hello"
    assert result["a_bool"] is True
    assert result["a_list"] == [1, 2, 3]
    assert result["a_dict"] == {"key": "value"}
    assert result["none_val"] is None


def test_export_variables_filters_non_serializable(
    session_model: NotebookSessionModel,
) -> None:
    """Lambda, functions, and arbitrary objects are excluded."""
    bridge = WorkspaceBridge(session_model)
    workspace = {
        "x": 1,
        "fn": lambda v: v,  # not JSON-serializable
        "obj": object(),  # not JSON-serializable
    }
    result = bridge.export_variables(workspace)
    assert "x" in result
    assert "fn" not in result
    assert "obj" not in result


def test_export_variables_does_not_mutate_input(
    session_model: NotebookSessionModel,
) -> None:
    """The input workspace dict is never modified."""
    bridge = WorkspaceBridge(session_model)
    workspace = {"x": 1, "fn": lambda v: v}
    original_keys = set(workspace.keys())
    bridge.export_variables(workspace)
    assert set(workspace.keys()) == original_keys


def test_bridge_accepts_empty_workspace(
    session_model: NotebookSessionModel,
) -> None:
    """export_variables({}) returns {}."""
    bridge = WorkspaceBridge(session_model)
    assert bridge.export_variables({}) == {}


def test_export_variables_requires_dict(
    session_model: NotebookSessionModel,
) -> None:
    """export_variables raises TypeError when workspace is not a dict (DbC)."""
    bridge = WorkspaceBridge(session_model)
    with pytest.raises(TypeError, match="workspace"):
        bridge.export_variables(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply_to_kernel_environment
# ---------------------------------------------------------------------------


def test_apply_to_kernel_environment_calls_session_model(
    session_model: NotebookSessionModel,
) -> None:
    """apply_to_kernel_environment calls set_kernel_environment on a widget."""
    bridge = WorkspaceBridge(session_model)

    # Patch set_kernel_environment on the session model (it's a dataclass,
    # so we attach a mock method directly for this test).
    received: list[dict] = []
    session_model.set_kernel_environment = lambda env: received.append(env)  # type: ignore[method-assign]

    bridge.apply_to_kernel_environment({"x": 1, "fn": lambda: None})

    assert len(received) == 1
    assert received[0] == {"x": 1}  # fn is filtered out


def test_apply_to_kernel_environment_empty_workspace(
    session_model: NotebookSessionModel,
) -> None:
    """apply_to_kernel_environment with empty workspace passes empty dict."""
    bridge = WorkspaceBridge(session_model)

    received: list[dict] = []
    session_model.set_kernel_environment = lambda env: received.append(env)  # type: ignore[method-assign]

    bridge.apply_to_kernel_environment({})
    assert received == [{}]

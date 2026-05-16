"""Tests for Phase 3 additions to ``SidekickNotebookWidget`` (Tools #2877).

Verifies that the widget accepts an optional WorkspaceBridge and calls
apply_to_kernel_environment when update_workspace() is invoked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sidekick.ui.tools_sidebar.jupyter_tab.notebook_session import (
    NotebookSessionModel,
)
from sidekick.ui.tools_sidebar.jupyter_tab.sidekick_notebook_widget import (
    SidekickNotebookWidget,
)
from sidekick.ui.tools_sidebar.jupyter_tab.workspace_bridge import WorkspaceBridge


@pytest.fixture
def session_model(tmp_path: Path) -> NotebookSessionModel:
    """A valid session model."""
    nb = tmp_path / "notebooks" / "test.ipynb"
    nb.parent.mkdir(parents=True)
    nb.touch()
    return NotebookSessionModel(
        notebook_path=nb,
        workspace_root=tmp_path,
        kernel_env=None,
    )


@pytest.fixture
def workspace_bridge(session_model: NotebookSessionModel) -> WorkspaceBridge:
    """A WorkspaceBridge backed by a real session model."""
    return WorkspaceBridge(session_model)


def test_widget_accepts_workspace_bridge(
    tmp_path: Path,
    workspace_bridge: WorkspaceBridge,
) -> None:
    """SidekickNotebookWidget(workspace_bridge=bridge) does not raise."""
    widget = SidekickNotebookWidget(
        workspace_root=tmp_path,
        workspace_bridge=workspace_bridge,
    )
    assert widget is not None


def test_widget_constructs_without_bridge(tmp_path: Path) -> None:
    """workspace_bridge is optional — widget must work without it."""
    widget = SidekickNotebookWidget(workspace_root=tmp_path)
    assert widget is not None


def test_widget_calls_bridge_on_workspace_update(
    tmp_path: Path,
    session_model: NotebookSessionModel,
) -> None:
    """update_workspace({'x': 1}) causes bridge.apply_to_kernel_environment to
    be called."""
    bridge_mock = MagicMock(spec=WorkspaceBridge)
    widget = SidekickNotebookWidget(
        workspace_root=tmp_path,
        workspace_bridge=bridge_mock,
    )
    widget.update_workspace({"x": 1, "y": "hello"})
    bridge_mock.apply_to_kernel_environment.assert_called_once_with(
        {"x": 1, "y": "hello"}
    )


def test_widget_update_workspace_without_bridge_does_not_raise(
    tmp_path: Path,
) -> None:
    """update_workspace when no bridge is attached is a no-op (no exception)."""
    widget = SidekickNotebookWidget(workspace_root=tmp_path)
    widget.update_workspace({"x": 42})  # should not raise

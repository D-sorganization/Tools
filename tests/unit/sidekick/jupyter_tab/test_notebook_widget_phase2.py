"""Tests for SidekickNotebookWidget — Phase 2 session integration.

Phase 2 (Tools #2876) RED tests.  These verify that the widget layer
validates paths and stores a ``NotebookSessionModel`` (not a bare dict).
No Qt objects are instantiated here so these run in headless CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[5] / "src" / "shared" / "python"),
)

from sidekick.ui.tools_sidebar.jupyter_tab.notebook_session import (  # noqa: E402
    NotebookSessionModel,
)
from sidekick.ui.tools_sidebar.jupyter_tab.sidekick_notebook_widget import (  # noqa: E402
    SidekickNotebookWidget,
)


@pytest.mark.unit
def test_open_notebook_rejects_traversal(tmp_path: Path) -> None:
    """open_notebook must raise ValueError for paths outside workspace_root."""
    widget = SidekickNotebookWidget.__new__(SidekickNotebookWidget)
    widget._session = None
    widget._workspace_root = tmp_path
    with pytest.raises(ValueError):
        widget.open_notebook(str(tmp_path / ".." / "etc" / "hosts"))


@pytest.mark.unit
def test_open_notebook_sets_session_model(tmp_path: Path) -> None:
    """After open_notebook, _session must be a NotebookSessionModel instance."""
    (tmp_path / "test.ipynb").touch()
    widget = SidekickNotebookWidget.__new__(SidekickNotebookWidget)
    widget._session = None
    widget._workspace_root = tmp_path
    widget.open_notebook(str(tmp_path / "test.ipynb"))
    assert widget._session is not None
    assert isinstance(widget._session, NotebookSessionModel)


@pytest.mark.unit
def test_open_notebook_records_path(tmp_path: Path) -> None:
    """The session stored by open_notebook must point to the opened file."""
    nb = tmp_path / "my_notebook.ipynb"
    nb.touch()
    widget = SidekickNotebookWidget.__new__(SidekickNotebookWidget)
    widget._session = None
    widget._workspace_root = tmp_path
    widget.open_notebook(str(nb))
    assert widget._session.notebook_path == nb


@pytest.mark.unit
def test_set_kernel_environment_updates_session(tmp_path: Path) -> None:
    """set_kernel_environment must update the session's kernel_env field."""
    nb = tmp_path / "nb.ipynb"
    nb.touch()
    widget = SidekickNotebookWidget.__new__(SidekickNotebookWidget)
    widget._session = None
    widget._workspace_root = tmp_path
    widget.open_notebook(str(nb))
    widget.set_kernel_environment("my-venv")
    assert widget._session.kernel_env == "my-venv"


@pytest.mark.unit
def test_set_kernel_environment_without_open_notebook_raises(
    tmp_path: Path,
) -> None:
    """set_kernel_environment without a prior open_notebook must raise RuntimeError."""
    widget = SidekickNotebookWidget.__new__(SidekickNotebookWidget)
    widget._session = None
    widget._workspace_root = tmp_path
    with pytest.raises(RuntimeError):
        widget.set_kernel_environment("venv")


@pytest.mark.unit
def test_session_metadata_property_returns_dict(tmp_path: Path) -> None:
    """session_metadata property must return a dict (Phase 1 compat surface)."""
    nb = tmp_path / "nb.ipynb"
    nb.touch()
    widget = SidekickNotebookWidget.__new__(SidekickNotebookWidget)
    widget._session = None
    widget._workspace_root = tmp_path
    widget.open_notebook(str(nb))
    meta = widget.session_metadata
    assert isinstance(meta, dict)
    assert "notebook_path" in meta

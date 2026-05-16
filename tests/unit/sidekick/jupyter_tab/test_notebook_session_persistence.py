"""Tests for NotebookSessionManager — save/load session persistence.

Phase 2 (Tools #2876) RED tests.  All tests here must fail before
the implementation is in place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[5] / "src" / "shared" / "python"),
)

from sidekick.ui.tools_sidebar.jupyter_tab.notebook_session import (  # noqa: E402
    NotebookSessionManager,
    NotebookSessionModel,
)


@pytest.mark.unit
def test_save_session_writes_json_file(tmp_path: Path) -> None:
    """save_session must create a <sid>.json file in the sessions dir."""
    model = NotebookSessionModel(
        notebook_path=tmp_path / "nb.ipynb",
        workspace_root=tmp_path,
        kernel_env="myenv",
    )
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    sid = mgr.save_session(model)
    assert (tmp_path / ".sessions" / f"{sid}.json").exists()


@pytest.mark.unit
def test_save_session_stores_relative_path_not_absolute(tmp_path: Path) -> None:
    """Session file must store notebook_path relative to workspace_root."""
    model = NotebookSessionModel(
        notebook_path=tmp_path / "notebooks" / "test.ipynb",
        workspace_root=tmp_path,
        kernel_env=None,
    )
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    sid = mgr.save_session(model)
    data = json.loads((tmp_path / ".sessions" / f"{sid}.json").read_text())
    assert data["notebook_path"] == "notebooks/test.ipynb"


@pytest.mark.unit
def test_save_session_does_not_embed_notebook_json(tmp_path: Path) -> None:
    """Notebook JSON content (cells, nbformat) must never appear in the session file."""
    model = NotebookSessionModel(
        notebook_path=tmp_path / "nb.ipynb",
        workspace_root=tmp_path,
        kernel_env=None,
    )
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    sid = mgr.save_session(model)
    data = json.loads((tmp_path / ".sessions" / f"{sid}.json").read_text())
    assert "cells" not in data
    assert "nbformat" not in data


@pytest.mark.unit
def test_load_session_round_trips(tmp_path: Path) -> None:
    """A saved session can be loaded back with the same kernel_env and path."""
    model = NotebookSessionModel(
        notebook_path=tmp_path / "nb.ipynb",
        workspace_root=tmp_path,
        kernel_env="py311",
    )
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    sid = mgr.save_session(model)
    loaded = mgr.load_session(sid, workspace_root=tmp_path)
    assert loaded.kernel_env == "py311"
    assert loaded.notebook_path == tmp_path / "nb.ipynb"


@pytest.mark.unit
def test_load_session_validates_path_on_load(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Loading a session with a different workspace_root must raise ValueError."""
    other_root = tmp_path_factory.mktemp("other")
    model = NotebookSessionModel(
        notebook_path=tmp_path / "nb.ipynb",
        workspace_root=tmp_path,
        kernel_env=None,
    )
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    sid = mgr.save_session(model)
    # Loading with a different workspace_root should fail path validation
    with pytest.raises(ValueError):
        mgr.load_session(sid, workspace_root=other_root)


@pytest.mark.unit
def test_save_session_creates_sessions_dir_if_missing(tmp_path: Path) -> None:
    """NotebookSessionManager must create the sessions directory if absent."""
    sessions_dir = tmp_path / "new" / "sessions"
    assert not sessions_dir.exists()
    model = NotebookSessionModel(
        notebook_path=tmp_path / "nb.ipynb",
        workspace_root=tmp_path,
        kernel_env=None,
    )
    mgr = NotebookSessionManager(sessions_dir=sessions_dir)
    mgr.save_session(model)
    assert sessions_dir.exists()


@pytest.mark.unit
def test_load_nonexistent_session_raises(tmp_path: Path) -> None:
    """Loading a session that does not exist must raise FileNotFoundError."""
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    with pytest.raises(FileNotFoundError):
        mgr.load_session("does-not-exist", workspace_root=tmp_path)


@pytest.mark.unit
def test_save_session_stores_kernel_env(tmp_path: Path) -> None:
    """kernel_env value is present in the persisted JSON."""
    model = NotebookSessionModel(
        notebook_path=tmp_path / "nb.ipynb",
        workspace_root=tmp_path,
        kernel_env="venv312",
    )
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    sid = mgr.save_session(model)
    data = json.loads((tmp_path / ".sessions" / f"{sid}.json").read_text())
    assert data["kernel_env"] == "venv312"


@pytest.mark.unit
def test_save_session_returns_stable_id_for_same_model(tmp_path: Path) -> None:
    """Saving the same model twice returns the same session ID (idempotent)."""
    model = NotebookSessionModel(
        notebook_path=tmp_path / "nb.ipynb",
        workspace_root=tmp_path,
        kernel_env="myenv",
    )
    mgr = NotebookSessionManager(sessions_dir=tmp_path / ".sessions")
    sid1 = mgr.save_session(model)
    sid2 = mgr.save_session(model)
    assert sid1 == sid2

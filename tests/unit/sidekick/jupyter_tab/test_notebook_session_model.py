"""Tests for NotebookSessionModel — path validation (DbC).

These are the RED tests for Phase 2 (Tools #2876).  They must fail
before the implementation is in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the shared package is importable from tests without install
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[5] / "src" / "shared" / "python"),
)

from sidekick.ui.tools_sidebar.jupyter_tab.notebook_session import (  # noqa: E402
    NotebookSessionModel,
)


@pytest.mark.unit
def test_valid_path_within_workspace_root_passes() -> None:
    """A notebook path clearly inside the workspace root must not raise."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/notebooks/test.ipynb"),
        workspace_root=Path("/workspace"),
        kernel_env="venv",
    )
    model.validate_path()  # must not raise


@pytest.mark.unit
def test_path_traversal_outside_root_raises() -> None:
    """A path that resolves outside the root must raise ValueError."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/../etc/passwd"),
        workspace_root=Path("/workspace"),
        kernel_env=None,
    )
    with pytest.raises(ValueError, match="outside workspace root"):
        model.validate_path()


@pytest.mark.unit
def test_path_with_dotdot_resolved_before_validation() -> None:
    """Paths with .. that escape the root after resolution must raise."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/subdir/../../etc/hosts"),
        workspace_root=Path("/workspace"),
        kernel_env=None,
    )
    with pytest.raises(ValueError):
        model.validate_path()


@pytest.mark.unit
def test_session_model_accepts_exact_root() -> None:
    """A notebook directly at the workspace root level is valid."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/a.ipynb"),
        workspace_root=Path("/workspace"),
        kernel_env=None,
    )
    model.validate_path()  # root-level file is fine


@pytest.mark.unit
def test_session_model_stores_kernel_env() -> None:
    """kernel_env is stored and accessible."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/nb.ipynb"),
        workspace_root=Path("/workspace"),
        kernel_env="py311",
    )
    assert model.kernel_env == "py311"


@pytest.mark.unit
def test_session_model_allows_none_kernel_env() -> None:
    """kernel_env may be None (no venv selected)."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/nb.ipynb"),
        workspace_root=Path("/workspace"),
        kernel_env=None,
    )
    assert model.kernel_env is None


@pytest.mark.unit
def test_last_saved_defaults_to_none() -> None:
    """last_saved should default to None if not supplied."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/nb.ipynb"),
        workspace_root=Path("/workspace"),
        kernel_env=None,
    )
    assert model.last_saved is None


@pytest.mark.unit
def test_deep_nested_path_inside_root_passes() -> None:
    """Deeply nested paths inside the root are valid."""
    model = NotebookSessionModel(
        notebook_path=Path("/workspace/a/b/c/d/nb.ipynb"),
        workspace_root=Path("/workspace"),
        kernel_env=None,
    )
    model.validate_path()  # must not raise

"""Regression tests for the movement optimizer maturin workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "maturin-movement-optimizer.yml"


@pytest.mark.unit
def test_movement_optimizer_maturin_reinstalls_pytest_safely() -> None:
    """Self-hosted Python tool caches can contain pytest without RECORD metadata."""
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "--ignore-installed" in content
    assert "--no-cache-dir" in content
    assert "numpy<2" in content
    assert "scipy>=1.10,<1.16" in content
    assert "pytest" in content


@pytest.mark.unit
def test_movement_optimizer_maturin_uses_job_local_virtualenv() -> None:
    """Parity gates must not import stale NumPy/SciPy wheels from runner tool caches."""
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m venv .venv" in content
    assert "$GITHUB_PATH" in content
    assert '"$PYTHON" -m pip install' in content


@pytest.mark.unit
def test_movement_optimizer_maturin_enables_pyo3_python_313_forward_compat() -> None:
    """PyO3 0.21 needs an explicit compatibility override for Python 3.13."""
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "PYO3_USE_ABI3_FORWARD_COMPATIBILITY" in content
    assert '"1"' in content or "'1'" in content

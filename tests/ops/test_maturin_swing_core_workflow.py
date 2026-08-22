"""Regression tests for the swing-core maturin workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "maturin-swing-core.yml"


def _workflow() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


@pytest.mark.unit
def test_swing_core_parity_disables_unrelated_pytest_plugin_autoload() -> None:
    """A Rust-only parity lane must not import GUI plugins from runner caches."""
    workflow = _workflow()
    parity_step = next(
        step
        for step in workflow["jobs"]["parity-gate"]["steps"]
        if step.get("name") == "Run Rust<->Python parity suite (gate, non-skipped)"
    )

    assert parity_step["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


@pytest.mark.unit
def test_swing_core_parity_uses_a_job_local_virtual_environment() -> None:
    """Shared setup-python caches must never supply wheel-test site packages."""
    workflow = _workflow()
    steps = workflow["jobs"]["parity-gate"]["steps"]
    names = [step.get("name") for step in steps]
    create_index = names.index("Create Isolated Python Environment")
    install_index = names.index("Install Build and Test Dependencies")

    assert create_index < install_index
    create = steps[create_index]["run"]
    assert "python -m venv .ci-venv" in create
    assert "GITHUB_PATH" in create
    assert "VIRTUAL_ENV" in create
    install = steps[install_index]["run"]
    assert "sys.prefix != sys.base_prefix" in install

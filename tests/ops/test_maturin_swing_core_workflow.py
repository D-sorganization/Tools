"""Regression tests for the swing-core maturin workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "maturin-swing-core.yml"


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


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

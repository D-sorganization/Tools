"""Epic #6019 registry acceptance: shared describe payloads and conformance (#6020)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("calc_backend")

from calc_backend.registry.store import describe_calculator, registered_calculators

from shared.python.calc_backend import agent_api, mcp_server
from shared.python.calc_backend.agent_api import CONTRACT_VERSION
from shared.python.lifecycle import LifecycleStatus

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_describe_calculator_includes_contract_version() -> None:
    """Registry describe is the schema source; MCP must not patch fields ad hoc."""
    mcp_server._ensure_registry()
    for spec in registered_calculators():
        payload = describe_calculator(spec.id)
        assert payload.get("contract_version") == CONTRACT_VERSION, spec.id


@pytest.mark.unit
def test_succeeded_run_includes_registry_calculation_status() -> None:
    """SUCCEEDED envelopes mirror design-manual calculation_status like FAILED."""
    mcp_server._ensure_registry()
    sample_id = "flow-rate"
    entry = mcp_server.CALCULATOR_REGISTRY[sample_id]
    example = dict(describe_calculator(sample_id)["worked_example"]["request"])
    expected_status = agent_api.calculation_status(entry.spec.formula_ids)
    result = mcp_server._run_calculator(sample_id, example)
    assert result["status"] == LifecycleStatus.SUCCEEDED.name
    provenance = result["provenance"]
    assert provenance["calculation_status"] == expected_status
    assert provenance["contract_version"] == CONTRACT_VERSION
    assert provenance["formula_ids"] == list(entry.spec.formula_ids)


@pytest.mark.unit
def test_fleet_agent_api_conformance_script_exits_zero() -> None:
    """Vendored RM conformance template must pass in CI and pytest (issue #6022)."""
    pytest.importorskip("mcp")
    script = _REPO_ROOT / "scripts" / "run_agent_api_conformance.py"
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""Epic #6019: FAILED MCP envelopes carry registry-backed calculation_status."""

from __future__ import annotations

import pytest

from shared.python.calc_backend import agent_api, mcp_server


@pytest.mark.unit
def test_unknown_calculator_failure_includes_default_calculation_status() -> None:
    mcp_server._ensure_registry()
    result = mcp_server._run_calculator("no-such-calculator", {})
    assert result["status"] == "FAILED"
    provenance = result["provenance"]
    assert provenance["calculation_status"] == agent_api.DEFAULT_CALCULATION_STATUS
    assert provenance["formula_ids"] == []
    assert provenance["contract_version"] == agent_api.CONTRACT_VERSION


@pytest.mark.unit
def test_validation_failure_includes_registry_calculation_status() -> None:
    mcp_server._ensure_registry()
    sample_id = "acid-gas-dewpoint"
    entry = mcp_server.CALCULATOR_REGISTRY[sample_id]
    expected_status = agent_api.calculation_status(entry.spec.formula_ids)
    invalid = {"temperature_c": 60.0, "pressure_bar": -1.0, "method": "antoine"}
    result = mcp_server._run_calculator(sample_id, invalid)
    assert result["status"] == "FAILED"
    provenance = result["provenance"]
    assert provenance["calculation_status"] == expected_status
    assert provenance["formula_ids"] == list(entry.spec.formula_ids)
    assert provenance["calculator_id"] == sample_id

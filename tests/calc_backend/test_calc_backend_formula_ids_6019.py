"""Epic #6019: CalculatorSpec formula_ids from calculation-registry.json (#6020)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("calc_backend")

from calc_backend.registry import store
from calc_backend.registry.formula_linkage import (
    FormulaLinkageError,
    expected_formula_ids,
    load_agent_api_bindings,
)
from calc_backend.registry.store import registered_calculators
from calc_backend.registry.wire import wire_calculator_registry


@pytest.fixture()
def wired_registry() -> None:
    store.CALCULATOR_REGISTRY.clear()
    load_agent_api_bindings.cache_clear()
    wire_calculator_registry()


@pytest.mark.unit
def test_flow_rate_formula_ids_match_calculation_registry(
    wired_registry: None,
) -> None:
    """Wave-1 calculators must expose design-manual ids, not empty tuples."""
    by_id = {spec.id: spec for spec in registered_calculators()}
    spec = by_id["flow-rate"]
    assert spec.formula_ids == expected_formula_ids("flow-rate")
    assert spec.formula_ids


@pytest.mark.unit
def test_every_bound_calculator_exposes_matching_formula_ids(
    wired_registry: None,
) -> None:
    bindings = load_agent_api_bindings()
    by_id = {spec.id: spec for spec in registered_calculators()}
    for calculator_id, expected in bindings.items():
        assert calculator_id in by_id, calculator_id
        assert by_id[calculator_id].formula_ids == expected, calculator_id


@pytest.mark.unit
def test_unbound_calculator_retains_empty_formula_ids(wired_registry: None) -> None:
    bindings = load_agent_api_bindings()
    for spec in registered_calculators():
        if spec.id not in bindings:
            assert spec.formula_ids == (), spec.id


@pytest.mark.unit
def test_unknown_binding_calculator_id_fails_loudly() -> None:
    bindings, registry = _synthetic_payloads()
    with pytest.raises(FormulaLinkageError, match="unknown calculator_id binding"):
        expected_formula_ids(
            "not-registered",
            bindings_payload=bindings,
            registry_payload=registry,
        )


@pytest.mark.unit
def test_unknown_calculation_id_in_binding_fails_loudly() -> None:
    bindings = {
        "schema_version": "tools/agent-api-calculator-bindings/1.0.0",
        "manual_id": "tools",
        "bindings": {"flow-rate": "TOOLS-NO-SUCH-CALC"},
    }
    _, registry = _synthetic_payloads()
    with pytest.raises(FormulaLinkageError, match="unknown calculation_id"):
        expected_formula_ids(
            "flow-rate",
            bindings_payload=bindings,
            registry_payload=registry,
        )


@pytest.mark.unit
def test_malformed_equation_entry_fails_loudly() -> None:
    bindings, registry = _synthetic_payloads()
    rows = registry["calculations"]
    assert isinstance(rows, list)
    rows[0]["equations"] = [{}]
    with pytest.raises(FormulaLinkageError, match="equation missing equation_id"):
        expected_formula_ids(
            "flow-rate",
            bindings_payload=bindings,
            registry_payload=registry,
        )


def _synthetic_payloads() -> tuple[dict[str, object], dict[str, object]]:
    registry: dict[str, object] = {
        "schema_version": "1.0.0",
        "manual_id": "tools",
        "repository": "D-sorganization/Tools",
        "release_status": "provisional",
        "inventory_commit": "916fb9f27f486486f0c071dcc6f15f81f68ecb18",
        "blockers": [],
        "calculations": [
            {
                "calculation_id": "TOOLS-CALC-FLOW-RATE",
                "title": "Flow rate unit conversion",
                "equations": [
                    {"equation_id": "TOOLS-CALC-FLOW-RATE-CONVERT", "description": "x"}
                ],
                "approval": {"state": "blocked-pending-human-approval"},
            }
        ],
    }
    bindings = {
        "schema_version": "tools/agent-api-calculator-bindings/1.0.0",
        "manual_id": "tools",
        "bindings": {"flow-rate": "TOOLS-CALC-FLOW-RATE"},
    }
    return bindings, registry

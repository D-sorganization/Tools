"""Contract tests for the canonical ball-flight result catalog (#4194)."""

from __future__ import annotations

import json

import pytest

from shared.python.swing_sim.flight.result_contract import (
    SCHEMA_VERSION,
    AvailabilityReason,
    ComparabilityStatus,
    FlightMetricCatalog,
    FlightMetricId,
    ValueStatus,
    flight_metric_catalog,
)


def test_catalog_is_complete_deterministic_and_strict() -> None:
    catalog = flight_metric_catalog()

    assert {item.metric_id for item in catalog.definitions} == set(FlightMetricId)
    assert all(
        item.label and item.definition and item.unit for item in catalog.definitions
    )
    assert all(item.frame_id and item.geometry_contract for item in catalog.definitions)
    assert all(len(item.coverage) == 3 for item in catalog.definitions)
    assert catalog.to_json() == flight_metric_catalog().to_json()

    payload = json.loads(catalog.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert FlightMetricCatalog.from_json(payload).to_json() == catalog.to_json()

    payload["unexpected"] = True
    with pytest.raises(ValueError, match="catalog fields"):
        FlightMetricCatalog.from_json(payload)

    payload = json.loads(catalog.to_json())
    payload["definitions"][0]["solver_objective"] = "false"
    with pytest.raises(ValueError, match="solver_objective"):
        FlightMetricCatalog.from_json(payload)


def test_unsupported_comparisons_and_unavailable_values_are_typed() -> None:
    catalog = flight_metric_catalog()
    terminal = catalog.definition(FlightMetricId.TERMINAL_SPEED)
    trackman = next(
        item
        for item in terminal.coverage
        if item.convention_id == "trackman_comparable"
    )
    total = catalog.definition(FlightMetricId.TOTAL_DISTANCE)

    assert trackman.status is ComparabilityStatus.NOT_COMPARABLE
    assert trackman.reason_code == "public_definition_not_established"
    assert total.default_status is ValueStatus.MODEL_DEPENDENT
    assert AvailabilityReason.GROUND_MODEL_REQUIRED.value == "ground_model_required"

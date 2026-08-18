"""Contract and wire-parity tests for scalar-ensemble/v1."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from rate_of_closure.variation.scalar_ensemble_contract import (
    SCALAR_ENSEMBLE_SCHEMA_VERSION,
    ScalarCohortDefinition,
    ScalarEnsembleDataset,
    ScalarEnsembleProvenance,
    ScalarEnsembleRow,
    ScalarEnsembleStage,
    ScalarVariableCategory,
    ScalarVariableDefinition,
    scalar_ensemble_row_id,
)
from shared.python.contracts import ContractViolationError

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe, pytest.mark.contract]


def _dataset() -> ScalarEnsembleDataset:
    variables = (
        ScalarVariableDefinition("speed", "Speed", "m/s", "input", "delivery"),
        ScalarVariableDefinition("carry", "Carry", "m", "result", "shot"),
    )
    rows = (
        ScalarEnsembleRow(
            scalar_ensemble_row_id(0, "baseline"),
            0,
            "complete",
            {"speed": 30.0, "carry": 27.4},
            "baseline",
            {"status": "complete"},
        ),
        ScalarEnsembleRow(
            scalar_ensemble_row_id(1, "baseline"),
            1,
            "failed",
            {"speed": 31.0, "carry": None},
            "baseline",
            {"status": "failed"},
        ),
        ScalarEnsembleRow(
            scalar_ensemble_row_id(2, "baseline"),
            2,
            "complete",
            {"speed": None, "carry": 28.1},
            "baseline",
        ),
        ScalarEnsembleRow(
            scalar_ensemble_row_id(0, "alternative"),
            0,
            "complete",
            {"speed": 29.0, "carry": 26.8},
            "alternative",
        ),
    )
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        "example-ensemble",
        ScalarEnsembleProvenance("test-adapter/v1", "source/v2", "seed=42"),
        (
            ScalarEnsembleStage("input", "Inputs"),
            ScalarEnsembleStage("result", "Results"),
        ),
        (
            ScalarVariableCategory("delivery", "Delivery"),
            ScalarVariableCategory("shot", "Shot"),
        ),
        variables,
        (
            ScalarCohortDefinition("complete", "Completed"),
            ScalarCohortDefinition("failed", "Failed"),
        ),
        rows,
    )


def test_contract_exposes_structured_provenance_definitions_and_nullable_rows() -> None:
    dataset = _dataset()

    assert dataset.schema_version == "scalar-ensemble/v1"
    assert dataset.provenance.adapter_id == "test-adapter/v1"
    assert dataset.stages[0] == ScalarEnsembleStage("input", "Inputs")
    assert dataset.variables[0].unit == "m/s"
    assert dataset.variables[0].category_key == "delivery"
    assert dataset.rows[1].value("carry") is None
    assert isinstance(dataset.rows[0].values, MappingProxyType)


def test_scatter_is_paired_finite_with_exact_overall_and_cohort_availability() -> None:
    scatter = _dataset().scatter("speed", "carry")

    assert [point.row_id for point in scatter.points] == [
        "series:baseline/trial:0",
        "series:alternative/trial:0",
    ]
    assert [(point.x, point.y) for point in scatter.points] == [
        (30.0, 27.4),
        (29.0, 26.8),
    ]
    assert scatter.availability.overall.total_rows == 4
    assert scatter.availability.overall.x_finite == 3
    assert scatter.availability.overall.y_finite == 3
    assert scatter.availability.overall.paired_finite == 2
    assert scatter.availability.overall.unavailable == 2
    complete = scatter.availability.by_cohort["complete"]
    assert (complete.total_rows, complete.x_finite, complete.y_finite) == (3, 2, 3)
    assert (complete.paired_finite, complete.unavailable) == (2, 1)
    assert scatter.availability.by_cohort["failed"].unavailable == 1


def test_canonical_row_id_matches_rfc3986_cross_runtime_fixture() -> None:
    assert scalar_ensemble_row_id(2) == "trial:2"
    assert (
        scalar_ensemble_row_id(7, "wedge/α!*") == "series:wedge%2F%CE%B1%21%2A/trial:7"
    )
    with pytest.raises(ContractViolationError, match="trial_index"):
        scalar_ensemble_row_id(-1)


def test_wire_representation_matches_exact_shared_v1_shape() -> None:
    wire = _dataset().to_wire()

    assert set(wire) == {
        "schema_version",
        "result_id",
        "provenance",
        "stages",
        "categories",
        "variables",
        "cohorts",
        "rows",
    }
    assert wire["provenance"] == {
        "adapter_id": "test-adapter/v1",
        "source_schema_version": "source/v2",
        "source_provenance": "seed=42",
    }
    assert wire["variables"][0] == {
        "key": "speed",
        "label": "Speed",
        "unit": "m/s",
        "stage_key": "input",
        "category_key": "delivery",
    }
    assert wire["rows"][0]["row_id"] == "series:baseline/trial:0"
    assert "attributes" not in wire["rows"][2]


def test_contract_rejects_mismatched_composite_id_and_duplicate_row_id() -> None:
    with pytest.raises(ContractViolationError, match="composite identity"):
        ScalarEnsembleRow("trial:0", 0, "complete", {"speed": 1.0}, "baseline")
    dataset = _dataset()
    with pytest.raises(ContractViolationError, match="row_id values must be unique"):
        ScalarEnsembleDataset(
            dataset.schema_version,
            dataset.result_id,
            dataset.provenance,
            dataset.stages,
            dataset.categories,
            dataset.variables,
            dataset.cohorts,
            dataset.rows + (dataset.rows[0],),
        )


def test_contract_rejects_nonfinite_and_inexact_row_values() -> None:
    with pytest.raises(ContractViolationError, match="finite or null"):
        ScalarEnsembleRow("trial:0", 0, "complete", {"speed": float("nan")})
    dataset = _dataset()
    incomplete = ScalarEnsembleRow(
        "series:new/trial:3", 3, "complete", {"speed": 1.0}, "new"
    )
    with pytest.raises(ContractViolationError, match="exactly the declared"):
        ScalarEnsembleDataset(
            dataset.schema_version,
            dataset.result_id,
            dataset.provenance,
            dataset.stages,
            dataset.categories,
            dataset.variables,
            dataset.cohorts,
            dataset.rows + (incomplete,),
        )


def test_contract_rejects_duplicate_and_unknown_definition_references() -> None:
    dataset = _dataset()
    with pytest.raises(ContractViolationError, match="stage keys must be unique"):
        ScalarEnsembleDataset(
            dataset.schema_version,
            dataset.result_id,
            dataset.provenance,
            dataset.stages + (dataset.stages[0],),
            dataset.categories,
            dataset.variables,
            dataset.cohorts,
            (),
        )
    bad_variable = ScalarVariableDefinition("bad", "Bad", "1", "missing", "shot")
    with pytest.raises(ContractViolationError, match="unknown stage"):
        ScalarEnsembleDataset(
            dataset.schema_version,
            dataset.result_id,
            dataset.provenance,
            dataset.stages,
            dataset.categories,
            (bad_variable,),
            dataset.cohorts,
            (),
        )

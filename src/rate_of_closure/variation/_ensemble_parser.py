"""Strict parser internals for the complete Rate ensemble JSON contract."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    APP_FRAME_ID,
    EVALUATED_HIT,
    NUMERICAL_FAILURE,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
    TrialEvaluationStatus,
)
from shared.python.contracts import require
from shared.python.swing_sim.variation.engine import VariationDataset
from shared.python.swing_sim.variation.ensemble_types import (
    EnsemblePositionTraces,
    require_point_ids,
    validated_sample_times,
)
from shared.python.swing_sim.variation.spec import VariationPlan

from ._ensemble_json_contract import (
    MAX_DECODED_NODES,
    MAX_NESTING_DEPTH,
    MAX_TEXT_CHARS,
    bool_matrix,
    bool_vector,
    exact_integer,
    integer,
    integer_vector,
    json_list,
    mapping,
    number,
    number_vector,
    numeric_matrix,
    optional_string,
    require_fields,
    string_tuple,
    validate_decoded_tree,
)

MAX_TRIALS = 100_000
MAX_SAMPLES = 100_000
MAX_POINTS = 256
MAX_POSITION_CELLS = 5_000_000

_ROOT_FIELDS = {
    "schema_version",
    "coordinate_frame",
    "position_unit",
    "time_unit",
    "point_ids",
    "sample_times_s",
    "sample_valid",
    "impact_sample_indices",
    "positions_m",
    "outcomes",
    "variation",
}
_OUTCOME_FIELDS = {
    "trial_index",
    "status",
    "values",
    "failure_type",
    "failure_message",
}
_VARIATION_FIELDS = {
    "schema_version",
    "plan",
    "input_names",
    "output_names",
    "inputs",
    "outputs",
    "success",
    "elapsed_s",
}
_PLAN_FIELDS = {
    "schema_version",
    "mode",
    "base_variables",
    "noise",
    "n_runs",
    "seed",
    "flight_model",
    "groups",
}
_NOISE_FIELDS = {
    "variable_key",
    "distribution",
    "scale",
    "lower",
    "upper",
    "spec_id",
    "time_window_s",
    "point_ids",
}
_GROUP_FIELDS = {"group_id", "spec_ids", "matrix_kind", "matrix"}


def parse_ensemble_document(
    document: object, schema_version: int
) -> SimulationEnsembleResult:
    """Parse one already-decoded document under exact v1 authority."""
    validate_decoded_tree(document)
    root = mapping(document, "ensemble root")
    require_fields(root, _ROOT_FIELDS, "root fields")
    exact_integer(root["schema_version"], "schema_version", schema_version)
    require(root["coordinate_frame"] == APP_FRAME_ID, "coordinate_frame is unsupported")
    require(root["position_unit"] == "m", "position_unit must be m")
    require(root["time_unit"] == "s", "time_unit must be s")

    variation = _parse_variation(root["variation"])
    outcomes = _parse_outcomes(root["outcomes"], variation)
    traces = _parse_traces(root, variation, outcomes)
    return SimulationEnsembleResult(outcomes, variation, traces)


def _parse_variation(value: object) -> VariationDataset:
    """Parse complete scalar authority and its reproducible plan provenance."""
    data = mapping(value, "variation")
    require_fields(data, _VARIATION_FIELDS, "variation fields")
    exact_integer(data["schema_version"], "variation schema_version", 1)
    plan = _parse_plan(data["plan"])
    require(plan.n_runs <= MAX_TRIALS, "trial limit exceeded", plan.n_runs)

    input_names = string_tuple(data["input_names"], "input_names")
    output_names = string_tuple(data["output_names"], "output_names")
    expected_inputs = tuple(spec.variable_key for spec in plan.noise)
    require(input_names == expected_inputs, "input_names must match plan provenance")
    require(output_names == ALL_OUTPUT_NAMES, "output_names must be canonical")

    inputs = numeric_matrix(data["inputs"], plan.n_runs, len(input_names), False)
    outputs = numeric_matrix(data["outputs"], plan.n_runs, len(output_names), True)
    success = bool_vector(data["success"], plan.n_runs, "success")
    elapsed_s = number(data["elapsed_s"], "elapsed_s")
    require(elapsed_s >= 0.0, "elapsed_s must be non-negative", elapsed_s)
    return VariationDataset(
        plan=plan,
        input_names=input_names,
        inputs=inputs,
        output_names=output_names,
        outputs=outputs,
        success=success,
        elapsed_s=elapsed_s,
    )


def _parse_plan(value: object) -> VariationPlan:
    """Parse only the current lossless plan-v2 representation."""
    data = mapping(value, "variation plan")
    require_fields(data, _PLAN_FIELDS, "variation plan fields")
    exact_integer(data["schema_version"], "plan schema_version", 2)
    integer(data["n_runs"], "n_runs")
    integer(data["seed"], "seed")
    _validate_plan_scalar_types(data)
    noise = json_list(data["noise"], "noise")
    groups = json_list(data["groups"], "groups")
    for entry in noise:
        noise_data = mapping(entry, "noise entry")
        require_fields(noise_data, _NOISE_FIELDS, "noise fields")
        _validate_noise_scalar_types(noise_data)
    for entry in groups:
        group_data = mapping(entry, "group entry")
        require_fields(group_data, _GROUP_FIELDS, "group fields")
        _validate_group_scalar_types(group_data)
    try:
        return VariationPlan.from_json_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        require(False, "variation plan is invalid", str(exc))
        raise AssertionError from exc


def _validate_plan_scalar_types(data: Mapping[str, object]) -> None:
    """Reject coercible strings and booleans in plan scalar authority."""
    require(isinstance(data["mode"], str), "plan mode must be a string")
    require(isinstance(data["flight_model"], str), "flight_model must be a string")
    base = mapping(data["base_variables"], "base_variables")
    for key, value in base.items():
        require(isinstance(key, str), "base variable keys must be strings")
        number(value, "base variable")


def _validate_noise_scalar_types(data: Mapping[str, object]) -> None:
    """Validate the exact plan-v2 noise scalar representation."""
    for name in ("variable_key", "distribution", "spec_id"):
        require(isinstance(data[name], str), f"{name} must be a string")
    number(data["scale"], "noise scale")
    for name in ("lower", "upper"):
        if data[name] is not None:
            number(data[name], name)
    window = data["time_window_s"]
    if window is not None:
        values = json_list(window, "time_window_s")
        require(len(values) == 2, "time_window_s must contain two values")
        for value in values:
            number(value, "time_window_s")
    string_tuple(data["point_ids"], "noise point_ids")


def _validate_group_scalar_types(data: Mapping[str, object]) -> None:
    """Validate stable IDs and a finite numeric group matrix."""
    require(isinstance(data["group_id"], str), "group_id must be a string")
    require(isinstance(data["matrix_kind"], str), "matrix_kind must be a string")
    spec_ids = string_tuple(data["spec_ids"], "group spec_ids")
    rows = json_list(data["matrix"], "group matrix")
    require(len(rows) == len(spec_ids), "group matrix row count is invalid")
    for row in rows:
        values = json_list(row, "group matrix row")
        require(len(values) == len(spec_ids), "group matrix column count is invalid")
        for value in values:
            number(value, "group matrix value")


def _parse_outcomes(
    value: object, variation: VariationDataset
) -> tuple[SimulationTrialOutcome, ...]:
    """Parse typed per-trial outcomes and bind them to scalar matrices."""
    entries = json_list(value, "outcomes")
    require(len(entries) == variation.plan.n_runs, "outcomes must align to trials")
    outcomes: list[SimulationTrialOutcome] = []
    for expected_index, entry in enumerate(entries):
        data = mapping(entry, "outcome")
        require_fields(data, _OUTCOME_FIELDS, "outcome fields")
        trial_index = integer(data["trial_index"], "trial_index")
        require(
            trial_index == expected_index, "outcomes must be in canonical trial order"
        )
        require(isinstance(data["status"], str), "status must be a string")
        try:
            status = TrialEvaluationStatus(data["status"])
        except ValueError as exc:
            require(False, "unknown trial status", data["status"])
            raise AssertionError from exc
        raw_values = mapping(data["values"], "outcome values")
        require_fields(raw_values, set(ALL_OUTPUT_NAMES), "scalar output fields")
        values = {
            name: None if raw_values[name] is None else number(raw_values[name], name)
            for name in ALL_OUTPUT_NAMES
        }
        outcome = SimulationTrialOutcome(
            trial_index=trial_index,
            status=status,
            values=values,
            failure_type=optional_string(data["failure_type"], "failure_type"),
            failure_message=optional_string(data["failure_message"], "failure_message"),
        )
        _require_outcome_scalar_binding(outcome, variation)
        outcomes.append(outcome)
    return tuple(outcomes)


def _require_outcome_scalar_binding(
    outcome: SimulationTrialOutcome, variation: VariationDataset
) -> None:
    """Reject a valid outcome crossed with another scalar dataset."""
    index = outcome.trial_index
    expected_success = outcome.status is not NUMERICAL_FAILURE
    require(
        bool(variation.success[index]) == expected_success,
        "outcome status must match variation success",
    )
    expected = np.array(
        [
            math.nan if outcome.value(name) is None else outcome.value(name)
            for name in ALL_OUTPUT_NAMES
        ],
        dtype=float,
    )
    require(
        bool(np.array_equal(variation.outputs[index], expected, equal_nan=True)),
        "outcome values must match variation outputs",
    )


def _parse_traces(
    root: Mapping[str, object],
    variation: VariationDataset,
    outcomes: tuple[SimulationTrialOutcome, ...],
) -> EnsemblePositionTraces:
    """Parse bounded common-grid geometry and enforce typed availability."""
    point_ids = string_tuple(root["point_ids"], "point_ids")
    require_point_ids(point_ids)
    times = number_vector(root["sample_times_s"], "sample_times_s")
    validated_sample_times(times)
    require(len(point_ids) <= MAX_POINTS, "point limit exceeded", len(point_ids))
    require(times.size <= MAX_SAMPLES, "sample limit exceeded", times.size)
    cell_count = variation.plan.n_runs * times.size * len(point_ids) * 3
    require(
        cell_count <= MAX_POSITION_CELLS, "position cell limit exceeded", cell_count
    )
    valid = bool_matrix(
        root["sample_valid"], variation.plan.n_runs, times.size, "sample_valid"
    )
    impacts = integer_vector(
        root["impact_sample_indices"], variation.plan.n_runs, "impact_sample_indices"
    )
    legal_impacts = (impacts == -1) | ((impacts >= 0) & (impacts < times.size))
    require(bool(np.all(legal_impacts)), "impact sample index is out of range")
    _require_trace_status_binding(outcomes, times, valid, impacts)
    positions = _position_tensor(
        root["positions_m"], variation.plan.n_runs, times.size, len(point_ids), valid
    )
    return EnsemblePositionTraces(
        variation=variation,
        sample_times_s=times,
        coordinate_frame=APP_FRAME_ID,
        point_ids=point_ids,
        positions_m=positions,
        sample_valid=valid,
        impact_sample_indices=impacts,
    )


def _require_trace_status_binding(
    outcomes: tuple[SimulationTrialOutcome, ...],
    times: np.ndarray,
    valid: np.ndarray,
    impacts: np.ndarray,
) -> None:
    """Bind hit/miss/failure status to full trace and impact availability."""
    for index, outcome in enumerate(outcomes):
        if outcome.status is NUMERICAL_FAILURE:
            require(
                not np.any(valid[index]), "numerical failure trace must be unavailable"
            )
            require(impacts[index] == -1, "numerical failure impact marker must be -1")
        else:
            require(np.all(valid[index]), "evaluated trial trace must be complete")
            expected_impact = outcome.status is EVALUATED_HIT
            require(
                bool(impacts[index] >= 0) == expected_impact,
                "impact marker must match typed trial status",
            )
            if expected_impact:
                impact_time = outcome.value("impact_time_s")
                assert impact_time is not None
                expected_index = int(np.argmin(np.abs(times - impact_time)))
                require(
                    impacts[index] == expected_index,
                    "impact marker must match impact-time provenance",
                )


def _position_tensor(
    value: object,
    trials: int,
    samples: int,
    points: int,
    valid: np.ndarray,
) -> np.ndarray:
    """Validate exact tensor axes before allocating the NumPy authority."""
    trial_rows = json_list(value, "positions_m")
    require(len(trial_rows) == trials, "positions_m trial axis is invalid")
    result = np.full((trials, samples, points, 3), np.nan)
    for trial_index, trial_value in enumerate(trial_rows):
        sample_rows = json_list(trial_value, "positions_m trial")
        require(len(sample_rows) == samples, "positions_m sample axis is invalid")
        for sample_index, sample_value in enumerate(sample_rows):
            point_rows = json_list(sample_value, "positions_m sample")
            require(len(point_rows) == points, "positions_m point axis is invalid")
            for point_index, point_value in enumerate(point_rows):
                coordinates = json_list(point_value, "position coordinates")
                require(len(coordinates) == 3, "position must have three coordinates")
                if valid[trial_index, sample_index]:
                    result[trial_index, sample_index, point_index] = [
                        number(component, "position coordinate")
                        for component in coordinates
                    ]
                else:
                    require(
                        all(component is None for component in coordinates),
                        "invalid trace samples must contain null coordinates",
                    )
    return result


__all__ = [
    "MAX_DECODED_NODES",
    "MAX_NESTING_DEPTH",
    "MAX_POINTS",
    "MAX_POSITION_CELLS",
    "MAX_SAMPLES",
    "MAX_TEXT_CHARS",
    "MAX_TRIALS",
    "parse_ensemble_document",
]

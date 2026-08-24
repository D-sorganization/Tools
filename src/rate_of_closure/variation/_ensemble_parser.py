"""Strict parser internals for the complete Rate ensemble JSON contract."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    APP_FRAME_ID,
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
from shared.python.swing_sim.variation.execution_metadata import (
    execution_document_from_json_dict,
)

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
from ._ensemble_limits import (
    MAX_POINTS,
    MAX_POSITION_CELLS,
    MAX_SAMPLES,
    MAX_TRIALS,
    require_ensemble_shape_limits,
)

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
    "plan_document",
    "input_names",
    "output_names",
    "inputs",
    "outputs",
    "success",
    "elapsed_s",
}


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
    outcomes = _parse_outcomes(root["outcomes"], variation.plan.n_runs)
    traces = _parse_traces(root, variation)
    return SimulationEnsembleResult(outcomes, variation, traces)


def _parse_variation(value: object) -> VariationDataset:
    """Parse complete scalar authority and its reproducible plan provenance."""
    data = mapping(value, "variation")
    require_fields(data, _VARIATION_FIELDS, "variation fields")
    exact_integer(data["schema_version"], "variation schema_version", 2)
    plan_document = mapping(data["plan_document"], "variation plan document")
    plan = execution_document_from_json_dict(dict(plan_document)).plan
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


def _parse_outcomes(
    value: object, trial_count: int
) -> tuple[SimulationTrialOutcome, ...]:
    """Parse canonical typed per-trial outcomes."""
    entries = json_list(value, "outcomes")
    require(len(entries) == trial_count, "outcomes must align to trials")
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
        outcomes.append(outcome)
    return tuple(outcomes)


def _parse_traces(
    root: Mapping[str, object],
    variation: VariationDataset,
) -> EnsemblePositionTraces:
    """Parse bounded common-grid geometry."""
    raw_point_ids = json_list(root["point_ids"], "point_ids")
    raw_times = json_list(root["sample_times_s"], "sample_times_s")
    require_ensemble_shape_limits(
        variation.plan.n_runs, len(raw_times), len(raw_point_ids)
    )
    point_ids = string_tuple(raw_point_ids, "point_ids")
    require_point_ids(point_ids)
    times = number_vector(raw_times, "sample_times_s")
    validated_sample_times(times)
    valid = bool_matrix(
        root["sample_valid"], variation.plan.n_runs, times.size, "sample_valid"
    )
    impacts = integer_vector(
        root["impact_sample_indices"], variation.plan.n_runs, "impact_sample_indices"
    )
    legal_impacts = (impacts == -1) | ((impacts >= 0) & (impacts < times.size))
    require(bool(np.all(legal_impacts)), "impact sample index is out of range")
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
    for trial_value in trial_rows:
        sample_rows = json_list(trial_value, "positions_m trial")
        require(len(sample_rows) == samples, "positions_m sample axis is invalid")
        for sample_value in sample_rows:
            point_rows = json_list(sample_value, "positions_m sample")
            require(len(point_rows) == points, "positions_m point axis is invalid")
            for point_value in point_rows:
                coordinates = json_list(point_value, "position coordinates")
                require(len(coordinates) == 3, "position must have three coordinates")
    result = np.full((trials, samples, points, 3), np.nan)
    for trial_index, trial_value in enumerate(trial_rows):
        sample_rows = json_list(trial_value, "positions_m trial")
        for sample_index, sample_value in enumerate(sample_rows):
            point_rows = json_list(sample_value, "positions_m sample")
            for point_index, point_value in enumerate(point_rows):
                coordinates = json_list(point_value, "position coordinates")
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

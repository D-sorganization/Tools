"""Thin Rate adapter from governed response pairs to shared attribution."""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from rate_of_closure.variation.simulation_types import (
    IMPACT_OUTPUT_NAMES,
    SHOT_OUTPUT_NAMES,
    SimulationTrialOutcome,
)
from shared.python.contracts import require
from shared.python.swing_sim.variation._execution_digest import canonical_sha256
from shared.python.swing_sim.variation.noise_response import ResponseFieldInput
from shared.python.swing_sim.variation.noise_response_types import ADEQUACY_ESTIMABLE
from shared.python.swing_sim.variation.paired_attribution import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NONFINITE,
    AttributionPair,
    AttributionRunContext,
    AttributionSource,
    AttributionTarget,
    PairedAttributionInput,
)

RATE_PAIRED_ATTRIBUTION_ADAPTER_ID = "rate-paired-attribution-adapter/v1"
_STATE_METRICS = {"position_x_m": 0, "position_y_m": 1, "position_z_m": 2}
_IMPACT_UNITS = {
    "impact_time_s": "s",
    "clubhead_speed_mps": "m/s",
    "spin_loft_deg": "deg",
    "face_to_path_deg": "deg",
    "spin_axis_tilt_deg": "deg",
}
_SHOT_UNITS = {
    "ball_speed_mph": "mph",
    "launch_angle_deg": "deg",
    "launch_azimuth_deg": "deg",
    "spin_rpm": "rpm",
    "carry_m": "m",
    "lateral_m": "m",
    "max_height_m": "m",
    "flight_time_s": "s",
    "landing_angle_deg": "deg",
}


def rate_attribution_target_registry() -> tuple[tuple[str, str, str], ...]:
    """Return the complete immutable Rate scalar target registry for R13.3."""
    return tuple(
        [(metric_id, "state", "m") for metric_id in _STATE_METRICS]
        + [
            (metric_id, "impact", _IMPACT_UNITS[metric_id])
            for metric_id in IMPACT_OUTPUT_NAMES
        ]
        + [
            (metric_id, "shot", _SHOT_UNITS[metric_id])
            for metric_id in SHOT_OUTPUT_NAMES
        ]
    )


def _validate_outcomes(
    outcomes: tuple[SimulationTrialOutcome, ...], trial_count: int, label: str
) -> tuple[SimulationTrialOutcome, ...]:
    values = tuple(outcomes)
    require(len(values) == trial_count, f"{label} outcome count mismatch")
    require(
        tuple(outcome.trial_index for outcome in values) == tuple(range(trial_count)),
        f"{label} outcome order mismatch",
    )
    return values


def _validate_targets(
    field_input: ResponseFieldInput, targets: tuple[AttributionTarget, ...]
) -> tuple[AttributionTarget, ...]:
    result = tuple(targets)
    require(bool(result), "attribution targets must be nonempty")
    for target in result:
        _validate_target(field_input, target)
    return result


def _validate_target(
    field_input: ResponseFieldInput, target: AttributionTarget
) -> None:
    metric_id = cast(str, target.metric_id)
    if target.kind == "state":
        require(
            metric_id in _STATE_METRICS and target.unit == "m", "invalid state target"
        )
        require(
            target.coordinate_frame == field_input.baseline.traces.coordinate_frame,
            "state target frame mismatch",
        )
        require(
            target.point_id in field_input.baseline.traces.point_ids,
            "state target point mismatch",
        )
        require(target.coordinate_unit == "s", "state target coordinate unit mismatch")
        require(
            bool(
                np.any(
                    field_input.baseline.traces.sample_times_s
                    == target.coordinate_value
                )
            ),
            "state target coordinate is absent from the governed grid",
        )
        return
    registry = _IMPACT_UNITS if target.kind == "impact" else _SHOT_UNITS
    names = IMPACT_OUTPUT_NAMES if target.kind == "impact" else SHOT_OUTPUT_NAMES
    require(
        metric_id in names and registry.get(metric_id) == target.unit,
        "invalid scalar target",
    )


def _source(field_input: ResponseFieldInput) -> AttributionSource:
    spec = field_input.spec
    require(len(spec.point_ids) <= 1, "multi-point source locus is unsupported")
    return AttributionSource(
        source_id=cast(str, spec.spec_id),
        variable_key=spec.variable_key,
        unit=field_input.input_unit,
        point_id=spec.point_ids[0] if spec.point_ids else None,
        time_window_s=spec.time_window_s,
    )


def _context(field_input: ResponseFieldInput) -> AttributionRunContext:
    metadata = field_input.execution_metadata
    grid_payload = {
        "policy_id": field_input.baseline.policy_id,
        "sample_times_s": field_input.baseline.traces.sample_times_s.tolist(),
    }
    return AttributionRunContext(
        model_id=field_input.source_layout_id,
        adapter_id=RATE_PAIRED_ATTRIBUTION_ADAPTER_ID,
        coordinate_frame=field_input.baseline.traces.coordinate_frame,
        trace_grid_sha256=canonical_sha256(grid_payload),
        plan_sha256=metadata.plan_sha256,
        registry_sha256=metadata.registry_sha256,
        execution_sha256=canonical_sha256(metadata.to_json_dict()),
        source_adapter_id=field_input.adapter_id,
    )


def _state_value(
    field_input: ResponseFieldInput,
    target: AttributionTarget,
    trial_index: int,
    *,
    perturbed: bool,
) -> tuple[float, str]:
    traces = field_input.perturbed.traces if perturbed else field_input.baseline.traces
    sample_index = int(
        np.flatnonzero(traces.sample_times_s == target.coordinate_value)[0]
    )
    if not traces.sample_valid[trial_index, sample_index]:
        return math.nan, AVAILABILITY_MISSING
    point_index = traces.point_index(cast(str, target.point_id))
    axis = _STATE_METRICS[cast(str, target.metric_id)]
    value = float(traces.positions_m[trial_index, sample_index, point_index, axis])
    return (
        (value, AVAILABILITY_AVAILABLE)
        if math.isfinite(value)
        else (math.nan, AVAILABILITY_NONFINITE)
    )


def _scalar_value(
    outcome: SimulationTrialOutcome, target: AttributionTarget
) -> tuple[float, str]:
    value = outcome.value(cast(str, target.metric_id))
    if value is None:
        return math.nan, AVAILABILITY_MISSING
    return (
        (value, AVAILABILITY_AVAILABLE)
        if math.isfinite(value)
        else (math.nan, AVAILABILITY_NONFINITE)
    )


def _target_value(
    field_input: ResponseFieldInput,
    outcome: SimulationTrialOutcome,
    target: AttributionTarget,
    trial_index: int,
    *,
    perturbed: bool,
) -> tuple[float, str]:
    if target.kind == "state":
        return _state_value(field_input, target, trial_index, perturbed=perturbed)
    return _scalar_value(outcome, target)


def _pair(
    field_input: ResponseFieldInput,
    targets: tuple[AttributionTarget, ...],
    baseline: SimulationTrialOutcome,
    perturbed: SimulationTrialOutcome,
) -> AttributionPair:
    index = baseline.trial_index
    baseline_cells = tuple(
        _target_value(field_input, baseline, target, index, perturbed=False)
        for target in targets
    )
    perturbed_cells = tuple(
        _target_value(field_input, perturbed, target, index, perturbed=True)
        for target in targets
    )
    names = field_input.input_names
    column = names.index(field_input.spec.variable_key)
    logical_id = field_input.trial_ids[index]
    return AttributionPair(
        pair_id=logical_id,
        baseline_trial_id=f"{field_input.baseline_trial_ids[index]}.baseline",
        perturbed_trial_id=f"{field_input.perturbed_trial_ids[index]}.perturbed",
        baseline_status=baseline.status.value,
        perturbed_status=perturbed.status.value,
        baseline_source_value=float(
            field_input.baseline.traces.variation.inputs[index, column]
        ),
        perturbed_source_value=float(
            field_input.perturbed.traces.variation.inputs[index, column]
        ),
        baseline_values=np.asarray([cell[0] for cell in baseline_cells]),
        perturbed_values=np.asarray([cell[0] for cell in perturbed_cells]),
        baseline_value_states=tuple(cell[1] for cell in baseline_cells),
        perturbed_value_states=tuple(cell[1] for cell in perturbed_cells),
    )


def build_rate_paired_attribution_input(
    field_input: ResponseFieldInput,
    targets: tuple[AttributionTarget, ...],
    baseline_outcomes: tuple[SimulationTrialOutcome, ...],
    perturbed_outcomes: tuple[SimulationTrialOutcome, ...],
) -> PairedAttributionInput:
    """Bind Rate scalar outcomes and R11.3 traces to the shared R13.3 contract."""
    require(isinstance(field_input, ResponseFieldInput), "invalid response field input")
    require(
        field_input.support_status == ADEQUACY_ESTIMABLE,
        "paired attribution requires an independently estimable OAT source design",
        field_input.support_status,
    )
    trial_count = field_input.baseline.traces.n_trials
    validated_targets = _validate_targets(field_input, targets)
    baseline = _validate_outcomes(baseline_outcomes, trial_count, "baseline")
    perturbed = _validate_outcomes(perturbed_outcomes, trial_count, "perturbed")
    context = _context(field_input)
    pairs = tuple(
        _pair(field_input, validated_targets, baseline[index], perturbed[index])
        for index in range(trial_count)
    )
    return PairedAttributionInput(
        _source(field_input),
        validated_targets,
        pairs,
        context,
        context,
        field_input.source_sha256,
    )


__all__ = [
    "RATE_PAIRED_ATTRIBUTION_ADAPTER_ID",
    "build_rate_paired_attribution_input",
    "rate_attribution_target_registry",
]

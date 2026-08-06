"""Adapt complete Rate simulations into miss-safe ensemble records.

This module owns no swing physics. It executes fully validated
:class:`~rate_of_closure.simulation.SimulationConfig` values through the
canonical simulation entry point, then projects their results onto the shared
variation and ensemble-geometry contracts.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from rate_of_closure.simulation import (
    BallSetup,
    BallSupportMode,
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.simulation.pipeline import configured_swing_sample_times
from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    APP_FRAME_ID,
    CONTACT_OUTPUT_NAMES,
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    IMPACT_OUTPUT_NAMES,
    NUMERICAL_FAILURE,
    SHOT_OUTPUT_NAMES,
    SimulationEnsembleRequest,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
    TrialEvaluationStatus,
)
from shared.python.contracts import ContractViolationError, require
from shared.python.swing_sim.solver.solve import (
    CancelledError,
    ProgressCallback,
    ProgressReport,
)
from shared.python.swing_sim.variation.engine import VariationDataset
from shared.python.swing_sim.variation.ensemble_types import EnsemblePositionTraces
from shared.python.swing_sim.variation.registry import CATEGORY_BALL_SETUP
from shared.python.swing_sim.variation.spec import VariationPlan

from .request_builder import build_simulation_ensemble_request

logger = logging.getLogger(__name__)

TEE_HEIGHT_VARIABLE_KEY = f"{CATEGORY_BALL_SETUP}.tee_height_m"

_TRIAL_FAILURES = (
    ValueError,
    ContractViolationError,
    RuntimeError,
    FloatingPointError,
    OverflowError,
)

_POINT_IDS_BY_SOURCE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "manual": ("swing.clubhead.reference",),
        "double_pendulum": (
            "swing.pivot",
            "swing.wrist",
            "swing.clubhead.reference",
        ),
        "triple_pendulum": (
            "swing.pivot",
            "swing.elbow",
            "swing.wrist",
            "swing.clubhead.reference",
        ),
    }
)


@dataclass(frozen=True)
class _TrialCapture:
    """Internal union of a completed run or a numerical failure."""

    run: SimulationRun | None
    error: Exception | None


SimulationExecutor = Callable[[SimulationConfig], SimulationRun]


def apply_ball_setup_sample(
    config: SimulationConfig,
    plan: VariationPlan,
    sampled_row: np.ndarray,
) -> SimulationConfig:
    """Apply the context-specific tee-height value from one sampled row.

    Other sampled variables are intentionally left to their owning adapters.
    This narrow seam prevents the scalar variation evaluator from pretending a
    tee-height perturbation affects its geometry-free impact calculation.
    """
    require(isinstance(config, SimulationConfig), "config must be SimulationConfig")
    require(isinstance(plan, VariationPlan), "plan must be a VariationPlan")
    row = np.asarray(sampled_row, dtype=float)
    require(
        row.shape == (len(plan.noise),),
        "sampled_row must align with plan.noise",
        row.shape,
    )
    require(bool(np.all(np.isfinite(row))), "sampled_row must be finite")
    keys = tuple(spec.variable_key for spec in plan.noise)
    if TEE_HEIGHT_VARIABLE_KEY not in keys:
        return config
    require(
        config.ball_setup.support_mode is BallSupportMode.TEE,
        "tee_height_m variation requires Tee support",
        config.ball_setup.support_mode,
    )
    height_m = float(row[keys.index(TEE_HEIGHT_VARIABLE_KEY)])
    return replace(config, ball_setup=BallSetup(BallSupportMode.TEE, height_m))


def spatial_point_ids(run: SimulationRun) -> tuple[str, ...]:
    """Return explicit spatial point IDs, separate from torque joint IDs.

    The clubhead trace is the reference/GC pose translation stored by the
    canonical simulation, not the face impact point.
    """
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    source_kind = run.config.source_kind
    require(source_kind in _POINT_IDS_BY_SOURCE, "unknown source_kind", source_kind)
    point_ids = _point_ids_for_source(source_kind)
    expected_joint_columns = 0 if source_kind == "manual" else len(point_ids)
    require(
        run.swing_joints.shape[1] == expected_joint_columns,
        "swing joint geometry is incompatible with source_kind",
        run.swing_joints.shape,
    )
    return point_ids


def run_simulation_ensemble(
    request: SimulationEnsembleRequest,
    executor: SimulationExecutor = run_simulation,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> SimulationEnsembleResult:
    """Execute complete configs and retain hits, misses, and failures.

    At least one trial must evaluate so its canonical sample grid can anchor
    failed rows. All evaluated trials must expose the same time grid and point
    IDs; resampling belongs in a later, explicit alignment layer.
    """
    require(
        isinstance(request, SimulationEnsembleRequest),
        "request must be a SimulationEnsembleRequest",
    )
    require(callable(executor), "executor must be callable")
    started = time.monotonic()
    cancellation = cancel_event or threading.Event()
    captures: list[_TrialCapture] = []
    failed = 0
    for index, config in enumerate(request.configs):
        if cancellation.is_set():
            raise CancelledError
        capture = _capture(config, executor)
        captures.append(capture)
        failed += int(capture.run is None)
        if progress_cb is not None:
            progress_cb(
                ProgressReport(
                    iteration=index + 1,
                    cost=float(failed),
                    best_cost=0.0,
                    improvement_pct=0.0,
                    elapsed_s=time.monotonic() - started,
                )
            )
    capture_tuple = tuple(captures)
    reference = next((item.run for item in capture_tuple if item.run is not None), None)
    outcomes = tuple(
        _outcome(index, capture) for index, capture in enumerate(capture_tuple)
    )
    variation = _variation_dataset(request, outcomes, time.monotonic() - started)
    traces = _ensemble_traces(request, variation, capture_tuple, reference)
    return SimulationEnsembleResult(outcomes, variation, traces)


def _capture(config: SimulationConfig, executor: SimulationExecutor) -> _TrialCapture:
    """Execute one trial while retaining ordinary numerical/model failures."""
    try:
        run = executor(config)
    except _TRIAL_FAILURES as error:
        logger.debug("ensemble simulation trial failed: %s", error)
        return _TrialCapture(None, error)
    require(isinstance(run, SimulationRun), "executor must return SimulationRun")
    return _TrialCapture(run, None)


def _outcome(index: int, capture: _TrialCapture) -> SimulationTrialOutcome:
    """Convert one capture into a validated scalar outcome."""
    if capture.run is None:
        assert capture.error is not None
        return SimulationTrialOutcome(
            index,
            NUMERICAL_FAILURE,
            _empty_values(),
            type(capture.error).__name__,
            str(capture.error),
        )
    run = capture.run
    status = EVALUATED_HIT if run.impact_outcome.is_hit else EVALUATED_NO_IMPACT
    values = _contact_values(run)
    if status is EVALUATED_HIT:
        values.update(_impact_values(run))
        values.update(_shot_values(run))
    return SimulationTrialOutcome(index, status, values)


def _empty_values() -> dict[str, float | None]:
    """Return the canonical scalar mapping with every value unavailable."""
    return dict.fromkeys(ALL_OUTPUT_NAMES)


def _contact_values(run: SimulationRun) -> dict[str, float | None]:
    """Return contact quantities available for every evaluated trial."""
    values = _empty_values()
    outcome = run.impact_outcome
    values.update(
        candidate_time_s=outcome.candidate_time_s,
        closest_approach_m=outcome.closest_approach_m,
        contact_margin_m=outcome.contact_margin_m,
    )
    return values


def _impact_values(run: SimulationRun) -> dict[str, float]:
    """Return impact quantities for a hit."""
    delivery = run.delivery
    assert run.impact_time_s is not None and delivery is not None
    return {
        "impact_time_s": run.impact_time_s,
        "clubhead_speed_mps": float(np.linalg.norm(delivery.clubhead_velocity)),
        "spin_loft_deg": delivery.spin_loft_deg,
        "face_to_path_deg": delivery.face_to_path_deg,
        "spin_axis_tilt_deg": delivery.spin_axis_tilt_deg,
    }


def _shot_values(run: SimulationRun) -> dict[str, float]:
    """Return launch/flight quantities for a hit, including lateral landing."""
    launch = run.launch
    assert launch is not None and len(run.flight_positions) > 0
    return {
        "ball_speed_mph": launch["ball_speed_mph"],
        "launch_angle_deg": launch["launch_angle_deg"],
        "launch_azimuth_deg": launch["launch_azimuth_deg"],
        "spin_rpm": launch["spin_rpm"],
        "carry_m": launch["carry_m"],
        "lateral_m": float(run.flight_positions[-1, 2]),
        "max_height_m": launch["max_height_m"],
        "flight_time_s": launch["flight_time_s"],
        "landing_angle_deg": launch["landing_angle_deg"],
    }


def _variation_dataset(
    request: SimulationEnsembleRequest,
    outcomes: tuple[SimulationTrialOutcome, ...],
    elapsed_s: float,
) -> VariationDataset:
    """Build the scalar matrix while preserving sampled inputs."""
    outputs = np.full((request.plan.n_runs, len(ALL_OUTPUT_NAMES)), np.nan)
    success = np.zeros(request.plan.n_runs, dtype=bool)
    for outcome in outcomes:
        outputs[outcome.trial_index] = [
            math.nan if outcome.value(name) is None else outcome.value(name)
            for name in ALL_OUTPUT_NAMES
        ]
        success[outcome.trial_index] = outcome.status is not NUMERICAL_FAILURE
    return VariationDataset(
        plan=request.plan,
        input_names=tuple(spec.variable_key for spec in request.plan.noise),
        inputs=request.sampled_inputs,
        output_names=ALL_OUTPUT_NAMES,
        outputs=outputs,
        success=success,
        elapsed_s=elapsed_s,
    )


def _ensemble_traces(
    request: SimulationEnsembleRequest,
    variation: VariationDataset,
    captures: tuple[_TrialCapture, ...],
    reference: SimulationRun | None,
) -> EnsemblePositionTraces:
    """Build common-grid positions, marking numerical failures invalid."""
    times, point_ids = _trace_layout(request.configs[0], reference)
    positions = np.full((len(captures), len(times), len(point_ids), 3), np.nan)
    valid = np.zeros((len(captures), len(times)), dtype=bool)
    impacts = np.full(len(captures), -1, dtype=int)
    for index, capture in enumerate(captures):
        run = capture.run
        if run is None:
            continue
        if reference is not None:
            _require_common_run(reference, run)
        positions[index] = _spatial_positions(run)
        valid[index] = True
        if run.impact_time_s is not None:
            impacts[index] = int(np.argmin(np.abs(times - run.impact_time_s)))
    return EnsemblePositionTraces(
        variation=variation,
        sample_times_s=times,
        coordinate_frame=APP_FRAME_ID,
        point_ids=point_ids,
        positions_m=positions,
        sample_valid=valid,
        impact_sample_indices=impacts,
    )


def _spatial_positions(run: SimulationRun) -> np.ndarray:
    """Return ``(sample, point, xyz)`` positions in stable point order."""
    if run.config.source_kind == "manual":
        positions: np.ndarray = np.asarray(run.swing_positions, dtype=float)
        return positions[:, np.newaxis, :]
    require(
        bool(np.allclose(run.swing_joints[:, -1], run.swing_positions, atol=1e-9)),
        "last spatial point must be the clubhead reference trajectory",
    )
    joints: np.ndarray = np.asarray(run.swing_joints, dtype=float)
    return joints


def _trace_layout(
    config: SimulationConfig, reference: SimulationRun | None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the canonical grid/IDs even when every execution failed."""
    if reference is not None:
        return reference.swing_times, spatial_point_ids(reference)
    return configured_swing_sample_times(config), _point_ids_for_source(
        config.source_kind
    )


def _point_ids_for_source(source_kind: str) -> tuple[str, ...]:
    """Return the stable spatial schema for a validated source kind."""
    require(source_kind in _POINT_IDS_BY_SOURCE, "unknown source_kind", source_kind)
    return _POINT_IDS_BY_SOURCE[source_kind]


def _require_common_run(reference: SimulationRun, run: SimulationRun) -> None:
    """Require exact common time and stable-point coordinates."""
    require(
        np.array_equal(run.swing_times, reference.swing_times),
        "evaluated runs must share one sample-time grid",
    )
    require(
        spatial_point_ids(run) == spatial_point_ids(reference),
        "evaluated runs must share stable spatial point IDs",
    )


__all__ = [
    "APP_FRAME_ID",
    "CONTACT_OUTPUT_NAMES",
    "EVALUATED_HIT",
    "EVALUATED_NO_IMPACT",
    "IMPACT_OUTPUT_NAMES",
    "NUMERICAL_FAILURE",
    "SHOT_OUTPUT_NAMES",
    "SimulationEnsembleRequest",
    "SimulationEnsembleResult",
    "SimulationTrialOutcome",
    "TrialEvaluationStatus",
    "TEE_HEIGHT_VARIABLE_KEY",
    "apply_ball_setup_sample",
    "build_simulation_ensemble_request",
    "run_simulation_ensemble",
    "spatial_point_ids",
]

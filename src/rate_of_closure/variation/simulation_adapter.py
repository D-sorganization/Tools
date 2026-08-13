"""Adapt complete Rate simulations into miss-safe ensemble records.

This module owns no swing physics. It executes fully validated
:class:`~rate_of_closure.simulation.SimulationConfig` values through the
canonical simulation entry point, then projects their results onto the shared
variation and ensemble-geometry contracts.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from rate_of_closure.simulation import (
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.simulation.pipeline import configured_swing_sample_times
from rate_of_closure.variation.simulation_types import (
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
from shared.python.contracts import require
from shared.python.swing_sim.solver.solve import (
    CancelledError,
    ProgressCallback,
    ProgressReport,
)
from shared.python.swing_sim.variation.registry import CATEGORY_BALL_SETUP
from shared.python.swing_sim.variation.spec import VariationPlan

from .ensemble_chunks import (
    MAX_CHUNK_POSITION_CELLS,
    CollectingEnsembleSink,
    EnsembleChunkSink,
    EnsembleStreamHeader,
    SimulationResultChunk,
)
from .request_builder import (
    apply_global_simulation_values,
    build_simulation_ensemble_request,
)
from .trial_projection import (
    SimulationExecutor,
    TrialCapture,
    capture_simulation,
    project_simulation_outcome,
)

TEE_HEIGHT_VARIABLE_KEY = f"{CATEGORY_BALL_SETUP}.tee_height_m"

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
    height_m = float(row[keys.index(TEE_HEIGHT_VARIABLE_KEY)])
    return apply_global_simulation_values(config, {TEE_HEIGHT_VARIABLE_KEY: height_m})


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
    """Materialize the legacy result through the bounded chunk executor."""
    return run_simulation_ensemble_chunks(
        request,
        CollectingEnsembleSink(),
        chunk_size=1,
        executor=executor,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )


TChunkResult = TypeVar("TChunkResult")


def run_simulation_ensemble_chunks(
    request: SimulationEnsembleRequest,
    sink: EnsembleChunkSink[TChunkResult],
    *,
    chunk_size: int | None = None,
    executor: SimulationExecutor = run_simulation,
    progress_cb: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> TChunkResult:
    """Execute bounded canonical chunks into one coordinator-owned sink.

    ``accept`` is provisional. Only ``commit`` returns a valid result; any
    cancellation or exception aborts the sink exactly once before propagating.
    """
    require(
        isinstance(request, SimulationEnsembleRequest),
        "request must be a SimulationEnsembleRequest",
    )
    require(callable(executor), "executor must be callable")
    started = time.monotonic()
    cancellation = cancel_event or threading.Event()
    times, point_ids = _trace_layout(request.configs[0], None)
    header = EnsembleStreamHeader(
        request.plan,
        request.sampled_inputs,
        times,
        point_ids,
        APP_FRAME_ID,
        request.execution_metadata,
    )
    cells_per_trial = times.size * len(point_ids) * 3
    maximum_rows = MAX_CHUNK_POSITION_CELLS // cells_per_trial
    require(maximum_rows > 0, "one trace row exceeds the chunk cell limit")
    rows_per_chunk = maximum_rows if chunk_size is None else chunk_size
    require(
        isinstance(rows_per_chunk, int)
        and not isinstance(rows_per_chunk, bool)
        and 0 < rows_per_chunk <= maximum_rows,
        "chunk_size exceeds the bounded trace capacity",
        rows_per_chunk,
    )
    failed = 0
    try:
        sink.begin(header)
        for start in range(0, request.plan.n_runs, rows_per_chunk):
            stop = min(start + rows_per_chunk, request.plan.n_runs)
            captures: list[TrialCapture] = []
            for config in request.configs[start:stop]:
                if cancellation.is_set():
                    raise CancelledError
                capture = capture_simulation(config, executor)
                if cancellation.is_set():
                    raise CancelledError
                captures.append(capture)
                failed += int(capture.run is None)
            chunk = _result_chunk(request, header, start, tuple(captures))
            if cancellation.is_set():
                raise CancelledError
            sink.accept(chunk)
            if progress_cb is not None:
                progress_cb(
                    ProgressReport(
                        iteration=stop,
                        cost=float(failed),
                        best_cost=0.0,
                        improvement_pct=0.0,
                        elapsed_s=time.monotonic() - started,
                    )
                )
        if cancellation.is_set():
            raise CancelledError
        return sink.commit(time.monotonic() - started)
    except BaseException as primary:
        try:
            sink.abort()
        except BaseException as abort_error:
            primary.add_note(f"sink abort also failed: {abort_error!r}")
        raise


def _result_chunk(
    request: SimulationEnsembleRequest,
    header: EnsembleStreamHeader,
    start_index: int,
    captures: tuple[TrialCapture, ...],
) -> SimulationResultChunk:
    """Project and release one bounded set of complete simulation captures."""
    times = header.sample_times_s
    point_ids = header.point_ids
    positions = np.full((len(captures), len(times), len(point_ids), 3), np.nan)
    valid: np.ndarray = np.zeros((len(captures), len(times)), dtype=bool)
    impacts: np.ndarray = np.full(len(captures), -1, dtype=int)
    outcomes = tuple(
        project_simulation_outcome(start_index + offset, capture)
        for offset, capture in enumerate(captures)
    )
    for offset, capture in enumerate(captures):
        run = capture.run
        if run is None:
            continue
        _require_header_run(header, run)
        positions[offset] = _spatial_positions(run)
        valid[offset] = True
        if run.impact_time_s is not None:
            impacts[offset] = int(np.argmin(np.abs(times - run.impact_time_s)))
    stop = start_index + len(captures)
    return SimulationResultChunk(
        start_index=start_index,
        sampled_inputs=request.sampled_inputs[start_index:stop],
        outcomes=outcomes,
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


def _require_header_run(header: EnsembleStreamHeader, run: SimulationRun) -> None:
    """Require exact announced time and stable-point coordinates."""
    require(
        np.array_equal(run.swing_times, header.sample_times_s),
        "evaluated runs must share one sample-time grid",
    )
    require(
        spatial_point_ids(run) == header.point_ids,
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
    "run_simulation_ensemble_chunks",
    "spatial_point_ids",
]

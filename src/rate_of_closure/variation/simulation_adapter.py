"""Adapt complete Rate simulations into miss-safe ensemble records.

This module owns no swing physics. It executes fully validated
:class:`~rate_of_closure.simulation.SimulationConfig` values through the
canonical simulation entry point, then projects their results onto the shared
variation and ensemble-geometry contracts.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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

from ._simulation_config_identity import simulation_configuration_stream_sha256
from .ensemble_chunks import (
    MAX_CHUNK_POSITION_CELLS,
    CollectingEnsembleSink,
    EnsembleChunkSink,
    EnsembleResumeState,
    EnsembleStreamHeader,
    ResumableEnsembleChunkSink,
    SimulationResultChunk,
)
from .ensemble_source import (
    EnsembleWorkChunk,
    SimulationEnsembleSource,
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
    request: SimulationEnsembleSource,
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


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    source: SimulationEnsembleSource
    header: EnsembleStreamHeader
    rows_per_chunk: int
    executor: SimulationExecutor
    progress_cb: ProgressCallback | None
    cancellation: threading.Event
    started: float


def build_ensemble_stream_header(
    request: SimulationEnsembleSource,
) -> EnsembleStreamHeader:
    """Build the immutable scientific identity announced to chunk sinks."""
    require(
        isinstance(request, SimulationEnsembleSource),
        "request must be a SimulationEnsembleSource",
    )
    times, point_ids = _trace_layout(request.reference_config(), None)
    input_sha256, configuration_sha256 = _source_identity(request)
    materialized_inputs = (
        request.sampled_inputs
        if isinstance(request, SimulationEnsembleRequest)
        else None
    )
    return EnsembleStreamHeader(
        request.plan,
        materialized_inputs,
        times,
        point_ids,
        APP_FRAME_ID,
        configuration_sha256,
        input_sha256,
    )


def run_simulation_ensemble_chunks(
    request: SimulationEnsembleSource,
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
        isinstance(request, SimulationEnsembleSource),
        "request must be a SimulationEnsembleSource",
    )
    require(callable(executor), "executor must be callable")
    started = time.monotonic()
    cancellation = cancel_event or threading.Event()
    header = build_ensemble_stream_header(request)
    rows_per_chunk = _bounded_rows_per_chunk(header, chunk_size)
    context = _ExecutionContext(
        request, header, rows_per_chunk, executor, progress_cb, cancellation, started
    )
    return _execute_stream(context, sink)


def _bounded_rows_per_chunk(header: EnsembleStreamHeader, requested: int | None) -> int:
    cells_per_trial = header.sample_times_s.size * len(header.point_ids) * 3
    maximum_rows = MAX_CHUNK_POSITION_CELLS // cells_per_trial
    require(maximum_rows > 0, "one trace row exceeds the chunk cell limit")
    rows = maximum_rows if requested is None else requested
    require(
        isinstance(rows, int)
        and not isinstance(rows, bool)
        and 0 < rows <= maximum_rows,
        "chunk_size exceeds the bounded trace capacity",
        rows,
    )
    return rows


def _execute_stream(
    context: _ExecutionContext, sink: EnsembleChunkSink[TChunkResult]
) -> TChunkResult:
    """Own one sink lifecycle around bounded production."""
    try:
        sink.begin(context.header)
        resume = _resume_state(sink, context.source.plan.n_runs)
        if context.progress_cb is not None and resume.next_index > 0:
            _report_progress(
                context.progress_cb,
                resume.next_index,
                resume.failed_count,
                context.started,
            )
        _produce_chunks(context, sink, resume)
        if context.cancellation.is_set():
            raise CancelledError
        return sink.commit(time.monotonic() - context.started)
    except BaseException as primary:
        try:
            sink.abort()
        except BaseException as abort_error:
            primary.add_note(f"sink abort also failed: {abort_error!r}")
        raise


def _produce_chunks(
    context: _ExecutionContext,
    sink: EnsembleChunkSink[object],
    resume: EnsembleResumeState,
) -> None:
    """Evaluate and commit each bounded work block after the durable prefix."""
    failed = resume.failed_count
    for work in context.source.work_chunks(
        chunk_size=context.rows_per_chunk, start_index=resume.next_index
    ):
        captures: list[TrialCapture] = []
        for config in work.configs:
            if context.cancellation.is_set():
                raise CancelledError
            capture = capture_simulation(config, context.executor)
            if context.cancellation.is_set():
                raise CancelledError
            captures.append(capture)
            failed += int(capture.run is None)
        chunk = _result_chunk(work, context.header, tuple(captures))
        if context.cancellation.is_set():
            raise CancelledError
        sink.accept(chunk)
        if context.progress_cb is not None:
            _report_progress(
                context.progress_cb,
                work.start_index + len(work.configs),
                failed,
                context.started,
            )


def _report_progress(
    callback: ProgressCallback, iteration: int, failed: int, started: float
) -> None:
    callback(
        ProgressReport(
            iteration=iteration,
            cost=float(failed),
            best_cost=0.0,
            improvement_pct=0.0,
            elapsed_s=time.monotonic() - started,
        )
    )


def _resume_state(sink: object, trial_count: int) -> EnsembleResumeState:
    """Return and bound an optional verified prefix after ``begin``."""
    state = (
        sink.resume_state()
        if isinstance(sink, ResumableEnsembleChunkSink)
        else EnsembleResumeState(0, 0)
    )
    require(
        state.next_index <= trial_count,
        "resume prefix exceeds the declared trial count",
    )
    return state


def _result_chunk(
    work: EnsembleWorkChunk,
    header: EnsembleStreamHeader,
    captures: tuple[TrialCapture, ...],
) -> SimulationResultChunk:
    """Project and release one bounded set of complete simulation captures."""
    times = header.sample_times_s
    point_ids = header.point_ids
    positions = np.full((len(captures), len(times), len(point_ids), 3), np.nan)
    valid: np.ndarray = np.zeros((len(captures), len(times)), dtype=bool)
    impacts: np.ndarray = np.full(len(captures), -1, dtype=int)
    outcomes = tuple(
        project_simulation_outcome(work.start_index + offset, capture)
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
    return SimulationResultChunk(
        start_index=work.start_index,
        sampled_inputs=work.sampled_inputs,
        outcomes=outcomes,
        positions_m=positions,
        sample_valid=valid,
        impact_sample_indices=impacts,
    )


def _source_identity(source: SimulationEnsembleSource) -> tuple[str, str]:
    """Hash inputs and configurations through one bounded deterministic scan."""
    input_digest = hashlib.sha256()
    next_index = 0

    def configurations() -> Iterator[object]:
        nonlocal next_index
        chunk_size = min(256, source.plan.n_runs)
        for work in source.work_chunks(chunk_size=chunk_size, start_index=0):
            require(
                work.start_index == next_index,
                "source identity stream is not contiguous",
            )
            input_digest.update(work.sampled_inputs.tobytes(order="C"))
            next_index += len(work.configs)
            yield from work.configs

    configuration_sha256 = simulation_configuration_stream_sha256(
        configurations(), count=source.plan.n_runs
    )
    require(next_index == source.plan.n_runs, "source identity stream is incomplete")
    return input_digest.hexdigest(), configuration_sha256


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
    "build_ensemble_stream_header",
    "run_simulation_ensemble",
    "run_simulation_ensemble_chunks",
    "spatial_point_ids",
]

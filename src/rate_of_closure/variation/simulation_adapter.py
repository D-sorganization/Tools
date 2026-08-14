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
from dataclasses import replace
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from rate_of_closure.simulation import (
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.simulation.pipeline import configured_swing_sample_times
from rate_of_closure.simulation.sources import (
    commanded_torque_joint_ids,
    generalized_state_layout,
)
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
from shared.python.swing_sim.variation.execution_metadata import (
    PYTHON_TEST_INJECTED_IMPLEMENTATION_IDENTITY,
)
from shared.python.swing_sim.variation.registry import CATEGORY_BALL_SETUP
from shared.python.swing_sim.variation.spec import VariationPlan

from ._ensemble_limits import MAX_CHUNK_AUTHORITY_BYTES
from .ensemble_archive_contracts import EnsembleResumeCursor
from .ensemble_chunk_builder import ChunkAccumulator
from .ensemble_chunks import (
    MAX_CHUNK_POSITION_CELLS,
    CollectingEnsembleSink,
    EnsembleChunkSink,
    EnsembleStreamHeader,
)
from .ensemble_request_identity import request_identity_sha256
from .ensemble_trace_authority import EnsembleAuthorityLayout
from .request_builder import (
    apply_global_simulation_values,
    build_simulation_ensemble_request,
)
from .trial_projection import (
    SimulationExecutor,
    capture_simulation,
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
    point_ids = spatial_point_ids_for_source(source_kind)
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
    state_ids, state_units = generalized_state_layout(request.configs[0].source_kind)
    authority_layout = EnsembleAuthorityLayout(
        state_ids,
        state_units,
        commanded_torque_joint_ids(request.configs[0].source_kind),
    )
    execution_metadata = request.execution_metadata
    if executor is not run_simulation and execution_metadata is not None:
        # An injected executor is never allowed to be reported as the pinned
        # production implementation. Requests with no metadata (explicit design
        # matrices) stay metadata-free rather than gaining a fabricated one.
        execution_metadata = replace(
            execution_metadata,
            implementation_identity=PYTHON_TEST_INJECTED_IMPLEMENTATION_IDENTITY,
        )
    header = EnsembleStreamHeader(
        request.plan,
        request.sampled_inputs,
        times,
        point_ids,
        APP_FRAME_ID,
        authority_layout,
        request_identity_sha256(request),
        execution_metadata,
    )
    cells_per_trial = times.size * len(point_ids) * 3
    authority_bytes_per_trial = times.size * (
        (16 + 6 + len(state_ids) + len(authority_layout.torque_joint_ids)) * 8 + 1
    )
    maximum_rows = min(
        MAX_CHUNK_POSITION_CELLS // cells_per_trial,
        MAX_CHUNK_AUTHORITY_BYTES // authority_bytes_per_trial,
    )
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
        resume = sink.begin(header)
        require(
            resume is None or isinstance(resume, EnsembleResumeCursor),
            "sink begin must return EnsembleResumeCursor or None",
        )
        next_index = 0 if resume is None else resume.next_trial_index
        failed = 0 if resume is None else resume.failure_count
        require(next_index <= request.plan.n_runs, "resume cursor exceeds trial count")
        require(failed <= next_index, "resume failure count exceeds completed trials")
        if resume is not None and next_index > 0 and progress_cb is not None:
            progress_cb(
                ProgressReport(
                    iteration=next_index,
                    cost=float(failed),
                    best_cost=0.0,
                    improvement_pct=0.0,
                    elapsed_s=time.monotonic() - started,
                )
            )
        for start in range(next_index, request.plan.n_runs, rows_per_chunk):
            stop = min(start + rows_per_chunk, request.plan.n_runs)
            accumulator = ChunkAccumulator(request, header, start, stop)
            for config in request.configs[start:stop]:
                if cancellation.is_set():
                    raise CancelledError
                capture = capture_simulation(config, executor)
                if cancellation.is_set():
                    raise CancelledError
                accumulator.append(capture)
            failed += accumulator.failure_count
            chunk = accumulator.finish()
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


def _trace_layout(
    config: SimulationConfig, reference: SimulationRun | None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the canonical grid/IDs even when every execution failed."""
    if reference is not None:
        return reference.swing_times, spatial_point_ids(reference)
    return configured_swing_sample_times(config), spatial_point_ids_for_source(
        config.source_kind
    )


def spatial_point_ids_for_source(source_kind: str) -> tuple[str, ...]:
    """Return the stable spatial schema for a validated source kind."""
    require(source_kind in _POINT_IDS_BY_SOURCE, "unknown source_kind", source_kind)
    return _POINT_IDS_BY_SOURCE[source_kind]


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
    "spatial_point_ids_for_source",
]

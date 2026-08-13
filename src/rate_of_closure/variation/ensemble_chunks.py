"""Bounded in-process chunk contracts for complete Rate ensembles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, cast

import numpy as np

from shared.python.contracts import require
from shared.python.swing_sim.variation.engine import VariationDataset
from shared.python.swing_sim.variation.ensemble_types import (
    EnsemblePositionTraces,
    require_coordinate_frame_id,
    require_point_ids,
)
from shared.python.swing_sim.variation.spec import VariationPlan

from ._ensemble_limits import MAX_INPUT_CELLS, require_ensemble_shape_limits
from .ensemble_archive_contracts import EnsembleResumeCursor
from .ensemble_trace_authority import (
    ChunkTraceAuthority,
    EnsembleAuthorityLayout,
    require_chunk_authority,
)
from .simulation_types import (
    ALL_OUTPUT_NAMES,
    APP_FRAME_ID,
    EVALUATED_HIT,
    NUMERICAL_FAILURE,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
)

MAX_CHUNK_POSITION_CELLS = 500_000


def _owned_array(value: object, dtype: Any) -> np.ndarray:
    result: np.ndarray = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _require_real_numeric_array(value: np.ndarray, name: str) -> None:
    require(
        np.issubdtype(value.dtype, np.number)
        and not np.issubdtype(value.dtype, np.bool_)
        and not np.issubdtype(value.dtype, np.complexfloating),
        f"{name} must contain real non-boolean numbers",
    )


@dataclass(frozen=True)
class EnsembleStreamHeader:
    """Immutable layout announced before the first result chunk."""

    plan: VariationPlan
    sampled_inputs: np.ndarray = field(repr=False)
    sample_times_s: np.ndarray = field(repr=False)
    point_ids: tuple[str, ...]
    coordinate_frame: str
    authority_layout: EnsembleAuthorityLayout | None = None
    request_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        require(isinstance(self.plan, VariationPlan), "plan must be a VariationPlan")
        raw_inputs = np.asarray(self.sampled_inputs)
        raw_times = np.asarray(self.sample_times_s)
        require(
            raw_inputs.shape == (self.plan.n_runs, len(self.plan.noise)),
            "sampled_inputs must match the declared plan",
        )
        require(
            raw_inputs.size <= MAX_INPUT_CELLS,
            "sampled input cell limit exceeded",
            raw_inputs.size,
        )
        _require_real_numeric_array(raw_inputs, "sampled_inputs")
        require(
            raw_times.ndim == 1 and raw_times.size > 0,
            "sample_times_s must be 1-D",
        )
        _require_real_numeric_array(raw_times, "sample_times_s")
        points = tuple(self.point_ids)
        require_point_ids(points)
        require_coordinate_frame_id(self.coordinate_frame)
        require(
            self.coordinate_frame == APP_FRAME_ID,
            "coordinate_frame is unsupported",
        )
        require_ensemble_shape_limits(self.plan.n_runs, raw_times.size, len(points))
        inputs = _owned_array(raw_inputs, float)
        times = _owned_array(raw_times, float)
        require(bool(np.all(np.isfinite(inputs))), "sampled_inputs must be finite")
        require(bool(np.all(np.isfinite(times))), "sample_times_s must be finite")
        require(bool(np.all(np.diff(times) > 0)), "sample_times_s must increase")
        object.__setattr__(self, "sampled_inputs", inputs)
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "point_ids", points)
        require(
            self.authority_layout is None
            or isinstance(self.authority_layout, EnsembleAuthorityLayout),
            "authority_layout must be an EnsembleAuthorityLayout",
        )
        require(
            self.request_identity_sha256 is None
            or re.fullmatch(r"[0-9a-f]{64}", self.request_identity_sha256) is not None,
            "request_identity_sha256 must be lowercase SHA-256",
        )


@dataclass(frozen=True)
class SimulationResultChunk:
    """One contiguous immutable prefix-independent result block."""

    start_index: int
    sampled_inputs: np.ndarray = field(repr=False)
    outcomes: tuple[SimulationTrialOutcome, ...]
    positions_m: np.ndarray = field(repr=False)
    sample_valid: np.ndarray = field(repr=False)
    impact_sample_indices: np.ndarray = field(repr=False)
    authority: ChunkTraceAuthority | None = None

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        rows = len(outcomes)
        require(
            type(self.start_index) is int and self.start_index >= 0,
            "start_index must be a non-negative integer",
        )
        require(rows > 0, "result chunks must be non-empty")
        require(
            tuple(item.trial_index for item in outcomes)
            == tuple(range(self.start_index, self.start_index + rows)),
            "chunk outcomes must be contiguous and canonical",
        )
        raw_inputs = np.asarray(self.sampled_inputs)
        raw_positions = np.asarray(self.positions_m)
        raw_valid = np.asarray(self.sample_valid)
        raw_impacts = np.asarray(self.impact_sample_indices)
        require(
            raw_inputs.ndim == 2 and raw_inputs.shape[0] == rows,
            "invalid chunk inputs",
        )
        require(
            raw_inputs.size <= MAX_INPUT_CELLS,
            "chunk input cell limit exceeded",
            raw_inputs.size,
        )
        _require_real_numeric_array(raw_inputs, "chunk inputs")
        require(
            raw_positions.ndim == 4 and raw_positions.shape[0] == rows,
            "invalid positions",
        )
        require(raw_positions.shape[3] == 3, "positions require xyz coordinates")
        _require_real_numeric_array(raw_positions, "positions")
        require(
            raw_positions.size <= MAX_CHUNK_POSITION_CELLS,
            "chunk position cell limit exceeded",
            raw_positions.size,
        )
        samples = raw_positions.shape[1]
        require(raw_valid.shape == (rows, samples), "invalid sample_valid shape")
        require(raw_impacts.shape == (rows,), "invalid impact index shape")
        require(
            raw_valid.dtype == np.dtype(bool),
            "sample_valid must contain genuine boolean values",
        )
        require(
            np.issubdtype(raw_impacts.dtype, np.integer)
            and raw_impacts.dtype != np.dtype(bool),
            "impact indices must contain genuine integer values",
        )
        impact_bounds = np.iinfo(np.int64)
        require(
            bool(np.all(raw_impacts <= impact_bounds.max))
            and bool(np.all(raw_impacts >= impact_bounds.min)),
            "impact indices exceed the supported integer range",
        )
        inputs = _owned_array(raw_inputs, float)
        positions = _owned_array(raw_positions, float)
        valid = _owned_array(raw_valid, bool)
        impacts = _owned_array(raw_impacts, int)
        require(bool(np.all(np.isfinite(inputs))), "chunk inputs must be finite")
        require(
            bool(np.all((impacts >= -1) & (impacts < samples))),
            "impact indices are outside the chunk sample grid",
        )
        require(
            bool(np.all(np.isfinite(positions) | np.isnan(positions))),
            "positions must be finite or unavailable NaN",
        )
        for row, outcome in enumerate(outcomes):
            if outcome.status is NUMERICAL_FAILURE:
                require(
                    not bool(np.any(valid[row]))
                    and bool(np.all(np.isnan(positions[row])))
                    and impacts[row] == -1,
                    "numerical failure chunk trace must be unavailable",
                )
                continue
            require(bool(np.any(valid[row])), "evaluated chunk trace must be available")
            require(
                bool(np.all(np.isfinite(positions[row][valid[row]])))
                and bool(np.all(np.isnan(positions[row][~valid[row]]))),
                "chunk positions must agree with sample validity",
            )
            require(
                (impacts[row] >= 0) == (outcome.status is EVALUATED_HIT),
                "chunk impact marker must match typed outcome",
            )
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "sampled_inputs", inputs)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "sample_valid", valid)
        object.__setattr__(self, "impact_sample_indices", impacts)
        require(
            self.authority is None or isinstance(self.authority, ChunkTraceAuthority),
            "authority must be ChunkTraceAuthority when supplied",
        )


TCommit_co = TypeVar("TCommit_co", covariant=True)


class EnsembleChunkSink(Protocol[TCommit_co]):
    """Coordinator-thread lifecycle for provisional result chunks."""

    def begin(self, header: EnsembleStreamHeader) -> EnsembleResumeCursor | None: ...

    def accept(self, chunk: SimulationResultChunk) -> None: ...

    def commit(self, elapsed_s: float) -> TCommit_co: ...

    def abort(self) -> None: ...


def require_chunk_matches_header(
    header: EnsembleStreamHeader, chunk: SimulationResultChunk, next_index: int
) -> None:
    """Bind a provisional chunk to the announced layout and canonical prefix."""
    require(chunk.start_index == next_index, "chunk stream contains a gap or overlap")
    require(
        chunk.start_index + len(chunk.outcomes) <= header.plan.n_runs,
        "chunk exceeds declared trial count",
    )
    require(
        chunk.sampled_inputs.shape[1] == len(header.plan.noise),
        "chunk input columns do not match plan",
    )
    stop = chunk.start_index + len(chunk.outcomes)
    require(
        np.array_equal(
            chunk.sampled_inputs,
            header.sampled_inputs[chunk.start_index : stop],
            equal_nan=False,
        ),
        "chunk sampled inputs do not match header authority",
    )
    require(
        chunk.positions_m.shape[1:3]
        == (header.sample_times_s.size, len(header.point_ids)),
        "chunk trace layout does not match header",
    )
    require(
        (chunk.authority is None) == (header.authority_layout is None),
        "chunk authority presence must match the announced header",
    )
    if chunk.authority is not None:
        assert header.authority_layout is not None
        require_chunk_authority(
            chunk.authority,
            header.authority_layout,
            chunk.outcomes,
            chunk.sample_valid,
            header.sample_times_s,
        )
    for row, outcome in enumerate(chunk.outcomes):
        impact_index = int(chunk.impact_sample_indices[row])
        if outcome.status is not EVALUATED_HIT:
            continue
        require(
            chunk.sample_valid[row, impact_index],
            "chunk impact marker must identify a valid sample",
        )
        impact_time = outcome.value("impact_time_s")
        assert impact_time is not None
        nearest = int(np.argmin(np.abs(header.sample_times_s - impact_time)))
        require(
            impact_index == nearest,
            "chunk impact marker must match impact-time provenance",
        )


class CollectingEnsembleSink:
    """Compatibility sink that reconstructs the existing materialized result."""

    def __init__(self) -> None:
        self._header: EnsembleStreamHeader | None = None
        self._next_index = 0
        self._inputs: np.ndarray | None = None
        self._positions: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        self._impacts: np.ndarray | None = None
        self._outcomes: list[SimulationTrialOutcome] = []
        self._finished = False

    def begin(self, header: EnsembleStreamHeader) -> None:
        require(self._header is None, "sink lifecycle has already begun")
        self._header = header
        rows = header.plan.n_runs
        samples = header.sample_times_s.size
        points = len(header.point_ids)
        self._inputs = np.empty((rows, len(header.plan.noise)), dtype=float)
        self._positions = np.full((rows, samples, points, 3), np.nan)
        self._valid = np.zeros((rows, samples), dtype=bool)
        self._impacts = np.full(rows, -1, dtype=int)

    def accept(self, chunk: SimulationResultChunk) -> None:
        header = self._require_active()
        require_chunk_matches_header(header, chunk, self._next_index)
        stop = chunk.start_index + len(chunk.outcomes)
        assert self._inputs is not None
        assert self._positions is not None
        assert self._valid is not None
        assert self._impacts is not None
        self._inputs[chunk.start_index : stop] = chunk.sampled_inputs
        self._positions[chunk.start_index : stop] = chunk.positions_m
        self._valid[chunk.start_index : stop] = chunk.sample_valid
        self._impacts[chunk.start_index : stop] = chunk.impact_sample_indices
        self._outcomes.extend(chunk.outcomes)
        self._next_index = stop

    def commit(self, elapsed_s: float) -> SimulationEnsembleResult:
        header = self._require_active()
        require(self._next_index == header.plan.n_runs, "cannot commit partial stream")
        require(math.isfinite(elapsed_s) and elapsed_s >= 0.0, "invalid elapsed_s")
        assert self._inputs is not None
        assert self._positions is not None
        assert self._valid is not None
        assert self._impacts is not None
        outputs = np.full((header.plan.n_runs, len(ALL_OUTPUT_NAMES)), np.nan)
        success = np.zeros(header.plan.n_runs, dtype=bool)
        outcomes = tuple(self._outcomes)
        for outcome in outcomes:
            outputs[outcome.trial_index] = [
                math.nan if outcome.value(name) is None else outcome.value(name)
                for name in ALL_OUTPUT_NAMES
            ]
            success[outcome.trial_index] = outcome.status is not NUMERICAL_FAILURE
        variation = VariationDataset(
            plan=header.plan,
            input_names=tuple(spec.variable_key for spec in header.plan.noise),
            inputs=self._inputs,
            output_names=ALL_OUTPUT_NAMES,
            outputs=outputs,
            success=success,
            elapsed_s=elapsed_s,
        )
        traces = EnsemblePositionTraces(
            variation=variation,
            sample_times_s=header.sample_times_s,
            coordinate_frame=header.coordinate_frame,
            point_ids=header.point_ids,
            positions_m=self._positions,
            sample_valid=self._valid,
            impact_sample_indices=self._impacts,
        )
        result = SimulationEnsembleResult(outcomes, variation, traces)
        self._finished = True
        return result

    def abort(self) -> None:
        if not self._finished:
            self._finished = True
            self._inputs = None
            self._positions = None
            self._valid = None
            self._impacts = None
            self._outcomes.clear()

    def _require_active(self) -> EnsembleStreamHeader:
        require(self._header is not None, "sink lifecycle has not begun")
        require(not self._finished, "sink lifecycle is finished")
        return cast(EnsembleStreamHeader, self._header)


__all__ = [
    "CollectingEnsembleSink",
    "EnsembleChunkSink",
    "EnsembleStreamHeader",
    "MAX_CHUNK_POSITION_CELLS",
    "SimulationResultChunk",
    "require_chunk_matches_header",
]

"""One-run-at-a-time projection into a bounded immutable ensemble chunk."""

from __future__ import annotations

from typing import cast

import numpy as np

from rate_of_closure.simulation import SimulationRun
from shared.python.contracts import require

from .ensemble_chunks import EnsembleStreamHeader, SimulationResultChunk
from .ensemble_trace_authority import (
    ChunkTraceAuthority,
    EnsembleAuthorityLayout,
    TrialContactEvent,
    event_for_grid,
)
from .simulation_types import SimulationEnsembleRequest, SimulationTrialOutcome
from .trial_projection import TrialCapture, project_simulation_outcome


class ChunkAccumulator:
    """Preallocated chunk buffer that never retains complete prior runs."""

    def __init__(
        self,
        request: SimulationEnsembleRequest,
        header: EnsembleStreamHeader,
        start_index: int,
        stop_index: int,
    ) -> None:
        require(0 <= start_index < stop_index <= request.plan.n_runs, "invalid chunk")
        layout = header.authority_layout
        require(
            layout is not None, "complete chunk projection requires authority layout"
        )
        layout = cast(EnsembleAuthorityLayout, layout)
        rows = stop_index - start_index
        samples = header.sample_times_s.size
        self._request = request
        self._header = header
        self._start = start_index
        self._stop = stop_index
        self._cursor = 0
        self._outcomes: list[SimulationTrialOutcome] = []
        self._events: list[TrialContactEvent | None] = []
        self._positions: np.ndarray = np.full(
            (rows, samples, len(header.point_ids), 3), np.nan
        )
        self._valid: np.ndarray = np.zeros((rows, samples), dtype=bool)
        self._impacts: np.ndarray = np.full(rows, -1, dtype=int)
        self._poses: np.ndarray = np.full((rows, samples, 4, 4), np.nan)
        self._twists: np.ndarray = np.full((rows, samples, 6), np.nan)
        self._states: np.ndarray = np.full(
            (rows, samples, len(layout.state_ids)), np.nan
        )
        self._torques: np.ndarray = np.full(
            (rows, samples, len(layout.torque_joint_ids)), np.nan
        )
        self._preimpact: np.ndarray = np.zeros((rows, samples), dtype=bool)

    @property
    def failure_count(self) -> int:
        """Return failures projected into this chunk so far."""
        return sum(outcome.failure_type is not None for outcome in self._outcomes)

    def append(self, capture: TrialCapture) -> None:
        """Project one capture immediately, retaining no ``SimulationRun`` reference."""
        require(self._cursor < self._stop - self._start, "chunk is already full")
        trial_index = self._start + self._cursor
        outcome = project_simulation_outcome(trial_index, capture)
        self._outcomes.append(outcome)
        run = capture.run
        if run is None:
            self._events.append(None)
            self._cursor += 1
            return
        self._append_run(self._cursor, run)
        self._events.append(
            event_for_grid(trial_index, run.impact_outcome, self._header.sample_times_s)
        )
        self._cursor += 1

    def _append_run(self, row: int, run: SimulationRun) -> None:
        header = self._header
        layout = header.authority_layout
        assert layout is not None
        require(
            np.array_equal(run.swing_times, header.sample_times_s),
            "evaluated runs must share one sample-time grid",
        )
        require(
            run.swing_state_ids == layout.state_ids, "state IDs changed within stream"
        )
        require(
            run.swing_state_units == layout.state_units,
            "state units changed within stream",
        )
        require(
            run.swing_joint_ids == layout.torque_joint_ids,
            "torque joint IDs changed within stream",
        )
        positions = _spatial_positions(run)
        require(
            positions.shape == self._positions[row].shape,
            "spatial point layout changed within stream",
        )
        self._positions[row] = positions
        self._poses[row] = run.swing_poses
        self._twists[row] = run.swing_twists
        self._states[row] = run.swing_generalized_states
        self._torques[row] = run.swing_applied_torques_nm
        self._valid[row] = True
        if run.impact_time_s is None:
            self._preimpact[row] = True
            return
        times = header.sample_times_s
        self._impacts[row] = int(np.argmin(np.abs(times - run.impact_time_s)))
        self._preimpact[row] = times <= run.impact_time_s

    def finish(self) -> SimulationResultChunk:
        """Freeze exactly one full contiguous chunk."""
        require(self._cursor == self._stop - self._start, "chunk is incomplete")
        authority = ChunkTraceAuthority(
            poses_app=self._poses,
            twists_app_si=self._twists,
            generalized_states=self._states,
            applied_torques_nm=self._torques,
            preimpact_valid=self._preimpact,
            events=tuple(self._events),
        )
        return SimulationResultChunk(
            start_index=self._start,
            sampled_inputs=self._request.sampled_inputs[self._start : self._stop],
            outcomes=tuple(self._outcomes),
            positions_m=self._positions,
            sample_valid=self._valid,
            impact_sample_indices=self._impacts,
            authority=authority,
        )


def _spatial_positions(run: SimulationRun) -> np.ndarray:
    """Return positions in stable proximal-to-distal point order."""
    if run.config.source_kind == "manual":
        return cast(
            np.ndarray,
            np.asarray(run.swing_positions, dtype=float)[:, np.newaxis, :],
        )
    require(
        bool(np.allclose(run.swing_joints[:, -1], run.swing_positions, atol=1e-9)),
        "last spatial point must be the clubhead reference trajectory",
    )
    return cast(np.ndarray, np.asarray(run.swing_joints, dtype=float))


__all__ = ["ChunkAccumulator"]

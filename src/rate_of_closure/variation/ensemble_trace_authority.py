"""Immutable per-trial contact-event authority for streamed ensembles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from rate_of_closure.simulation.contact import ImpactOutcome, ImpactStatus
from shared.python.contracts import require

AUTHORITY_SCHEMA_VERSION = 1
POSE_FRAME = "app_frame:x_target,y_up,z_right/world_from_clubhead"
TWIST_COMPONENT_IDS = ("wx", "wy", "wz", "vx", "vy", "vz")
TWIST_UNITS = ("rad/s", "rad/s", "rad/s", "m/s", "m/s", "m/s")
CONTINUATION_POLICY = (
    "samples_after_impact_are_open_loop_source_continuation_not_collision_coupled"
)


def _owned_float_array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    require(
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.bool_)
        and not np.issubdtype(raw.dtype, np.complexfloating),
        f"{name} must contain real non-boolean numbers",
    )
    result = np.array(raw, dtype=np.float64, copy=True)
    require(
        bool(np.all(np.isfinite(result) | np.isnan(result))),
        f"{name} must be finite or unavailable NaN",
    )
    result.setflags(write=False)
    return cast(np.ndarray, result)


@dataclass(frozen=True)
class EnsembleAuthorityLayout:
    """Stable state and applied-torque schema announced once per stream."""

    state_ids: tuple[str, ...]
    state_units: tuple[str, ...]
    torque_joint_ids: tuple[str, ...]
    pose_frame: str = POSE_FRAME
    twist_component_ids: tuple[str, ...] = TWIST_COMPONENT_IDS
    twist_units: tuple[str, ...] = TWIST_UNITS
    continuation_policy: str = CONTINUATION_POLICY

    def __post_init__(self) -> None:
        state_ids = tuple(self.state_ids)
        state_units = tuple(self.state_units)
        torque_ids = tuple(self.torque_joint_ids)
        require(len(state_ids) == len(state_units), "state IDs and units must align")
        for values, name in (
            (state_ids, "state IDs"),
            (state_units, "state units"),
            (torque_ids, "torque joint IDs"),
            (tuple(self.twist_component_ids), "twist component IDs"),
            (tuple(self.twist_units), "twist units"),
        ):
            require(
                all(isinstance(value, str) and bool(value.strip()) for value in values),
                f"{name} must be nonempty strings",
            )
        require(len(set(state_ids)) == len(state_ids), "state IDs must be unique")
        require(len(set(torque_ids)) == len(torque_ids), "torque IDs must be unique")
        require(
            tuple(self.twist_component_ids) == TWIST_COMPONENT_IDS
            and tuple(self.twist_units) == TWIST_UNITS,
            "twist schema is unsupported",
        )
        require(self.pose_frame == POSE_FRAME, "pose frame is unsupported")
        require(
            self.continuation_policy == CONTINUATION_POLICY,
            "continuation policy is unsupported",
        )
        object.__setattr__(self, "state_ids", state_ids)
        object.__setattr__(self, "state_units", state_units)
        object.__setattr__(self, "torque_joint_ids", torque_ids)


@dataclass(frozen=True)
class ChunkTraceAuthority:
    """Owned full state/command/event arrays for one contiguous result chunk."""

    poses_app: np.ndarray = field(repr=False)
    twists_app_si: np.ndarray = field(repr=False)
    generalized_states: np.ndarray = field(repr=False)
    applied_torques_nm: np.ndarray = field(repr=False)
    preimpact_valid: np.ndarray = field(repr=False)
    events: tuple[TrialContactEvent | None, ...]

    def __post_init__(self) -> None:
        poses = _owned_float_array(self.poses_app, "poses_app")
        twists = _owned_float_array(self.twists_app_si, "twists_app_si")
        states = _owned_float_array(self.generalized_states, "generalized_states")
        torques = _owned_float_array(self.applied_torques_nm, "applied_torques_nm")
        raw_valid = np.asarray(self.preimpact_valid)
        require(raw_valid.dtype == np.dtype(bool), "preimpact_valid must be boolean")
        valid = np.array(raw_valid, dtype=bool, copy=True)
        valid.setflags(write=False)
        events = tuple(self.events)
        require(
            all(
                event is None or isinstance(event, TrialContactEvent)
                for event in events
            ),
            "events must contain TrialContactEvent or None",
        )
        object.__setattr__(self, "poses_app", poses)
        object.__setattr__(self, "twists_app_si", twists)
        object.__setattr__(self, "generalized_states", states)
        object.__setattr__(self, "applied_torques_nm", torques)
        object.__setattr__(self, "preimpact_valid", valid)
        object.__setattr__(self, "events", events)


@dataclass(frozen=True)
class TrialContactEvent:
    """Exact contact assessment plus its relationship to the common grid."""

    trial_index: int
    outcome: ImpactOutcome
    left_sample_index: int
    right_sample_index: int
    nearest_sample_index: int

    def __post_init__(self) -> None:
        require(
            type(self.trial_index) is int and self.trial_index >= 0,
            "event trial_index must be a non-negative integer",
        )
        require(isinstance(self.outcome, ImpactOutcome), "invalid contact outcome")
        for name in (
            "left_sample_index",
            "right_sample_index",
            "nearest_sample_index",
        ):
            value = getattr(self, name)
            require(type(value) is int and value >= 0, f"{name} must be non-negative")
        require(
            self.left_sample_index <= self.right_sample_index,
            "event sample bracket must be ordered",
        )

    @property
    def kind(self) -> str:
        """Return the scientific event label without fabricating impact."""
        return (
            "impact" if self.outcome.status is ImpactStatus.HIT else "closest_approach"
        )


def event_for_grid(
    trial_index: int, outcome: ImpactOutcome, sample_times_s: np.ndarray
) -> TrialContactEvent:
    """Bind one exact event time to bracketing and nearest grid samples."""
    times = np.asarray(sample_times_s, dtype=float)
    require(times.ndim == 1 and times.size > 0, "sample grid must be non-empty")
    event_time = outcome.candidate_time_s
    require(
        float(times[0]) <= event_time <= float(times[-1]),
        "contact event must lie within the sample grid",
        event_time,
    )
    right = int(np.searchsorted(times, event_time, side="left"))
    if right == times.size:
        right = times.size - 1
    if float(times[right]) == event_time:
        left = right
    else:
        left = max(right - 1, 0)
    nearest = int(np.argmin(np.abs(times - event_time)))
    return TrialContactEvent(trial_index, outcome, left, right, nearest)


def require_event_matches_grid(
    event: TrialContactEvent, sample_times_s: np.ndarray
) -> None:
    """Require exact canonical indices for an event and common grid."""
    expected = event_for_grid(event.trial_index, event.outcome, sample_times_s)
    require(event == expected, "contact event grid provenance is not canonical")


def require_chunk_authority(
    authority: ChunkTraceAuthority,
    layout: EnsembleAuthorityLayout,
    outcomes: tuple[object, ...],
    sample_valid: np.ndarray,
    sample_times_s: np.ndarray,
) -> None:
    """Bind owned authority arrays to outcomes, layout, and common time grid."""
    from rate_of_closure.variation._ensemble_limits import (
        MAX_CHUNK_AUTHORITY_BYTES,
    )
    from rate_of_closure.variation.simulation_types import (
        EVALUATED_HIT,
        NUMERICAL_FAILURE,
        SimulationTrialOutcome,
    )

    rows, samples = sample_valid.shape
    require(authority.poses_app.shape == (rows, samples, 4, 4), "invalid pose shape")
    require(authority.twists_app_si.shape == (rows, samples, 6), "invalid twist shape")
    require(
        authority.generalized_states.shape == (rows, samples, len(layout.state_ids)),
        "invalid generalized-state shape",
    )
    require(
        authority.applied_torques_nm.shape
        == (rows, samples, len(layout.torque_joint_ids)),
        "invalid applied-torque shape",
    )
    require(authority.preimpact_valid.shape == (rows, samples), "invalid mask shape")
    require(len(authority.events) == rows, "events must align to chunk rows")
    total_bytes = sum(
        values.nbytes
        for values in (
            authority.poses_app,
            authority.twists_app_si,
            authority.generalized_states,
            authority.applied_torques_nm,
            authority.preimpact_valid,
        )
    )
    require(
        total_bytes <= MAX_CHUNK_AUTHORITY_BYTES,
        "chunk authority byte limit exceeded",
        total_bytes,
    )
    require(
        not np.any(authority.preimpact_valid & ~sample_valid),
        "preimpact samples must be a subset of available source samples",
    )
    typed_outcomes = cast(tuple[SimulationTrialOutcome, ...], outcomes)
    for row, outcome in enumerate(typed_outcomes):
        status = outcome.status
        event = authority.events[row]
        arrays = (
            authority.poses_app[row],
            authority.twists_app_si[row],
            authority.generalized_states[row],
            authority.applied_torques_nm[row],
        )
        if status is NUMERICAL_FAILURE:
            require(event is None, "numerical failure cannot fabricate an event")
            require(
                not np.any(authority.preimpact_valid[row]), "failure mask unavailable"
            )
            require(
                all(np.all(np.isnan(values)) for values in arrays),
                "failure data unavailable",
            )
            continue
        require(event is not None, "evaluated trial requires a contact event")
        assert event is not None
        require(
            event.trial_index == outcome.trial_index,
            "event trial index must match outcome",
        )
        require_event_matches_grid(event, sample_times_s)
        require(
            event.outcome.is_hit == (status is EVALUATED_HIT),
            "event contact status must match typed trial outcome",
        )
        for name, expected_value in (
            ("candidate_time_s", event.outcome.candidate_time_s),
            ("closest_approach_m", event.outcome.closest_approach_m),
            ("contact_margin_m", event.outcome.contact_margin_m),
        ):
            require(
                outcome.value(name) == expected_value,
                f"event {name} must match typed trial outcome",
            )
        if status is EVALUATED_HIT:
            require(
                outcome.value("impact_time_s") == event.outcome.candidate_time_s,
                "impact time must match exact contact event",
            )
        for values in arrays:
            require(
                bool(np.all(np.isfinite(values[sample_valid[row]])))
                and bool(np.all(np.isnan(values[~sample_valid[row]]))),
                "authority arrays must agree with sample availability",
            )
        valid_poses = authority.poses_app[row][sample_valid[row]]
        require(
            bool(
                np.allclose(
                    valid_poses[:, 3, :],
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    atol=1e-12,
                )
            ),
            "pose bottom rows must be homogeneous",
        )
        rotations = valid_poses[:, :3, :3]
        require(
            bool(
                np.allclose(
                    np.swapaxes(rotations, 1, 2) @ rotations,
                    np.eye(3),
                    atol=1e-9,
                )
            ),
            "pose rotations must be orthonormal",
        )
        if status is EVALUATED_HIT:
            expected = sample_valid[row] & (
                sample_times_s <= event.outcome.candidate_time_s
            )
        else:
            expected = sample_valid[row]
        require(
            np.array_equal(authority.preimpact_valid[row], expected),
            "preimpact mask must match exact contact provenance",
        )


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "ChunkTraceAuthority",
    "CONTINUATION_POLICY",
    "POSE_FRAME",
    "EnsembleAuthorityLayout",
    "TWIST_COMPONENT_IDS",
    "TWIST_UNITS",
    "TrialContactEvent",
    "event_for_grid",
    "require_event_matches_grid",
    "require_chunk_authority",
]

"""Top-level request and result records for flight-to-ground v1."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contract_types import (
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    UNIT_SYSTEM_SI,
    GroundCalibration,
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundFrame,
    GroundPhase,
    GroundProvenance,
    GroundResultStatus,
    GroundSurfaceProfile,
    GroundTrajectoryPoint,
    _integer,
    _positive,
    _raw_finite,
    _text,
    _WireRecord,
)
from .result_types import GroundSummary, GroundTermination, GroundWarning
from .unavailable_types import GroundUnavailableField
from .validation import (
    close as _close,
)
from .validation import (
    validate_status_termination as _validate_status_termination,
)
from .validation import (
    validate_terminal_state as _validate_terminal_state,
)
from .validation import (
    vector_close as _vector_close,
)

_CONTACT_SPEED_TOLERANCE_M_S = 1e-12
_PHASE_TRANSITIONS = {
    GroundPhase.IMPACT: frozenset(GroundPhase),
    GroundPhase.BOUNCE: frozenset(
        {GroundPhase.BOUNCE, GroundPhase.SKID, GroundPhase.ROLL, GroundPhase.REST}
    ),
    GroundPhase.SKID: frozenset({GroundPhase.SKID, GroundPhase.ROLL, GroundPhase.REST}),
    GroundPhase.ROLL: frozenset({GroundPhase.ROLL, GroundPhase.REST}),
    GroundPhase.REST: frozenset({GroundPhase.REST}),
}
_EVENT_TRANSITIONS = {
    GroundEventType.FIRST_CONTACT: frozenset(
        {
            GroundEventType.BOUNCE,
            GroundEventType.SKID_TO_ROLL,
            GroundEventType.REST,
            GroundEventType.LEFT_SURFACE,
        }
    ),
    GroundEventType.BOUNCE: frozenset(
        {
            GroundEventType.BOUNCE,
            GroundEventType.SKID_TO_ROLL,
            GroundEventType.REST,
            GroundEventType.LEFT_SURFACE,
        }
    ),
    GroundEventType.SKID_TO_ROLL: frozenset(
        {GroundEventType.REST, GroundEventType.LEFT_SURFACE}
    ),
    GroundEventType.REST: frozenset(),
    GroundEventType.LEFT_SURFACE: frozenset(),
}


def _schema(value: str, expected: str) -> str:
    if value != expected:
        raise ValueError(f"unsupported schema_version: {value}")
    return value


def _unit_system(value: str) -> str:
    if value != UNIT_SYSTEM_SI:
        raise ValueError(f"unsupported unit_system: {value}")
    return value


def _require_exact(value: object, expected: type[object], name: str) -> None:
    if type(value) is not expected:
        raise ValueError(f"{name} must be a {expected.__name__}")


def _require_collection(value: object, name: str) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")


@dataclass(frozen=True)
class GroundSimulationRequest(_WireRecord):
    """Strict physical transfer request with two states bracketing contact."""

    request_id: str
    surface: GroundSurfaceProfile
    last_separated_state: GroundContactState
    first_penetrating_state: GroundContactState
    ball_radius_m: float
    ball_mass_kg: float
    rotational_inertia_factor: float
    max_time_s: float
    output_interval_s: float
    max_events: int
    calibration: GroundCalibration
    provenance: GroundProvenance
    unit_system: str = UNIT_SYSTEM_SI
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_nested_types()
        inertia_raw = _raw_finite(
            self.rotational_inertia_factor,
            "rotational_inertia_factor",
        )
        max_time_raw = _raw_finite(self.max_time_s, "max_time_s")
        interval_raw = _raw_finite(self.output_interval_s, "output_interval_s")
        if inertia_raw > 1.0:
            raise ValueError("rotational_inertia_factor must lie within (0, 1]")
        if interval_raw > max_time_raw:
            raise ValueError("output_interval_s must not exceed max_time_s")
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(
            self, "ball_radius_m", _positive(self.ball_radius_m, "ball_radius_m")
        )
        object.__setattr__(
            self, "ball_mass_kg", _positive(self.ball_mass_kg, "ball_mass_kg")
        )
        inertia = _positive(self.rotational_inertia_factor, "rotational_inertia_factor")
        object.__setattr__(self, "rotational_inertia_factor", inertia)
        object.__setattr__(self, "max_time_s", _positive(self.max_time_s, "max_time_s"))
        interval = _positive(self.output_interval_s, "output_interval_s")
        object.__setattr__(self, "output_interval_s", interval)
        object.__setattr__(
            self, "max_events", _integer(self.max_events, "max_events", 1)
        )
        object.__setattr__(self, "unit_system", _unit_system(self.unit_system))
        object.__setattr__(
            self, "schema_version", _schema(self.schema_version, REQUEST_SCHEMA_VERSION)
        )
        self._validate_contact_bracket()

    def _validate_nested_types(self) -> None:
        _require_exact(self.surface, GroundSurfaceProfile, "surface")
        _require_exact(
            self.last_separated_state,
            GroundContactState,
            "last_separated_state",
        )
        _require_exact(
            self.first_penetrating_state,
            GroundContactState,
            "first_penetrating_state",
        )
        _require_exact(self.calibration, GroundCalibration, "calibration")
        _require_exact(self.provenance, GroundProvenance, "provenance")

    def _validate_contact_bracket(self) -> None:
        separated = self.last_separated_state
        penetrating = self.first_penetrating_state
        if (
            separated.frame is not self.surface.frame
            or penetrating.frame is not self.surface.frame
        ):
            raise ValueError("contact frame must match surface frame")
        if penetrating.time_s <= separated.time_s:
            raise ValueError("contact bracket times must be strictly increasing")
        first_gap = self.surface.signed_gap_m(separated, self.ball_radius_m)
        second_gap = self.surface.signed_gap_m(penetrating, self.ball_radius_m)
        if first_gap <= 0.0 or second_gap > 0.0:
            raise ValueError("contact states must straddle the physical sphere surface")
        speeds = (
            self.surface.relative_normal_speed_m_s(separated),
            self.surface.relative_normal_speed_m_s(penetrating),
        )
        if any(speed >= -_CONTACT_SPEED_TOLERANCE_M_S for speed in speeds):
            raise ValueError(
                "both bracket states require incoming relative normal velocity"
            )


def _validate_trajectory(
    points: tuple[GroundTrajectoryPoint, ...], frame: GroundFrame
) -> None:
    if not points:
        raise ValueError("trajectory must be nonempty")
    if any(point.frame is not frame for point in points):
        raise ValueError("trajectory frame must match result frame")
    if any(
        right.time_s <= left.time_s
        for left, right in zip(points, points[1:], strict=False)
    ):
        raise ValueError("trajectory times must be strictly increasing")
    for left, right in zip(points, points[1:], strict=False):
        if right.phase not in _PHASE_TRANSITIONS[left.phase]:
            raise ValueError(
                f"invalid ground phase transition: {left.phase}->{right.phase}"
            )


def _validate_events(events: tuple[GroundEvent, ...], frame: GroundFrame) -> None:
    if tuple(event.sequence for event in events) != tuple(range(len(events))):
        raise ValueError("event sequence must be contiguous from zero")
    if any(event.frame is not frame for event in events):
        raise ValueError("event frame must match result frame")
    if any(
        right.time_s < left.time_s
        for left, right in zip(events, events[1:], strict=False)
    ):
        raise ValueError("event times must be nondecreasing")
    for left, right in zip(events, events[1:], strict=False):
        if right.event_type not in _EVENT_TRANSITIONS[left.event_type]:
            transition = f"{left.event_type}->{right.event_type}"
            raise ValueError(f"invalid ground event transition: {transition}")


def _validate_first_contact(
    points: tuple[GroundTrajectoryPoint, ...],
    events: tuple[GroundEvent, ...],
) -> None:
    if not events or events[0].event_type is not GroundEventType.FIRST_CONTACT:
        raise ValueError("event ledger must begin with first_contact")
    point = points[0]
    event = events[0]
    if point.phase is not GroundPhase.IMPACT:
        raise ValueError("ground trajectory must begin in the impact phase")
    if not _close(point.time_s, event.time_s):
        raise ValueError("first_contact time must match the initial trajectory point")
    if not _vector_close(point.position_m, event.position_m):
        raise ValueError(
            "first_contact position must match the initial trajectory point"
        )
    if not _vector_close(point.velocity_m_s, event.velocity_after_m_s):
        raise ValueError(
            "first_contact output velocity must match the initial trajectory point"
        )
    if not _vector_close(
        point.angular_velocity_rad_s,
        event.angular_velocity_after_rad_s,
    ):
        raise ValueError(
            "first_contact output spin must match the initial trajectory point"
        )


def _validate_summary(
    summary: GroundSummary,
    points: tuple[GroundTrajectoryPoint, ...],
    events: tuple[GroundEvent, ...],
) -> None:
    first = points[0].position_m
    final = points[-1].position_m
    expected = (math.hypot(first[0], first[2]), final[0], final[2])
    actual = (
        summary.carry_distance_m,
        summary.final_downrange_m,
        summary.final_offline_m,
    )
    if not all(
        _close(left, right) for left, right in zip(actual, expected, strict=True)
    ):
        raise ValueError("summary displacement metrics must match trajectory geometry")
    if not _close(summary.total_distance_m, math.hypot(final[0], final[2])):
        raise ValueError("summary total distance must match final horizontal position")
    path_sum = summary.skid_distance_m + summary.roll_distance_m
    if not _close(summary.surface_path_distance_m, path_sum):
        raise ValueError("surface path must equal skid plus roll distance")
    bounce_count = sum(event.event_type is GroundEventType.BOUNCE for event in events)
    if summary.bounce_count != bounce_count:
        raise ValueError(
            "summary bounce_count must match post-first-contact bounce events"
        )


@dataclass(frozen=True)
class GroundSimulationResult(_WireRecord):
    """Qualified ground-run output and ordered event ledger."""

    request_id: str
    surface_id: str
    frame: GroundFrame
    model_id: str
    model_version: str
    status: GroundResultStatus
    trajectory: tuple[GroundTrajectoryPoint, ...]
    events: tuple[GroundEvent, ...]
    summary: GroundSummary | None
    termination: GroundTermination
    calibration: GroundCalibration
    warnings: tuple[GroundWarning, ...]
    unavailable_fields: tuple[GroundUnavailableField, ...]
    provenance: GroundProvenance
    unit_system: str = UNIT_SYSTEM_SI
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._validate_nested_types()
        for name in ("request_id", "surface_id", "model_id", "model_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "frame", GroundFrame(self.frame))
        object.__setattr__(self, "status", GroundResultStatus(self.status))
        points = tuple(self.trajectory)
        events = tuple(self.events)
        warnings = tuple(self.warnings)
        unavailable_fields = tuple(self.unavailable_fields)
        self._validate_status_payload(points, events)
        object.__setattr__(self, "trajectory", points)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "unavailable_fields", unavailable_fields)
        object.__setattr__(self, "unit_system", _unit_system(self.unit_system))
        object.__setattr__(
            self, "schema_version", _schema(self.schema_version, RESULT_SCHEMA_VERSION)
        )

    def _validate_nested_types(self) -> None:
        _require_collection(self.trajectory, "trajectory")
        _require_collection(self.events, "events")
        _require_collection(self.warnings, "warnings")
        _require_collection(self.unavailable_fields, "unavailable_fields")
        for point in self.trajectory:
            _require_exact(point, GroundTrajectoryPoint, "trajectory point")
        for event in self.events:
            _require_exact(event, GroundEvent, "ground event")
        for warning in self.warnings:
            _require_exact(warning, GroundWarning, "ground warning")
        for field in self.unavailable_fields:
            _require_exact(field, GroundUnavailableField, "unavailable field")
        field_ids = tuple(field.field_id for field in self.unavailable_fields)
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("unavailable field IDs must be unique")
        if self.summary is not None:
            _require_exact(self.summary, GroundSummary, "summary")
        _require_exact(self.termination, GroundTermination, "termination")
        _require_exact(self.calibration, GroundCalibration, "calibration")
        _require_exact(self.provenance, GroundProvenance, "provenance")

    def _validate_status_payload(
        self,
        points: tuple[GroundTrajectoryPoint, ...],
        events: tuple[GroundEvent, ...],
    ) -> None:
        _validate_status_termination(self.status, self.termination)
        has_unavailable = bool(self.unavailable_fields)
        if (self.status is GroundResultStatus.UNAVAILABLE) is not has_unavailable:
            raise ValueError("only unavailable results require unavailable_fields")
        if self.status in {GroundResultStatus.FAILED, GroundResultStatus.UNAVAILABLE}:
            if (
                points
                or events
                or self.summary is not None
                or self.termination.completed
            ):
                raise ValueError(
                    "failed or unavailable results cannot contain fabricated output"
                )
            return
        _validate_trajectory(points, self.frame)
        _validate_events(events, self.frame)
        _validate_first_contact(points, events)
        if self.summary is None:
            raise ValueError("complete or partial results require a summary")
        _validate_summary(self.summary, points, events)
        if abs(self.termination.time_s - points[-1].time_s) > 1e-9:
            raise ValueError("termination time_s must match the final trajectory point")
        if any(
            not points[0].time_s <= event.time_s <= points[-1].time_s
            for event in events
        ):
            raise ValueError("event times must lie within the trajectory interval")
        if (
            self.status is GroundResultStatus.COMPLETE
            and not self.termination.completed
        ):
            raise ValueError("complete result requires completed termination")
        if self.status is GroundResultStatus.PARTIAL and self.termination.completed:
            raise ValueError("partial result requires incomplete termination")
        _validate_terminal_state(self.status, points, events, self.termination)


__all__ = ["GroundSimulationRequest", "GroundSimulationResult"]

"""Typed configuration and prefix result for repeated rigid impacts."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

from .contract_types import (
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundFrame,
    GroundPhase,
    GroundTrajectoryPoint,
    Vector3,
)
from .impact_types import ImpactImpulseResult
from .request_identity import validate_request_fingerprint

GROUND_IMPACT_MODEL_ID = "tools-ground-impact-bounce"
GROUND_IMPACT_MODEL_VERSION = "1.0.0"
BOUNCE_MATERIAL_LIMITATION = (
    "Rigid restitution v1 does not use firmness_pa, hardness_fraction, "
    "grass_height_m, compressibility_fraction, compression_damping_fraction, "
    "turf_density_kg_m3, moisture_fraction, or rolling_resistance."
)
BOUNCE_HANDOFF_NOTICE = (
    "The impact/bounce prefix ends at settled-to-skid and requires the qualified "
    "#4271 surface continuation before final distance is available."
)
STANDARD_GRAVITY_M_S2: Vector3 = (0.0, -9.80665, 0.0)
CancellationCheck = Callable[[], bool]


class BounceTerminationReason(StrEnum):
    """Bounded terminal states for the bounce-prefix solver."""

    SETTLED_TO_SKID = "settled_to_skid"
    CANCELLED = "cancelled"
    TIME_LIMIT = "time_limit"
    EVENT_LIMIT = "event_limit"
    NO_RECONTACT = "no_recontact"
    NUMERICAL_FAILURE = "numerical_failure"


_SETTLED_TO_SKID = BounceTerminationReason("settled_to_skid")
_TIME_LIMIT = BounceTerminationReason("time_limit")


@dataclass(frozen=True)
class BounceAirSegment:
    """Exact horizontal-distance evidence for one airborne prefix segment."""

    start_time_s: float
    end_time_s: float
    start_position_m: Vector3
    end_position_m: Vector3
    horizontal_distance_m: float
    completed_at_contact: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time_s) or self.start_time_s < 0.0:
            raise ValueError("airborne start time must be finite and nonnegative")
        if not math.isfinite(self.end_time_s) or self.end_time_s <= self.start_time_s:
            raise ValueError("airborne end time must be finite and after start")
        if len(self.start_position_m) != 3 or len(self.end_position_m) != 3:
            raise ValueError("airborne positions must have three components")
        if not all(
            math.isfinite(value)
            for value in self.start_position_m + self.end_position_m
        ):
            raise ValueError("airborne positions must be finite")
        if (
            not math.isfinite(self.horizontal_distance_m)
            or self.horizontal_distance_m < 0.0
        ):
            raise ValueError(
                "airborne horizontal distance must be finite and nonnegative"
            )
        expected = math.hypot(
            self.end_position_m[0] - self.start_position_m[0],
            self.end_position_m[2] - self.start_position_m[2],
        )
        if not math.isclose(
            self.horizontal_distance_m, expected, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("airborne horizontal distance must match x-z displacement")


@dataclass(frozen=True)
class BounceModelSettings:
    """Versioned numerical settings for deterministic repeated hops."""

    gravity_m_s2: Vector3 = STANDARD_GRAVITY_M_S2
    capture_speed_m_s: float = 0.05
    velocity_tolerance_m_s: float = 1e-12
    time_tolerance_s: float = 1e-12
    model_id: str = GROUND_IMPACT_MODEL_ID
    model_version: str = GROUND_IMPACT_MODEL_VERSION

    def __post_init__(self) -> None:
        if len(self.gravity_m_s2) != 3 or not all(
            math.isfinite(value) for value in self.gravity_m_s2
        ):
            raise ValueError("gravity_m_s2 must contain three finite components")
        if tuple(self.gravity_m_s2) != STANDARD_GRAVITY_M_S2:
            raise ValueError("gravity_m_s2 must equal versioned standard gravity")
        object.__setattr__(self, "gravity_m_s2", STANDARD_GRAVITY_M_S2)
        for name in ("capture_speed_m_s", "velocity_tolerance_m_s", "time_tolerance_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.model_id or not self.model_version:
            raise ValueError("bounce model identity must be nonempty")


@dataclass(frozen=True)
class BounceTermination:
    """Reason and exact time for a bounded bounce-prefix termination."""

    reason: BounceTerminationReason
    time_s: float
    elapsed_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", BounceTerminationReason(self.reason))
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("bounce termination time_s must be finite and nonnegative")
        if not math.isfinite(self.elapsed_time_s) or self.elapsed_time_s < 0.0:
            raise ValueError("bounce elapsed_time_s must be finite and nonnegative")


@dataclass(frozen=True)
class RepeatedBounceResult:
    """Validated impact/hop prefix; intentionally not a final v1 ground result."""

    request_id: str
    surface_id: str
    frame: GroundFrame
    model_id: str
    model_version: str
    request_fingerprint_sha256: str
    trajectory: tuple[GroundTrajectoryPoint, ...]
    events: tuple[GroundEvent, ...]
    impacts: tuple[ImpactImpulseResult, ...]
    airborne_segments: tuple[BounceAirSegment, ...]
    handoff_state: GroundContactState | None
    termination: BounceTermination
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_fingerprint_sha256",
            validate_request_fingerprint(self.request_fingerprint_sha256),
        )
        self._validate_nested_types()
        self._normalize_collections()
        self._validate_event_ledger()
        self._validate_trajectory()
        self._validate_handoff()
        self._validate_airborne_segments()

    def _validate_nested_types(self) -> None:
        collections = {
            "trajectory": self.trajectory,
            "events": self.events,
            "impacts": self.impacts,
            "airborne_segments": self.airborne_segments,
            "warnings": self.warnings,
        }
        if any(not isinstance(value, (list, tuple)) for value in collections.values()):
            raise ValueError("bounce result collections must be lists or tuples")
        if any(type(point) is not GroundTrajectoryPoint for point in self.trajectory):
            raise ValueError("bounce trajectory requires exact trajectory points")
        if any(type(event) is not GroundEvent for event in self.events):
            raise ValueError("bounce events require exact ground events")
        if any(type(impact) is not ImpactImpulseResult for impact in self.impacts):
            raise ValueError("bounce impacts require exact impact results")
        if any(
            type(segment) is not BounceAirSegment for segment in self.airborne_segments
        ):
            raise ValueError("bounce air segments require exact segment records")
        if type(self.termination) is not BounceTermination:
            raise ValueError("bounce result requires exact termination record")
        if (
            self.handoff_state is not None
            and type(self.handoff_state) is not GroundContactState
        ):
            raise ValueError("bounce handoff requires an exact contact state")
        if any(
            not isinstance(warning, str) or not warning for warning in self.warnings
        ):
            raise ValueError("bounce warnings must be nonempty strings")

    def _normalize_collections(self) -> None:
        object.__setattr__(self, "frame", GroundFrame(self.frame))
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "impacts", tuple(self.impacts))
        object.__setattr__(self, "airborne_segments", tuple(self.airborne_segments))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def _validate_event_ledger(self) -> None:
        if len(self.events) != len(self.impacts):
            raise ValueError("each bounce event requires one impact result")
        if tuple(event.sequence for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("bounce event sequence must be contiguous from zero")
        if (
            self.events
            and self.events[0].event_type is not GroundEventType.FIRST_CONTACT
        ):
            raise ValueError("bounce ledger must begin with first_contact")
        if any(
            event.event_type is not GroundEventType.BOUNCE for event in self.events[1:]
        ):
            raise ValueError("post-first-contact bounce events must be bounce typed")
        if any(event.frame is not self.frame for event in self.events):
            raise ValueError("bounce event frame must match result frame")
        if any(
            right.time_s < left.time_s
            for left, right in zip(self.events, self.events[1:], strict=False)
        ):
            raise ValueError("bounce event times must be nondecreasing")
        if any(
            not _event_matches_impact(event, impact)
            for event, impact in zip(self.events, self.impacts, strict=True)
        ):
            raise ValueError("bounce event states must match their impact result")

    def _validate_trajectory(self) -> None:
        if any(point.frame is not self.frame for point in self.trajectory):
            raise ValueError("bounce trajectory frame must match result frame")
        if any(
            right.time_s <= left.time_s
            for left, right in zip(self.trajectory, self.trajectory[1:], strict=False)
        ):
            raise ValueError("bounce trajectory times must be strictly increasing")
        allowed_phases = {GroundPhase.IMPACT, GroundPhase.BOUNCE, GroundPhase.SKID}
        if any(point.phase not in allowed_phases for point in self.trajectory):
            raise ValueError("bounce prefix contains an unsupported phase")
        skid_points = tuple(
            point for point in self.trajectory if point.phase is GroundPhase.SKID
        )
        if skid_points and (
            len(skid_points) != 1 or skid_points[-1] is not self.trajectory[-1]
        ):
            raise ValueError("a bounce prefix may contain only one terminal skid point")

    def _validate_handoff(self) -> None:
        if self.handoff_state is not None:
            if self.termination.reason is not _SETTLED_TO_SKID:
                raise ValueError(
                    "only settled bounce prefixes may expose handoff state"
                )
            if not self.impacts or self.impacts[-1].effective_restitution != 0.0:
                raise ValueError("settled handoff requires a zero-restitution impact")
            if not self.trajectory or self.trajectory[-1].phase is not GroundPhase.SKID:
                raise ValueError("settled handoff requires a terminal skid point")
            if not _state_matches_point(self.handoff_state, self.trajectory[-1]):
                raise ValueError("handoff state must match the terminal skid point")

    def _validate_airborne_segments(self) -> None:
        tolerance = 1e-10
        completed_count = sum(
            segment.completed_at_contact for segment in self.airborne_segments
        )
        if completed_count != max(0, len(self.events) - 1):
            raise ValueError("each recontact event requires one completed air segment")
        partial_count = len(self.airborne_segments) - completed_count
        if partial_count > 1:
            raise ValueError("a bounce prefix may have at most one partial air segment")
        if self.termination.reason is _TIME_LIMIT and partial_count != 1:
            raise ValueError("time-limit termination requires one partial air segment")
        for index, segment in enumerate(self.airborne_segments):
            if index >= len(self.events):
                raise ValueError("airborne segment requires a preceding impact event")
            start_event = self.events[index]
            if abs(segment.start_time_s - start_event.time_s) > tolerance:
                raise ValueError("airborne segment start must match its impact event")
            if not _positions_close(segment.start_position_m, start_event.position_m):
                raise ValueError("airborne segment start position must match its event")
            if segment.completed_at_contact:
                self._validate_completed_segment(index, segment, tolerance)
            elif index != len(self.airborne_segments) - 1:
                raise ValueError("only the final airborne segment may be partial")
        if (
            self.airborne_segments
            and not self.airborne_segments[-1].completed_at_contact
        ):
            self._validate_partial_segment(self.airborne_segments[-1], tolerance)

    def _validate_completed_segment(
        self,
        index: int,
        segment: BounceAirSegment,
        tolerance: float,
    ) -> None:
        if index + 1 >= len(self.events):
            raise ValueError("completed airborne segment requires a contact event")
        end_event = self.events[index + 1]
        if abs(segment.end_time_s - end_event.time_s) > tolerance:
            raise ValueError("completed segment end must match contact time")
        if not _positions_close(segment.end_position_m, end_event.position_m):
            raise ValueError("completed segment end must match contact position")

    def _validate_partial_segment(
        self,
        segment: BounceAirSegment,
        tolerance: float,
    ) -> None:
        if abs(segment.end_time_s - self.termination.time_s) > tolerance:
            raise ValueError("partial segment end must match termination time")
        if not self.trajectory or not _positions_close(
            segment.end_position_m,
            self.trajectory[-1].position_m,
        ):
            raise ValueError("partial segment end must match final trajectory point")

    @property
    def bounce_air_distance_m(self) -> float:
        """Return accumulated x-z arc length of emitted airborne segments."""
        return sum(segment.horizontal_distance_m for segment in self.airborne_segments)


def _positions_close(left: Vector3, right: Vector3) -> bool:
    return all(
        math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
        for a, b in zip(left, right, strict=True)
    )


def _event_matches_impact(
    event: GroundEvent,
    impact: ImpactImpulseResult,
) -> bool:
    before = impact.state_before
    after = impact.state_after
    return (
        math.isclose(event.time_s, before.time_s, abs_tol=1e-10)
        and _positions_close(event.position_m, before.position_m)
        and _positions_close(event.velocity_before_m_s, before.velocity_m_s)
        and _positions_close(event.velocity_after_m_s, after.velocity_m_s)
        and _positions_close(
            event.angular_velocity_before_rad_s,
            before.angular_velocity_rad_s,
        )
        and _positions_close(
            event.angular_velocity_after_rad_s,
            after.angular_velocity_rad_s,
        )
    )


def _state_matches_point(
    state: GroundContactState,
    point: GroundTrajectoryPoint,
) -> bool:
    return (
        math.isclose(state.time_s, point.time_s, abs_tol=1e-10)
        and state.frame is point.frame
        and _positions_close(state.position_m, point.position_m)
        and _positions_close(state.velocity_m_s, point.velocity_m_s)
        and _positions_close(
            state.angular_velocity_rad_s,
            point.angular_velocity_rad_s,
        )
    )


__all__ = [
    "BOUNCE_HANDOFF_NOTICE",
    "BOUNCE_MATERIAL_LIMITATION",
    "BounceModelSettings",
    "BounceAirSegment",
    "BounceTermination",
    "BounceTerminationReason",
    "CancellationCheck",
    "GROUND_IMPACT_MODEL_ID",
    "GROUND_IMPACT_MODEL_VERSION",
    "RepeatedBounceResult",
    "STANDARD_GRAVITY_M_S2",
]

"""Validated internal result contracts for the ground skid/roll suffix."""

from __future__ import annotations

from dataclasses import dataclass

from .contract_types import (
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundFrame,
    GroundTrajectoryPoint,
)
from .regional_surface_types import SurfaceRegionTransition
from .request_identity import validate_request_fingerprint
from .surface_motion_types import (
    SkidRollEnergyLedger,
    SkidRollTermination,
    _finite,
)


@dataclass(frozen=True)
class SkidRollResult:
    """Validated non-wire suffix evidence beginning after a #4270 handoff."""

    request_id: str
    surface_id: str
    frame: GroundFrame
    model_id: str
    model_version: str
    request_fingerprint_sha256: str
    trajectory: tuple[GroundTrajectoryPoint, ...]
    events: tuple[GroundEvent, ...]
    surface_transitions: tuple[SurfaceRegionTransition, ...]
    final_state: GroundContactState
    skid_distance_m: float
    roll_distance_m: float
    energy: SkidRollEnergyLedger
    termination: SkidRollTermination

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_fingerprint_sha256",
            validate_request_fingerprint(self.request_fingerprint_sha256),
        )
        points = tuple(self.trajectory)
        events = tuple(self.events)
        transitions = tuple(self.surface_transitions)
        if any(type(point) is not GroundTrajectoryPoint for point in points):
            raise ValueError("suffix trajectory requires exact points")
        if any(type(event) is not GroundEvent for event in events):
            raise ValueError("suffix event ledger requires exact events")
        if any(type(item) is not SurfaceRegionTransition for item in transitions):
            raise ValueError("suffix transition ledger requires exact records")
        if type(self.final_state) is not GroundContactState:
            raise ValueError("suffix final_state must be an exact contact state")
        if type(self.energy) is not SkidRollEnergyLedger:
            raise ValueError("suffix requires an exact energy ledger")
        if type(self.termination) is not SkidRollTermination:
            raise ValueError("suffix requires an exact termination")
        object.__setattr__(self, "frame", GroundFrame(self.frame))
        object.__setattr__(self, "trajectory", points)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "surface_transitions", transitions)
        object.__setattr__(
            self, "skid_distance_m", _finite(self.skid_distance_m, "skid_distance_m")
        )
        object.__setattr__(
            self, "roll_distance_m", _finite(self.roll_distance_m, "roll_distance_m")
        )
        if self.skid_distance_m < 0.0 or self.roll_distance_m < 0.0:
            raise ValueError("suffix path distances must be nonnegative")
        self._validate_sequence()

    def _validate_sequence(self) -> None:
        if not self.request_id or not self.surface_id:
            raise ValueError("suffix identities must be nonempty")
        if not self.model_id or not self.model_version:
            raise ValueError("suffix model identity must be nonempty")
        if self.final_state.frame is not self.frame:
            raise ValueError("suffix final state frame must match result frame")
        if any(point.frame is not self.frame for point in self.trajectory):
            raise ValueError("suffix trajectory frame must match result frame")
        if any(event.frame is not self.frame for event in self.events):
            raise ValueError("suffix event frame must match result frame")
        if any(
            right.time_s <= left.time_s
            for left, right in zip(self.trajectory, self.trajectory[1:], strict=False)
        ):
            raise ValueError("suffix trajectory times must be strictly increasing")
        if self.termination.time_s != self.final_state.time_s:
            raise ValueError("suffix termination must match final state time")
        self._validate_events()
        if self.trajectory and not _point_matches_state(
            self.trajectory[-1], self.final_state
        ):
            raise ValueError("suffix final state must match terminal trajectory point")

    def _validate_events(self) -> None:
        if self.events:
            first_sequence = self.events[0].sequence
            expected = tuple(range(first_sequence, first_sequence + len(self.events)))
            if tuple(event.sequence for event in self.events) != expected:
                raise ValueError("suffix event sequence must be contiguous")
            if any(
                right.time_s < left.time_s
                for left, right in zip(self.events, self.events[1:], strict=False)
            ):
                raise ValueError("suffix event times must be nondecreasing")
            if self.events[-1].time_s > self.termination.time_s:
                raise ValueError("suffix event cannot follow termination")
        regional_events = tuple(
            event
            for event in self.events
            if event.event_type is GroundEventType.SURFACE_TRANSITION
        )
        if len(regional_events) != len(self.surface_transitions):
            raise ValueError("surface transition ledger must match emitted events")
        for event, transition in zip(
            regional_events,
            self.surface_transitions,
            strict=True,
        ):
            if (
                transition.event_sequence != event.sequence
                or transition.time_s != event.time_s
                or transition.position_m != event.position_m
            ):
                raise ValueError("surface transition evidence must match its event")


def _point_matches_state(
    point: GroundTrajectoryPoint,
    state: GroundContactState,
) -> bool:
    return (
        point.time_s == state.time_s
        and point.frame is state.frame
        and point.position_m == state.position_m
        and point.velocity_m_s == state.velocity_m_s
        and point.angular_velocity_rad_s == state.angular_velocity_rad_s
    )


__all__ = ["SkidRollResult"]

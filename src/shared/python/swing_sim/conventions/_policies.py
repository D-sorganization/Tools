"""Measurement policies and provenance for all audited launch-monitor parameters."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import (
    AvailabilityRule,
    ConventionId,
    EventTime,
    ParameterId,
    QuantityStatus,
    ReferencePoint,
    SignRule,
)

FRAME = "target_frame:x_target,y_up,z_right"
RETRIEVED = "2026-08-05"

APP_SOURCE = (
    "https://github.com/D-sorganization/Tools/blob/main/docs/specs/D_PLANE_GEOMETRY.md"
)
TRACKMAN_CLUB_SOURCE = "https://www.trackman.com/blog/golf/club-data-definitions"
TRACKMAN_PARAMETERS_SOURCE = "https://www.trackman.com/blog/golf/40-trackman-parameters"
FORESIGHT_CLUB_SOURCE = (
    "https://help.foresightsports.com/hc/en-us/articles/"
    "47214673873811-Club-Head-Data-Measurements-Definitions"
)
FORESIGHT_BALL_SOURCE = (
    "https://help.foresightsports.com/hc/en-us/articles/"
    "47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results"
)


@dataclass(frozen=True)
class Policy:
    """Convention-specific measurement policy and provenance."""

    reference: ReferencePoint
    event: EventTime
    status: QuantityStatus
    availability: AvailabilityRule
    source: str
    sign: SignRule | None = None


def _app_club(
    ref: ReferencePoint, ev: EventTime = EventTime.INSPECTION_EVENT
) -> Policy:
    return Policy(
        ref,
        ev,
        QuantityStatus.DERIVED,
        AvailabilityRule.NONZERO_CLUB_TRAVEL,
        APP_SOURCE,
    )


def _app_face(ref: ReferencePoint) -> Policy:
    return Policy(
        ref,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        APP_SOURCE,
    )


def _app_ball_launch(ev: EventTime = EventTime.JUST_AFTER_SEPARATION) -> Policy:
    return Policy(
        ReferencePoint.BALL_CENTER,
        ev,
        QuantityStatus.MODELED,
        AvailabilityRule.COLLISION_COMPLETE,
        APP_SOURCE,
    )


def _app_flight(ev: EventTime) -> Policy:
    return Policy(
        ReferencePoint.BALL_CENTER,
        ev,
        QuantityStatus.MODELED,
        AvailabilityRule.TRAJECTORY_COMPLETE,
        APP_SOURCE,
    )


APP_POLICIES: dict[ParameterId, Policy] = {
    ParameterId.CLUB_SPEED: _app_club(ReferencePoint.TRACKED_HEAD_REFERENCE),
    ParameterId.CLUB_PATH: _app_club(ReferencePoint.TRACKED_HEAD_REFERENCE),
    ParameterId.ATTACK_ANGLE: _app_club(ReferencePoint.TRACKED_HEAD_REFERENCE),
    ParameterId.DYNAMIC_LIE: _app_face(ReferencePoint.FACE_CENTER),
    ParameterId.CLOSURE_RATE: _app_face(ReferencePoint.FACE_CENTER),
    ParameterId.SWING_DIRECTION: _app_club(ReferencePoint.TRACKED_HEAD_REFERENCE),
    ParameterId.LOW_POINT: _app_club(ReferencePoint.TRACKED_HEAD_REFERENCE),
    ParameterId.FACE_ANGLE: _app_face(ReferencePoint.FACE_CENTER),
    ParameterId.DYNAMIC_LOFT: _app_face(ReferencePoint.FACE_CENTER),
    ParameterId.FACE_TO_PATH: _app_face(ReferencePoint.MIXED_CLUB_DELIVERY),
    ParameterId.SPIN_LOFT: _app_face(ReferencePoint.MIXED_CLUB_DELIVERY),
    ParameterId.IMPACT_OFFSET: _app_face(ReferencePoint.IMPACT_LOCATION),
    ParameterId.IMPACT_HEIGHT: _app_face(ReferencePoint.IMPACT_LOCATION),
    ParameterId.BALL_SPEED: _app_ball_launch(),
    ParameterId.LAUNCH_ANGLE: _app_ball_launch(),
    ParameterId.LAUNCH_DIRECTION: _app_ball_launch(),
    ParameterId.SMASH_FACTOR: Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.JUST_AFTER_SEPARATION,
        QuantityStatus.DERIVED,
        AvailabilityRule.COLLISION_COMPLETE,
        APP_SOURCE,
    ),
    ParameterId.TOTAL_SPIN: _app_ball_launch(),
    ParameterId.SPIN_AXIS: _app_ball_launch(),
    ParameterId.BACK_SPIN: _app_ball_launch(),
    ParameterId.SIDE_SPIN: _app_ball_launch(),
    ParameterId.APEX_HEIGHT: _app_flight(EventTime.APEX),
    ParameterId.CARRY_DISTANCE: _app_flight(EventTime.LANDING),
    ParameterId.TOTAL_DISTANCE: _app_flight(EventTime.LANDING),
    ParameterId.CARRY_OFFLINE: _app_flight(EventTime.LANDING),
    ParameterId.CURVE: _app_flight(EventTime.LANDING),
    ParameterId.FLIGHT_TIME: _app_flight(EventTime.FLIGHT_DURATION),
    ParameterId.LANDING_ANGLE: _app_flight(EventTime.LANDING),
}


def _tm_club(ref: ReferencePoint, ev: EventTime, src: str) -> Policy:
    return Policy(
        ref,
        ev,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.NONZERO_CLUB_TRAVEL,
        src,
    )


def _tm_face(ref: ReferencePoint, ev: EventTime, src: str) -> Policy:
    return Policy(
        ref, ev, QuantityStatus.MEASURED_COMPARABLE, AvailabilityRule.FACE_GEOMETRY, src
    )


def _tm_ball(ev: EventTime = EventTime.JUST_AFTER_SEPARATION) -> Policy:
    return Policy(
        ReferencePoint.BALL_CENTER,
        ev,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.COLLISION_COMPLETE,
        TRACKMAN_PARAMETERS_SOURCE,
    )


def _tm_flight(ev: EventTime) -> Policy:
    return Policy(
        ReferencePoint.BALL_CENTER,
        ev,
        QuantityStatus.MODELED,
        AvailabilityRule.TRAJECTORY_COMPLETE,
        TRACKMAN_PARAMETERS_SOURCE,
    )


TRACKMAN_POLICIES: dict[ParameterId, Policy] = {
    ParameterId.CLUB_SPEED: _tm_club(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.JUST_BEFORE_FIRST_CONTACT,
        TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.CLUB_PATH: _tm_club(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.MAXIMUM_COMPRESSION,
        TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.ATTACK_ANGLE: _tm_club(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.MAXIMUM_COMPRESSION,
        TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.DYNAMIC_LIE: _tm_face(
        ReferencePoint.FACE_CENTER, EventTime.MAXIMUM_COMPRESSION, TRACKMAN_CLUB_SOURCE
    ),
    ParameterId.CLOSURE_RATE: Policy(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.SWING_DIRECTION: _tm_club(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.JUST_BEFORE_FIRST_CONTACT,
        TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.LOW_POINT: _tm_club(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.MAXIMUM_COMPRESSION,
        TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.FACE_ANGLE: _tm_face(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.MAXIMUM_COMPRESSION,
        TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.DYNAMIC_LOFT: _tm_face(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.MAXIMUM_COMPRESSION,
        TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.FACE_TO_PATH: Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.SPIN_LOFT: Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.IMPACT_OFFSET: _tm_face(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.MAXIMUM_COMPRESSION,
        TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.IMPACT_HEIGHT: _tm_face(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.MAXIMUM_COMPRESSION,
        TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.BALL_SPEED: _tm_ball(),
    ParameterId.LAUNCH_ANGLE: _tm_ball(),
    ParameterId.LAUNCH_DIRECTION: _tm_ball(),
    ParameterId.SMASH_FACTOR: Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.JUST_AFTER_SEPARATION,
        QuantityStatus.DERIVED,
        AvailabilityRule.COLLISION_COMPLETE,
        TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.TOTAL_SPIN: _tm_ball(),
    ParameterId.SPIN_AXIS: _tm_ball(),
    ParameterId.BACK_SPIN: _tm_ball(),
    ParameterId.SIDE_SPIN: _tm_ball(),
    ParameterId.APEX_HEIGHT: _tm_flight(EventTime.APEX),
    ParameterId.CARRY_DISTANCE: _tm_flight(EventTime.LANDING),
    ParameterId.TOTAL_DISTANCE: _tm_flight(EventTime.LANDING),
    ParameterId.CARRY_OFFLINE: _tm_flight(EventTime.LANDING),
    ParameterId.CURVE: _tm_flight(EventTime.LANDING),
    ParameterId.FLIGHT_TIME: _tm_flight(EventTime.FLIGHT_DURATION),
    ParameterId.LANDING_ANGLE: _tm_flight(EventTime.LANDING),
}


def _fs_club(ref: ReferencePoint, ev: EventTime, src: str) -> Policy:
    return Policy(
        ref,
        ev,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.NONZERO_CLUB_TRAVEL,
        src,
    )


def _fs_face(ref: ReferencePoint, ev: EventTime, src: str) -> Policy:
    return Policy(
        ref, ev, QuantityStatus.MEASURED_COMPARABLE, AvailabilityRule.FACE_GEOMETRY, src
    )


def _fs_ball(
    ev: EventTime = EventTime.JUST_AFTER_SEPARATION, sign: SignRule | None = None
) -> Policy:
    return Policy(
        ReferencePoint.BALL_CENTER,
        ev,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.COLLISION_COMPLETE,
        FORESIGHT_BALL_SOURCE,
        sign=sign,
    )


def _fs_flight(ev: EventTime) -> Policy:
    return Policy(
        ReferencePoint.BALL_CENTER,
        ev,
        QuantityStatus.MODELED,
        AvailabilityRule.TRAJECTORY_COMPLETE,
        FORESIGHT_BALL_SOURCE,
    )


FORESIGHT_POLICIES: dict[ParameterId, Policy] = {
    ParameterId.CLUB_SPEED: _fs_club(
        ReferencePoint.FACE_CENTER,
        EventTime.JUST_BEFORE_FIRST_CONTACT,
        FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.CLUB_PATH: _fs_club(
        ReferencePoint.FACE_CENTER, EventTime.IMPACT, FORESIGHT_CLUB_SOURCE
    ),
    ParameterId.ATTACK_ANGLE: _fs_club(
        ReferencePoint.FACE_CENTER, EventTime.IMPACT, FORESIGHT_CLUB_SOURCE
    ),
    ParameterId.DYNAMIC_LIE: _fs_face(
        ReferencePoint.FACE_CENTER, EventTime.IMPACT, FORESIGHT_CLUB_SOURCE
    ),
    ParameterId.CLOSURE_RATE: Policy(
        ReferencePoint.FACE_CENTER,
        EventTime.IMPACT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.SWING_DIRECTION: Policy(
        ReferencePoint.FACE_CENTER,
        EventTime.IMPACT,
        QuantityStatus.UNAVAILABLE,
        AvailabilityRule.UNAVAILABLE,
        FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.LOW_POINT: Policy(
        ReferencePoint.FACE_CENTER,
        EventTime.IMPACT,
        QuantityStatus.UNAVAILABLE,
        AvailabilityRule.UNAVAILABLE,
        FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.FACE_ANGLE: _fs_face(
        ReferencePoint.IMPACT_LOCATION, EventTime.IMPACT, FORESIGHT_CLUB_SOURCE
    ),
    ParameterId.DYNAMIC_LOFT: _fs_face(
        ReferencePoint.IMPACT_LOCATION, EventTime.IMPACT, FORESIGHT_CLUB_SOURCE
    ),
    ParameterId.FACE_TO_PATH: Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.IMPACT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.SPIN_LOFT: Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.IMPACT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.IMPACT_OFFSET: _fs_face(
        ReferencePoint.IMPACT_LOCATION, EventTime.IMPACT, FORESIGHT_CLUB_SOURCE
    ),
    ParameterId.IMPACT_HEIGHT: _fs_face(
        ReferencePoint.IMPACT_LOCATION, EventTime.IMPACT, FORESIGHT_CLUB_SOURCE
    ),
    ParameterId.BALL_SPEED: _fs_ball(),
    ParameterId.LAUNCH_ANGLE: _fs_ball(),
    ParameterId.LAUNCH_DIRECTION: _fs_ball(sign=SignRule.UNSPECIFIED),
    ParameterId.SMASH_FACTOR: Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.JUST_AFTER_SEPARATION,
        QuantityStatus.DERIVED,
        AvailabilityRule.COLLISION_COMPLETE,
        FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.TOTAL_SPIN: _fs_ball(),
    ParameterId.SPIN_AXIS: _fs_ball(),
    ParameterId.BACK_SPIN: _fs_ball(),
    ParameterId.SIDE_SPIN: _fs_ball(),
    ParameterId.APEX_HEIGHT: _fs_flight(EventTime.APEX),
    ParameterId.CARRY_DISTANCE: _fs_flight(EventTime.LANDING),
    ParameterId.TOTAL_DISTANCE: _fs_flight(EventTime.LANDING),
    ParameterId.CARRY_OFFLINE: _fs_flight(EventTime.LANDING),
    ParameterId.CURVE: Policy(
        ReferencePoint.BALL_CENTER,
        EventTime.LANDING,
        QuantityStatus.UNAVAILABLE,
        AvailabilityRule.UNAVAILABLE,
        FORESIGHT_BALL_SOURCE,
    ),
    ParameterId.FLIGHT_TIME: _fs_flight(EventTime.FLIGHT_DURATION),
    ParameterId.LANDING_ANGLE: _fs_flight(EventTime.LANDING),
}

POLICIES: dict[ConventionId, dict[ParameterId, Policy]] = {
    ConventionId.APP_NATIVE: APP_POLICIES,
    ConventionId.TRACKMAN_COMPARABLE: TRACKMAN_POLICIES,
    ConventionId.FORESIGHT_COMPARABLE: FORESIGHT_POLICIES,
}

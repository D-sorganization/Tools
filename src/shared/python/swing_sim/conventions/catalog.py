"""Source-backed foundation catalog for convention-aware calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .registry import (
    AvailabilityRule,
    ConventionId,
    ConventionRegistry,
    EventTime,
    ParameterDefinition,
    ParameterId,
    QuantityStatus,
    ReferencePoint,
    SignRule,
)

_FRAME = "target_frame:x_target,y_up,z_right"
_RETRIEVED = "2026-08-05"
_APP_SOURCE = (
    "https://github.com/D-sorganization/Tools/blob/main/docs/specs/D_PLANE_GEOMETRY.md"
)
_TRACKMAN_CLUB_SOURCE = "https://www.trackman.com/blog/golf/club-data-definitions"
_TRACKMAN_PARAMETERS_SOURCE = (
    "https://www.trackman.com/blog/golf/40-trackman-parameters"
)
_FORESIGHT_CLUB_SOURCE = "https://help.foresightsports.com/hc/en-us/articles/47214673873811-Club-Head-Data-Measurements-Definitions"
_FORESIGHT_BALL_SOURCE = "https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results"


@dataclass(frozen=True)
class _Identity:
    label: str
    unit: str
    sign: SignRule
    geometry: str


@dataclass(frozen=True)
class _Policy:
    reference: ReferencePoint
    event: EventTime
    status: QuantityStatus
    availability: AvailabilityRule
    source: str
    sign: SignRule | None = None


_IDENTITIES = {
    ParameterId.CLUB_SPEED: _Identity(
        "Club Speed", "m/s", SignRule.NONNEGATIVE, "magnitude(club_velocity)"
    ),
    ParameterId.CLUB_PATH: _Identity(
        "Club Path", "deg", SignRule.POSITIVE_RIGHT, "heading(club_velocity)"
    ),
    ParameterId.ATTACK_ANGLE: _Identity(
        "Attack Angle", "deg", SignRule.POSITIVE_UP, "elevation(club_velocity)"
    ),
    ParameterId.FACE_ANGLE: _Identity(
        "Face Angle", "deg", SignRule.POSITIVE_RIGHT, "heading(face_normal)"
    ),
    ParameterId.DYNAMIC_LOFT: _Identity(
        "Dynamic Loft", "deg", SignRule.POSITIVE_UP, "elevation(face_normal)"
    ),
    ParameterId.FACE_TO_PATH: _Identity(
        "Face to Path", "deg", SignRule.POSITIVE_RIGHT, "wrapped(face_angle-club_path)"
    ),
    ParameterId.SPIN_LOFT: _Identity(
        "Spin Loft", "deg", SignRule.NONNEGATIVE, "angle_3d(club_velocity,face_normal)"
    ),
    ParameterId.LAUNCH_DIRECTION: _Identity(
        "Launch Direction",
        "deg",
        SignRule.POSITIVE_RIGHT,
        "heading(initial_ball_velocity)",
    ),
}


def _club_policy(
    reference: ReferencePoint,
    event: EventTime,
    status: QuantityStatus,
    source: str,
) -> _Policy:
    return _Policy(
        reference, event, status, AvailabilityRule.NONZERO_CLUB_TRAVEL, source
    )


_APP_POLICIES = {
    ParameterId.CLUB_SPEED: _club_policy(
        ReferencePoint.TRACKED_HEAD_REFERENCE,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        _APP_SOURCE,
    ),
    ParameterId.CLUB_PATH: _club_policy(
        ReferencePoint.TRACKED_HEAD_REFERENCE,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        _APP_SOURCE,
    ),
    ParameterId.ATTACK_ANGLE: _club_policy(
        ReferencePoint.TRACKED_HEAD_REFERENCE,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        _APP_SOURCE,
    ),
    ParameterId.FACE_ANGLE: _Policy(
        ReferencePoint.FACE_CENTER,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _APP_SOURCE,
    ),
    ParameterId.DYNAMIC_LOFT: _Policy(
        ReferencePoint.FACE_CENTER,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _APP_SOURCE,
    ),
    ParameterId.FACE_TO_PATH: _Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _APP_SOURCE,
    ),
    ParameterId.SPIN_LOFT: _Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.INSPECTION_EVENT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _APP_SOURCE,
    ),
    ParameterId.LAUNCH_DIRECTION: _Policy(
        ReferencePoint.BALL_CENTER,
        EventTime.JUST_AFTER_SEPARATION,
        QuantityStatus.MODELED,
        AvailabilityRule.COLLISION_COMPLETE,
        _APP_SOURCE,
    ),
}


_TRACKMAN_POLICIES = {
    ParameterId.CLUB_SPEED: _club_policy(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.JUST_BEFORE_FIRST_CONTACT,
        QuantityStatus.MEASURED_COMPARABLE,
        _TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.CLUB_PATH: _club_policy(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.MEASURED_COMPARABLE,
        _TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.ATTACK_ANGLE: _club_policy(
        ReferencePoint.GEOMETRIC_CENTER,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.MEASURED_COMPARABLE,
        _TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.FACE_ANGLE: _Policy(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.FACE_GEOMETRY,
        _TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.DYNAMIC_LOFT: _Policy(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.FACE_GEOMETRY,
        _TRACKMAN_CLUB_SOURCE,
    ),
    ParameterId.FACE_TO_PATH: _Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.SPIN_LOFT: _Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.MAXIMUM_COMPRESSION,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _TRACKMAN_PARAMETERS_SOURCE,
    ),
    ParameterId.LAUNCH_DIRECTION: _Policy(
        ReferencePoint.BALL_CENTER,
        EventTime.JUST_AFTER_SEPARATION,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.COLLISION_COMPLETE,
        _TRACKMAN_PARAMETERS_SOURCE,
    ),
}


_FORESIGHT_POLICIES = {
    ParameterId.CLUB_SPEED: _club_policy(
        ReferencePoint.FACE_CENTER,
        EventTime.JUST_BEFORE_FIRST_CONTACT,
        QuantityStatus.MEASURED_COMPARABLE,
        _FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.CLUB_PATH: _club_policy(
        ReferencePoint.FACE_CENTER,
        EventTime.IMPACT,
        QuantityStatus.MEASURED_COMPARABLE,
        _FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.ATTACK_ANGLE: _club_policy(
        ReferencePoint.FACE_CENTER,
        EventTime.IMPACT,
        QuantityStatus.MEASURED_COMPARABLE,
        _FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.FACE_ANGLE: _Policy(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.IMPACT,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.FACE_GEOMETRY,
        _FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.DYNAMIC_LOFT: _Policy(
        ReferencePoint.IMPACT_LOCATION,
        EventTime.IMPACT,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.FACE_GEOMETRY,
        _FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.FACE_TO_PATH: _Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.IMPACT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.SPIN_LOFT: _Policy(
        ReferencePoint.MIXED_CLUB_DELIVERY,
        EventTime.IMPACT,
        QuantityStatus.DERIVED,
        AvailabilityRule.FACE_GEOMETRY,
        _FORESIGHT_CLUB_SOURCE,
    ),
    ParameterId.LAUNCH_DIRECTION: _Policy(
        ReferencePoint.BALL_CENTER,
        EventTime.JUST_AFTER_SEPARATION,
        QuantityStatus.MEASURED_COMPARABLE,
        AvailabilityRule.COLLISION_COMPLETE,
        _FORESIGHT_BALL_SOURCE,
        SignRule.UNSPECIFIED,
    ),
}


_POLICIES = {
    ConventionId.APP_NATIVE: _APP_POLICIES,
    ConventionId.TRACKMAN_COMPARABLE: _TRACKMAN_POLICIES,
    ConventionId.FORESIGHT_COMPARABLE: _FORESIGHT_POLICIES,
}


@lru_cache(maxsize=1)
def convention_registry() -> ConventionRegistry:
    """Return the immutable source-backed foundation registry."""
    definitions = []
    for convention, policies in _POLICIES.items():
        for parameter, identity in _IDENTITIES.items():
            policy = policies[parameter]
            definitions.append(
                ParameterDefinition(
                    convention,
                    parameter,
                    identity.label,
                    policy.source,
                    _RETRIEVED,
                    policy.reference,
                    policy.event,
                    _FRAME,
                    identity.geometry,
                    policy.sign or identity.sign,
                    identity.unit,
                    policy.status,
                    policy.availability,
                )
            )
    return ConventionRegistry(tuple(definitions))

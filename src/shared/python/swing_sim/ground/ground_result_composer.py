"""Compose #4270 impact/bounce and #4271 surface evidence into v1 results."""

from __future__ import annotations

import math

from .bounce_types import (
    BOUNCE_HANDOFF_NOTICE,
    BounceTerminationReason,
    RepeatedBounceResult,
)
from .contract_records import GroundSimulationRequest, GroundSimulationResult
from .contract_types import (
    GroundEventType,
    GroundPhase,
    GroundResultStatus,
    GroundTerminationReason,
    GroundTrajectoryPoint,
    GroundWarningSeverity,
)
from .request_identity import ground_request_fingerprint
from .result_types import GroundSummary, GroundTermination, GroundWarning
from .skid_roll_result_types import SkidRollResult
from .surface_motion_types import SkidRollTerminationReason


class GroundCompositionError(ValueError):
    """Reject internal outcomes that v1 cannot serialize without fabrication."""


_TERMINATION_MAP = {
    SkidRollTerminationReason.REST: (
        GroundResultStatus.COMPLETE,
        GroundTerminationReason.REST,
    ),
    SkidRollTerminationReason.LEFT_SURFACE: (
        GroundResultStatus.COMPLETE,
        GroundTerminationReason.LEFT_SURFACE,
    ),
    SkidRollTerminationReason.TIME_LIMIT: (
        GroundResultStatus.PARTIAL,
        GroundTerminationReason.TIME_LIMIT,
    ),
    SkidRollTerminationReason.EVENT_LIMIT: (
        GroundResultStatus.PARTIAL,
        GroundTerminationReason.EVENT_LIMIT,
    ),
}


def _validate_inputs(
    request: GroundSimulationRequest,
    prefix: RepeatedBounceResult,
    suffix: SkidRollResult,
) -> None:
    if type(request) is not GroundSimulationRequest:
        raise GroundCompositionError("composer requires an exact request")
    if type(prefix) is not RepeatedBounceResult or type(suffix) is not SkidRollResult:
        raise GroundCompositionError(
            "composer requires exact prefix and suffix results"
        )
    if prefix.termination.reason is not BounceTerminationReason.SETTLED_TO_SKID:
        raise GroundCompositionError("composer requires a settled bounce prefix")
    handoff = prefix.handoff_state
    if handoff is None:
        raise GroundCompositionError("composer requires an exact physical handoff")
    if (
        request.request_id != prefix.request_id
        or request.request_id != suffix.request_id
    ):
        raise GroundCompositionError("request and result identities must match")
    if (
        request.surface.surface_id != prefix.surface_id
        or prefix.surface_id != suffix.surface_id
    ):
        raise GroundCompositionError("request and result surface identities must match")
    request_fingerprint = ground_request_fingerprint(request)
    if (
        prefix.request_fingerprint_sha256 != request_fingerprint
        or suffix.request_fingerprint_sha256 != request_fingerprint
    ):
        raise GroundCompositionError(
            "phase request fingerprints must match the request"
        )
    if handoff != suffix.final_state and suffix.termination.time_s == handoff.time_s:
        raise GroundCompositionError("zero-duration suffix cannot change handoff state")
    if suffix.termination.reason not in _TERMINATION_MAP:
        raise GroundCompositionError(
            "internal skid/roll outcome is not representable in v1"
        )


def _impact_point_from_first_event(
    prefix: RepeatedBounceResult,
) -> GroundTrajectoryPoint:
    event = prefix.events[0]
    return GroundTrajectoryPoint(
        event.time_s,
        event.frame,
        event.position_m,
        event.velocity_after_m_s,
        event.angular_velocity_after_rad_s,
        GroundPhase.IMPACT,
    )


def _prefix_trajectory(
    prefix: RepeatedBounceResult,
) -> tuple[GroundTrajectoryPoint, ...]:
    points = _typed_trajectory(prefix.trajectory)
    if not points:
        raise GroundCompositionError("bounce prefix trajectory must be nonempty")
    if points[0].phase is GroundPhase.IMPACT:
        return points
    if len(points) != 1 or points[0].phase is not GroundPhase.SKID:
        raise GroundCompositionError("bounce prefix lacks a representable first impact")
    reconstructed = _impact_point_from_first_event(prefix)
    point = points[0]
    same_numeric_state = (
        reconstructed.time_s == point.time_s
        and reconstructed.position_m == point.position_m
        and reconstructed.velocity_m_s == point.velocity_m_s
        and reconstructed.angular_velocity_rad_s == point.angular_velocity_rad_s
    )
    if not same_numeric_state:
        raise GroundCompositionError(
            "immediate capture does not match first-contact state"
        )
    return (reconstructed,)


def _typed_trajectory(
    points: tuple[GroundTrajectoryPoint, ...],
) -> tuple[GroundTrajectoryPoint, ...]:
    """Keep the imported prefix boundary explicit under isolated MyPy."""
    return points


def _trajectory(
    prefix: RepeatedBounceResult,
    suffix: SkidRollResult,
) -> tuple[GroundTrajectoryPoint, ...]:
    prefix_points = _prefix_trajectory(prefix)
    handoff = prefix.handoff_state
    if handoff is None:
        raise GroundCompositionError("composer requires a handoff")
    if suffix.termination.reason is SkidRollTerminationReason.REST:
        if suffix.termination.time_s <= handoff.time_s:
            raise GroundCompositionError(
                "zero-duration rest is not representable in v1"
            )
    if any(point.time_s <= handoff.time_s for point in suffix.trajectory):
        raise GroundCompositionError("suffix points must begin strictly after handoff")
    return prefix_points + suffix.trajectory


def _summary(
    prefix: RepeatedBounceResult,
    suffix: SkidRollResult,
    trajectory: tuple[GroundTrajectoryPoint, ...],
) -> GroundSummary:
    first = trajectory[0].position_m
    final = trajectory[-1].position_m
    bounce_count = sum(
        event.event_type is GroundEventType.BOUNCE for event in prefix.events
    )
    return GroundSummary(
        carry_distance_m=math.hypot(first[0], first[2]),
        bounce_air_distance_m=prefix.bounce_air_distance_m,
        skid_distance_m=suffix.skid_distance_m,
        roll_distance_m=suffix.roll_distance_m,
        surface_path_distance_m=suffix.skid_distance_m + suffix.roll_distance_m,
        total_distance_m=math.hypot(final[0], final[2]),
        final_downrange_m=final[0],
        final_offline_m=final[2],
        bounce_count=bounce_count,
    )


def _warnings(suffix: SkidRollResult) -> tuple[GroundWarning, ...]:
    has_regions = any(
        event.event_type is GroundEventType.SURFACE_TRANSITION
        for event in suffix.events
    )
    domain_warning = (
        GroundWarning(
            "REGIONAL_PLANAR_V1",
            GroundWarningSeverity.INFO,
            "Qualified for deterministic coplanar material regions; changing "
            "normals and surface velocity at boundaries remain unsupported.",
        )
        if has_regions
        else GroundWarning(
            "STATIC_PLANE_V1",
            GroundWarningSeverity.INFO,
            "Qualified for one immutable planar profile; material regions "
            "and changing normals are unsupported.",
        )
    )
    warnings = [
        domain_warning,
        GroundWarning(
            "AXIAL_SPIN_UNDAMPED",
            GroundWarningSeverity.INFO,
            "Normal-axis spin is preserved because v1 has no calibrated "
            "torsional damping law.",
        ),
    ]
    if suffix.termination.reason is not SkidRollTerminationReason.REST:
        warnings.append(
            GroundWarning(
                "CENSORED_ENDPOINT",
                GroundWarningSeverity.WARNING,
                "Distance totals describe only the observed endpoint and are "
                "not projected final-rest metrics.",
            )
        )
    return tuple(warnings)


def _composed_warnings(
    prefix: RepeatedBounceResult,
    suffix: SkidRollResult,
) -> tuple[GroundWarning, ...]:
    prefix_warnings = tuple(
        GroundWarning(
            f"IMPACT_PREFIX_LIMITATION_{index:03d}",
            GroundWarningSeverity.INFO,
            message,
        )
        for index, message in enumerate(
            (
                warning
                for warning in prefix.warnings
                if warning != BOUNCE_HANDOFF_NOTICE
            ),
            start=1,
        )
    )
    return prefix_warnings + _warnings(suffix)


def compose_ground_result(
    request: GroundSimulationRequest,
    prefix: RepeatedBounceResult,
    suffix: SkidRollResult,
) -> GroundSimulationResult:
    """Build one honest v1 result without duplicate or epsilon-time samples."""
    _validate_inputs(request, prefix, suffix)
    trajectory = _trajectory(prefix, suffix)
    if not trajectory or trajectory[-1].time_s != suffix.termination.time_s:
        raise GroundCompositionError(
            "suffix must provide the exact terminal trajectory point"
        )
    status, reason = _TERMINATION_MAP[suffix.termination.reason]
    events = prefix.events + suffix.events
    return GroundSimulationResult(
        request_id=request.request_id,
        surface_id=request.surface.surface_id,
        frame=request.surface.frame,
        model_id=f"{prefix.model_id}+{suffix.model_id}",
        model_version=f"{prefix.model_version}+{suffix.model_version}",
        status=status,
        trajectory=trajectory,
        events=events,
        summary=_summary(prefix, suffix, trajectory),
        termination=GroundTermination(
            reason,
            suffix.termination.time_s,
            status is GroundResultStatus.COMPLETE,
        ),
        calibration=request.calibration,
        warnings=_composed_warnings(prefix, suffix),
        unavailable_fields=(),
        provenance=request.provenance,
    )


__all__ = ["GroundCompositionError", "compose_ground_result"]

"""Project qualified ground results into target- and solver-safe studies."""

from __future__ import annotations

import hashlib

from shared.python.swing_sim.solver.spatial_targets import SpatialTarget

from .contract_records import GroundSimulationRequest, GroundSimulationResult
from .contract_types import GroundResultStatus, GroundTerminationReason
from .profile_binding import BoundGroundSurface
from .request_identity import ground_request_fingerprint
from .study_derivation import derive_solver_eligibility, derive_study_status
from .study_geometry import (
    endpoint_contacts_surface,
    surface_contact_point,
    target_evaluation,
    validate_nonpenetrating_endpoint,
    validate_target,
)
from .study_record import GroundStudyProjection
from .study_target import canonical_ground_target
from .study_types import (
    GroundEndpointKind,
    GroundStudyMetrics,
    GroundStudyProfile,
    GroundTargetEvaluation,
    GroundTargetUnavailableReason,
)


def _validate_inputs(
    request: GroundSimulationRequest,
    result: GroundSimulationResult,
    bound_surface: BoundGroundSurface | None,
    target: SpatialTarget | None,
) -> None:
    if type(request) is not GroundSimulationRequest:
        raise TypeError("request must use the exact GroundSimulationRequest type")
    if type(result) is not GroundSimulationResult:
        raise TypeError("result must use the exact GroundSimulationResult type")
    if request.request_id != result.request_id:
        raise ValueError("request_id does not match ground result")
    if request.surface.surface_id != result.surface_id:
        raise ValueError("request surface does not match ground result")
    if request.surface.frame is not result.frame:
        raise ValueError("request frame does not match ground result")
    if request.calibration != result.calibration:
        raise ValueError("request calibration does not match ground result")
    if request.provenance != result.provenance:
        raise ValueError("request provenance does not match ground result")
    if bound_surface is not None:
        if type(bound_surface) is not BoundGroundSurface:
            raise TypeError("bound_surface must use the exact contract type")
        if bound_surface.surface != request.surface:
            raise ValueError("bound surface does not match request surface")
    if target is not None:
        if type(target) is not SpatialTarget:
            raise TypeError("target must use the exact SpatialTarget type")
        validate_target(target, request.surface)
    if result.trajectory:
        surface_contact_point(
            result.trajectory[0].position_m, request.surface, request.ball_radius_m
        )
        if result.status is GroundResultStatus.COMPLETE:
            surface_contact_point(
                result.trajectory[-1].position_m,
                request.surface,
                request.ball_radius_m,
            )
        elif result.status is GroundResultStatus.PARTIAL:
            validate_nonpenetrating_endpoint(
                result.trajectory[-1].position_m,
                request.surface,
                request.ball_radius_m,
            )


def _profile(bound_surface: BoundGroundSurface | None) -> GroundStudyProfile | None:
    if bound_surface is None:
        return None
    return GroundStudyProfile(
        bound_surface.profile,
        bound_surface.operating_condition,
        bound_surface.warnings,
    )


def _metrics(result: GroundSimulationResult) -> GroundStudyMetrics | None:
    if result.summary is None:
        return None
    points = result.trajectory
    return GroundStudyMetrics(
        result.summary,
        points[0].position_m,
        points[-1].position_m,
        result.termination.time_s - points[0].time_s,
    )


def _target_evaluations(
    request: GroundSimulationRequest,
    result: GroundSimulationResult,
    metrics: GroundStudyMetrics | None,
    target: SpatialTarget | None,
) -> tuple[
    GroundTargetEvaluation | None,
    GroundTargetEvaluation | None,
    GroundTargetUnavailableReason | None,
]:
    if target is None or metrics is None:
        return None, None, None
    first = target_evaluation(
        target,
        metrics.first_contact_position_m,
        GroundEndpointKind.FIRST_CONTACT,
        request,
    )
    endpoint_kind = (
        GroundEndpointKind.FINAL_REST
        if result.termination.reason is GroundTerminationReason.REST
        else GroundEndpointKind.FINAL_OBSERVED
    )
    if endpoint_contacts_surface(
        metrics.final_observed_position_m,
        request.surface,
        request.ball_radius_m,
    ):
        final = target_evaluation(
            target,
            metrics.final_observed_position_m,
            endpoint_kind,
            request,
        )
        return first, final, None
    return first, None, GroundTargetUnavailableReason.ENDPOINT_AIRBORNE


def project_ground_study(
    request: GroundSimulationRequest,
    result: GroundSimulationResult,
    *,
    bound_surface: BoundGroundSurface | None = None,
    target: SpatialTarget | None = None,
) -> GroundStudyProjection:
    """Build one honest study record without inferring a final-rest endpoint."""
    _validate_inputs(request, result, bound_surface, target)
    canonical_target = canonical_ground_target(target) if target is not None else None
    metrics = _metrics(result)
    profile = _profile(bound_surface)
    first_target, final_target, target_unavailable = _target_evaluations(
        request, result, metrics, canonical_target
    )
    return GroundStudyProjection(
        request.request_id,
        ground_request_fingerprint(request),
        hashlib.sha256(result.to_json().encode("utf-8")).hexdigest(),
        request.surface,
        request.ball_radius_m,
        result.model_id,
        result.model_version,
        result.calibration,
        result.provenance,
        result.status,
        derive_study_status(result.status, result.termination.reason),
        result.termination.reason,
        metrics,
        canonical_target,
        first_target,
        final_target,
        target_unavailable,
        derive_solver_eligibility(
            result.status,
            result.termination.reason,
            profile,
            result.calibration,
        ),
        profile,
        result.warnings,
        result.unavailable_fields,
    )


__all__ = ["project_ground_study"]

"""Fail-closed preconditions for the #4270 to #4271 physical handoff."""

from __future__ import annotations

from .bounce_types import BounceTerminationReason, RepeatedBounceResult
from .contract_records import GroundSimulationRequest
from .contract_types import GroundContactState
from .request_identity import ground_request_fingerprint
from .surface_motion_types import SkidRollSettings
from .surface_resolver import SurfaceResolver


def validate_surface_run_inputs(
    request: GroundSimulationRequest,
    prefix: RepeatedBounceResult,
    resolver: SurfaceResolver,
    settings: SkidRollSettings,
) -> GroundContactState:
    """Return the exact handoff after validating identity and contact geometry."""
    if type(request) is not GroundSimulationRequest:
        raise ValueError("skid/roll simulation requires an exact request")
    if type(prefix) is not RepeatedBounceResult:
        raise ValueError("skid/roll simulation requires an exact bounce prefix")
    if type(resolver) is not SurfaceResolver or type(settings) is not SkidRollSettings:
        raise ValueError("resolver and settings must be exact skid/roll records")
    if prefix.termination.reason is not BounceTerminationReason.SETTLED_TO_SKID:
        raise ValueError("skid/roll simulation requires SETTLED_TO_SKID")
    handoff = _typed_handoff(prefix.handoff_state)
    if handoff is None or prefix.request_id != request.request_id:
        raise ValueError("bounce prefix must expose the request handoff")
    if prefix.surface_id != request.surface.surface_id:
        raise ValueError("bounce prefix surface must match the request")
    if prefix.request_fingerprint_sha256 != ground_request_fingerprint(request):
        raise ValueError("bounce prefix request fingerprint must match the request")
    if prefix.frame is not request.surface.frame:
        raise ValueError("bounce prefix frame must match the request")
    if abs(request.surface.signed_gap_m(handoff, request.ball_radius_m)) > 1e-9:
        raise ValueError("skid/roll handoff must be at exact physical contact")
    if abs(request.surface.relative_normal_speed_m_s(handoff)) > 1e-9:
        raise ValueError("skid/roll handoff must have zero relative normal speed")
    resolver.validate_request(request)
    if not resolver.domain.contains(handoff.position_m):
        raise ValueError("skid/roll handoff must lie inside the surface domain")
    resolver.validate_handoff(handoff.position_m)
    return handoff


def _typed_handoff(value: GroundContactState | None) -> GroundContactState | None:
    """Keep the imported bounce boundary explicit under isolated MyPy."""
    return value


__all__ = ["validate_surface_run_inputs"]

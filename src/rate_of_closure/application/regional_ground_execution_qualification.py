"""Launch-origin qualification for regional-ground execution plans."""

from __future__ import annotations

from dataclasses import replace

from rate_of_closure.application._regional_ground_execution_job_values import sha256
from shared.python.swing_sim.flight import LaunchConditions, launch_relative_surface
from shared.python.swing_sim.ground import (
    GroundProvenance,
    GroundRegionalMaterialPlanRequest,
    GroundRegionalMaterialRegion,
    GroundSurfaceProfile,
)
from shared.python.swing_sim.ground.contract_wire import record_to_dict
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)

LAUNCH_ORIGIN_QUALIFIER_ID = "tools.rate_of_closure.launch-origin-qualification"
LAUNCH_ORIGIN_QUALIFIER_VERSION = "1.0.0"
LAUNCH_ORIGIN_QUALIFICATION_SCHEMA = (
    "rate-of-closure/regional-ground-launch-origin-qualification/v1"
)


def qualification_input_sha256(
    source_plan: GroundRegionalMaterialPlanRequest,
    launch: LaunchConditions,
    transfer_surface: GroundSurfaceProfile,
) -> str:
    """Hash every source and launch authority used by the translation."""
    _validate_authorities(source_plan, launch, transfer_surface)
    qualification_digest: str = sha256(
        {
            "schema_version": LAUNCH_ORIGIN_QUALIFICATION_SCHEMA,
            "source_plan_sha256": regional_plan_request_sha256(source_plan),
            "transfer_surface": record_to_dict(transfer_surface),
            "ball_radius_m": launch.ball_radius,
            "ball_setup": launch.ball_setup.to_json_dict(),
        }
    )
    return qualification_digest


def _validate_authorities(
    source_plan: GroundRegionalMaterialPlanRequest,
    launch: LaunchConditions,
    transfer_surface: GroundSurfaceProfile,
) -> None:
    if type(source_plan) is not GroundRegionalMaterialPlanRequest:
        raise TypeError("source_plan must be an exact regional material plan")
    if type(launch) is not LaunchConditions:
        raise TypeError("launch must be an exact LaunchConditions")
    if type(transfer_surface) is not GroundSurfaceProfile:
        raise TypeError("transfer_surface must be an exact GroundSurfaceProfile")
    if source_plan.base_surface != transfer_surface:
        raise ValueError("source plan base surface must equal transfer surface")


def _translated_region(
    region: GroundRegionalMaterialRegion,
    vertical_translation_m: float,
) -> GroundRegionalMaterialRegion:
    return replace(
        region,
        surface=replace(
            region.surface,
            height_m=region.surface.height_m + vertical_translation_m,
        ),
    )


def qualify_regional_plan_for_launch(
    source_plan: GroundRegionalMaterialPlanRequest,
    launch: LaunchConditions,
    transfer_surface: GroundSurfaceProfile,
    *,
    source_revision: str,
) -> GroundRegionalMaterialPlanRequest:
    """Translate one source plan into exact launch-origin coordinates.

    Preconditions:
        The source base surface is the exact transfer surface and every source
        overlay is already coplanar under the regional-plan v1 contract.
    Postconditions:
        Base, overlays, and axis origin share one vertical translation, while
        qualification provenance hashes the complete source plan and launch.
    """
    input_digest = qualification_input_sha256(source_plan, launch, transfer_surface)
    qualified_base = launch_relative_surface(
        transfer_surface,
        launch.ball_radius,
        launch.ball_setup,
    )
    translation = qualified_base.height_m - transfer_surface.height_m
    axis = source_plan.axis_origin_m
    return replace(
        source_plan,
        base_surface=qualified_base,
        axis_origin_m=(axis[0], axis[1] + translation, axis[2]),
        regions=tuple(
            _translated_region(region, translation) for region in source_plan.regions
        ),
        provenance=GroundProvenance(
            LAUNCH_ORIGIN_QUALIFIER_ID,
            LAUNCH_ORIGIN_QUALIFIER_VERSION,
            source_revision,
            input_digest,
        ),
    )


__all__ = [
    "LAUNCH_ORIGIN_QUALIFICATION_SCHEMA",
    "LAUNCH_ORIGIN_QUALIFIER_ID",
    "LAUNCH_ORIGIN_QUALIFIER_VERSION",
    "qualification_input_sha256",
    "qualify_regional_plan_for_launch",
]

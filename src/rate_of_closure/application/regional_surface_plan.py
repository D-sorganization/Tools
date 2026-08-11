"""UI-neutral adapter from an editor draft to the strict regional wire contract."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground.regional_plan_records import (
    REGIONAL_PLAN_GEOMETRY_MODEL,
    REGIONAL_PLAN_LIMITATIONS,
    REGIONAL_PLAN_REQUEST_SCHEMA_VERSION,
    GroundRegionalMaterialPlanRequest,
)
from shared.python.swing_sim.ground.regional_plan_wire import (
    regional_material_plan_request_from_dict,
)

MAX_EDITOR_REGIONS = 8
EDITOR_PROVIDER_ID = "tools.rate_of_closure.regional_surface_editor"
EDITOR_PROVIDER_VERSION = "1.0.0"
TARGET_FRAME = "target_frame:x_downrange,y_up,z_right"


@dataclass(frozen=True)
class SurfaceMaterialDraft:
    """Editable SI material values for one static coplanar surface."""

    surface_id: str
    normal_restitution: float
    static_friction: float
    kinetic_friction: float
    rolling_resistance: float
    firmness_pa: float
    hardness_fraction: float
    grass_height_m: float
    compressibility_fraction: float
    compression_damping_fraction: float
    turf_density_kg_m3: float
    moisture_fraction: float


@dataclass(frozen=True)
class RegionalOverlayDraft:
    """One finite overlay interval and its editable material values."""

    region_id: str
    precedence: int
    lower_coordinate_m: float
    upper_coordinate_m: float
    surface: SurfaceMaterialDraft


@dataclass(frozen=True)
class RegionalSurfacePlanDraft:
    """Session-only regional plan draft with explicit evidence qualification."""

    request_id: str
    lower_coordinate_m: float
    upper_coordinate_m: float
    source_revision: str
    calibration_kind: str
    base_surface: SurfaceMaterialDraft
    regions: tuple[RegionalOverlayDraft, ...]


def _illustrative_surface(surface_id: str, *, rough: bool) -> SurfaceMaterialDraft:
    """Return disclosed synthetic values for editor discovery and tests."""
    if rough:
        return SurfaceMaterialDraft(
            surface_id,
            0.31,
            0.52,
            0.41,
            0.08,
            700_000.0,
            0.48,
            0.035,
            0.34,
            0.38,
            240.0,
            0.42,
        )
    return SurfaceMaterialDraft(
        surface_id,
        0.42,
        0.35,
        0.28,
        0.04,
        1_200_000.0,
        0.70,
        0.012,
        0.20,
        0.25,
        180.0,
        0.30,
    )


def illustrative_regional_surface_plan_draft() -> RegionalSurfacePlanDraft:
    """Create the visibly unvalidated example shown on first editor launch."""
    region = RegionalOverlayDraft(
        "illustrative-rough-band",
        10,
        120.0,
        150.0,
        _illustrative_surface("illustrative-rough", rough=True),
    )
    return RegionalSurfacePlanDraft(
        "illustrative-regional-plan",
        0.0,
        300.0,
        "interactive-illustrative-draft-v1",
        "unvalidated",
        _illustrative_surface("illustrative-fairway", rough=False),
        (region,),
    )


def _surface_payload(surface: SurfaceMaterialDraft) -> dict[str, object]:
    """Bind editable material values to the qualified static flat geometry."""
    return {
        "surface_id": surface.surface_id,
        "provider_id": EDITOR_PROVIDER_ID,
        "provider_version": EDITOR_PROVIDER_VERSION,
        "frame": TARGET_FRAME,
        "height_m": 0.0,
        "normal_unit": [0.0, 1.0, 0.0],
        "surface_velocity_m_s": [0.0, 0.0, 0.0],
        "normal_restitution": surface.normal_restitution,
        "static_friction": surface.static_friction,
        "kinetic_friction": surface.kinetic_friction,
        "rolling_resistance": surface.rolling_resistance,
        "firmness_pa": surface.firmness_pa,
        "hardness_fraction": surface.hardness_fraction,
        "grass_height_m": surface.grass_height_m,
        "compressibility_fraction": surface.compressibility_fraction,
        "compression_damping_fraction": surface.compression_damping_fraction,
        "turf_density_kg_m3": surface.turf_density_kg_m3,
        "moisture_fraction": surface.moisture_fraction,
    }


def _source_digest(draft: RegionalSurfacePlanDraft) -> str:
    """Bind provenance to the actual editor draft, including qualification."""
    text = str(canonical_numeric_json(asdict(draft)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _region_payload(region: RegionalOverlayDraft) -> dict[str, object]:
    """Create the exact regional wire mapping for one editor row."""
    return {
        "region_id": region.region_id,
        "precedence": region.precedence,
        "lower_coordinate_m": region.lower_coordinate_m,
        "upper_coordinate_m": region.upper_coordinate_m,
        "surface": _surface_payload(region.surface),
    }


def validate_regional_surface_plan_draft(
    draft: RegionalSurfacePlanDraft,
) -> GroundRegionalMaterialPlanRequest:
    """Delegate a bounded editor draft to the authoritative wire validator."""
    if type(draft) is not RegionalSurfacePlanDraft:
        raise TypeError("draft must be an exact RegionalSurfacePlanDraft")
    if draft.calibration_kind != "unvalidated":
        raise ValueError("this editor slice supports unvalidated drafts only")
    if not 1 <= len(draft.regions) <= MAX_EDITOR_REGIONS:
        raise ValueError(f"editor supports one to at most {MAX_EDITOR_REGIONS} regions")
    payload = {
        "request_id": draft.request_id,
        "base_surface": _surface_payload(draft.base_surface),
        "axis_origin_m": [0.0, 0.0, 0.0],
        "axis_unit": [1.0, 0.0, 0.0],
        "lower_coordinate_m": draft.lower_coordinate_m,
        "upper_coordinate_m": draft.upper_coordinate_m,
        "regions": [_region_payload(region) for region in draft.regions],
        "provenance": {
            "producer": EDITOR_PROVIDER_ID,
            "producer_version": EDITOR_PROVIDER_VERSION,
            "source_revision": draft.source_revision,
            "input_sha256": _source_digest(draft),
        },
        "geometry_model": REGIONAL_PLAN_GEOMETRY_MODEL,
        "limitations": list(REGIONAL_PLAN_LIMITATIONS),
        "unit_system": "SI",
        "schema_version": REGIONAL_PLAN_REQUEST_SCHEMA_VERSION,
    }
    return regional_material_plan_request_from_dict(payload)


__all__ = [
    "MAX_EDITOR_REGIONS",
    "RegionalOverlayDraft",
    "RegionalSurfacePlanDraft",
    "SurfaceMaterialDraft",
    "illustrative_regional_surface_plan_draft",
    "validate_regional_surface_plan_draft",
]

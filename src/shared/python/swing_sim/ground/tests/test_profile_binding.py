"""Tests for binding qualified material profiles to solver surfaces."""

from __future__ import annotations

from dataclasses import replace

import pytest

from shared.python.swing_sim.ground.contract_types import GroundFrame
from shared.python.swing_sim.ground.profile_binding import (
    PROFILE_ILLUSTRATIVE_WARNING,
    PROFILE_UNQUALIFIED_WARNING,
    BoundGroundSurface,
    ProfileOperatingCondition,
    SurfacePlacement,
    bind_material_profile,
)
from shared.python.swing_sim.ground.profile_types import (
    GroundEvidenceKind,
    GroundProfileRights,
    GroundQualificationStatus,
)

from .test_profile_contract import _profile


def _condition() -> ProfileOperatingCondition:
    return ProfileOperatingCondition("fairway", 290.0, 0.24)


def test_binding_maps_all_eleven_values_without_defaults() -> None:
    profile = _profile()
    placement = SurfacePlacement(
        "run-surface-001",
        1.25,
        (0.0, 1.0, 0.0),
        (2.0, 0.0, -1.0),
    )

    bound = bind_material_profile(profile, placement, _condition())
    surface = bound.surface

    assert surface.frame is GroundFrame.TARGET
    assert surface.surface_id == "run-surface-001"
    assert surface.provider_id == profile.provenance.producer
    assert surface.provider_version == profile.provenance.producer_version
    assert surface.height_m == 1.25
    assert surface.surface_velocity_m_s == (2.0, 0.0, -1.0)
    assert (
        surface.normal_restitution,
        surface.static_friction,
        surface.kinetic_friction,
        surface.rolling_resistance,
        surface.firmness_pa,
        surface.hardness_fraction,
        surface.grass_height_m,
        surface.compressibility_fraction,
        surface.compression_damping_fraction,
        surface.turf_density_kg_m3,
        surface.moisture_fraction,
    ) == tuple(item.value_si for item in profile.parameters)
    assert bound.profile_sha256 == profile.canonical_sha256()
    assert bound.applicability == profile.applicability
    assert bound.operating_condition == _condition()
    assert bound.warnings == ()


def test_binding_preserves_unqualified_status_as_explicit_warning() -> None:
    profile = replace(
        _profile(),
        rights=GroundProfileRights(
            "LicenseRef-internal", "Fixture authors", False, False
        ),
    )
    bound = bind_material_profile(
        profile,
        SurfacePlacement("illustrative", 0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        _condition(),
    )

    assert bound.qualification == profile.qualification
    assert bound.warnings == (PROFILE_UNQUALIFIED_WARNING,)


def test_binding_preserves_illustrative_model_use_as_distinct_warning() -> None:
    profile = _profile(evidence_kind=GroundEvidenceKind.ENGINEERING_ESTIMATE)
    bound = bind_material_profile(
        profile,
        SurfacePlacement("illustrative", 0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        _condition(),
    )

    assert bound.qualification.status is GroundQualificationStatus.QUALIFIED
    assert bound.warnings == (PROFILE_ILLUSTRATIVE_WARNING,)


def test_binding_rejects_operating_conditions_outside_applicability() -> None:
    placement = SurfacePlacement(
        "run-surface-001", 0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)
    )
    conditions = (
        ProfileOperatingCondition("bunker", 290.0, 0.24),
        ProfileOperatingCondition("fairway", 320.0, 0.24),
    )

    for condition in conditions:
        with pytest.raises(ValueError, match="applicability"):
            bind_material_profile(_profile(), placement, condition)

    profile = _profile()
    narrow = replace(
        profile,
        applicability=replace(profile.applicability, moisture_max_fraction=0.5),
    )
    with pytest.raises(ValueError, match="applicability"):
        bind_material_profile(
            narrow,
            placement,
            ProfileOperatingCondition("fairway", 290.0, 0.99),
        )


def test_bound_surface_rejects_forged_output_evidence() -> None:
    bound = bind_material_profile(
        _profile(),
        SurfacePlacement("bound", 0.0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
        _condition(),
    )

    with pytest.raises(ValueError, match="profile_sha256"):
        replace(bound, profile_sha256="0" * 64)
    with pytest.raises(ValueError, match="warnings"):
        replace(bound, warnings=("forged",))
    with pytest.raises(ValueError, match="applicability"):
        replace(
            bound,
            operating_condition=replace(bound.operating_condition, temperature_k=999.0),
        )
    with pytest.raises(ValueError, match="surface material"):
        replace(
            bound,
            surface=replace(bound.surface, normal_restitution=0.99),
        )
    with pytest.raises(TypeError, match="applicability"):
        BoundGroundSurface(
            bound.surface,
            bound.profile,
            bound.profile_id,
            bound.profile_revision,
            bound.profile_sha256,
            bound.qualification,
            "forged",  # type: ignore[arg-type]
            bound.operating_condition,
            bound.warnings,
        )

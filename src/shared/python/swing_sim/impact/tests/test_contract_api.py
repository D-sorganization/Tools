"""Contract test pinning the public API surface of swing_sim.impact.

Downstream consumers (UpstreamDrift, the app's club package, web backend)
import from this self-façaded subpackage only; this test fails loudly
when the surface changes so removals are always deliberate. The parent
``swing_sim`` façade is wired during epic integration and intentionally
does NOT re-export impact symbols yet.
"""

from __future__ import annotations

import dataclasses

import pytest

import shared.python.swing_sim.impact as impact

EXPECTED_PUBLIC_API = {
    "DPlaneAnalysis",
    "DPlaneStatus",
    "DRIVER_CG_DEPTH_M",
    "DRIVER_COR",
    "DRIVER_MASS_KG",
    "DRIVER_MOI_KG_M2",
    "GOLF_BALL_MASS_KG",
    "GOLF_BALL_MOMENT_OF_INERTIA_KG_M2",
    "GOLF_BALL_RADIUS_M",
    "SPHERE_ROLLING_CAP_FACTOR",
    "TYPICAL_CONTACT_DURATION_S",
    "DeliveryDerived",
    "DeliveryParameters",
    "FaceNormalAtOffset",
    "FiniteTimeImpactModel",
    "GearEffectResult",
    "ImpactEvent",
    "ImpactModel",
    "ImpactModelType",
    "ImpactParameters",
    "ImpactRecorder",
    "ImpactSolverAPI",
    "PostImpactState",
    "PreImpactState",
    "RigidBodyImpactModel",
    "SpringDamperImpactModel",
    "analyze_dplane",
    "compute_gear_effect",
    "create_impact_model",
    "derive_delivery",
    "face_basis",
    "offset_to_face_vector",
    "resolve_contact_normal",
    "spin_loft_sector_directions",
    "to_pre_impact_state",
    "validate_energy_balance",
}


@pytest.mark.contract
def test_public_api_surface_is_pinned() -> None:
    assert set(impact.__all__) == EXPECTED_PUBLIC_API


@pytest.mark.contract
def test_all_exports_resolve() -> None:
    for name in impact.__all__:
        assert getattr(impact, name) is not None, f"{name} did not resolve"


@pytest.mark.contract
def test_parent_facade_untouched() -> None:
    """Integration wires the parent façade later (epic #4103)."""
    import shared.python.swing_sim as swing_sim

    assert "ImpactSolverAPI" not in swing_sim.__all__


@pytest.mark.contract
def test_delivery_types_are_frozen_dataclasses() -> None:
    for cls in (impact.DeliveryParameters, impact.DeliveryDerived):
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


@pytest.mark.contract
def test_pre_impact_state_has_tensor_field() -> None:
    fields = {f.name for f in dataclasses.fields(impact.PreImpactState)}
    assert "impact_offset" in fields
    assert "clubhead_moi_tensor" in fields

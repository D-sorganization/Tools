from __future__ import annotations

from shared.python.golf_club import (
    CATEGORY_TURF,
    TURF_FRICTION_KEY,
    TURF_STIFFNESS_KEY,
    TurfCalibrationStatus,
    TurfPreset,
    TurfVariationPlan,
    turf_profile_preset,
    turf_profiles_for_variation_plan,
)
from shared.python.swing_sim.variation import (
    NoiseSpec,
    variables_in_category,
)


def test_turf_parameters_are_registered_with_visible_evidence_limits() -> None:
    definitions = variables_in_category(CATEGORY_TURF)

    assert {definition.key for definition in definitions} >= {
        TURF_STIFFNESS_KEY,
        TURF_FRICTION_KEY,
    }
    assert all(
        "calibrated" in definition.guidance.lower() for definition in definitions
    )


def test_seeded_turf_plan_is_deterministic_and_downgrades_claims() -> None:
    plan = TurfVariationPlan(
        n_runs=4,
        seed=42,
        base_variables={},
        noise=(NoiseSpec(TURF_STIFFNESS_KEY, scale=100.0),),
    )
    base = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)

    first = turf_profiles_for_variation_plan(plan, base)
    second = turf_profiles_for_variation_plan(plan, base)

    assert first == second
    assert len(first) == 4
    assert all(
        profile.calibration_status is TurfCalibrationStatus.ILLUSTRATIVE
        for profile in first
    )
    assert all(
        "variation plan" in profile.provenance.parameter_basis for profile in first
    )

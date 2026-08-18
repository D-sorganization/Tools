"""Scientific and contract tests for the reduced compliant-turf model."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club import (
    GroundPlane,
    TurfCalibrationStatus,
    TurfContactKinematics,
    TurfContactStatus,
    TurfPreset,
    evaluate_turf_contact,
    simulate_reduced_turf_contact,
    turf_profile_preset,
)


def _kinematics(
    *,
    penetration_m: float = 0.002,
    velocity_mps: tuple[float, float, float] = (1.0, -0.5, 0.0),
) -> TurfContactKinematics:
    return TurfContactKinematics(
        frame_id="ground_frame:x_target,y_up,z_right",
        reference_point_m=(0.0, 0.1, 0.0),
        application_point_m=(0.2, 0.0, 0.0),
        surface_normal_unit=(0.0, 1.0, 0.0),
        surface_velocity_mps=(0.0, 0.0, 0.0),
        contact_point_velocity_mps=velocity_mps,
        penetration_m=penetration_m,
    )


def test_illustrative_presets_disclose_claim_boundary() -> None:
    profile = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)

    assert profile.calibration_status is TurfCalibrationStatus.ILLUSTRATIVE
    assert "illustrative" in profile.provenance.source_name.lower()
    assert profile.supports_turf_rankings is False


def test_kelvin_voigt_normal_force_and_wrench_are_exact() -> None:
    profile = replace(
        turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
        normal_stiffness_n_m=20_000.0,
        normal_damping_n_s_m=100.0,
        friction_coefficient=0.0,
    )

    response = evaluate_turf_contact(profile, _kinematics())

    assert response.status is TurfContactStatus.ACTIVE
    assert response.normal_force_n == pytest.approx(90.0)
    np.testing.assert_allclose(response.force_world_n, [0.0, 90.0, 0.0])
    np.testing.assert_allclose(response.torque_at_reference_n_m, [0.0, 0.0, 18.0])
    assert response.stored_elastic_energy_j == pytest.approx(0.04)
    assert response.dissipated_power_w == pytest.approx(25.0)


def test_regularized_friction_opposes_slip_and_respects_coulomb_bound() -> None:
    profile = replace(
        turf_profile_preset(TurfPreset.SOFT_TURF),
        normal_stiffness_n_m=10_000.0,
        normal_damping_n_s_m=0.0,
        friction_coefficient=0.6,
        friction_regularization_mps=0.05,
    )
    state = _kinematics(velocity_mps=(3.0, 0.0, 4.0))

    response = evaluate_turf_contact(profile, state)
    force = np.asarray(response.tangential_force_world_n)
    slip = np.array([3.0, 0.0, 4.0])

    assert float(force @ slip) < 0.0
    assert np.linalg.norm(force) <= 0.6 * response.normal_force_n
    assert response.dissipated_power_w == pytest.approx(-float(force @ slip))


def test_zero_stiffness_and_damping_is_an_explicit_no_response_limit() -> None:
    profile = replace(
        turf_profile_preset(TurfPreset.SOFT_TURF),
        normal_stiffness_n_m=0.0,
        normal_damping_n_s_m=0.0,
    )

    response = evaluate_turf_contact(profile, _kinematics())
    result = simulate_reduced_turf_contact(
        profile,
        initial_contact_velocity_mps=(1.0, -2.0, 0.0),
        surface_normal_unit=(0.0, 1.0, 0.0),
        effective_mass_kg=0.3,
    )

    assert response.status is TurfContactStatus.NO_RESPONSE
    assert response.normal_force_n == 0.0
    assert result.status is TurfContactStatus.NO_RESPONSE
    assert result.impulse_world_n_s == (0.0, 0.0, 0.0)
    assert result.final_contact_velocity_mps == (1.0, -2.0, 0.0)


def test_reduced_contact_is_passive_and_frictionless_tangent_is_preserved() -> None:
    profile = replace(
        turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
        friction_coefficient=0.0,
        max_penetration_m=0.05,
    )

    result = simulate_reduced_turf_contact(
        profile,
        initial_contact_velocity_mps=(1.25, -2.0, 0.0),
        surface_normal_unit=(0.0, 1.0, 0.0),
        effective_mass_kg=0.3,
        time_step_s=2.5e-6,
    )

    assert result.status is TurfContactStatus.SEPARATED
    assert result.final_contact_velocity_mps[0] == pytest.approx(1.25)
    assert result.final_contact_velocity_mps[1] > 0.0
    assert result.dissipated_energy_j >= 0.0
    assert result.separation_loss_energy_j > 0.0
    assert result.final_kinetic_energy_j <= result.initial_kinetic_energy_j + 1e-4
    assert abs(result.energy_balance_residual_j) < 3e-3


def test_reduced_contact_converges_under_timestep_refinement() -> None:
    profile = replace(
        turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
        friction_coefficient=0.0,
        max_penetration_m=0.05,
    )
    kwargs = {
        "initial_contact_velocity_mps": (0.0, -1.5, 0.0),
        "surface_normal_unit": (0.0, 1.0, 0.0),
        "effective_mass_kg": 0.3,
    }

    coarse = simulate_reduced_turf_contact(profile, time_step_s=1e-5, **kwargs)
    fine = simulate_reduced_turf_contact(profile, time_step_s=2.5e-6, **kwargs)

    assert coarse.normal_impulse_n_s == pytest.approx(
        fine.normal_impulse_n_s, rel=0.015
    )
    assert coarse.peak_penetration_m == pytest.approx(
        fine.peak_penetration_m, rel=0.015
    )


def test_lower_stiffness_increases_peak_penetration() -> None:
    firm = replace(turf_profile_preset(TurfPreset.FIRM_FAIRWAY), max_penetration_m=0.08)
    soft = replace(
        firm,
        profile_id="softer-test-profile",
        normal_stiffness_n_m=0.25 * firm.normal_stiffness_n_m,
    )
    kwargs = {
        "initial_contact_velocity_mps": (0.0, -1.0, 0.0),
        "surface_normal_unit": (0.0, 1.0, 0.0),
        "effective_mass_kg": 0.3,
        "time_step_s": 5e-6,
    }

    assert (
        simulate_reduced_turf_contact(soft, **kwargs).peak_penetration_m
        > simulate_reduced_turf_contact(firm, **kwargs).peak_penetration_m
    )


def test_reduced_contact_honors_cooperative_cancellation() -> None:
    calls = 0

    def cancel_after_five_checks() -> bool:
        nonlocal calls
        calls += 1
        return calls > 5

    result = simulate_reduced_turf_contact(
        turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
        initial_contact_velocity_mps=(0.0, -1.0, 0.0),
        surface_normal_unit=(0.0, 1.0, 0.0),
        effective_mass_kg=0.3,
        cancel_check=cancel_after_five_checks,
    )

    assert result.status is TurfContactStatus.CANCELLED
    assert result.step_count == 5


def test_rotation_equivariance_preserves_force_and_torque() -> None:
    profile = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)
    state = _kinematics()
    response = evaluate_turf_contact(profile, state)
    angle = math.radians(37.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def rotated(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        values = rotation @ np.asarray(vector)
        return (float(values[0]), float(values[1]), float(values[2]))

    rotated_state = TurfContactKinematics(
        frame_id=state.frame_id,
        reference_point_m=rotated(state.reference_point_m),
        application_point_m=rotated(state.application_point_m),
        surface_normal_unit=rotated(state.surface_normal_unit),
        surface_velocity_mps=rotated(state.surface_velocity_mps),
        contact_point_velocity_mps=rotated(state.contact_point_velocity_mps),
        penetration_m=state.penetration_m,
    )
    rotated_response = evaluate_turf_contact(profile, rotated_state)

    np.testing.assert_allclose(
        rotated_response.force_world_n,
        rotation @ np.asarray(response.force_world_n),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rotated_response.torque_at_reference_n_m,
        rotation @ np.asarray(response.torque_at_reference_n_m),
        atol=1e-10,
    )


def test_ground_plane_normal_can_drive_the_reduced_model() -> None:
    ground = GroundPlane(normal_unit=(0.0, math.sqrt(0.5), math.sqrt(0.5)))
    result = simulate_reduced_turf_contact(
        turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
        initial_contact_velocity_mps=(0.0, -1.0, -1.0),
        surface_normal_unit=ground.normal_unit,
        effective_mass_kg=0.3,
        time_step_s=5e-6,
    )

    assert result.status is TurfContactStatus.SEPARATED
    assert result.normal_impulse_n_s > 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normal_stiffness_n_m", -1.0),
        ("friction_coefficient", 1.1),
        ("max_penetration_m", 0.0),
    ],
)
def test_profile_rejects_nonphysical_values(field: str, value: float) -> None:
    profile = turf_profile_preset(TurfPreset.FIRM_FAIRWAY)

    with pytest.raises(ValueError, match=field):
        replace(profile, **{field: value})

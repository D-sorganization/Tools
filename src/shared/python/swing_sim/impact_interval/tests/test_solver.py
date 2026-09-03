"""Scientific limits and DbC tests for interval rigid-club dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.impact import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    ImpactParameters,
    PreImpactState,
    RigidBodyImpactModel,
)
from shared.python.swing_sim.impact_interval import (
    BoundaryKind,
    ClubRigidBody,
    ImpactIntervalConfig,
    ImpactIntervalInitialState,
    KelvinVoigtContactLaw,
    solve_impact_interval,
)


def _reduced_mass(club_mass: float = 0.2) -> float:
    return float(GOLF_BALL_MASS_KG * club_mass / (GOLF_BALL_MASS_KG + club_mass))


def _club(contact_offset: np.ndarray | None = None) -> ClubRigidBody:
    return ClubRigidBody(
        mass_kg=0.2,
        inertia_body_kg_m2=np.diag([4.5e-4, 4.5e-4, 4.5e-4]),
        cg_to_contact_body_m=(
            np.zeros(3) if contact_offset is None else contact_offset
        ),
        # Simplified vertical shaft: attachment is 0.9 m above the head.
        # A toe-side normal strike then twists about the shaft/y axis.
        cg_to_attachment_body_m=np.array([0.0, 0.90, 0.0]),
        face_normal_body=np.array([1.0, 0.0, 0.0]),
    )


def _initial(contact_offset: np.ndarray | None = None) -> ImpactIntervalInitialState:
    offset = np.zeros(3) if contact_offset is None else contact_offset
    return ImpactIntervalInitialState(
        club_position_m=np.zeros(3),
        club_orientation=np.eye(3),
        club_velocity_mps=np.array([50.0, 0.0, 0.0]),
        club_angular_velocity_rad_s=np.zeros(3),
        ball_position_m=offset + np.array([GOLF_BALL_RADIUS_M, 0.0, 0.0]),
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
    )


def _pinned_initial(
    contact_offset: np.ndarray | None = None,
) -> ImpactIntervalInitialState:
    """Initial velocity consistent with a fixed attachment-point constraint."""
    state = _initial(contact_offset)
    state.club_angular_velocity_rad_s = np.array([0.0, 0.0, 50.0 / 0.90])
    return state


def _config(
    boundary: BoundaryKind = BoundaryKind.FREE,
    *,
    torsional_stiffness: float = 0.0,
) -> ImpactIntervalConfig:
    return ImpactIntervalConfig(
        contact_law=KelvinVoigtContactLaw.from_restitution(
            stiffness_n_per_m=5.0e7,
            restitution=0.83,
            effective_mass_kg=_reduced_mass(),
        ),
        time_step_s=1.0e-7,
        maximum_time_s=2.0e-3,
        friction_coefficient=0.4,
        boundary=boundary,
        torsional_stiffness_n_m_per_rad=torsional_stiffness,
        torsional_damping_n_m_s_per_rad=0.02 if torsional_stiffness else 0.0,
    )


@pytest.mark.unit
@pytest.mark.physics
class TestScientificLimits:
    def test_symmetric_strike_has_zero_twist(self) -> None:
        result = solve_impact_interval(_initial(), _club(), _config())
        assert result.did_contact
        assert np.max(np.abs(result.twist_angle_rad)) < 1.0e-10
        assert np.max(np.linalg.norm(result.club_angular_velocity_rad_s, axis=1)) < 1e-8

    def test_stiff_contact_recovers_instantaneous_impulse_limit(self) -> None:
        result = solve_impact_interval(_initial(), _club(), _config())
        pre = PreImpactState(
            clubhead_velocity=np.array([50.0, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([1.0, 0.0, 0.0]),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
            clubhead_mass=0.2,
        )
        instantaneous = RigidBodyImpactModel().solve(pre, ImpactParameters(cor=0.83))
        assert result.ball_velocity_mps[-1, 0] == pytest.approx(
            instantaneous.ball_velocity[0], rel=0.025
        )
        assert result.club_velocity_mps[-1, 0] == pytest.approx(
            instantaneous.clubhead_velocity[0], rel=0.04
        )

    def test_off_center_strike_develops_twist_over_time(self) -> None:
        offset = np.array([0.0, 0.0, 0.02])
        result = solve_impact_interval(_initial(offset), _club(offset), _config())
        assert abs(result.twist_angle_rad[-1]) > 1.0e-5
        assert abs(result.club_angular_velocity_rad_s[-1, 1]) > 1.0
        assert np.count_nonzero(result.normal_force_n > 0.0) > 10

    def test_free_body_audit_closes_energy_and_momentum(self) -> None:
        result = solve_impact_interval(_initial(), _club(), _config())
        assert abs(result.audit.linear_momentum_residual_n_s) < 2.0e-8
        assert abs(result.audit.energy_residual_j) < 0.08
        assert result.audit.dissipated_energy_j > 0.0
        assert result.audit.integrated_normal_impulse_n_s > 0.0


@pytest.mark.unit
@pytest.mark.physics
class TestBoundaryConditions:
    def test_pinned_attachment_point_does_not_translate(self) -> None:
        offset = np.array([0.0, 0.0, 0.02])
        result = solve_impact_interval(
            _pinned_initial(offset), _club(offset), _config(BoundaryKind.PINNED)
        )
        anchor = result.attachment_position_m[0]
        assert (
            np.max(np.linalg.norm(result.attachment_position_m - anchor, axis=1))
            < 2e-10
        )

    def test_torsional_grip_reduces_face_twist(self) -> None:
        offset = np.array([0.0, 0.0, 0.02])
        pinned = solve_impact_interval(
            _pinned_initial(offset), _club(offset), _config(BoundaryKind.PINNED)
        )
        sprung = solve_impact_interval(
            _pinned_initial(offset),
            _club(offset),
            _config(BoundaryKind.TORSIONAL_GRIP, torsional_stiffness=2_000.0),
        )
        assert abs(sprung.twist_angle_rad[-1]) < abs(pinned.twist_angle_rad[-1])
        assert sprung.audit.boundary_stored_energy_j >= 0.0


@pytest.mark.unit
class TestQueryableHistoryAndContracts:
    def test_named_channel_and_nearest_sample_are_available(self) -> None:
        result = solve_impact_interval(_initial(), _club(), _config())
        np.testing.assert_array_equal(
            result.channel("normal_force_n"), result.normal_force_n
        )
        sample = result.at_time(result.time_s[len(result.time_s) // 2])
        assert sample.time_s >= 0.0
        assert sample.club_orientation.shape == (3, 3)

    def test_unknown_channel_is_rejected(self) -> None:
        result = solve_impact_interval(_initial(), _club(), _config())
        with pytest.raises(ValueError, match="Unknown impact-interval channel"):
            result.channel("not-a-channel")

    def test_post_impact_adapter_preserves_final_state(self) -> None:
        result = solve_impact_interval(_initial(), _club(), _config())
        post = result.to_post_impact_state()
        np.testing.assert_allclose(post.ball_velocity, result.ball_velocity_mps[-1])
        assert post.contact_duration == pytest.approx(result.contact_duration_s)

    def test_non_orthonormal_orientation_is_rejected(self) -> None:
        state = _initial()
        state.club_orientation[0, 0] = 2.0
        with pytest.raises(ValueError, match="orthonormal"):
            solve_impact_interval(state, _club(), _config())

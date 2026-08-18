"""Delivery front-end tests: angle mapping, offsets, D-plane diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.swing_sim.impact.delivery import (
    DeliveryParameters,
    derive_delivery,
    to_pre_impact_state,
)
from shared.python.swing_sim.impact.models import RigidBodyImpactModel
from shared.python.swing_sim.impact.types import ImpactParameters


def _params(**kwargs: float) -> DeliveryParameters:
    defaults: dict[str, float] = {"clubhead_speed_mps": 50.0}
    defaults.update(kwargs)
    return DeliveryParameters(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestDeliveryValidation:
    def test_rejects_non_positive_speed(self) -> None:
        with pytest.raises(ValueError, match="clubhead_speed_mps"):
            DeliveryParameters(clubhead_speed_mps=0.0)

    def test_rejects_non_finite_angle(self) -> None:
        with pytest.raises(ValueError, match="face_angle_deg"):
            _params(face_angle_deg=float("nan"))

    def test_rejects_out_of_range_angle(self) -> None:
        with pytest.raises(ValueError, match="club_path_deg"):
            _params(club_path_deg=120.0)

    def test_is_frozen(self) -> None:
        params = _params()
        with pytest.raises(AttributeError):
            params.club_path_deg = 1.0  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.physics
class TestDeliveryVectors:
    def test_neutral_delivery_is_straight_normal(self) -> None:
        """Zero path/face/AoA: velocity along +x, normal lofted in x-y."""
        derived = derive_delivery(_params(dynamic_loft_deg=10.5))
        loft = math.radians(10.5)
        np.testing.assert_allclose(
            derived.clubhead_velocity, [50.0, 0.0, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            derived.face_normal, [math.cos(loft), math.sin(loft), 0.0], atol=1e-12
        )
        np.testing.assert_allclose(derived.impact_offset, [0.0, 0.0], atol=1e-15)
        assert derived.spin_loft_deg == pytest.approx(10.5)
        assert derived.face_to_path_deg == pytest.approx(0.0)
        np.testing.assert_allclose(derived.spin_axis, [0.0, 0.0, 1.0], atol=1e-9)
        assert derived.spin_axis_tilt_deg == pytest.approx(0.0, abs=1e-9)

    def test_in_to_out_path_moves_right(self) -> None:
        derived = derive_delivery(_params(club_path_deg=4.0))
        assert derived.clubhead_velocity[2] > 0.0  # +z = right

    def test_open_face_points_right(self) -> None:
        derived = derive_delivery(_params(face_angle_deg=3.0))
        assert derived.face_normal[2] > 0.0

    def test_positive_attack_angle_moves_up(self) -> None:
        derived = derive_delivery(_params(attack_angle_deg=5.0))
        assert derived.clubhead_velocity[1] > 0.0
        assert float(np.linalg.norm(derived.clubhead_velocity)) == pytest.approx(50.0)

    def test_spin_loft_is_loft_minus_attack_for_square_face(self) -> None:
        derived = derive_delivery(_params(dynamic_loft_deg=12.0, attack_angle_deg=-3.0))
        assert derived.spin_loft_deg == pytest.approx(15.0, abs=1e-9)

    def test_open_face_tilts_spin_axis_fade_side(self) -> None:
        """Face open of path (D-plane): positive tilt = fade/slice spin."""
        fade = derive_delivery(_params(dynamic_loft_deg=10.5, face_angle_deg=4.0))
        draw = derive_delivery(_params(dynamic_loft_deg=10.5, face_angle_deg=-4.0))
        assert fade.spin_axis_tilt_deg > 0.0
        assert draw.spin_axis_tilt_deg < 0.0
        assert fade.spin_axis_tilt_deg == pytest.approx(
            -draw.spin_axis_tilt_deg, rel=1e-9
        )

    def test_offset_millimetre_conversion(self) -> None:
        derived = derive_delivery(
            _params(impact_offset_toe_mm=12.0, impact_offset_high_mm=-5.0)
        )
        np.testing.assert_allclose(derived.impact_offset, [0.012, -0.005])

    def test_lie_rotates_offset_axes(self) -> None:
        """+90 deg toe-up lie maps a pure toe offset onto the vertical axis."""
        derived = derive_delivery(_params(lie_deg=89.0, impact_offset_toe_mm=10.0))
        assert derived.impact_offset[1] == pytest.approx(
            0.010 * math.sin(math.radians(89.0))
        )

    def test_angular_velocity_passthrough(self) -> None:
        omega = np.array([0.0, 1.0, 2.0])
        derived = derive_delivery(_params(), clubhead_angular_velocity=omega)
        np.testing.assert_allclose(derived.clubhead_angular_velocity, omega)


@pytest.mark.unit
@pytest.mark.physics
class TestToPreImpactState:
    def test_round_trip_neutral_delivery_launches_near_normal(self) -> None:
        """End-to-end: neutral delivery -> impact solve -> straight launch."""
        pre = to_pre_impact_state(_params(dynamic_loft_deg=10.5))
        post = RigidBodyImpactModel().solve(pre, ImpactParameters())
        v = post.ball_velocity / np.linalg.norm(post.ball_velocity)
        # Ball launches along the face normal (rigid-body normal impulse).
        np.testing.assert_allclose(v, pre.clubhead_orientation, atol=1e-12)
        assert abs(v[2]) < 1e-12  # no sideways component
        # Backspin about +z (sign-fixed friction spin axis).
        assert post.ball_angular_velocity[2] > 0.0

    def test_offset_carried_into_state(self) -> None:
        pre = to_pre_impact_state(_params(impact_offset_toe_mm=20.0))
        assert pre.impact_offset is not None
        assert pre.impact_offset[0] == pytest.approx(0.020)

    def test_tensor_passthrough(self) -> None:
        tensor = 4.5e-4 * np.eye(3)
        pre = to_pre_impact_state(_params(), clubhead_moi_tensor=tensor)
        assert pre.clubhead_moi_tensor is tensor

    def test_ball_defaults_to_rest(self) -> None:
        pre = to_pre_impact_state(_params())
        np.testing.assert_allclose(pre.ball_velocity, 0.0)
        np.testing.assert_allclose(pre.ball_angular_velocity, 0.0)

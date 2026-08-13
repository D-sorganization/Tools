"""Physics-based gear-effect tests: qualitative signatures + bulge seam.

Signatures validated (right-handed player, AffineDrift frame
x target / y up / z right; backspin axis = +z, draw-side spin = +y):

- toe hit -> draw-side spin (+y)
- heel hit -> fade-side spin (-y)
- high hit -> reduced backspin (-z delta)
- low hit -> added backspin (+z delta)
- bulge partially offsets the toe-hit pull (starts the ball further right)
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.impact.constants import (
    DRIVER_MOI_KG_M2,
    GOLF_BALL_MASS_KG,
)
from shared.python.swing_sim.impact.gear_effect import (
    GearEffectResult,
    compute_gear_effect,
    resolve_contact_normal,
)
from shared.python.swing_sim.impact.models import SPHERE_ROLLING_CAP_FACTOR
from shared.python.swing_sim.impact.solver import ImpactSolverAPI

_N_SQUARE = np.array([1.0, 0.0, 0.0])
_V_CLUB = np.array([50.0, 0.0, 0.0])
_J_TYPICAL = 3.0  # [N.s] ~ (1+e) mu v for a 50 m/s driver strike


def _gear(offset: np.ndarray, cg_depth_m: float | None = None) -> GearEffectResult:
    if cg_depth_m is None:
        return compute_gear_effect(
            impact_offset=offset,
            face_normal=_N_SQUARE,
            normal_impulse=_J_TYPICAL,
            clubhead_moi=DRIVER_MOI_KG_M2,
        )
    return compute_gear_effect(
        impact_offset=offset,
        face_normal=_N_SQUARE,
        normal_impulse=_J_TYPICAL,
        clubhead_moi=DRIVER_MOI_KG_M2,
        cg_depth_m=cg_depth_m,
    )


@pytest.mark.unit
@pytest.mark.physics
class TestGearEffectSignatures:
    def test_toe_hit_gives_draw_side_spin(self) -> None:
        result = _gear(np.array([0.02, 0.0]))
        assert result.ball_spin_delta[1] > 0.0  # +y = draw (RH)
        assert abs(result.ball_spin_delta[1]) > abs(result.ball_spin_delta[2])

    def test_heel_hit_gives_fade_side_spin(self) -> None:
        result = _gear(np.array([-0.02, 0.0]))
        assert result.ball_spin_delta[1] < 0.0  # -y = fade (RH)

    def test_high_hit_reduces_backspin(self) -> None:
        result = _gear(np.array([0.0, 0.015]))
        assert result.ball_spin_delta[2] < 0.0  # opposes +z backspin

    def test_low_hit_adds_backspin(self) -> None:
        result = _gear(np.array([0.0, -0.015]))
        assert result.ball_spin_delta[2] > 0.0

    def test_toe_and_heel_are_antisymmetric(self) -> None:
        toe = _gear(np.array([0.02, 0.0]))
        heel = _gear(np.array([-0.02, 0.0]))
        np.testing.assert_allclose(
            toe.ball_spin_delta, -heel.ball_spin_delta, atol=1e-12
        )

    def test_center_hit_produces_no_gear_spin(self) -> None:
        result = _gear(np.zeros(2))
        np.testing.assert_allclose(result.ball_spin_delta, 0.0)
        assert result.tangential_surface_speed == 0.0

    def test_zero_cg_depth_produces_no_gear_spin(self) -> None:
        """No front-to-back lever arm -> no tangential sweep -> no spin."""
        result = _gear(np.array([0.02, 0.0]), cg_depth_m=0.0)
        np.testing.assert_allclose(result.ball_spin_delta, 0.0, atol=1e-12)

    def test_toe_hit_head_recoil_opens_face(self) -> None:
        """Toe hit twists the head about -y: the toe rotates backward."""
        result = _gear(np.array([0.02, 0.0]))
        assert result.head_angular_velocity_delta[1] < 0.0

    def test_spin_scales_with_offset(self) -> None:
        small = _gear(np.array([0.005, 0.0]))
        large = _gear(np.array([0.02, 0.0]))
        assert abs(large.ball_spin_delta[1]) > abs(small.ball_spin_delta[1])

    def test_higher_moi_reduces_gear_spin(self) -> None:
        """Forgiving (high-MOI) heads twist less and gear less."""
        low = _gear(np.array([0.02, 0.0]))
        high = compute_gear_effect(
            impact_offset=np.array([0.02, 0.0]),
            face_normal=_N_SQUARE,
            normal_impulse=_J_TYPICAL,
            clubhead_moi=4.0 * DRIVER_MOI_KG_M2,
        )
        assert abs(high.ball_spin_delta[1]) < abs(low.ball_spin_delta[1])

    def test_tensor_moi_matches_matching_scalar(self) -> None:
        scalar = _gear(np.array([0.02, 0.01]))
        tensor = compute_gear_effect(
            impact_offset=np.array([0.02, 0.01]),
            face_normal=_N_SQUARE,
            normal_impulse=_J_TYPICAL,
            clubhead_moi=DRIVER_MOI_KG_M2 * np.eye(3),
        )
        np.testing.assert_allclose(
            tensor.ball_spin_delta, scalar.ball_spin_delta, rtol=1e-12
        )

    def test_friction_cap_bounds_spin(self) -> None:
        """The friction impulse never exceeds the 2/7 rolling cap."""
        result = _gear(np.array([0.03, 0.0]))
        i_ball_over_r = GOLF_BALL_MASS_KG * (2.0 / 5.0) * 0.021335
        cap = (
            GOLF_BALL_MASS_KG
            * result.tangential_surface_speed
            * SPHERE_ROLLING_CAP_FACTOR
        ) / i_ball_over_r
        assert float(np.linalg.norm(result.ball_spin_delta)) <= cap * (1.0 + 1e-9)

    def test_negative_impulse_rejected(self) -> None:
        with pytest.raises(ValueError, match="normal_impulse"):
            compute_gear_effect(
                impact_offset=np.array([0.02, 0.0]),
                face_normal=_N_SQUARE,
                normal_impulse=-1.0,
                clubhead_moi=DRIVER_MOI_KG_M2,
            )


@pytest.mark.unit
@pytest.mark.physics
class TestBulgeRollSeam:
    @staticmethod
    def _bulge_normal(toe_m: float, high_m: float) -> np.ndarray:
        """Toy bulge/roll: convex face with 0.25 m radius of curvature."""
        radius = 0.25
        n = np.array([1.0, high_m / radius, toe_m / radius])
        return n / np.linalg.norm(n)

    def test_local_normal_resolution(self) -> None:
        n = resolve_contact_normal(np.array([0.02, 0.0]), _N_SQUARE, self._bulge_normal)
        assert n[2] > 0.0  # toe-side bulge normal points right (open)
        assert float(np.linalg.norm(n)) == pytest.approx(1.0)

    def test_bulge_partially_offsets_toe_hit_pull(self) -> None:
        """Toe hit with bulge launches further right than with a flat face,
        so the gear-effect draw spin curves it back toward the target."""
        api_flat = ImpactSolverAPI()
        flat = api_flat.solve_with_gear_effect(
            0.0,
            _V_CLUB,
            _N_SQUARE,
            impact_offset=np.array([0.02, 0.0]),
            record=False,
        )
        api_bulge = ImpactSolverAPI()
        bulged = api_bulge.solve_with_gear_effect(
            0.0,
            _V_CLUB,
            _N_SQUARE,
            impact_offset=np.array([0.02, 0.0]),
            face_normal_at_offset=self._bulge_normal,
            record=False,
        )
        # Launch direction gains a rightward (+z) component from bulge...
        assert bulged.ball_velocity[2] > flat.ball_velocity[2]
        # ...while the gear effect still adds draw-side spin on top of the
        # open-local-normal friction spin (compare against the same local
        # normal solved WITHOUT gear effect).
        no_gear = ImpactSolverAPI().solve_impact(
            0.0,
            _V_CLUB,
            self._bulge_normal(0.02, 0.0),
            impact_offset=np.array([0.02, 0.0]),
            record=False,
        )
        assert bulged.ball_angular_velocity[1] > no_gear.ball_angular_velocity[1]

    def test_gear_solver_adds_head_recoil(self) -> None:
        api = ImpactSolverAPI()
        post = api.solve_with_gear_effect(
            0.0,
            _V_CLUB,
            _N_SQUARE,
            impact_offset=np.array([0.02, 0.0]),
            record=False,
        )
        assert post.clubhead_angular_velocity[1] < 0.0  # toe twists open

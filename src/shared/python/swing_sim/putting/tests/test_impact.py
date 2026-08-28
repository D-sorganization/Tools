"""Putter-ball impact tests (#4125 H3; 2-D stroke extension #4800 P1)."""

from __future__ import annotations

import itertools
import math

import pytest

from shared.python.swing_sim.impact import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
    SPHERE_ROLLING_CAP_FACTOR,
)
from shared.python.swing_sim.putting import (
    DEFAULT_PUTTER_COR,
    DEFAULT_PUTTER_MOI_KG_M2,
    MINIMAL_PUTTERS,
    PutterSpec,
    clubhead_speed_from_backstroke,
    strike,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

BLADE = MINIMAL_PUTTERS["Blade Putter"]


def _center_transfer(spec: PutterSpec) -> float:
    """Center-strike momentum transfer (1 + e) M / (M + m)."""
    return (
        (1.0 + spec.cor) * spec.head_mass_kg / (spec.head_mass_kg + GOLF_BALL_MASS_KG)
    )


class TestPutterSpec:
    def test_minimal_putters_are_plausible(self) -> None:
        for spec in MINIMAL_PUTTERS.values():
            assert 0.3 <= spec.head_mass_kg <= 0.4
            assert 2.0 <= spec.loft_deg <= 4.0
            assert spec.cor == DEFAULT_PUTTER_COR

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            PutterSpec(name="x", head_mass_kg=5.0, loft_deg=3.0)
        with pytest.raises(ValueError):
            PutterSpec(name="x", head_mass_kg=0.35, loft_deg=45.0)
        with pytest.raises(ValueError):
            PutterSpec(name="x", head_mass_kg=0.35, loft_deg=3.0, cor=1.2)


class TestStrike:
    def test_normal_transfer_matches_impulse_momentum(self) -> None:
        """Zero loft: pure 1-D COR impulse, no spin, no launch."""
        flat = PutterSpec(name="Flat", head_mass_kg=0.350, loft_deg=0.0)
        launch = strike(flat, 2.0)
        expected = (
            2.0
            * (1.0 + flat.cor)
            * flat.head_mass_kg
            / (flat.head_mass_kg + GOLF_BALL_MASS_KG)
        )
        assert launch.ball_speed_mps == pytest.approx(expected, rel=1e-12)
        assert launch.launch_angle_deg == pytest.approx(0.0, abs=1e-12)
        assert launch.spin_rad_s == pytest.approx(0.0, abs=1e-12)

    def test_lofted_strike_launches_up_with_topspin(self) -> None:
        launch = strike(BLADE, 2.0)
        assert launch.launch_angle_deg > 0.0
        assert launch.launch_angle_deg < 2.0 * BLADE.loft_deg
        assert launch.spin_rad_s > 0.0  # backspin, topspin-positive sign
        # 2/7 cap: surface backspin speed is (5/7) * v * sin(loft).
        u = 2.0 * math.sin(math.radians(BLADE.loft_deg))
        assert launch.spin_rad_s * GOLF_BALL_RADIUS_M == pytest.approx(
            (5.0 / 7.0) * u, rel=1e-12
        )

    def test_smash_factor_is_physical(self) -> None:
        launch = strike(BLADE, 1.5)
        smash = launch.ball_speed_mps / 1.5
        assert 1.4 < smash < 1.7  # typical published putter smash ~1.5-1.7

    def test_forward_press_reduces_launch(self) -> None:
        pressed = strike(BLADE, 2.0, shaft_lean_deg=-2.0)
        neutral = strike(BLADE, 2.0)
        assert pressed.launch_angle_deg < neutral.launch_angle_deg
        assert pressed.effective_loft_deg == pytest.approx(1.0)

    def test_rejects_bad_speeds(self) -> None:
        with pytest.raises(ValueError):
            strike(BLADE, 0.0)
        with pytest.raises(ValueError):
            strike(BLADE, 50.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, shaft_lean_deg=-30.0)


class TestStrike2DAnalyticGates:
    """Analytic gates for the #4800 P1 stroke/impact extension.

    Written before the implementation (TDD): every assertion here is a
    closed-form consequence of the documented model, not a pinned
    output of the code under test.
    """

    def test_square_face_square_path_is_straight(self) -> None:
        """Gate 1: square face + square path -> no azimuth, no sidespin."""
        launch = strike(BLADE, 1.8)
        assert launch.start_azimuth_deg == 0.0
        assert launch.sidespin_rad_s == 0.0

    def test_aim_rotates_start_line_without_touching_the_solve(self) -> None:
        """Aim is a pure rotation: face/path are measured off the aim line."""
        neutral = strike(BLADE, 1.8)
        aimed = strike(BLADE, 1.8, aim_deg=4.0)
        assert aimed.start_azimuth_deg == 4.0
        assert aimed.sidespin_rad_s == 0.0
        assert aimed.ball_speed_mps == neutral.ball_speed_mps
        assert aimed.launch_angle_deg == neutral.launch_angle_deg
        assert aimed.spin_rad_s == neutral.spin_rad_s

    def test_face_only_vs_path_only_split_matches_rolling_cap_ratio(self) -> None:
        """Gate 2: start-line split follows the impact package's 2/7 cap.

        The tangential (path-following) share of the start line is the
        documented ``SPHERE_ROLLING_CAP_FACTOR`` (2/7) against the
        normal (face-following) COR transfer: for small angles,
        ``start ~= (1 - k) face + k path`` with
        ``k = (2/7) / ((1 + e) M / (M + m))``.
        """
        transfer = _center_transfer(BLADE)
        k = SPHERE_ROLLING_CAP_FACTOR / transfer
        angle = 2.0
        face_only = strike(BLADE, 1.8, face_angle_deg=angle)
        path_only = strike(BLADE, 1.8, path_angle_deg=angle)
        # Exact closed form (atan2 of the two impulse components).
        expected_path = math.degrees(
            math.atan2(
                SPHERE_ROLLING_CAP_FACTOR * math.sin(math.radians(angle)),
                transfer * math.cos(math.radians(angle)),
            )
        )
        assert path_only.start_azimuth_deg == pytest.approx(expected_path, rel=1e-12)
        # Small-angle sensitivities: (1 - k) for face, k for path.
        assert face_only.start_azimuth_deg / angle == pytest.approx(1.0 - k, abs=2e-4)
        assert path_only.start_azimuth_deg / angle == pytest.approx(k, abs=2e-4)
        # The face keeps the dominant share, as in the full-swing model.
        assert face_only.start_azimuth_deg > path_only.start_azimuth_deg > 0.0

    def test_face_and_path_deflections_are_exact_complements(self) -> None:
        """Face-only and path-only starts sum exactly to the face angle.

        ``atan2`` is odd in its first argument, so the friction
        deflections for face-only (face-to-path -f) and path-only
        (face-to-path +f) cancel exactly: no small-angle tolerance.
        """
        for angle in (0.5, 2.0, 8.0):
            face_only = strike(BLADE, 1.8, face_angle_deg=angle)
            path_only = strike(BLADE, 1.8, path_angle_deg=angle)
            total = face_only.start_azimuth_deg + path_only.start_azimuth_deg
            assert total == pytest.approx(angle, rel=1e-12)

    def test_sidespin_sign_and_magnitude_follow_face_to_path(self) -> None:
        """Path right of face -> +sidespin (draw side); magnitude (5/7)u/R."""
        v = 2.0
        draw = strike(BLADE, v, path_angle_deg=3.0)
        fade = strike(BLADE, v, face_angle_deg=3.0)
        assert draw.sidespin_rad_s > 0.0
        assert fade.sidespin_rad_s < 0.0
        expected = (
            (1.0 - SPHERE_ROLLING_CAP_FACTOR)
            * v
            * math.sin(math.radians(3.0))
            / GOLF_BALL_RADIUS_M
        )
        assert draw.sidespin_rad_s == pytest.approx(expected, rel=1e-12)
        assert fade.sidespin_rad_s == pytest.approx(-expected, rel=1e-12)
        # Sidespin depends on face-to-path only: aim shifts neither.
        aimed = strike(BLADE, v, aim_deg=5.0, path_angle_deg=3.0)
        assert aimed.sidespin_rad_s == draw.sidespin_rad_s

    def test_ball_speed_monotone_decreasing_in_strike_offset(self) -> None:
        """Gate 3: |strike offset| up -> effective mass down -> speed down."""
        toe_speeds = [
            strike(BLADE, 2.0, strike_offset_toe_mm=r).ball_speed_mps
            for r in (0.0, 4.0, 8.0, 16.0, 32.0)
        ]
        assert all(a > b for a, b in itertools.pairwise(toe_speeds))
        high_speeds = [
            strike(BLADE, 2.0, strike_offset_high_mm=r).ball_speed_mps
            for r in (0.0, 3.0, 6.0, 12.0)
        ]
        assert all(a > b for a, b in itertools.pairwise(high_speeds))
        combined = strike(
            BLADE, 2.0, strike_offset_toe_mm=8.0, strike_offset_high_mm=6.0
        )
        assert (
            combined.ball_speed_mps
            < strike(BLADE, 2.0, strike_offset_toe_mm=8.0).ball_speed_mps
        )

    def test_off_center_effective_mass_matches_impact_package_formula(self) -> None:
        """The reduction is the scalar 1/(1/M + r^2/I) of swing_sim.impact."""
        flat = PutterSpec(name="Flat", head_mass_kg=0.350, loft_deg=0.0)
        r_m = 10.0e-3
        launch = strike(flat, 2.0, strike_offset_toe_mm=10.0)
        m_eff = 1.0 / (1.0 / flat.head_mass_kg + r_m**2 / DEFAULT_PUTTER_MOI_KG_M2)
        expected = 2.0 * (1.0 + flat.cor) * m_eff / (m_eff + GOLF_BALL_MASS_KG)
        assert launch.ball_speed_mps == pytest.approx(expected, rel=1e-12)

    def test_head_moi_hook_scales_off_center_loss(self) -> None:
        """P3 hook: explicit head MOI drives the off-center speed loss."""
        default_moi = strike(BLADE, 2.0, strike_offset_toe_mm=10.0)
        explicit = strike(
            BLADE,
            2.0,
            strike_offset_toe_mm=10.0,
            head_moi_kg_m2=DEFAULT_PUTTER_MOI_KG_M2,
        )
        assert explicit.ball_speed_mps == default_moi.ball_speed_mps
        low_moi = strike(BLADE, 2.0, strike_offset_toe_mm=10.0, head_moi_kg_m2=2.0e-4)
        high_moi = strike(BLADE, 2.0, strike_offset_toe_mm=10.0, head_moi_kg_m2=9.0e-4)
        assert low_moi.ball_speed_mps < default_moi.ball_speed_mps
        assert high_moi.ball_speed_mps > default_moi.ball_speed_mps

    def test_center_strike_ignores_head_moi(self) -> None:
        """MOI only matters off-center: centered strikes are unchanged."""
        neutral = strike(BLADE, 2.0)
        with_moi = strike(BLADE, 2.0, head_moi_kg_m2=2.0e-4)
        assert with_moi.ball_speed_mps == neutral.ball_speed_mps
        assert with_moi.spin_rad_s == neutral.spin_rad_s

    def test_attack_angle_square_to_face_kills_spin(self) -> None:
        """Attack angle equal to the effective loft -> zero spin loft.

        The head velocity is then parallel to the face normal: no
        tangential slip, no backspin, launch exactly along the attack
        angle.
        """
        launch = strike(BLADE, 2.0, attack_angle_deg=BLADE.loft_deg)
        assert launch.spin_rad_s == pytest.approx(0.0, abs=1e-12)
        assert launch.launch_angle_deg == pytest.approx(BLADE.loft_deg, rel=1e-12)

    def test_hitting_down_adds_topspin(self) -> None:
        """Spin loft (loft - attack) grows when hitting down."""
        down = strike(BLADE, 2.0, attack_angle_deg=-3.0)
        level = strike(BLADE, 2.0)
        up = strike(BLADE, 2.0, attack_angle_deg=2.0)
        assert down.spin_rad_s > level.spin_rad_s > up.spin_rad_s > 0.0

    def test_energy_never_created(self) -> None:
        """Gate 4: ball KE (linear + spin) never exceeds head KE."""
        for face, path, attack, toe, speed in itertools.product(
            (-10.0, 0.0, 10.0),
            (-10.0, 0.0, 10.0),
            (-5.0, 0.0, 5.0),
            (0.0, 15.0),
            (0.5, 3.0),
        ):
            launch = strike(
                BLADE,
                speed,
                face_angle_deg=face,
                path_angle_deg=path,
                attack_angle_deg=attack,
                strike_offset_toe_mm=toe,
            )
            ball_ke = 0.5 * GOLF_BALL_MASS_KG * launch.ball_speed_mps**2 + (
                0.5
                * GOLF_BALL_MOMENT_OF_INERTIA_KG_M2
                * (launch.spin_rad_s**2 + launch.sidespin_rad_s**2)
            )
            head_ke = 0.5 * BLADE.head_mass_kg * speed**2
            assert ball_ke < head_ke

    def test_defaults_bit_identical_to_legacy_1d(self) -> None:
        """Regression gate: at defaults the H3 1-D results are bit-exact.

        The expected values reproduce the pre-#4800 closed form in the
        same operation order; comparisons are ``==``, not approx.
        """
        cap = 2.0 / 7.0
        for speed, lean, spec in itertools.product(
            (0.3, 1.0, 1.8, 3.2),
            (-2.0, 0.0, 1.5),
            MINIMAL_PUTTERS.values(),
        ):
            launch = strike(spec, speed, shaft_lean_deg=lean)
            delta = math.radians(spec.loft_deg + lean)
            mass_ratio = spec.head_mass_kg / (spec.head_mass_kg + GOLF_BALL_MASS_KG)
            transfer = (1.0 + spec.cor) * mass_ratio
            v_normal = transfer * speed * math.cos(delta)
            u_tangential = -speed * math.sin(delta)
            v_tangential = cap * u_tangential
            spin = -(1.0 - cap) * u_tangential / GOLF_BALL_RADIUS_M
            horizontal = v_normal * math.cos(delta) - v_tangential * math.sin(delta)
            vertical = v_normal * math.sin(delta) + v_tangential * math.cos(delta)
            assert launch.ball_speed_mps == math.hypot(horizontal, vertical)
            assert launch.launch_angle_deg == math.degrees(
                math.atan2(vertical, horizontal)
            )
            assert launch.horizontal_speed_mps == horizontal
            assert launch.spin_rad_s == spin
            assert launch.start_azimuth_deg == 0.0
            assert launch.sidespin_rad_s == 0.0

    def test_rejects_out_of_range_2d_parameters(self) -> None:
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, aim_deg=60.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, face_angle_deg=25.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, path_angle_deg=-25.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, attack_angle_deg=15.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, strike_offset_toe_mm=50.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, strike_offset_high_mm=-30.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, head_moi_kg_m2=0.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, head_moi_kg_m2=float("nan"))


class TestBackstrokeProxy:
    def test_pendulum_speed_scales_linearly_with_amplitude(self) -> None:
        v1 = clubhead_speed_from_backstroke(0.2)
        v2 = clubhead_speed_from_backstroke(0.4)
        assert v2 == pytest.approx(2.0 * v1, rel=1e-12)

    def test_matches_simple_pendulum_formula(self) -> None:
        v = clubhead_speed_from_backstroke(0.3, putter_length_m=0.889)
        assert v == pytest.approx(0.3 * math.sqrt(9.80665 / 0.889), rel=1e-12)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            clubhead_speed_from_backstroke(0.0)
        with pytest.raises(ValueError):
            clubhead_speed_from_backstroke(0.3, putter_length_m=3.0)

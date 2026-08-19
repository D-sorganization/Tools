"""Limit and realism gates for shaft delivery dynamics (C2, #4551)."""

from __future__ import annotations

import math

import pytest

from shared.python.golf_club import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftProfileScaling,
    ShaftStation,
    scale_shaft_profile,
    solve_shaft_bending_modes,
)
from shared.python.golf_club.shaft_delivery import (
    GripKinematics,
    ShaftTipMass,
    solve_shaft_delivery,
)
from shared.python.golf_club.shaft_dynamics import ShaftModalSettings
from shared.python.golf_club.shaft_statics import (
    ShaftTipLoad,
    solve_cantilever_tip_response,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _uniform_profile(
    *,
    length_m: float = 1.0,
    ei_n_m2: float = 25.0,
    gj_n_m2: float = 18.0,
    density_kg_m: float = 0.08,
) -> ShaftProfile:
    def station(position_m: float) -> ShaftStation:
        return ShaftStation(
            position_m=position_m,
            outer_diameter_m=0.012,
            inner_diameter_m=0.010,
            linear_density_kg_m=density_kg_m,
            ei_about_x_n_m2=ei_n_m2,
            ei_about_y_n_m2=ei_n_m2,
            gj_n_m2=gj_n_m2,
            damping_ratio=0.025,
        )

    return ShaftProfile(
        shaft_id="uniform-delivery-reference",
        frame_id="shaft",
        raw_length_m=length_m,
        cut_length_m=length_m,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=(station(0.0), station(length_m)),
        provenance=ShaftProfileProvenance(
            source_name="analytic fixture",
            measurement_method="uniform reference",
            uncertainty_note="exact synthetic values",
        ),
    )


def _driver_profile() -> ShaftProfile:
    """Driver-like uniform reference: 65 g, EI 80 N·m², GJ 60 N·m², 1.12 m."""
    return _uniform_profile(
        length_m=1.12, ei_n_m2=80.0, gj_n_m2=60.0, density_kg_m=0.06
    )


_DRIVER_HEAD = ShaftTipMass(
    mass_kg=0.200, cg_back_m=0.012, cg_toe_m=0.030, cg_drop_m=0.040
)
_DRIVER_GRIP = GripKinematics(
    omega_rad_s=39.0,
    alpha_rad_s2=-80.0,
    swing_radius_m=1.15,
    downswing_duration_s=0.30,
    release_recovery=0.5,
)


class TestStaticsMomentExtension:
    """Closed forms for the additive tip-moment terms (uniform EI)."""

    def test_unit_moment_rotation_and_deflection_match_closed_forms(self) -> None:
        profile = _uniform_profile()
        response = solve_cantilever_tip_response(
            profile, ShaftTipLoad(moment_about_y_nm=1.0)
        )
        # theta = M*L/EI, delta = M*L^2/(2*EI) for a uniform cantilever.
        assert response.rotation_about_y_rad == pytest.approx(1.0 / 25.0, rel=1e-9)
        assert response.deflection_x_m == pytest.approx(1.0 / 50.0, rel=1e-9)

    def test_zero_moments_leave_the_force_response_unchanged(self) -> None:
        profile = _uniform_profile()
        force_only = solve_cantilever_tip_response(profile, ShaftTipLoad(force_x_n=2.5))
        with_zero_moments = solve_cantilever_tip_response(
            profile,
            ShaftTipLoad(force_x_n=2.5, moment_about_x_nm=0.0, moment_about_y_nm=0.0),
        )
        assert force_only == with_zero_moments


class TestLimits:
    def test_rigid_shaft_produces_no_deltas(self) -> None:
        rigid = scale_shaft_profile(
            _driver_profile(),
            ShaftProfileScaling(
                ei_about_x_scale=1e6, ei_about_y_scale=1e6, gj_scale=1e6
            ),
            shaft_id="rigid-limit",
        )
        deltas = solve_shaft_delivery(rigid, _DRIVER_HEAD, _DRIVER_GRIP)
        assert abs(deltas.dynamic_loft_add_deg) < 1e-4
        assert abs(deltas.face_closure_deg) < 1e-4
        assert abs(deltas.lie_toe_down_deg) < 1e-4
        assert abs(deltas.lead_deflection_m) < 1e-6
        assert abs(deltas.droop_deflection_m) < 1e-6
        assert deltas.kick_speed_mps < 1e-2

    def test_static_limit_reproduces_the_cantilever_response(self) -> None:
        """omega = 0 kills tension, alignment, and centrifugal moments; a huge
        downswing duration sends the amplification to 1 — leaving exactly the
        statics response for the tangential tip force."""
        profile = _uniform_profile()
        head = ShaftTipMass(mass_kg=0.2, cg_back_m=0.0, cg_toe_m=0.0, cg_drop_m=0.0)
        grip = GripKinematics(
            omega_rad_s=0.0,
            alpha_rad_s2=-50.0,
            swing_radius_m=1.0,
            downswing_duration_s=1e6,
            release_recovery=0.0,
        )
        deltas = solve_shaft_delivery(profile, head, grip)
        expected = solve_cantilever_tip_response(
            profile, ShaftTipLoad(force_x_n=0.2 * 50.0 * 1.0)
        )
        assert deltas.lead_deflection_m == pytest.approx(
            expected.deflection_x_m, rel=1e-9
        )
        assert deltas.dynamic_loft_add_deg == pytest.approx(
            math.degrees(expected.rotation_about_y_rad), rel=1e-9
        )
        assert deltas.lie_toe_down_deg == 0.0
        assert deltas.face_closure_deg == 0.0
        assert deltas.dynamic_amplification == pytest.approx(1.0, abs=1e-9)

    def test_rayleigh_first_mode_matches_the_modal_fe(self) -> None:
        """With a vanishing head mass the Rayleigh estimate must agree with
        the modal FE's first bending frequency (closed-form uniform shaft)."""
        profile = _uniform_profile()
        head = ShaftTipMass(mass_kg=1e-9, cg_back_m=0.0, cg_toe_m=0.0, cg_drop_m=0.0)
        grip = GripKinematics(omega_rad_s=1.0, alpha_rad_s2=0.0, swing_radius_m=1.0)
        deltas = solve_shaft_delivery(profile, head, grip)
        modal = solve_shaft_bending_modes(
            profile, ShaftModalSettings(element_count=24, mode_count=1)
        )
        assert deltas.first_mode_hz == pytest.approx(
            modal.frequencies_x_hz[0], rel=0.02
        )


class TestDriverRealism:
    def test_deltas_land_in_published_fitting_ranges(self) -> None:
        deltas = solve_shaft_delivery(_driver_profile(), _DRIVER_HEAD, _DRIVER_GRIP)
        assert 0.5 < deltas.dynamic_loft_add_deg < 6.0
        assert 0.2 < deltas.lie_toe_down_deg < 4.0
        assert 0.05 < deltas.face_closure_deg < 8.0  # decelerating -> closed
        assert 0.005 < deltas.lead_deflection_m < 0.06
        assert 0.0 < deltas.kick_speed_mps < 5.0
        assert deltas.dynamic_amplification > 1.0

    def test_stiffer_shaft_delivers_strictly_smaller_deltas(self) -> None:
        """The core clubfitting property: stiffness monotonicity."""
        base = solve_shaft_delivery(_driver_profile(), _DRIVER_HEAD, _DRIVER_GRIP)
        stiff = solve_shaft_delivery(
            scale_shaft_profile(
                _driver_profile(),
                ShaftProfileScaling(
                    ei_about_x_scale=1.5, ei_about_y_scale=1.5, gj_scale=1.5
                ),
                shaft_id="stiff-variant",
            ),
            _DRIVER_HEAD,
            _DRIVER_GRIP,
        )
        assert stiff.dynamic_loft_add_deg < base.dynamic_loft_add_deg
        assert stiff.lie_toe_down_deg < base.lie_toe_down_deg
        assert stiff.face_closure_deg < base.face_closure_deg
        assert abs(stiff.lead_deflection_m) < abs(base.lead_deflection_m)
        # Kick is deliberately NOT asserted monotone: v ~ f1*|delta| with
        # f1 ~ sqrt(k) and delta ~ 1/k-ish, so stiffness mostly cancels --
        # matching the fitting literature (flex changes kick timing far more
        # than kick velocity). It must merely stay in the same small band.
        assert stiff.kick_speed_mps == pytest.approx(base.kick_speed_mps, rel=0.15)
        assert stiff.first_mode_hz > base.first_mode_hz

    def test_face_is_held_open_while_still_accelerating(self) -> None:
        accelerating = GripKinematics(
            omega_rad_s=39.0,
            alpha_rad_s2=+80.0,
            swing_radius_m=1.15,
        )
        deltas = solve_shaft_delivery(_driver_profile(), _DRIVER_HEAD, accelerating)
        assert deltas.face_closure_deg < 0.0


class TestContracts:
    def test_wrong_types_raise(self) -> None:
        with pytest.raises(TypeError):
            solve_shaft_delivery("nope", _DRIVER_HEAD, _DRIVER_GRIP)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            solve_shaft_delivery(_driver_profile(), "nope", _DRIVER_GRIP)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            solve_shaft_delivery(_driver_profile(), _DRIVER_HEAD, "nope")  # type: ignore[arg-type]

    def test_invalid_fields_raise_value_errors(self) -> None:
        with pytest.raises(ValueError):
            ShaftTipMass(mass_kg=0.0, cg_back_m=0.0, cg_toe_m=0.0, cg_drop_m=0.0)
        with pytest.raises(ValueError):
            ShaftTipMass(mass_kg=0.2, cg_back_m=-0.01, cg_toe_m=0.0, cg_drop_m=0.0)
        with pytest.raises(ValueError):
            GripKinematics(
                omega_rad_s=39.0,
                alpha_rad_s2=0.0,
                swing_radius_m=1.15,
                release_recovery=1.5,
            )

    def test_whippy_shaft_beyond_validity_refuses(self) -> None:
        whippy = scale_shaft_profile(
            _uniform_profile(),
            ShaftProfileScaling(ei_about_x_scale=0.2, ei_about_y_scale=0.2),
            shaft_id="whippy-limit",
        )
        with pytest.raises(ValueError, match="quasi-static validity"):
            solve_shaft_delivery(whippy, _DRIVER_HEAD, _DRIVER_GRIP)

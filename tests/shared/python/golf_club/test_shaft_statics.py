from __future__ import annotations

import math

import pytest

from shared.python.golf_club import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
    ShaftTipLoad,
    solve_cantilever_tip_response,
)


def _uniform_profile(*, length_m: float = 1.0) -> ShaftProfile:
    station = {
        "outer_diameter_m": 0.012,
        "inner_diameter_m": 0.010,
        "linear_density_kg_m": 0.08,
        "ei_about_x_n_m2": 40.0,
        "ei_about_y_n_m2": 50.0,
        "gj_n_m2": 25.0,
        "damping_ratio": 0.02,
        "spine_angle_rad": 0.0,
    }
    return ShaftProfile(
        shaft_id="uniform-reference",
        frame_id="shaft:butt_to_tip",
        raw_length_m=length_m,
        cut_length_m=length_m,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=0.0,
        stations=(
            ShaftStation(position_m=0.0, **station),
            ShaftStation(position_m=length_m / 2.0, **station),
            ShaftStation(position_m=length_m, **station),
        ),
        provenance=ShaftProfileProvenance(
            source_name="analytic-uniform-beam",
            measurement_method="synthetic reference fixture",
            uncertainty_note="Exact analytic fixture; not an equipment preset.",
        ),
    )


def test_uniform_cantilever_matches_closed_form_bending_and_torsion() -> None:
    profile = _uniform_profile()
    load = ShaftTipLoad(force_x_n=10.0, force_y_n=12.0, torque_about_shaft_nm=5.0)

    response = solve_cantilever_tip_response(profile, load)

    assert response.deflection_x_m == pytest.approx(10.0 / (3.0 * 50.0))
    assert response.rotation_about_y_rad == pytest.approx(10.0 / (2.0 * 50.0))
    assert response.deflection_y_m == pytest.approx(12.0 / (3.0 * 40.0))
    assert response.rotation_about_x_rad == pytest.approx(-12.0 / (2.0 * 40.0))
    assert response.twist_about_shaft_rad == pytest.approx(5.0 / 25.0)
    assert response.model_name == "euler_bernoulli_cantilever_static/1"


def test_response_scales_linearly_with_load_and_cubically_with_length() -> None:
    base = solve_cantilever_tip_response(
        _uniform_profile(length_m=1.0),
        ShaftTipLoad(force_x_n=3.0),
    )
    doubled_load = solve_cantilever_tip_response(
        _uniform_profile(length_m=1.0),
        ShaftTipLoad(force_x_n=6.0),
    )
    doubled_length = solve_cantilever_tip_response(
        _uniform_profile(length_m=2.0),
        ShaftTipLoad(force_x_n=3.0),
    )

    assert doubled_load.deflection_x_m == pytest.approx(2.0 * base.deflection_x_m)
    assert doubled_length.deflection_x_m == pytest.approx(8.0 * base.deflection_x_m)


def test_tip_response_rejects_nonfinite_loads() -> None:
    with pytest.raises(ValueError, match="force_x_n"):
        ShaftTipLoad(force_x_n=math.inf)


def test_zero_load_has_exact_zero_response() -> None:
    response = solve_cantilever_tip_response(_uniform_profile(), ShaftTipLoad())

    assert response.deflection_x_m == 0.0
    assert response.deflection_y_m == 0.0
    assert response.twist_about_shaft_rad == 0.0

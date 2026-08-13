"""Modal finite-element reference tests for flexible shafts."""

from __future__ import annotations

import math

import pytest

from shared.python.golf_club import (
    ShaftModalSettings,
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
    solve_shaft_bending_modes,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _uniform_profile(*, insertion_depth_m: float = 0.0) -> ShaftProfile:
    def station(position_m: float) -> ShaftStation:
        return ShaftStation(
            position_m=position_m,
            outer_diameter_m=0.012,
            inner_diameter_m=0.010,
            linear_density_kg_m=0.08,
            ei_about_x_n_m2=40.0,
            ei_about_y_n_m2=25.0,
            gj_n_m2=18.0,
            damping_ratio=0.025,
        )

    return ShaftProfile(
        shaft_id="uniform-reference",
        frame_id="shaft",
        raw_length_m=1.0,
        cut_length_m=1.0,
        tip_trim_m=0.0,
        butt_trim_m=0.0,
        insertion_depth_m=insertion_depth_m,
        stations=(station(0.0), station(1.0)),
        provenance=ShaftProfileProvenance(
            source_name="analytic fixture",
            measurement_method="uniform reference",
            uncertainty_note="exact synthetic values",
        ),
    )


def test_uniform_first_modes_match_euler_bernoulli_closed_form() -> None:
    profile = _uniform_profile()
    settings = ShaftModalSettings(element_count=24, mode_count=3)

    response = solve_shaft_bending_modes(profile, settings)

    beta_1 = 1.875104068711961
    expected_x = beta_1**2 / (2.0 * math.pi) * math.sqrt(25.0 / 0.08)
    expected_y = beta_1**2 / (2.0 * math.pi) * math.sqrt(40.0 / 0.08)
    assert response.frequencies_x_hz[0] == pytest.approx(expected_x, rel=2e-4)
    assert response.frequencies_y_hz[0] == pytest.approx(expected_y, rel=2e-4)
    assert response.frequencies_x_hz == tuple(sorted(response.frequencies_x_hz))
    assert response.frequencies_y_hz == tuple(sorted(response.frequencies_y_hz))
    assert response.model_name == "euler_bernoulli_bending_fem/1"
    assert response.element_count == 24


def test_anisotropic_frequency_ratio_follows_axis_stiffness() -> None:
    response = solve_shaft_bending_modes(
        _uniform_profile(), ShaftModalSettings(element_count=16, mode_count=1)
    )

    assert response.frequencies_y_hz[0] / response.frequencies_x_hz[0] == pytest.approx(
        math.sqrt(40.0 / 25.0), rel=1e-10
    )


def test_hosel_insertion_shortens_span_and_raises_frequency() -> None:
    settings = ShaftModalSettings(element_count=16, mode_count=1)

    exposed = solve_shaft_bending_modes(_uniform_profile(), settings)
    inserted = solve_shaft_bending_modes(
        _uniform_profile(insertion_depth_m=0.1), settings
    )

    assert inserted.flexible_length_m == pytest.approx(0.9)
    assert inserted.frequencies_x_hz[0] > exposed.frequencies_x_hz[0]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"element_count": 1}, ValueError, "element_count"),
        ({"mode_count": 0}, ValueError, "mode_count"),
        ({"element_count": 2, "mode_count": 5}, ValueError, "mode_count"),
        ({"element_count": 2.5}, TypeError, "element_count"),
    ],
)
def test_modal_settings_reject_invalid_discretizations(
    kwargs: dict[str, object], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        ShaftModalSettings(**kwargs)  # type: ignore[arg-type]


def test_solver_rejects_wrong_contract_types() -> None:
    with pytest.raises(TypeError, match="profile"):
        solve_shaft_bending_modes(object(), ShaftModalSettings())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="settings"):
        solve_shaft_bending_modes(_uniform_profile(), object())  # type: ignore[arg-type]

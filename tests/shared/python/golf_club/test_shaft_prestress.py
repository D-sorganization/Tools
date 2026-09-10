"""Independent energy, beam-limit and rotating out-of-plane benchmarks."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club.shaft_dynamics import (
    ShaftModalSettings,
    solve_shaft_bending_modes,
)
from shared.python.golf_club.shaft_prestress import (
    ShaftPrestress,
    radial_shaft_tension,
    shaft_geometric_stiffness,
    solve_prestressed_shaft_modes,
)
from shared.python.golf_club.shaft_profile import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.scientific]


@pytest.fixture
def profile() -> ShaftProfile:
    """Dimensionless benchmark represented explicitly in SI units."""
    stations = tuple(
        ShaftStation(x, 0.012, 0.010, 1.0, 1.0, 1.0, 1.0, 0.0) for x in (0.0, 1.0)
    )
    return ShaftProfile(
        "prestress-reference",
        "shaft",
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        stations,
        ShaftProfileProvenance("synthetic", "analytic beam", "no measurement"),
    )


def test_constant_tension_matrix_and_energy(profile: ShaftProfile) -> None:
    settings = ShaftModalSettings(element_count=2)
    result = shaft_geometric_stiffness(
        profile, ShaftPrestress(tip_tension_n=7), settings
    )
    length = 0.5
    # Independently integrated cubic-Hermite slope products for constant N.
    local = (
        7
        / (30 * length)
        * np.array(
            [
                [36, 3 * length, -36, 3 * length],
                [3 * length, 4 * length**2, -3 * length, -(length**2)],
                [-36, -3 * length, 36, -3 * length],
                [3 * length, -(length**2), -3 * length, 4 * length**2],
            ]
        )
    )
    expected = np.zeros((6, 6))
    expected[:4, :4] += local
    expected[2:, 2:] += local
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result @ [1, 0, 1, 0, 1, 0], 0, atol=1e-12)
    assert np.linalg.eigvalsh(result)[0] >= -1e-12
    # w(s)=s: U_g = N L / 2; all nodal slopes are one.
    displacement = np.array([0, 1, 0.5, 1, 1, 1])
    assert displacement @ result @ displacement / 2 == pytest.approx(3.5)


def test_radial_tension_has_tip_mass_and_hub_offset(profile: ShaftProfile) -> None:
    load = ShaftPrestress(
        angular_speed_rad_s=3, hub_radius_m=0.4, tip_mass_kg=0.2, tip_tension_n=5
    )
    for position in (0.0, 0.2, 0.9, 1.0):
        expected = 5 + 9 * (0.2 * 1.4 + 0.4 * (1 - position) + (1 - position**2) / 2)
        assert radial_shaft_tension(profile, load, position) == pytest.approx(expected)


def test_tapered_density_respects_trimmed_coordinate(profile: ShaftProfile) -> None:
    stations = tuple(
        replace(station, linear_density_kg_m=1 + station.position_m)
        for station in profile.stations
    )
    trimmed = replace(
        profile,
        stations=stations,
        butt_trim_m=0.2,
        tip_trim_m=0.1,
        cut_length_m=0.7,
        insertion_depth_m=0.1,
    )
    load = ShaftPrestress(angular_speed_rad_s=2, hub_radius_m=0.3)

    # Integral (1.2+s)(0.3+s) ds over exposed coordinates [x, 0.6].
    def integral(x: float) -> float:
        return 0.36 * x + 0.75 * x * x + x**3 / 3

    assert radial_shaft_tension(trimmed, load, 0.17) == pytest.approx(
        4 * (integral(0.6) - integral(0.17)), rel=1e-12
    )


def test_zero_load_recovers_existing_modes(profile: ShaftProfile) -> None:
    settings = ShaftModalSettings(element_count=12)
    baseline = solve_shaft_bending_modes(profile, settings)
    response = solve_prestressed_shaft_modes(profile, ShaftPrestress(), settings)
    assert response.frequencies_x_hz == pytest.approx(
        baseline.frequencies_x_hz, rel=1e-12
    )
    assert response.model_name == "prescribed_tension_bending_fem/1"
    assert any("Coriolis" in item for item in response.assumptions)


@pytest.mark.parametrize(
    ("spin", "expected"), [(0, 3.5160), (3, 4.7973), (6, 7.3604), (12, 13.1702)]
)
def test_out_of_plane_published_first_mode(
    profile: ShaftProfile, spin: float, expected: float
) -> None:
    # Rodrigues et al. arXiv:2401.17519v1 Table 5, Exact column; eta=Omega
    # because L=EI=linear density=1. Only the radial out-of-plane limit.
    response = solve_prestressed_shaft_modes(
        profile,
        ShaftPrestress(angular_speed_rad_s=spin),
        ShaftModalSettings(element_count=24, mode_count=1),
    )
    assert 2 * math.pi * response.frequencies_x_hz[0] == pytest.approx(
        expected, abs=6e-5
    )


def test_large_tip_mass_recovers_static_tip_spring_limit(profile: ShaftProfile) -> None:
    load = ShaftPrestress(tip_mass_kg=1e5)
    response = solve_prestressed_shaft_modes(
        profile, load, ShaftModalSettings(element_count=8, mode_count=1)
    )
    assert (2 * math.pi * response.frequencies_x_hz[0]) ** 2 == pytest.approx(
        3 / 1e5, rel=1e-5
    )


def test_mesh_refinement_converges(profile: ShaftProfile) -> None:
    frequencies = [
        solve_prestressed_shaft_modes(
            profile,
            ShaftPrestress(angular_speed_rad_s=12),
            ShaftModalSettings(element_count=count, mode_count=3),
        ).frequencies_x_hz[2]
        for count in (4, 8, 16, 32)
    ]
    differences = np.abs(np.diff(frequencies))
    assert differences[2] < differences[1] < differences[0]
    assert abs(2 * math.pi * frequencies[-1] - 79.6145) < 0.002


@pytest.mark.parametrize("name", ["tip_mass_kg", "tip_tension_n", "hub_radius_m"])
def test_load_rejects_negative_parameters(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        ShaftPrestress(**{name: -1.0})


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_load_rejects_nonfinite_speed(value: float) -> None:
    with pytest.raises(ValueError, match="angular_speed"):
        ShaftPrestress(angular_speed_rad_s=value)


def test_coordinate_and_contract_refusal(profile: ShaftProfile) -> None:
    with pytest.raises(ValueError, match="position"):
        radial_shaft_tension(profile, ShaftPrestress(), 1.1)
    with pytest.raises(TypeError, match="profile"):
        solve_prestressed_shaft_modes(object(), ShaftPrestress())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prestress"):
        shaft_geometric_stiffness(profile, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="settings"):
        solve_prestressed_shaft_modes(profile, ShaftPrestress(), object())  # type: ignore[arg-type]


def test_density_knot_in_element_retains_exact_energy(profile: ShaftProfile) -> None:
    from scipy.integrate import quad

    base = profile.stations[0]
    piecewise = replace(
        profile,
        stations=(
            base,
            replace(base, position_m=0.37, linear_density_kg_m=2),
            replace(base, position_m=1.0, linear_density_kg_m=0.7),
        ),
    )
    load = ShaftPrestress(angular_speed_rad_s=4, hub_radius_m=0.2, tip_mass_kg=0.3)
    matrix = shaft_geometric_stiffness(
        piecewise, load, ShaftModalSettings(element_count=2)
    )

    # w=s^3 is represented exactly. Reverse the order of integration in
    # integral N(s)*9s^4 ds, giving integral mu(r)*(h+r)*9r^5/5 dr.
    def integrand(position: float) -> float:
        density = np.interp(position, [0, 0.37, 1], [1, 2, 0.7])
        return float(density * (0.2 + position) * 9 * position**5 / 5)

    distributed = quad(integrand, 0, 1, points=[0.37], epsabs=1e-11)[0]
    expected_energy = 16 * (distributed + 0.3 * 1.2 * 9 / 5) / 2
    displacement = np.array([0, 0, 0.125, 0.75, 1, 3])
    assert displacement @ matrix @ displacement / 2 == pytest.approx(
        expected_energy, rel=1e-12
    )


def test_speed_sign_and_independent_matrix_storage(profile: ShaftProfile) -> None:
    load = ShaftPrestress(angular_speed_rad_s=3)
    forward = shaft_geometric_stiffness(profile, load)
    reverse = shaft_geometric_stiffness(profile, replace(load, angular_speed_rad_s=-3))
    np.testing.assert_array_equal(forward, reverse)
    forward[:] = 0
    assert np.linalg.norm(shaft_geometric_stiffness(profile, load)) > 0


def test_unsupported_numeric_inputs_are_refused(profile: ShaftProfile) -> None:
    with pytest.raises(TypeError, match="angular_speed"):
        ShaftPrestress(angular_speed_rad_s=True)
    with pytest.raises(ValueError, match="nonfinite"):
        radial_shaft_tension(profile, ShaftPrestress(angular_speed_rad_s=1e200), 0)


def test_public_facade_exposes_the_qualified_model() -> None:
    from shared.python import golf_club

    assert golf_club.ShaftPrestress is ShaftPrestress
    assert golf_club.solve_prestressed_shaft_modes is solve_prestressed_shaft_modes


def test_uncoupled_mode_solver_refuses_rotated_principal_axes(
    profile: ShaftProfile,
) -> None:
    rotated = replace(
        profile,
        stations=tuple(
            replace(station, spine_angle_rad=0.3) for station in profile.stations
        ),
    )
    with pytest.raises(ValueError, match="spine_angle"):
        solve_prestressed_shaft_modes(rotated, ShaftPrestress())

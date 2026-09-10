"""Distributed rotation benchmarks independent of the point quadrature helper."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigvals

from shared.python.golf_club.grip_impedance import PassiveGripImpedance
from shared.python.golf_club.rotating_body import (
    RotatingFrameState,
    rotating_body_tangent,
)
from shared.python.golf_club.shaft_dynamics import ShaftModalSettings
from shared.python.golf_club.shaft_linear_system import (
    ShaftAttachments,
    ShaftRodProperties,
    assemble_shaft_linear_system,
)
from shared.python.golf_club.shaft_prestress import (
    ShaftPrestress,
    shaft_geometric_stiffness,
)
from shared.python.golf_club.shaft_profile import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)
from shared.python.golf_club.shaft_rotating_transport import (
    ShaftRotaryInertia,
    ShaftRotatingModel,
    assemble_shaft_rotating_transport,
)
from shared.python.golf_club.types import ComponentMassProperties, ComponentRole

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.scientific]


@pytest.fixture
def model() -> ShaftRotatingModel:
    source = ShaftProfileProvenance("synthetic", "analytic", "not measured")
    stations = tuple(ShaftStation(s, 0.02, 0.01, 2, 3, 5, 7, 0) for s in (0, 1))
    profile = ShaftProfile("rod", "shaft", 1, 1, 0, 0, 0, stations, source)
    rod = ShaftRodProperties(profile, (1000, 1000), (0.03, 0.03), source)
    inertia = ShaftRotaryInertia("rod", "shaft", ((0.01, 0.02, 0.001),) * 2, source)
    return ShaftRotatingModel(rod, inertia)


def _motion(omega: tuple[float, float, float]) -> RotatingFrameState:
    return RotatingFrameState("shaft", omega, (0, 0, 0), (0, 0, 0))


def test_zero_rotation_rayleigh_mass_and_stationary_stiffness(
    model: ShaftRotatingModel,
) -> None:
    settings = ShaftModalSettings(2)
    result = assemble_shaft_rotating_transport(model, _motion((0, 0, 0)), settings)
    stationary = assemble_shaft_linear_system(model.rod, settings)
    np.testing.assert_allclose(result.elastic_stiffness, stationary.stiffness)
    for axis, rotary in ((3, 0.01), (4, 0.02), (5, 0.03)):
        omega = np.eye(3)[axis - 3]
        state = np.concatenate(
            [np.r_[np.cross(omega, [0, 0, s]), omega] for s in result.positions_m]
        )
        expected = (2 / 3 if axis != 5 else 0) + rotary
        assert state @ result.mass @ state == pytest.approx(expected)
    difference = result.mass - stationary.mass
    assert np.linalg.eigvalsh(difference)[0] >= -1e-12
    assert np.linalg.norm(difference) > 0
    for name in (
        "gyroscopic",
        "centrifugal_stiffness",
        "euler_stiffness",
        "equilibrium_force",
    ):
        np.testing.assert_array_equal(getattr(result, name), 0)


def test_uniform_translation_coriolis_and_spin_softening(
    model: ShaftRotatingModel,
) -> None:
    omega = np.array([0.2, 0.7, -0.1])
    result = assemble_shaft_rotating_transport(
        model, _motion(tuple(omega)), ShaftModalSettings(4)
    )
    translation = np.tile(np.r_[np.eye(3), np.zeros((3, 3))], (5, 1))
    expected_cross = np.cross(omega, np.eye(3)).T
    np.testing.assert_allclose(
        translation.T @ result.gyroscopic @ translation, 4 * expected_cross, atol=1e-12
    )
    np.testing.assert_allclose(
        translation.T @ result.centrifugal_stiffness @ translation,
        2 * expected_cross @ expected_cross,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.gyroscopic + result.gyroscopic.T, 0, atol=1e-12)


def test_radial_equilibrium_resultant_is_outboard_centrifugal_force(
    model: ShaftRotatingModel,
) -> None:
    frame = RotatingFrameState("shaft", (0, 3, 0), (0, 0, 0), (0, 0, -9 * 0.4))
    result = assemble_shaft_rotating_transport(model, frame, ShaftModalSettings(4))
    # Integral mu*Omega²*(hub+s) ds = 2*9*(0.4+0.5).
    assert result.equilibrium_force[2::6].sum() == pytest.approx(16.2)
    assert result.equilibrium_force[0::6].sum() == pytest.approx(0)


def test_full_head_and_relative_grip_assembly(model: ShaftRotatingModel) -> None:
    head = ComponentMassProperties(
        "head",
        ComponentRole.HEAD,
        "shaft",
        0.4,
        (0.03, -0.02, 0.04),
        ((0.2, 0.02, 0.01), (0.02, 0.3, 0.03), (0.01, 0.03, 0.4)),
    )
    factor = np.eye(6)
    grip = PassiveGripImpedance("shaft", factor, 2 * factor, 3 * factor, "synthetic")
    loaded = replace(model, attachments=ShaftAttachments(head, grip))
    settings, frame = ShaftModalSettings(2), _motion((0, 3, 0))
    bare = assemble_shaft_rotating_transport(model, frame, settings)
    result = assemble_shaft_rotating_transport(loaded, frame, settings)
    head_terms = rotating_body_tangent(head, frame, (0, 0, 1))
    for name in (
        "mass",
        "gyroscopic",
        "centrifugal_stiffness",
        "euler_stiffness",
        "acceleration_stiffness",
    ):
        difference = getattr(result, name) - getattr(bare, name)
        np.testing.assert_allclose(
            difference[-6:, -6:], getattr(head_terms, name), atol=1e-12
        )
        np.testing.assert_allclose(
            difference[:6, :6], np.eye(6) if name == "mass" else 0, atol=1e-12
        )
    np.testing.assert_allclose(result.damping[:6, :6], 4 * np.eye(6))
    np.testing.assert_allclose(
        (result.elastic_stiffness - bare.elastic_stiffness)[:6, :6], 9 * np.eye(6)
    )


def test_piecewise_density_integration_respects_station_knots(
    model: ShaftRotatingModel,
) -> None:
    stations = tuple(
        replace(model.rod.profile.stations[0], position_m=s, linear_density_kg_m=mu)
        for s, mu in ((0, 1), (0.3, 4), (1, 2))
    )
    profile = replace(model.rod.profile, stations=stations)
    rod = replace(
        model.rod,
        profile=profile,
        axial_stiffness_n=(1000,) * 3,
        polar_mass_per_length_kg_m=(0.03,) * 3,
    )
    variable = replace(
        model,
        rod=rod,
        rotary_inertia=replace(
            model.rotary_inertia, transverse_kg_m=((0.01, 0.02, 0.001),) * 3
        ),
    )
    result = assemble_shaft_rotating_transport(
        variable, _motion((0, 0, 0)), ShaftModalSettings(2)
    )
    translation = np.tile([1, 0, 0, 0, 0, 0], 3)
    assert translation @ result.mass @ translation == pytest.approx(0.3 * 2.5 + 0.7 * 3)


def test_section_inertia_cannot_disagree_with_polar_data(
    model: ShaftRotatingModel,
) -> None:
    inertia = replace(model.rotary_inertia, transverse_kg_m=((0.02, 0.02, 0),) * 2)
    with pytest.raises(ValueError, match="polar"):
        replace(model, rotary_inertia=inertia)


@pytest.mark.parametrize(
    "values",
    [
        ((0, 0.03, 0),) * 2,
        ((0.01, 0.02, 0.02),) * 2,
        ((True, 0.02, 0),) * 2,
        ((0.01, np.inf, 0),) * 2,
    ],
)
def test_invalid_section_tensor_is_refused(
    model: ShaftRotatingModel, values: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(model.rotary_inertia, transverse_kg_m=values)


def test_frame_and_profile_identifiers_are_checked(model: ShaftRotatingModel) -> None:
    with pytest.raises(ValueError, match="profile"):
        replace(model, rotary_inertia=replace(model.rotary_inertia, profile_id="other"))
    with pytest.raises(ValueError, match="frame"):
        assemble_shaft_rotating_transport(
            model, replace(_motion((0, 0, 0)), frame_id="other")
        )


def _lowest_gyroscopic_frequency(
    mass: np.ndarray, gyro: np.ndarray, stiffness: np.ndarray
) -> float:
    """Independent quadratic eigenproblem for the clamped reference only."""
    count = len(mass)
    zero, identity = np.zeros_like(mass), np.eye(count)
    dynamic = np.block([[zero, identity], [-stiffness, -gyro]])
    metric = np.block([[identity, zero], [zero, mass]])
    values = eigvals(dynamic, metric)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values.real)) < 1e-5
    return float(np.min(values.imag[values.imag > 1e-5]))


@pytest.mark.parametrize(
    "speed,out_of_plane,in_plane",
    [(3, 4.7973, 3.7434), (6, 7.3604, 4.2625), (12, 13.1702, 5.4233)],
)
def test_complete_radial_reference_has_distinct_in_and_out_of_plane_modes(
    model: ShaftRotatingModel, speed: float, out_of_plane: float, in_plane: float
) -> None:
    """Rodrigues et al. 2024 Table 5: exact out-of-plane, five-TITOP in-plane.

    This deliberately assembles the known clamped radial limit; it does not
    substitute a radial tensile field for a general loaded-shaft equilibrium.
    """
    original = model.rod.profile
    stations = tuple(
        replace(s, linear_density_kg_m=1, ei_about_x_n_m2=1, ei_about_y_n_m2=1)
        for s in original.stations
    )
    rod = replace(
        model.rod,
        profile=replace(original, stations=stations),
        axial_stiffness_n=(1e6, 1e6),
        polar_mass_per_length_kg_m=(2e-8, 2e-8),
    )
    rotary = replace(model.rotary_inertia, transverse_kg_m=((1e-8, 1e-8, 0),) * 2)
    radial = replace(model, rod=rod, rotary_inertia=rotary)
    settings = ShaftModalSettings(24)
    result = assemble_shaft_rotating_transport(radial, _motion((0, speed, 0)), settings)
    geometric = shaft_geometric_stiffness(
        rod.profile, ShaftPrestress(angular_speed_rad_s=speed), settings
    )
    stiffness = result.elastic_stiffness + result.centrifugal_stiffness
    for axes, signs in (((0, 4), (1, 1)), ((1, 3), (1, -1))):
        indices = np.array([6 * node + axis for node in range(25) for axis in axes])
        factors = np.tile(signs, 25)
        stiffness[np.ix_(indices, indices)] += geometric * np.outer(factors, factors)
    # Conservative nominal strain bound is N_root / min(EA).
    assert speed**2 / (2 * 1e6) < 1e-4
    for axes, expected, tolerance in (
        ((1, 3, 5), out_of_plane, 2e-4),
        ((0, 2, 4), in_plane, 2e-3),
    ):
        indices = np.array([6 * node + axis for node in range(1, 25) for axis in axes])
        block = np.ix_(indices, indices)
        frequency = _lowest_gyroscopic_frequency(
            result.mass[block], result.gyroscopic[block], stiffness[block]
        )
        assert frequency == pytest.approx(expected, rel=tolerance)

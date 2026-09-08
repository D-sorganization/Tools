"""Independent static, rigid-motion and rod limits of a stationary 3-D shaft."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

from shared.python.golf_club.grip_impedance import PassiveGripImpedance
from shared.python.golf_club.shaft_dynamics import (
    ShaftModalSettings,
    solve_shaft_bending_modes,
)
from shared.python.golf_club.shaft_linear_system import (
    ShaftAttachments,
    ShaftRodProperties,
    assemble_shaft_linear_system,
)
from shared.python.golf_club.shaft_profile import (
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)
from shared.python.golf_club.types import ComponentMassProperties, ComponentRole

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.scientific]


@pytest.fixture
def rod() -> ShaftRodProperties:
    provenance = ShaftProfileProvenance("synthetic", "analytic", "not measured")
    stations = tuple(ShaftStation(s, 0.02, 0.01, 2, 3, 5, 7, 0) for s in (0, 1))
    profile = ShaftProfile("rod", "shaft", 1, 1, 0, 0, 0, stations, provenance)
    return ShaftRodProperties(profile, (1000, 1000), (0.02, 0.02), provenance)


def test_six_rigid_modes_and_independent_rigid_kinetic_energy(
    rod: ShaftRodProperties,
) -> None:
    result = assemble_shaft_linear_system(rod, ShaftModalSettings(4))
    mass, stiffness = np.asarray(result.mass), np.asarray(result.stiffness)
    assert result.frame_id == "shaft"
    assert result.model_name == "stationary_3d_euler_bernoulli/1"
    assert result.profile_id == "rod"
    for axis in range(6):
        velocity, omega = np.eye(6)[axis, :3], np.eye(6)[axis, 3:]
        state = np.concatenate(
            [
                np.r_[velocity + np.cross(omega, [0, 0, s]), omega]
                for s in result.positions_m
            ]
        )
        np.testing.assert_allclose(stiffness @ state, 0, atol=1e-10)
        expected = (2 if axis < 3 else 0.02 if axis == 5 else 2 / 3) / 2
        assert state @ mass @ state / 2 == pytest.approx(expected)
    assert np.linalg.eigvalsh(mass)[0] > 0
    assert np.count_nonzero(np.abs(np.linalg.eigvalsh(stiffness)) < 1e-8) == 6


@pytest.mark.parametrize(
    "axis,tip_compliance", [(0, 1 / 15), (1, 1 / 9), (2, 1 / 1000), (5, 1 / 7)]
)
def test_cantilever_tip_static_compliance(
    rod: ShaftRodProperties, axis: int, tip_compliance: float
) -> None:
    result = assemble_shaft_linear_system(rod, ShaftModalSettings(4))
    stiffness = np.asarray(result.stiffness)[6:, 6:]
    force = np.zeros(len(stiffness))
    force[-6 + axis] = 1
    displacement = np.linalg.solve(stiffness, force)
    assert displacement[-6 + axis] == pytest.approx(tip_compliance, rel=1e-11)


def test_existing_bending_modal_limit(rod: ShaftRodProperties) -> None:
    settings = ShaftModalSettings(8)
    result = assemble_shaft_linear_system(rod, settings)
    expected = solve_shaft_bending_modes(rod.profile, settings)
    for indices, frequencies in (
        ((0, 4), expected.frequencies_x_hz),
        ((1, 3), expected.frequencies_y_hz),
    ):
        dofs = np.array([6 * node + dof for node in range(1, 9) for dof in indices])
        values = eigh(
            np.asarray(result.stiffness)[np.ix_(dofs, dofs)],
            np.asarray(result.mass)[np.ix_(dofs, dofs)],
            eigvals_only=True,
        )
        np.testing.assert_allclose(
            np.sqrt(values[:3]) / (2 * np.pi), frequencies, rtol=1e-9
        )


@pytest.mark.parametrize(
    "axis,exact",
    [(2, np.pi / 2 * np.sqrt(1000 / 2)), (5, np.pi / 2 * np.sqrt(7 / 0.02))],
)
def test_axial_and_torsional_mesh_convergence(
    rod: ShaftRodProperties, axis: int, exact: float
) -> None:
    errors = []
    for count in (4, 8, 16):
        result = assemble_shaft_linear_system(rod, ShaftModalSettings(count))
        dofs = np.arange(6 + axis, 6 * (count + 1), 6)
        values = eigh(
            np.asarray(result.stiffness)[np.ix_(dofs, dofs)],
            np.asarray(result.mass)[np.ix_(dofs, dofs)],
            eigvals_only=True,
        )
        errors.append(abs(np.sqrt(values[0]) / exact - 1))
    assert errors[2] < errors[1] / 3 < errors[0] / 9
    assert errors[-1] < 0.0005


def test_head_offset_and_full_inertia_kinetic_energy(rod: ShaftRodProperties) -> None:
    inertia = ((0.2, 0.02, 0.01), (0.02, 0.3, 0.03), (0.01, 0.03, 0.4))
    head = ComponentMassProperties(
        "head", ComponentRole.HEAD, "shaft", 0.4, (0.03, -0.02, 0.04), inertia
    )
    settings = ShaftModalSettings(2)
    bare = assemble_shaft_linear_system(rod, settings)
    loaded = assemble_shaft_linear_system(rod, settings, ShaftAttachments(head))
    state = np.zeros(18)
    velocity, omega = np.array([1, 2, 3]), np.array([-2, 1, 4])
    state[-6:] = np.r_[velocity, omega]
    com_velocity = velocity + np.cross(omega, head.center_of_mass_m)
    expected = (
        head.mass_kg * com_velocity @ com_velocity + omega @ np.asarray(inertia) @ omega
    ) / 2
    delta = np.asarray(loaded.mass) - bare.mass
    assert state @ delta @ state / 2 == pytest.approx(expected)
    assert np.any(np.abs(delta[-6:-3, -3:]) > 0)


def test_passive_grip_assembles_at_butt_only(rod: ShaftRodProperties) -> None:
    factor = np.zeros((6, 6))
    factor[0, [0, 5]] = [2, 3]
    grip = PassiveGripImpedance("shaft", factor, factor * 2, factor * 3, "synthetic")
    settings = ShaftModalSettings(2)
    bare = assemble_shaft_linear_system(rod, settings)
    loaded = assemble_shaft_linear_system(rod, settings, ShaftAttachments(grip=grip))
    for name, scale in (("mass", 1), ("damping", 4), ("stiffness", 9)):
        difference = np.asarray(getattr(loaded, name)) - getattr(bare, name)
        expected = np.zeros((18, 18))
        expected[:6, :6] = scale * factor.T @ factor
        np.testing.assert_allclose(difference, expected)


def test_rotated_spine_produces_bending_coupling(rod: ShaftRodProperties) -> None:
    profile = replace(
        rod.profile,
        stations=tuple(
            replace(station, spine_angle_rad=np.pi / 4)
            for station in rod.profile.stations
        ),
    )
    result = assemble_shaft_linear_system(
        replace(rod, profile=profile), ShaftModalSettings(2)
    )
    force = np.zeros(12)
    force[-6] = 1
    displacement = np.linalg.solve(np.asarray(result.stiffness)[6:, 6:], force)
    assert displacement[-6] == pytest.approx((1 / 15 + 1 / 9) / 2)
    assert displacement[-5] == pytest.approx((1 / 15 - 1 / 9) / 2)


@pytest.mark.parametrize("values", [(0, 1), (-1, 1), (1, np.inf), (1,), (True, 2)])
def test_invalid_unmeasured_rod_properties_are_refused(
    rod: ShaftRodProperties, values: tuple[float, ...]
) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(rod, axial_stiffness_n=values)
    with pytest.raises((ValueError, TypeError)):
        replace(rod, polar_mass_per_length_kg_m=values)


def test_attachment_frame_mismatch_is_refused(rod: ShaftRodProperties) -> None:
    zero = np.zeros((6, 6))
    grip = PassiveGripImpedance("other", zero, zero, zero, "synthetic")
    with pytest.raises(ValueError, match="frame"):
        assemble_shaft_linear_system(
            rod, ShaftModalSettings(2), ShaftAttachments(grip=grip)
        )


def test_raw_station_interpolation_and_exposed_trim(rod: ShaftRodProperties) -> None:
    stations = tuple(
        replace(s, linear_density_kg_m=1 + s.position_m) for s in rod.profile.stations
    )
    profile = replace(
        rod.profile,
        cut_length_m=0.8,
        butt_trim_m=0.1,
        tip_trim_m=0.1,
        insertion_depth_m=0.1,
        stations=stations,
    )
    trimmed = replace(
        rod,
        profile=profile,
        axial_stiffness_n=(1000, 2000),
        polar_mass_per_length_kg_m=(0.01, 0.02),
    )
    result = assemble_shaft_linear_system(trimmed, ShaftModalSettings(2))
    assert result.positions_m[-1] == pytest.approx(0.7)
    translation = np.tile([1, 0, 0, 0, 0, 0], 3)
    assert translation @ result.mass @ translation == pytest.approx(1.45 * 0.7)
    axial = np.zeros(18)
    axial[2::6] = result.positions_m
    assert axial @ result.stiffness @ axial == pytest.approx(1450 * 0.7)
    spin = np.tile([0, 0, 0, 0, 0, 1], 3)
    assert spin @ result.mass @ spin == pytest.approx(0.0145 * 0.7)


def test_full_stationary_operator_work_closure(rod: ShaftRodProperties) -> None:
    """Independent ODE integration; production assembly supplies only M/C/K."""
    grip = PassiveGripImpedance(
        "shaft", np.eye(6), 0.5 * np.eye(6), 2 * np.eye(6), "synthetic"
    )
    result = assemble_shaft_linear_system(
        rod, ShaftModalSettings(2), ShaftAttachments(grip=grip)
    )
    mass, stiffness, damping = result.mass, result.stiffness, result.damping
    size = len(mass)
    initial = np.r_[np.linspace(-1e-3, 1e-3, size), np.zeros(size + 2)]

    def rate(time: float, state: np.ndarray) -> np.ndarray:
        q, velocity = state[:size], state[size : 2 * size]
        force = np.zeros(size)
        force[-6], force[-1] = 2 * np.sin(20 * time), 0.03 * np.cos(20 * time)
        acceleration = np.linalg.solve(mass, force - damping @ velocity - stiffness @ q)
        return np.r_[
            velocity, acceleration, velocity @ damping @ velocity, force @ velocity
        ]

    initial_energy = initial[:size] @ stiffness @ initial[:size] / 2
    solution = solve_ivp(
        rate,
        (0, 0.05),
        initial,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.001,
    )
    assert solution.success
    final = solution.y[:, -1]
    q, velocity = final[:size], final[size : 2 * size]
    energy = (q @ stiffness @ q + velocity @ mass @ velocity) / 2
    assert final[-2] > 0
    assert energy + final[-2] - final[-1] == pytest.approx(initial_energy, abs=1e-10)


def test_outputs_do_not_share_mutable_matrix_storage(rod: ShaftRodProperties) -> None:
    first = assemble_shaft_linear_system(rod, ShaftModalSettings(2))
    second = assemble_shaft_linear_system(rod, ShaftModalSettings(2))
    first.mass[:] = 0
    assert np.linalg.eigvalsh(second.mass)[0] > 0


def test_nonfinite_assembly_is_refused(rod: ShaftRodProperties) -> None:
    profile = replace(
        rod.profile,
        stations=tuple(replace(s, ei_about_x_n_m2=1e308) for s in rod.profile.stations),
    )
    with pytest.raises(ValueError, match="finite"):
        assemble_shaft_linear_system(
            replace(rod, profile=profile), ShaftModalSettings(2)
        )


def test_underflowed_torsional_mass_is_refused(rod: ShaftRodProperties) -> None:
    tiny = replace(rod, polar_mass_per_length_kg_m=(1e-323, 1e-323))
    with pytest.raises(ValueError, match="mass"):
        assemble_shaft_linear_system(tiny, ShaftModalSettings(16))


def test_unrepresentable_element_length_is_refused(rod: ShaftRodProperties) -> None:
    length = 1e-200
    stations = tuple(
        replace(s, position_m=s.position_m * length) for s in rod.profile.stations
    )
    profile = replace(
        rod.profile, raw_length_m=length, cut_length_m=length, stations=stations
    )
    with pytest.raises(ValueError, match="length"):
        assemble_shaft_linear_system(
            replace(rod, profile=profile), ShaftModalSettings(2)
        )

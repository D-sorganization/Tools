"""Work, dynamics and lost-mode controls for explicit real shaft reduction."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_affine_transient import affine_state_at
from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_galerkin import GalerkinReduction

from .test_shaft_affine_transient import _scales


def _plant() -> DampedPencil:
    return DampedPencil(
        [[2, 0.2, 0], [0.2, 1, 0.1], [0, 0.1, 3]],
        [[0, 0.7, -0.2], [-0.7, 0, 0.4], [0.2, -0.4, 0]],
        [[0.3, 0, 0], [0, 0.4, 0], [0, 0, 0.1]],
        [[7, -2, 0], [-1, 3, 0.5], [0.2, 0, 8]],
    )


def test_reduction_preserves_virtual_work_mass_gyro_and_dissipation() -> None:
    basis = np.array([[1, 0.2], [0.3, 1], [-0.4, 0.5]])
    model = GalerkinReduction(_plant(), basis, _scales())
    motion, velocity, acceleration = np.array([[0.2, -0.7], [0.4, 0.8], [-0.1, 0.3]])
    force = np.array([0.5, -0.2, 0.8])
    full_motion = model.lift_motion(motion)
    full_velocity = model.lift_motion(velocity)
    full_acceleration = model.lift_motion(acceleration)
    mass, gyro, damping, stiffness = _plant().arrays()
    reduced_mass, reduced_gyro, reduced_damping, reduced_stiffness = (
        model.pencil.arrays()
    )
    assert np.dot(velocity, model.reduce_force(force)) == pytest.approx(
        np.dot(full_velocity, force)
    )
    for original, reduced in ((mass, reduced_mass), (damping, reduced_damping)):
        assert velocity @ reduced @ velocity == pytest.approx(
            full_velocity @ original @ full_velocity
        )
    assert abs(velocity @ reduced_gyro @ velocity) < 1e-15
    assert not np.allclose(reduced_stiffness, reduced_stiffness.T)
    residual = (
        mass @ full_acceleration
        + (gyro + damping) @ full_velocity
        + stiffness @ full_motion
        - force
    )
    reduced_residual = (
        reduced_mass @ acceleration
        + (reduced_gyro + reduced_damping) @ velocity
        + reduced_stiffness @ motion
        - model.reduce_force(force)
    )
    np.testing.assert_allclose(reduced_residual, basis.T @ residual, atol=2e-15)
    assert model.stability_status == "unqualified"


def test_complete_nonorthogonal_basis_preserves_forced_transient() -> None:
    basis = np.array([[1, 0.2, 0], [0.3, 1, 0.1], [-0.4, 0.5, 1]])
    scales = _scales(0.3)
    model = GalerkinReduction(_plant(), basis, scales)
    initial = np.array([0.2, -0.7, 0.4, 0.8, -0.1, 0.3])
    residual = np.array([0.1, -0.2, 0.3])
    reduced_initial = np.linalg.solve(basis, initial.reshape(2, 3).T).T.ravel()
    for time in (0, 0.07, 1.1):
        full = affine_state_at(_plant(), residual, scales, initial, time)
        reduced = affine_state_at(
            model.pencil, model.reduce_force(residual), scales, reduced_initial, time
        )
        lifted = np.r_[model.lift_motion(reduced[:3]), model.lift_motion(reduced[3:])]
        np.testing.assert_allclose(lifted, full, rtol=2e-12, atol=2e-13)


def test_discarded_resonance_is_not_a_bandwidth_certificate() -> None:
    plant = DampedPencil(
        np.eye(2), np.zeros((2, 2)), 0.02 * np.eye(2), np.diag([1, 100])
    )
    model = GalerkinReduction(plant, [[1], [0]], _scales())
    # Equal collocated loading/observation excites both modes. Keeping the low
    # mode preserves its dynamics but entirely loses the resonance at 10 rad/s.
    load = np.ones(2)
    mass, gyro, damping, stiffness = model.pencil.arrays()
    reduced_load = model.reduce_force(load)
    errors = []
    for frequency in (0.1, 10):
        exact = np.sum(1 / (np.array([1, 100]) - frequency**2 + 0.02j * frequency))
        dynamic = stiffness - frequency**2 * mass + 1j * frequency * (gyro + damping)
        predicted = reduced_load @ np.linalg.solve(dynamic, reduced_load)
        errors.append(abs(predicted - exact) / abs(exact))
    assert errors[0] < 0.01
    assert errors[1] > 0.99
    assert model.stability_status == "unqualified"


@pytest.mark.parametrize(
    "basis",
    [
        [],
        [1, 2, 3],
        np.ones((2, 1)),
        np.ones((3, 4)),
        np.zeros((3, 1)),
        np.ones((3, 2)),
        np.diag([1, 1, 1e-8]),
        [[np.nan], [0], [1]],
    ],
)
def test_empty_malformed_rank_deficient_or_unresolved_basis_is_refused(
    basis: object,
) -> None:
    with pytest.raises(ValueError):
        GalerkinReduction(_plant(), basis, _scales())


@pytest.mark.parametrize(
    "basis", [[[True], [0], [1]], [["1"], ["0"], ["1"]], np.eye(3, dtype=complex)]
)
def test_basis_requires_strictly_real_numeric_values(basis: object) -> None:
    with pytest.raises(TypeError):
        GalerkinReduction(_plant(), basis, _scales())


@pytest.mark.parametrize(
    "field,values",
    [
        ("mass", np.diag([1, -1, 1])),
        ("damping", np.diag([1, -1, 1])),
        ("gyroscopic", np.diag([0, 1, 0])),
    ],
)
def test_invalid_discarded_direction_cannot_be_hidden(
    field: str, values: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        GalerkinReduction(
            replace(_plant(), **{field: values}), [[1], [0], [0]], _scales()
        )


def test_basis_and_results_are_owned_and_malformed_ports_are_refused() -> None:
    basis = np.eye(3)
    model = GalerkinReduction(_plant(), basis, _scales())
    basis[:] = 0
    np.testing.assert_array_equal(model.basis_array(), np.eye(3))
    copied = model.basis_array()
    copied[:] = 0
    np.testing.assert_array_equal(model.basis_array(), np.eye(3))
    assert isinstance(model.lift_motion([1, 2, 3]), tuple)
    assert isinstance(model.reduce_force([1, 2, 3]), tuple)
    with pytest.raises(FrozenInstanceError):
        model.basis = ()
    for method in (model.lift_motion, model.reduce_force):
        with pytest.raises(ValueError):
            method([1, 2])
        with pytest.raises(TypeError):
            method([True, 1, 2])
        with pytest.raises(ValueError):
            method([1, np.inf, 2])


@pytest.mark.parametrize("pencil,scales", [(None, _scales()), (_plant(), None)])
def test_wrong_plant_and_scale_types_are_refused(
    pencil: object, scales: object
) -> None:
    with pytest.raises(TypeError):
        GalerkinReduction(pencil, np.eye(3), scales)


def test_omitting_an_unstable_mode_does_not_qualify_original_stability() -> None:
    plant = DampedPencil(
        np.eye(2), np.zeros((2, 2)), np.zeros((2, 2)), np.diag([1, -4])
    )
    model = GalerkinReduction(plant, [[1], [0]], _scales())
    full = affine_state_at(plant, [0, 0], _scales(), [0, 1, 0, 0], 1)
    reduced = affine_state_at(model.pencil, [0], _scales(), [0, 0], 1)
    assert full[1] == pytest.approx(np.cosh(2))
    assert reduced == (0, 0)
    assert model.stability_status == "unqualified"


@pytest.mark.parametrize("frequency", [0.0, 0.7, 3.0])
def test_complete_basis_preserves_complex_force_to_observation_transfer(
    frequency: float,
) -> None:
    basis = np.array([[1, 0.2, 0], [0.3, 1, 0.1], [-0.4, 0.5, 1]])
    model = GalerkinReduction(_plant(), basis, _scales())
    load = np.array([0.1, -0.2, 0.3])
    observed = np.array([0.5, 0.2, -0.1])
    transfers = []
    for pencil, input_vector, output_vector in (
        (_plant(), load, observed),
        (model.pencil, model.reduce_force(load), basis.T @ observed),
    ):
        mass, gyro, damping, stiffness = pencil.arrays()
        dynamic = stiffness - frequency**2 * mass + 1j * frequency * (gyro + damping)
        transfers.append(output_vector @ np.linalg.solve(dynamic, input_vector))
    assert transfers[1] == pytest.approx(transfers[0], rel=1e-12, abs=1e-14)


def test_projection_and_port_overflow_are_refused_but_tiny_loads_survive() -> None:
    with pytest.raises(ValueError):
        GalerkinReduction(_plant(), 1e200 * np.eye(3), _scales())
    model = GalerkinReduction(_plant(), 100 * np.eye(3), _scales())
    for method in (model.reduce_force, model.lift_motion):
        with pytest.raises(ValueError):
            method([1e308, 0, 0])
    identity = GalerkinReduction(_plant(), np.eye(3), _scales())
    assert identity.reduce_force([1e-300, 0, 0]) == (1e-300, 0, 0)

"""Independent shear/rotary beam boundary-value oracle for the spatial shaft."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigh, expm

from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model
from shared.python.golf_club._shaft_gripped_response import gripped_tip_compliance

from .test_shaft_equilibrium import _controls
from .test_shaft_galerkin_rod import _transfer
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model
from .test_shaft_harmonic_response import _request

# Existing unit-length synthetic rod: EI=10 N m^2, shear rigidity=500 N,
# line mass=.2 kg/m, transverse rotary inertia density=1e-4 kg m.
_BENDING_RIGIDITY = 10.0
_LINE_MASS = 0.2
_ROOT_STIFFNESS = np.diag([300.0, 50.0])
_ROOT_DAMPING = np.diag([1.5, 0.03])
_ROOT_INERTANCE = np.diag([0.03, 0.0008])
_TIP_MASS = np.diag([0.1, 0.005])


def _continuum_response(
    frequency: float, shear_compliance: float = 1 / 500, rotary_density: float = 1e-4
) -> np.ndarray:
    # Continuum balance for s=[w, theta, Q, M], with exp(+i*omega*t):
    # w'=theta+Q/S, theta'=M/EI, Q'=-mu*omega^2*w, M'=-Q-j*omega^2*theta.
    spatial = np.array(
        [
            [0, 1, shear_compliance, 0],
            [0, 0, 0, 1 / _BENDING_RIGIDITY],
            [-_LINE_MASS * frequency**2, 0, 0, 0],
            [0, -rotary_density * frequency**2, -1, 0],
        ]
    )
    boundary = (
        _ROOT_STIFFNESS
        + 1j * frequency * _ROOT_DAMPING
        - frequency**2 * _ROOT_INERTANCE
    )
    tip = expm(spatial) @ np.vstack([np.eye(2), boundary])
    dynamic = tip[2:] - frequency**2 * _TIP_MASS @ tip[:2]
    return tip[:2] @ np.linalg.solve(dynamic, np.eye(2))


def _bending_model(count: int) -> tuple:
    chain, poses = _model(count)
    attachment = chain.grips[0]
    damping, inertance = (
        np.array(attachment.grip.damping_factor),
        np.array(attachment.grip.inertance_factor),
    )
    bending = np.ix_([0, 4], [0, 4])
    damping[bending] = np.sqrt(_ROOT_DAMPING)
    inertance[bending] = np.sqrt(_ROOT_INERTANCE)
    grip = replace(attachment.grip, damping_factor=damping, inertance_factor=inertance)
    return replace(chain, grips=(replace(attachment, grip=grip),)), poses


def test_continuum_oracle_recovers_static_force_couple_and_shear_deflection() -> None:
    expected = np.array(
        [
            [1 / 300 + 1 / 50 + 1 / 30 + 1 / 500, 1 / 50 + 1 / 20],
            [1 / 50 + 1 / 20, 1 / 50 + 1 / 10],
        ]
    )
    np.testing.assert_allclose(_continuum_response(0), expected, rtol=1e-14, atol=1e-14)
    shear_free = _continuum_response(0, shear_compliance=0)
    np.testing.assert_allclose(
        expected - shear_free, [[1 / 500, 0], [0, 0]], atol=1e-14
    )


@pytest.mark.parametrize("frequency", [0, 4, 8, 20, 40])
def test_spatial_shaft_mesh_converges_to_finite_grip_bending_continuum(
    frequency: float, record_property: Callable[[str, object], None]
) -> None:
    exact = _continuum_response(frequency)
    np.testing.assert_allclose(exact, exact.T, rtol=1e-12, atol=1e-14)
    assert np.min(np.abs(exact)) > 1e-8
    errors = []
    for count in (4, 8, 16):
        chain, poses = _bending_model(count)
        result = gripped_tip_compliance(chain, poses, _controls(), _request(frequency))
        predicted = result.displacement_compliance[np.ix_([0, 4], [0, 4])]
        errors.append(float(np.max(np.abs(predicted - exact) / np.abs(exact))))
    record_property("angular_frequency_rad_s", frequency)
    record_property("componentwise_relative_errors_4_8_16", errors)
    assert 3 < errors[1] / errors[2] < 5
    assert errors[-1] < 0.02
    mobility = 1j * frequency * exact
    power = (mobility + mobility.conj().T) / 2
    assert np.min(np.linalg.eigvalsh(power)) >= -1e-12


@pytest.mark.parametrize("count", [8, 16])
def test_joint_bending_mesh_and_modal_errors_are_kept_separate(
    count: int, record_property: Callable[[str, object], None]
) -> None:
    chain, poses = _bending_model(count)
    scales = _scales()
    full = constant_gripped_model(chain, poses, _controls(), scales)
    mass, _, _, stiffness = full.pencil.arrays()
    axes = (6 * np.arange(count + 1)[:, None] + [0, 4]).ravel()
    _, modes = eigh(stiffness[np.ix_(axes, axes)], mass[np.ix_(axes, axes)])
    loads = np.zeros((len(mass), 2))
    loads[axes[-2], 0], loads[axes[-1], 1] = scales.length_m, 1
    frequencies = np.array([0, 4, 8, 20, 40.0])
    continuum = np.array([_continuum_response(frequency) for frequency in frequencies])
    reference = _transfer(full.pencil, loads, frequencies)
    assert np.min(np.abs(reference)) > 1e-8
    mesh_error = float(np.max(np.abs(reference - continuum) / np.abs(continuum)))
    record_property("elements", count)
    record_property("maximum_componentwise_mesh_error", mesh_error)
    assert mesh_error < 0.08 / (count / 8) ** 2
    for retained in (2, 4, 8, len(axes)):
        basis = np.zeros((len(mass), retained))
        basis[axes] = modes[:, :retained]
        reduced = GalerkinReduction(full.pencil, basis, scales)
        predicted = _transfer(reduced.pencil, basis.T @ loads, frequencies)
        error = float(np.max(np.abs(predicted - reference) / np.abs(reference)))
        total = float(np.max(np.abs(predicted - continuum) / np.abs(continuum)))
        record_property(f"modes_{retained}_maximum_modal_error", error)
        record_property(f"modes_{retained}_maximum_total_error", total)
        if retained == len(axes):
            np.testing.assert_allclose(predicted, reference, rtol=1e-9, atol=1e-13)
        if retained == 8:
            assert error < 0.03

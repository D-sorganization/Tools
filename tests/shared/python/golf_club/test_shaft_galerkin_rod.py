"""Sampled modal convergence of the assembled finite-grip axial rod control."""

from collections.abc import Callable

import numpy as np
import pytest
from scipy.linalg import eigh

from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model
from shared.python.golf_club._shaft_gripped_response import gripped_tip_compliance

from .test_shaft_equilibrium import _controls
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model
from .test_shaft_harmonic_response import _request


def _transfer(
    pencil: DampedPencil, load: np.ndarray, frequencies: np.ndarray
) -> np.ndarray:
    mass, gyro, damping, stiffness = pencil.arrays()
    dynamic = (
        stiffness
        - frequencies[:, None, None] ** 2 * mass
        + 1j * frequencies[:, None, None] * (gyro + damping)
    )
    inputs = np.broadcast_to(load, (len(frequencies), len(load)))
    motions = np.linalg.solve(dynamic, inputs[..., None])[..., 0]
    return motions @ load


def _record_errors(
    exact: np.ndarray, predicted: np.ndarray, record: Callable[[str, object], None]
) -> tuple[np.ndarray, np.ndarray]:
    assert np.min(np.abs(exact)) > 1e-8  # Phase resolves at these fixture samples.
    assert np.min(np.abs(predicted)) > 1e-8
    errors = np.abs(predicted - exact) / np.abs(exact)
    phase = np.abs(np.angle(predicted / exact))
    magnitude = np.abs(np.abs(predicted) / np.abs(exact) - 1)
    record("maximum_sampled_complex_relative_error", float(np.max(errors)))
    record("maximum_sampled_magnitude_relative_error", float(np.max(magnitude)))
    record("maximum_sampled_phase_error_rad", float(np.max(phase)))
    return errors, phase


@pytest.mark.parametrize("retained", [1, 2, 4, 9])
def test_modal_refinement_against_full_finite_grip_rod(
    retained: int, record_property: Callable[[str, object], None]
) -> None:
    chain, poses = _model(8)
    scales = _scales()
    full = constant_gripped_model(chain, poses, _controls(), scales)
    mass, _, _, stiffness = full.pencil.arrays()
    axial = np.arange(2, len(mass), 6)
    # This straight stationary principal-axis fixture decouples axial motion.
    # Its conservative modes define a basis; projected damping stays coupled.
    squared, modes = eigh(stiffness[np.ix_(axial, axial)], mass[np.ix_(axial, axial)])
    assert np.all(squared > 0)
    basis = np.zeros((len(mass), retained))
    basis[axial] = modes[:, :retained]
    reduced = GalerkinReduction(full.pencil, basis, scales)
    load = np.zeros(len(mass))
    load[axial[-1]] = scales.length_m  # SI unit tip force, conjugate scaled load.
    frequencies = np.array([0, 10, 25, 50, 75, 100, 140, 200.0])
    exact = _transfer(full.pencil, load, frequencies)
    predicted = _transfer(
        reduced.pencil, np.array(reduced.reduce_force(load)), frequencies
    )
    record_property("retained_axial_modes", retained)
    errors, phase = _record_errors(exact, predicted, record_property)
    if retained == 9:
        np.testing.assert_allclose(predicted, exact, rtol=1e-9, atol=1e-13)
    if retained == 4:
        assert np.max(errors) < 0.05
        assert np.max(phase) < 0.05
    if retained == 1:
        assert np.max(errors) > 0.1
    physical = gripped_tip_compliance(chain, poses, _controls(), _request(25))
    assert exact[2] == pytest.approx(physical.displacement_compliance[2, 2], rel=1e-10)
    assert reduced.stability_status == "unqualified"

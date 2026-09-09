"""Complete-band reduction errors for an assembled finite-grip shaft control."""

from collections.abc import Callable

import numpy as np
import pytest
from scipy.linalg import eigh

from shared.python.golf_club._shaft_frequency_band import FrequencyBandControls
from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model
from shared.python.golf_club._shaft_reduction_band import (
    ReductionBandControls,
    assess_reduction_band,
)
from shared.python.golf_club._shaft_transfer_ports import DisplacementPorts

from .test_shaft_equilibrium import _controls
from .test_shaft_galerkin_rod import _transfer
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model


def _axial_reduction(retained: int) -> tuple[GalerkinReduction, DisplacementPorts]:
    """Share the unchanged axial modal fixture with magnitude/phase controls."""
    chain, poses = _model(4)
    scales = _scales()
    full = constant_gripped_model(chain, poses, _controls(), scales)
    mass, _, _, stiffness = full.pencil.arrays()
    axial = np.arange(2, len(mass), 6)
    _, modes = eigh(stiffness[np.ix_(axial, axial)], mass[np.ix_(axial, axial)])
    basis = np.zeros((len(mass), retained))
    basis[axial] = modes[:, :retained]
    reduced = GalerkinReduction(full.pencil, basis, scales)
    load = np.zeros((len(mass), 1))
    load[axial[-1]] = scales.length_m  # S.T unit SI axial tip force.
    ports = DisplacementPorts(load, load.T, [1], [0.001])
    return reduced, ports


@pytest.mark.parametrize("retained", [2, 4, 5])
def test_assembled_rod_complete_band_meets_declared_normalized_error(
    retained: int,
    record_property: Callable[[str, object], None],
) -> None:
    reduced, ports = _axial_reduction(retained)
    load, _ = ports.normalized_arrays()
    basis = reduced.basis_array()
    # Full transverse modes near 10.19/10.23 rad/s are undamped in this axial
    # fixture. The full-inverse method legitimately refuses a band across them.
    controls = ReductionBandControls(
        FrequencyBandControls((0, 8), 0, 0.6, 1023), 0, 0.05
    )
    result = assess_reduction_band(reduced, ports, controls)
    # 1 N reference input and 1 mm reference output: 0.05 means 50 um/N.
    assert result.maximum_absolute_error_bound <= 0.05
    for cell in result.cells:
        frequencies = np.array([cell.lower_rad_s, cell.upper_rad_s])
        exact = _transfer(reduced.full_pencil, load, frequencies)
        predicted = _transfer(reduced.pencil, basis.T @ load, frequencies)
        assert (
            np.max(np.abs(exact - predicted)) / 0.001
            <= cell.assessment.absolute_error_bound + 1e-10
        )
    record_property("retained_axial_modes", retained)
    record_property("paired_cell_evaluations", result.evaluation_count)
    record_property(
        "complete_band_absolute_error_bound", result.maximum_absolute_error_bound
    )


def test_assembled_rod_unobserved_transverse_poles_prevent_full_inverse_cover() -> None:
    chain, poses = _model(4)
    scales = _scales()
    full = constant_gripped_model(chain, poses, _controls(), scales)
    mass, _, damping, stiffness = full.pencil.arrays()
    axial = np.arange(2, len(mass), 6)
    transverse = np.setdiff1d(np.arange(len(mass)), axial)
    np.testing.assert_array_equal(damping[np.ix_(transverse, transverse)], 0)
    for matrix in (mass, stiffness):
        np.testing.assert_allclose(matrix[np.ix_(transverse, axial)], 0, atol=1e-12)
    frequencies = np.sqrt(eigh(stiffness, mass, eigvals_only=True))
    assert 10.18 < frequencies[0] < frequencies[1] < 10.24
    load = np.zeros((len(mass), 1))
    load[-4] = scales.length_m
    ports = DisplacementPorts(load, load.T, [1], [0.001])
    # Even a complete basis cannot make the full inverse exist at these poles.
    reduction = GalerkinReduction(full.pencil, np.eye(len(mass)), scales)
    controls = ReductionBandControls(
        FrequencyBandControls((0, 20), 0, 0.6, 1023), 0, 0.05
    )
    with pytest.raises(ValueError, match="budget|representable|numerical"):
        assess_reduction_band(reduction, ports, controls)

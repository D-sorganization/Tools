"""Complete magnitude/phase bands of assembled axial, bending and torsion ports."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigh

from shared.python.golf_club._shaft_frequency_band import FrequencyBandControls
from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model
from shared.python.golf_club._shaft_reduction_band import ReductionBandControls
from shared.python.golf_club._shaft_siso_band import (
    SisoReductionBandAssessment,
    SisoReductionBandControls,
    assess_siso_reduction_band,
)
from shared.python.golf_club._shaft_transfer_ports import DisplacementPorts

from .test_shaft_equilibrium import _controls
from .test_shaft_galerkin_rod import _transfer
from .test_shaft_gripped_operating import _scales
from .test_shaft_gripped_response import _model
from .test_shaft_reduction_rod import _axial_reduction


def _check_and_record(
    model: GalerkinReduction,
    ports: DisplacementPorts,
    result: SisoReductionBandAssessment,
    record: Callable[[str, object], None],
) -> None:
    load, _ = ports.normalized_arrays()
    reduced_load = model.basis_array().T @ load
    for cell in result.cells:
        frequencies = np.linspace(cell.lower_rad_s, cell.upper_rad_s, 3)
        full = _transfer(model.full_pencil, load, frequencies) / 0.001
        reduced = _transfer(model.pencil, reduced_load, frequencies) / 0.001
        checked = cell.assessment
        assert np.min(np.abs(full)) >= checked.full_response_lower_bound
        assert (
            np.max(np.abs(reduced / full - 1))
            <= checked.relative_complex_error_bound + 1e-10
        )
        assert (
            np.max(np.abs(np.angle(reduced / full)))
            <= checked.phase_error_bound_rad + 1e-10
        )
    record("paired_cell_evaluations", result.evaluation_count)
    record(
        "minimum_full_response_lower_bound", result.minimum_full_response_lower_bound
    )
    record("relative_complex_error_bound", result.maximum_relative_complex_error_bound)
    record("phase_error_bound_rad", result.maximum_phase_error_bound_rad)


@pytest.mark.parametrize("retained", [2, 4, 5])
def test_original_axial_fixture_meets_one_percent_and_phase_targets(
    retained: int,
    record_property: Callable[[str, object], None],
) -> None:
    model, ports = _axial_reduction(retained)
    absolute = ReductionBandControls(
        FrequencyBandControls((0, 8), 0, 0.6, 1023), 0, 0.05
    )
    result = assess_siso_reduction_band(
        model, ports, SisoReductionBandControls(absolute, 0.01, 0.01)
    )
    assert result.maximum_relative_magnitude_error_bound <= 0.01
    assert result.maximum_phase_error_bound_rad <= 0.01
    _check_and_record(model, ports, result, record_property)


def _damped_reduction(
    retained: int, axis: int
) -> tuple[GalerkinReduction, DisplacementPorts]:
    """Separate synthetic six-axis damping prescription; never a fitted hand law."""
    chain, poses = _model(4)
    attachment = chain.grips[0]
    # Translation damping: 2 N s/m; rotation damping: 0.02 N m s/rad.
    grip = replace(
        attachment.grip, damping_factor=np.diag(np.sqrt([2, 2, 2, 0.02, 0.02, 0.02]))
    )
    chain = replace(chain, grips=(replace(attachment, grip=grip),))
    scales = _scales()
    full = constant_gripped_model(chain, poses, _controls(), scales)
    mass, _, _, stiffness = full.pencil.arrays()
    _, modes = eigh(stiffness, mass)
    reduction = GalerkinReduction(full.pencil, modes[:, :retained], scales)
    load = np.zeros((len(mass), 1))
    load[-6 + axis] = scales.length_m if axis < 3 else 1
    return reduction, DisplacementPorts(load, load.T, [1], [0.001])


def _with_static_interfaces(
    model: GalerkinReduction,
    ports: DisplacementPorts,
    axis: int,
) -> GalerkinReduction:
    """Add tip and grip static-load residuals to this M-orthonormal modal basis."""
    mass, _, _, stiffness = model.full_pencil.arrays()
    basis = model.basis_array()
    np.testing.assert_allclose(
        basis.T @ mass @ basis, np.eye(basis.shape[1]), atol=1e-12
    )
    load, _ = ports.normalized_arrays()
    root_axes = [0, 4] if axis == 0 else [5]
    root = np.eye(len(mass))[:, root_axes]
    root *= np.array([model.scales.length_m if index < 3 else 1 for index in root_axes])
    static = np.linalg.solve(stiffness, np.column_stack((load, root)))
    residual = static - basis @ (basis.T @ mass @ static)
    factor = np.linalg.cholesky(mass)
    left, values, _ = np.linalg.svd(factor.T @ residual, full_matrices=False)
    rank = int(np.count_nonzero(values > values[0] * model.scales.residual_tolerance))
    assert rank == len(root_axes) + 1
    # Residual cancellation can leave small cross-terms with retained modes.
    # Explicitly orthonormalize the chosen span in the M metric before use.
    orthogonal, _ = np.linalg.qr(np.column_stack((factor.T @ basis, left[:, :rank])))
    combined = np.linalg.solve(factor.T, orthogonal)
    # The selected tip/grip static responses are preserved, not just tip gain.
    np.testing.assert_allclose(
        combined @ (combined.T @ mass @ static), static, atol=1e-12
    )
    return GalerkinReduction(model.full_pencil, combined, model.scales)


@pytest.mark.parametrize(
    "axis,kind",
    [
        (0, "interfaces"),
        (0, "full"),
        (2, "modal"),
        (2, "full"),
        (5, "interfaces"),
        (5, "full"),
    ],
)
def test_explicit_six_axis_damped_fixture_across_resonant_band(
    axis: int,
    kind: str,
    record_property: Callable[[str, object], None],
) -> None:
    model, ports = _damped_reduction(30 if kind == "full" else 16, axis)
    if kind == "interfaces":
        model = _with_static_interfaces(model, ports, axis)
    absolute = ReductionBandControls(FrequencyBandControls((0, 40), 0, 0.6, 4095), 0, 1)
    result = assess_siso_reduction_band(
        model, ports, SisoReductionBandControls(absolute, 0.02, 0.02)
    )
    assert result.maximum_relative_magnitude_error_bound <= 0.02
    assert result.maximum_phase_error_bound_rad <= 0.02
    _check_and_record(model, ports, result, record_property)
    record_property("basis_kind", kind)
    record_property("retained_coordinates", len(model.pencil.mass))


@pytest.mark.parametrize("axis,frequency", [(0, 10.2), (5, 23.4)])
def test_sixteen_modes_violate_absolute_target_despite_small_relative_error(
    axis: int,
    frequency: float,
) -> None:
    model, ports = _damped_reduction(16, axis)
    load, _ = ports.normalized_arrays()
    full = _transfer(model.full_pencil, load, np.array([frequency])) / 0.001
    reduced = (
        _transfer(model.pencil, model.basis_array().T @ load, np.array([frequency]))
        / 0.001
    )
    assert np.max(np.abs(reduced / full - 1)) < 0.02
    assert np.max(np.abs(reduced - full)) > 1
    absolute = ReductionBandControls(FrequencyBandControls((0, 40), 0, 0.6, 4095), 0, 1)
    with pytest.raises(ValueError, match="budget|representable"):
        assess_siso_reduction_band(
            model, ports, SisoReductionBandControls(absolute, 0.02, 0.02)
        )

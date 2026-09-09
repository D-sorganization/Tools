"""Constant-input prescription and physical rod oracles for Tools #5072."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from shared.python.golf_club import _shaft_gripped_operating as operating
from shared.python.golf_club._shaft_affine_response import (
    ForcedResponseControls,
    assess_affine_response,
)
from shared.python.golf_club._shaft_autonomous_decay import DecayControls
from shared.python.golf_club._shaft_chain import IndexedPointLoad
from shared.python.golf_club._shaft_gripped_equilibrium import solve_gripped_chain
from shared.python.golf_club._shaft_point_load import SpatialPointLoad
from shared.python.golf_club._shaft_spectrum import SpectrumScales

from .test_shaft_equilibrium import _controls
from .test_shaft_gripped_response import _model


def _scales() -> SpectrumScales:
    return SpectrumScales(0.2, 0.02, 1e-12, 1e-9)


def _loaded_model(force: float = 5e-9) -> tuple:
    chain, poses = _model()
    load = IndexedPointLoad(1, SpatialPointLoad([0, 0, force], [0, 0, 0], [0, 0, 0]))
    elastic = replace(chain.shaft.elastic, loads=(load,))
    return replace(chain, shaft=replace(chain.shaft, elastic=elastic)), poses


def test_two_node_rod_retains_physical_mass_stiffness_damping_and_residual() -> None:
    chain, poses = _loaded_model()
    result = operating.constant_gripped_model(chain, poses, _controls(), _scales())
    mass, gyro, damping, stiffness = result.pencil.arrays()
    axes = np.ix_([2, 8], [2, 8])
    physical_mass = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0.03, 0.1])
    physical_stiffness = 1000 * np.array([[1, -1], [-1, 1]]) + np.diag([400, 0])
    np.testing.assert_allclose(mass[axes], 0.2**2 * physical_mass)
    np.testing.assert_allclose(stiffness[axes], 0.2**2 * physical_stiffness)
    np.testing.assert_allclose(damping[axes], np.diag([2 * 0.2**2, 0]))
    np.testing.assert_array_equal(gyro, np.zeros((12, 12)))
    expected_residual = np.zeros(12)
    expected_residual[8] = -0.2 * 5e-9
    np.testing.assert_allclose(result.scaled_residual, expected_residual, atol=0)
    assert result.input_scope == "constant_observer_inputs_and_anchor_poses"
    assert result.stability_status == "unqualified"


def test_adapter_retains_forcing_when_nominal_balance_is_only_within_tolerance() -> (
    None
):
    chain, poses = _loaded_model()
    model = operating.constant_gripped_model(chain, poses, _controls(), _scales())
    controls = ForcedResponseControls(DecayControls(1e-10, 1e-10, 0), 0.0)
    response = model.assess_response(controls)
    expected = assess_affine_response(
        model.pencil, model.scaled_residual, model.scales, controls
    )
    assert response == expected
    assert any(value != 0 for value in response.input_per_s)


def test_model_owns_reference_poses_and_records_frame_and_grip_sources() -> None:
    chain, poses = _model()
    original = poses.copy()
    model = operating.constant_gripped_model(chain, poses, _controls(), _scales())
    poses[:] = 0
    np.testing.assert_array_equal(model.reference_poses, original)
    assert model.frame == chain.shaft.frame
    assert model.grip_source_ids == ("synthetic",)
    with pytest.raises(FrozenInstanceError):
        model.frame = None


@pytest.mark.parametrize("spin", [0.0, 2.0])
def test_prescribed_constant_spin_uses_the_loaded_supported_configuration(
    spin: float,
) -> None:
    chain, seed = _model(spin=spin)
    equilibrium = solve_gripped_chain(chain, seed, _controls())
    model = operating.constant_gripped_model(
        chain, equilibrium.poses, _controls(), _scales()
    )
    assert model.frame.angular_velocity_rad_s == (spin, 0.0, 0.0)
    assert model.frame.angular_acceleration_rad_s2 == (0.0, 0.0, 0.0)
    if spin:
        assert np.linalg.norm(model.pencil.gyroscopic) > 0


@pytest.mark.parametrize("acceleration", [1.0, 1e-300])
def test_nonzero_angular_acceleration_cannot_be_prescribed_as_constant_spin(
    acceleration: float,
) -> None:
    chain, poses = _model()
    frame = replace(chain.shaft.frame, angular_acceleration_rad_s2=(acceleration, 0, 0))
    chain = replace(chain, shaft=replace(chain.shaft, frame=frame))
    with pytest.raises(ValueError, match="angular acceleration"):
        operating.constant_gripped_model(chain, poses, _controls(), _scales())


def test_operating_adapter_does_not_relax_the_existing_balance_contract() -> None:
    chain, poses = _loaded_model(force=1.0)
    with pytest.raises(ValueError, match="balance"):
        operating.constant_gripped_model(chain, poses, _controls(), _scales())


@pytest.mark.parametrize("bad", [None, True, "snapshot"])
def test_operating_adapter_requires_a_gripped_rotating_chain(bad: object) -> None:
    _, poses = _model()
    with pytest.raises(TypeError):
        operating.constant_gripped_model(bad, poses, _controls(), _scales())


def test_constant_observer_acceleration_is_an_explicit_prescription() -> None:
    chain, poses = _model()
    frame = replace(chain.shaft.frame, origin_acceleration_m_s2=(0.02, -0.01, 0.03))
    chain = replace(chain, shaft=replace(chain.shaft, frame=frame))
    equilibrium = solve_gripped_chain(chain, poses, _controls())
    model = operating.constant_gripped_model(
        chain, equilibrium.poses, _controls(), _scales()
    )
    assert model.frame.origin_acceleration_m_s2 == (0.02, -0.01, 0.03)
    assert not np.array_equal(model.reference_poses, poses)


@pytest.mark.parametrize("sources", ["synthetic", b"synthetic", {"synthetic": 1}])
def test_source_identity_sequence_cannot_be_a_string_or_mapping(
    sources: object,
) -> None:
    chain, poses = _model()
    model = operating.constant_gripped_model(chain, poses, _controls(), _scales())
    with pytest.raises(TypeError):
        replace(model, grip_source_ids=sources)


def test_length_scaling_cannot_erase_a_nonzero_residual() -> None:
    chain, poses = _loaded_model(force=1e-300)
    with pytest.raises(ValueError, match="scaling"):
        operating.constant_gripped_model(
            chain, poses, _controls(), replace(_scales(), length_m=1e-100)
        )

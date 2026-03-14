import numpy as np
import pytest

from double_pendulum_golf.golfer_dynamics import (
    _mass_point_positions,
    potential_energy_from_q,
    analytical_fk_jacobians,
    analytical_mass_matrix,
    analytical_coriolis,
    analytical_gravity_vector,
    kinetic_energy,
)
from double_pendulum_golf.physics_golfer import GolferParams, N_DOF


@pytest.fixture
def params():
    return GolferParams(
        m_hub=0.01,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        L_hub=0.2,
        L_r_upper=0.3,
        L_r_fore=0.3,
        L_l_upper=0.3,
        L_l_fore=0.3,
        L_club=1.0,
        d_rs=0.1,
        d_ls=0.1,
        grip_right=0.2,
        grip_left=0.3,
    )


@pytest.fixture
def valid_q():
    return np.zeros(N_DOF)


@pytest.fixture
def valid_qdot():
    return np.zeros(N_DOF)


def test_mass_point_positions_dbc(params, valid_q):
    """Test standard shape output and DbC checks for mass points."""
    points = _mass_point_positions(valid_q, params)
    assert len(points) == 7

    with pytest.raises((AssertionError, TypeError)):
        _mass_point_positions("wrong", params)
    with pytest.raises(ValueError):
        _mass_point_positions(np.zeros(2), params)
    with pytest.raises((AssertionError, TypeError)):
        _mass_point_positions(valid_q, "wrong")


def test_potential_energy_from_q_dbc(params, valid_q):
    """Test standard scalar output and DbC checks."""
    pe = potential_energy_from_q(valid_q, params)
    assert isinstance(pe, float)

    with pytest.raises((AssertionError, TypeError)):
        potential_energy_from_q(list(valid_q), params)
    with pytest.raises(ValueError):
        potential_energy_from_q(np.zeros(2), params)
    with pytest.raises((AssertionError, TypeError)):
        potential_energy_from_q(valid_q, None)


def test_jacobians_dbc(params, valid_q):
    """Test standard dict output and DbC checks."""
    J = analytical_fk_jacobians(valid_q, params)
    assert isinstance(J, dict)
    assert "lh" in J
    assert J["lh"].shape == (2, N_DOF)

    with pytest.raises((AssertionError, TypeError)):
        analytical_fk_jacobians(None, params)
    with pytest.raises(ValueError):
        analytical_fk_jacobians(np.zeros(2), params)


def test_mass_matrix_dbc(params, valid_q):
    """Test standard matrix output and DbC checks."""
    M = analytical_mass_matrix(valid_q, params)
    assert M.shape == (N_DOF, N_DOF)

    # None input: native_backend now has a DbC assertion that fires first,
    # so accept either AssertionError (DbC) or TypeError (numpy ops on None)
    with pytest.raises((AssertionError, TypeError)):
        analytical_mass_matrix(None, params)
    with pytest.raises(ValueError):
        analytical_mass_matrix(np.zeros(2), params)


def test_coriolis_dbc(params, valid_q, valid_qdot):
    """Test standard vector output and DbC checks for coriolis."""
    C = analytical_coriolis(valid_q, valid_qdot, params)
    assert C.shape == (N_DOF,)

    with pytest.raises(TypeError):
        analytical_coriolis(list(valid_q), valid_qdot, params)
    with pytest.raises(TypeError):
        analytical_coriolis(valid_q, list(valid_qdot), params)
    with pytest.raises(ValueError):
        analytical_coriolis(np.zeros(2), valid_qdot, params)
    with pytest.raises(ValueError):
        analytical_coriolis(valid_q, np.zeros(2), params)


def test_gravity_vector_dbc(params, valid_q):
    """Test standard vector output and DbC checks for gravity."""
    G = analytical_gravity_vector(valid_q, params)
    assert G.shape == (N_DOF,)

    with pytest.raises((AssertionError, TypeError)):
        analytical_gravity_vector(None, params)
    with pytest.raises(ValueError):
        analytical_gravity_vector(np.zeros(2), params)


def test_kinetic_energy_dbc(params, valid_q, valid_qdot):
    """Test standard scalar output and DbC checks for kinetic energy."""
    ke = kinetic_energy(valid_q, valid_qdot, params)
    assert isinstance(ke, float)

    with pytest.raises(TypeError):
        kinetic_energy(list(valid_q), valid_qdot, params)
    with pytest.raises(ValueError):
        kinetic_energy(np.zeros(2), valid_qdot, params)

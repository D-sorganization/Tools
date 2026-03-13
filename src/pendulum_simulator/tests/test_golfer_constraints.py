import numpy as np
import pytest

from double_pendulum_golf.golfer_constraints import (
    constraint_vector,
    numerical_constraint_jacobian,
    analytical_constraint_jacobian,
    linear_accelerations,
    friction_torque_vector,
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


@pytest.fixture
def valid_qddot():
    return np.zeros(N_DOF)


def test_constraint_vector_shape_and_dbc(params, valid_q):
    """Test standard shape output and DbC checks for constraint vector calculation."""
    phi = constraint_vector(valid_q, params)
    assert phi.shape == (4,)
    assert isinstance(phi, np.ndarray)

    with pytest.raises(TypeError):
        constraint_vector([0, 0, 0, 0, 0, 0, 0, 0], params)  # Not ndarray

    with pytest.raises(ValueError):
        constraint_vector(np.array([0, 0, 0]), params)  # Wrong shape

    with pytest.raises(TypeError):
        constraint_vector(valid_q, "fake_params")  # Wrong params


def test_numerical_jacobian_shape_and_dbc(params, valid_q):
    """Test standard shape output and DbC for numerical jacobian."""
    J = numerical_constraint_jacobian(valid_q, params)
    assert J.shape == (4, N_DOF)

    with pytest.raises(TypeError):
        numerical_constraint_jacobian("wrong_type", params)
    with pytest.raises(ValueError):
        numerical_constraint_jacobian(np.ones(N_DOF - 1), params)


def test_analytical_jacobian_shape_and_dbc(params, valid_q):
    """Test standard shape output and DbC for analytical jacobian."""
    J = analytical_constraint_jacobian(valid_q, params)
    assert J.shape == (4, N_DOF)

    with pytest.raises(TypeError):
        analytical_constraint_jacobian("wrong_type", params)
    with pytest.raises(ValueError):
        analytical_constraint_jacobian(np.ones(N_DOF - 1), params)


def test_analytical_vs_numerical_jacobian(params):
    """TDD check: analytical jacobian closely matches analytical jacobian."""
    # Use a random reachable pose
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, 0.05])
    J_num = numerical_constraint_jacobian(q, params)
    J_ana = analytical_constraint_jacobian(q, params)

    np.testing.assert_allclose(J_num, J_ana, rtol=1e-4, atol=1e-4)


def test_linear_accelerations_dbc(params, valid_q, valid_qdot, valid_qddot):
    """Test DbC type enforcement on linear acceleration vectors."""
    acc = linear_accelerations(valid_q, valid_qdot, valid_qddot, params)
    assert isinstance(acc, dict)
    assert "rh" in acc
    assert "lh" in acc

    with pytest.raises(TypeError):
        linear_accelerations(list(valid_q), valid_qdot, valid_qddot, params)
    with pytest.raises(TypeError):
        linear_accelerations(valid_q, list(valid_qdot), valid_qddot, params)
    with pytest.raises(TypeError):
        linear_accelerations(valid_q, valid_qdot, list(valid_qddot), params)

    with pytest.raises(ValueError):
        linear_accelerations(np.zeros(2), valid_qdot, valid_qddot, params)
    with pytest.raises(ValueError):
        linear_accelerations(valid_q, np.zeros(2), valid_qddot, params)
    with pytest.raises(ValueError):
        linear_accelerations(valid_q, valid_qdot, np.zeros(2), params)


def test_friction_torque_vector_dbc(params, valid_qdot):
    """Test DbC checks for friction torques."""
    tau_f = friction_torque_vector(valid_qdot, params)
    assert tau_f.shape == (N_DOF,)

    with pytest.raises(TypeError):
        friction_torque_vector(list(valid_qdot), params)
    with pytest.raises(ValueError):
        friction_torque_vector(np.ones(5), params)

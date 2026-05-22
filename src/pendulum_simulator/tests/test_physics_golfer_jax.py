# ruff: noqa: E501
"""Parity tests between JAX and numpy golfer physics implementations.

Ensures that the JAX implementations produce results consistent with the
analytical numpy implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests in this file if JAX is not available
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from double_pendulum_golf.physics_golfer import (  # noqa: E402
    GolferParams,
    N_DOF,
    analytical_constraint_jacobian,
    analytical_coriolis,
    analytical_fk_jacobians,
    analytical_gravity_vector,
    analytical_mass_matrix,
    constraint_vector,
    forward_kinematics,
)
from double_pendulum_golf.physics_golfer_jax import (  # noqa: E402
    GolferParamsJAX,
    N_DOF,
    _hub_jacobian,
    _left_arm_base_jacobian,
    _right_arm_base_jacobian,
    analytical_fk_jacobians_jax,
    constraint_jacobian_jax,
    constraint_vector_jax,
    coriolis_jax,
    dict_to_golfer_params,
    golfer_params_to_dict,
    gravity_vector_jax,
    mass_matrix_jax,
)
from double_pendulum_golf.constraint_solver import project_to_constraints  # noqa: E402

# Test parameters
_PARAMS_NUMPY = GolferParams(
    m_hub=2.0,
    m_r_upper=3.0,
    m_r_fore=2.0,
    m_l_upper=3.0,
    m_l_fore=2.0,
    m_club=0.5,
    L_hub=0.15,
    L_r_upper=0.35,
    L_r_fore=0.30,
    L_l_upper=0.35,
    L_l_fore=0.30,
    L_club=1.1,
    d_rs=0.20,
    d_ls=0.20,
    grip_right=0.05,
    grip_left=0.25,
    m_clubhead=0.2,
)

_PARAMS_JAX = GolferParamsJAX(
    m_hub=2.0,
    m_r_upper=3.0,
    m_r_fore=2.0,
    m_l_upper=3.0,
    m_l_fore=2.0,
    m_club=0.5,
    L_hub=0.15,
    L_r_upper=0.35,
    L_r_fore=0.30,
    L_l_upper=0.35,
    L_l_fore=0.30,
    L_club=1.1,
    d_rs=0.20,
    d_ls=0.20,
    grip_right=0.05,
    grip_left=0.25,
    m_clubhead=0.2,
)


def _random_configs(n: int = 10, seed: int = 42) -> list[np.ndarray]:
    """Generate n random constrained configurations."""
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        q = project_to_constraints(rng.uniform(-0.3, 0.3, N_DOF), _PARAMS_NUMPY)
        configs.append(q)
    return configs


@pytest.fixture(params=_random_configs(n=5))
def random_config(request) -> np.ndarray:
    """Random constrained configuration."""
    return request.param


@pytest.fixture(params=_random_configs(n=5))
def random_state(request) -> np.ndarray:
    """Random state with non-zero velocities."""
    q = request.param
    qdot = np.random.default_rng(42).uniform(-0.5, 0.5, N_DOF)
    return np.concatenate([q, qdot])


class TestParameterConversion:
    """Test conversion between numpy and JAX parameter types."""

    def test_golfer_params_to_dict(self) -> None:
        """NamedTuple can be converted to dict."""
        d = golfer_params_to_dict(_PARAMS_JAX)
        assert isinstance(d, dict)
        assert d["m_hub"] == 2.0
        assert d["L_club"] == 1.1

    def test_dict_to_golfer_params(self) -> None:
        """Dict can be converted to NamedTuple."""
        d = golfer_params_to_dict(_PARAMS_JAX)
        p = dict_to_golfer_params(d)
        assert isinstance(p, GolferParamsJAX)
        assert p.m_hub == 2.0
        assert p.L_club == 1.1

    def test_roundtrip_conversion(self) -> None:
        """Conversion is invertible."""
        d1 = golfer_params_to_dict(_PARAMS_JAX)
        p = dict_to_golfer_params(d1)
        d2 = golfer_params_to_dict(p)
        assert d1 == d2


class TestForwardKinematics:
    """Test JAX forward kinematics against numpy."""

    def test_fk_hanging_position(self) -> None:
        """Zero configuration gives expected positions."""
        q_np = np.zeros(N_DOF)
        q_jax = jnp.array(q_np)

        fk_np = forward_kinematics(q_np, _PARAMS_NUMPY)
        fk_jax = forward_kinematics(q_np=q_jax, p=_PARAMS_JAX)

        # Compare hub position
        hub_np = np.array(fk_np["hub"])
        hub_jax = np.array(fk_jax["hub"])
        np.testing.assert_allclose(hub_np, hub_jax, rtol=1e-10)

        # Compare right shoulder
        rs_np = np.array(fk_np["rs"])
        rs_jax = np.array(fk_jax["rs"])
        np.testing.assert_allclose(rs_np, rs_jax, rtol=1e-10)

    def test_fk_parity_random_configs(self, random_config: np.ndarray) -> None:
        """JAX FK matches numpy FK on random configs."""
        q_jax = jnp.array(random_config)

        fk_np = forward_kinematics(random_config, _PARAMS_NUMPY)
        fk_jax = forward_kinematics(q_np=q_jax, p=_PARAMS_JAX)

        # Compare all positions
        for key in ["hub", "rs", "re", "rh", "ls", "le", "lh", "club_tip"]:
            pos_np = np.array(fk_np[key])
            pos_jax = np.array(fk_jax[key])
            np.testing.assert_allclose(pos_np, pos_jax, rtol=1e-10, atol=1e-12)


class TestFKJacobians:
    """Test JAX FK Jacobians against numpy."""

    def test_jacobian_shapes(self, random_config: np.ndarray) -> None:
        """Jacobians have correct shapes."""
        q_jax = jnp.array(random_config)
        jacobians = analytical_fk_jacobians_jax(q_jax, _PARAMS_JAX)

        assert jacobians["hub"].shape == (2, N_DOF)
        assert jacobians["re"].shape == (2, N_DOF)
        assert jacobians["rh"].shape == (2, N_DOF)
        assert jacobians["le"].shape == (2, N_DOF)
        assert jacobians["lh"].shape == (2, N_DOF)
        assert jacobians["club_com"].shape == (2, N_DOF)
        assert jacobians["club_tip"].shape == (2, N_DOF)

    def test_jacobian_parity_random_configs(self, random_config: np.ndarray) -> None:
        """JAX Jacobians match numpy on random configs."""
        q_jax = jnp.array(random_config)

        jacs_np = analytical_fk_jacobians(random_config, _PARAMS_NUMPY)
        jacs_jax = analytical_fk_jacobians_jax(q_jax, _PARAMS_JAX)

        # Compare all Jacobians
        for key in ["hub", "re", "rh", "le", "lh", "club_com", "club_tip"]:
            J_np = jacs_np[key]
            J_jax = np.array(jacs_jax[key])
            np.testing.assert_allclose(J_np, J_jax, rtol=1e-8, atol=1e-10)


class TestMassMatrix:
    """Test JAX mass matrix against numpy."""

    def test_mass_matrix_shape(self, random_config: np.ndarray) -> None:
        """Mass matrix has correct shape."""
        q_jax = jnp.array(random_config)
        M_jax = mass_matrix_jax(q_jax, _PARAMS_JAX)
        assert M_jax.shape == (N_DOF, N_DOF)

    def test_mass_matrix_symmetry(self, random_config: np.ndarray) -> None:
        """Mass matrix is symmetric."""
        q_jax = jnp.array(random_config)
        M_jax = mass_matrix_jax(q_jax, _PARAMS_JAX)
        M_array = np.array(M_jax)
        np.testing.assert_allclose(M_array, M_array.T, rtol=1e-10)

    def test_mass_matrix_parity_random_configs(self, random_config: np.ndarray) -> None:
        """JAX mass matrix matches numpy."""
        q_jax = jnp.array(random_config)

        M_np = analytical_mass_matrix(random_config, _PARAMS_NUMPY)
        M_jax = mass_matrix_jax(q_jax, _PARAMS_JAX)
        M_jax_array = np.array(M_jax)

        np.testing.assert_allclose(M_np, M_jax_array, rtol=1e-8, atol=1e-10)

    def test_mass_matrix_positive_semidefinite(self, random_config: np.ndarray) -> None:
        """Mass matrix is positive semi-definite."""
        q_jax = jnp.array(random_config)
        M_jax = mass_matrix_jax(q_jax, _PARAMS_JAX)
        M_array = np.array(M_jax)

        eigenvalues = np.linalg.eigvalsh(M_array)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors


class TestGravityVector:
    """Test JAX gravity vector against numpy."""

    def test_gravity_vector_shape(self, random_config: np.ndarray) -> None:
        """Gravity vector has correct shape."""
        q_jax = jnp.array(random_config)
        G_jax = gravity_vector_jax(q_jax, _PARAMS_JAX)
        assert G_jax.shape == (N_DOF,)

    def test_gravity_vector_parity_random_configs(
        self, random_config: np.ndarray
    ) -> None:
        """JAX gravity vector matches numpy."""
        q_jax = jnp.array(random_config)

        G_np = analytical_gravity_vector(random_config, _PARAMS_NUMPY)
        G_jax = gravity_vector_jax(q_jax, _PARAMS_JAX)
        G_jax_array = np.array(G_jax)

        np.testing.assert_allclose(G_np, G_jax_array, rtol=1e-7, atol=1e-9)


class TestCoriolisForces:
    """Test JAX Coriolis forces against numpy."""

    def test_coriolis_shape(self, random_state: np.ndarray) -> None:
        """Coriolis vector has correct shape."""
        q = random_state[:N_DOF]
        qdot = random_state[N_DOF:]
        q_jax = jnp.array(q)
        qdot_jax = jnp.array(qdot)

        C_jax = coriolis_jax(q_jax, qdot_jax, _PARAMS_JAX)
        assert C_jax.shape == (N_DOF,)

    def test_coriolis_parity_random_states(self, random_state: np.ndarray) -> None:
        """JAX Coriolis matches numpy."""
        q = random_state[:N_DOF]
        qdot = random_state[N_DOF:]
        q_jax = jnp.array(q)
        qdot_jax = jnp.array(qdot)

        C_np = analytical_coriolis(q, qdot, _PARAMS_NUMPY)
        C_jax = coriolis_jax(q_jax, qdot_jax, _PARAMS_JAX)
        C_jax_array = np.array(C_jax)

        np.testing.assert_allclose(C_np, C_jax_array, rtol=1e-6, atol=1e-8)


class TestConstraintVector:
    """Test JAX constraint equations against numpy."""

    def test_constraint_shape(self, random_config: np.ndarray) -> None:
        """Constraint vector has correct shape."""
        q_jax = jnp.array(random_config)
        phi_jax = constraint_vector_jax(q_jax, _PARAMS_JAX)
        assert phi_jax.shape == (4,)

    def test_constraint_parity_random_configs(self, random_config: np.ndarray) -> None:
        """JAX constraint matches numpy."""
        q_jax = jnp.array(random_config)

        phi_np = constraint_vector(random_config, _PARAMS_NUMPY)
        phi_jax = constraint_vector_jax(q_jax, _PARAMS_JAX)
        phi_jax_array = np.array(phi_jax)

        np.testing.assert_allclose(phi_np, phi_jax_array, rtol=1e-8, atol=1e-10)

    def test_constraint_near_zero_for_projected_configs(
        self, random_config: np.ndarray
    ) -> None:
        """Constraint is near zero for properly projected configs."""
        q_jax = jnp.array(random_config)
        phi_jax = constraint_vector_jax(q_jax, _PARAMS_JAX)
        phi_array = np.array(phi_jax)

        # Should be very small (within numerical error of projection)
        np.testing.assert_allclose(phi_array, 0.0, atol=1e-6)


class TestConstraintJacobian:
    """Test JAX constraint Jacobian against numpy."""

    def test_constraint_jacobian_shape(self, random_config: np.ndarray) -> None:
        """Constraint Jacobian has correct shape."""
        q_jax = jnp.array(random_config)
        Phi_q_jax = constraint_jacobian_jax(q_jax, _PARAMS_JAX)
        assert Phi_q_jax.shape == (4, N_DOF)

    def test_constraint_jacobian_parity_random_configs(
        self, random_config: np.ndarray
    ) -> None:
        """JAX Jacobian matches numpy."""
        q_jax = jnp.array(random_config)

        Phi_q_np = analytical_constraint_jacobian(random_config, _PARAMS_NUMPY)
        Phi_q_jax = constraint_jacobian_jax(q_jax, _PARAMS_JAX)
        Phi_q_jax_array = np.array(Phi_q_jax)

        np.testing.assert_allclose(Phi_q_np, Phi_q_jax_array, rtol=1e-6, atol=1e-8)


class TestJITCompilation:
    """Test that JAX functions are JIT-compilable."""

    def test_mass_matrix_jit(self, random_config: np.ndarray) -> None:
        """Mass matrix can be JIT-compiled."""
        q_jax = jnp.array(random_config)

        M_jitted = jax.jit(lambda q: mass_matrix_jax(q, _PARAMS_JAX))
        M = M_jitted(q_jax)

        assert M.shape == (N_DOF, N_DOF)

    def test_fk_jacobians_jit(self, random_config: np.ndarray) -> None:
        """FK Jacobians can be JIT-compiled."""
        q_jax = jnp.array(random_config)

        def jac_hub(q):
            jacs = analytical_fk_jacobians_jax(q, _PARAMS_JAX)
            return jacs["hub"]

        jac_jitted = jax.jit(jac_hub)
        J = jac_jitted(q_jax)

        assert J.shape == (2, N_DOF)

    def test_gravity_vector_jit(self, random_config: np.ndarray) -> None:
        """Gravity vector can be JIT-compiled."""
        q_jax = jnp.array(random_config)

        G_jitted = jax.jit(lambda q: gravity_vector_jax(q, _PARAMS_JAX))
        G = G_jitted(q_jax)

        assert G.shape == (N_DOF,)

    def test_constraint_vector_jit(self, random_config: np.ndarray) -> None:
        """Constraint vector can be JIT-compiled."""
        q_jax = jnp.array(random_config)

        phi_jitted = jax.jit(lambda q: constraint_vector_jax(q, _PARAMS_JAX))
        phi = phi_jitted(q_jax)

        assert phi.shape == (4,)


# ---------------------------------------------------------------------------
# Tests for refactored Jacobian helper functions (issue #2011)
# ---------------------------------------------------------------------------


class TestJacobianHelpers:
    """Unit tests for the extracted Jacobian helper functions.

    These helpers were extracted from analytical_fk_jacobians_jax to reduce
    its LOC from ~184 to ~99 lines (issue #2011 P1 refactoring).
    """

    def test_hub_jacobian_shape(self) -> None:
        """_hub_jacobian returns a (2, N_DOF) matrix."""
        cos_hub = jnp.cos(jnp.array(0.3))
        sin_hub = jnp.sin(jnp.array(0.3))
        J = _hub_jacobian(_PARAMS_JAX, cos_hub, sin_hub)
        assert J.shape == (2, N_DOF)

    def test_hub_jacobian_zero_angle(self) -> None:
        """At th_hub=0, hub Jacobian col-0 is (L_hub, 0) for x/y."""
        cos_hub = jnp.array(1.0)
        sin_hub = jnp.array(0.0)
        J = _hub_jacobian(_PARAMS_JAX, cos_hub, sin_hub)
        assert float(J[0, 0]) == pytest.approx(_PARAMS_JAX.L_hub)
        assert float(J[1, 0]) == pytest.approx(0.0)
        # All other DOF columns should be zero
        assert jnp.allclose(J[:, 1:], 0.0)

    def test_hub_jacobian_only_dof0_nonzero(self) -> None:
        """_hub_jacobian only sets DOF column 0 (hub DOF)."""
        cos_hub = jnp.cos(jnp.array(1.1))
        sin_hub = jnp.sin(jnp.array(1.1))
        J = _hub_jacobian(_PARAMS_JAX, cos_hub, sin_hub)
        assert jnp.allclose(J[:, 1:], 0.0)

    def test_right_arm_base_jacobian_shape(self) -> None:
        """_right_arm_base_jacobian returns (2, N_DOF)."""
        q = jnp.zeros(8)
        J = _right_arm_base_jacobian(
            _PARAMS_JAX,
            jnp.cos(q[0]),
            jnp.sin(q[0]),
            jnp.cos(q[0] + q[1]),
            jnp.sin(q[0] + q[1]),
            jnp.cos(q[0] + q[1] + q[2]),
            jnp.sin(q[0] + q[1] + q[2]),
        )
        assert J.shape == (2, N_DOF)

    def test_right_arm_dofs_3_to_7_are_zero(self) -> None:
        """Right-arm base Jacobian touches only DOFs 0, 1, 2."""
        cos_hub = jnp.cos(jnp.array(0.5))
        sin_hub = jnp.sin(jnp.array(0.5))
        cos_rs = jnp.cos(jnp.array(0.3))
        sin_rs = jnp.sin(jnp.array(0.3))
        cos_re = jnp.cos(jnp.array(-0.2))
        sin_re = jnp.sin(jnp.array(-0.2))
        J = _right_arm_base_jacobian(
            _PARAMS_JAX, cos_hub, sin_hub, cos_rs, sin_rs, cos_re, sin_re
        )
        assert jnp.allclose(J[:, 3:], 0.0)

    def test_left_arm_base_jacobian_shape(self) -> None:
        """_left_arm_base_jacobian returns (2, N_DOF)."""
        q = jnp.zeros(8)
        J = _left_arm_base_jacobian(
            _PARAMS_JAX,
            jnp.cos(q[0]),
            jnp.sin(q[0]),
            jnp.cos(q[0] + q[4]),
            jnp.sin(q[0] + q[4]),
            jnp.cos(q[0] + q[4] + q[5]),
            jnp.sin(q[0] + q[4] + q[5]),
        )
        assert J.shape == (2, N_DOF)

    def test_left_arm_dofs_1_2_3_6_7_are_zero(self) -> None:
        """Left-arm base Jacobian only touches DOFs 0, 4, 5."""
        cos_hub = jnp.cos(jnp.array(0.1))
        sin_hub = jnp.sin(jnp.array(0.1))
        cos_ls = jnp.cos(jnp.array(0.2))
        sin_ls = jnp.sin(jnp.array(0.2))
        cos_le = jnp.cos(jnp.array(0.3))
        sin_le = jnp.sin(jnp.array(0.3))
        J = _left_arm_base_jacobian(
            _PARAMS_JAX, cos_hub, sin_hub, cos_ls, sin_ls, cos_le, sin_le
        )
        zero_cols = [1, 2, 3, 6, 7]
        for col in zero_cols:
            assert jnp.allclose(J[:, col], 0.0), f"DOF {col} should be zero"

    def test_helpers_agree_with_full_jacobians_rh(self) -> None:
        """_right_arm_base_jacobian matches rh entry from analytical_fk_jacobians_jax."""
        q = jnp.array([0.3, 0.1, -0.2, 0.0, 0.1, -0.1, 0.0, 0.5])
        full = analytical_fk_jacobians_jax(q, _PARAMS_JAX)
        J_expected = full["rh"]

        cos_hub, sin_hub = jnp.cos(q[0]), jnp.sin(q[0])
        cos_rs = jnp.cos(q[0] + q[1])
        sin_rs = jnp.sin(q[0] + q[1])
        cos_re = jnp.cos(q[0] + q[1] + q[2])
        sin_re = jnp.sin(q[0] + q[1] + q[2])
        J_helper = _right_arm_base_jacobian(
            _PARAMS_JAX, cos_hub, sin_hub, cos_rs, sin_rs, cos_re, sin_re
        )
        assert jnp.allclose(J_helper, J_expected, atol=1e-6)

    def test_helpers_agree_with_full_jacobians_lh(self) -> None:
        """_left_arm_base_jacobian matches lh entry from analytical_fk_jacobians_jax."""
        q = jnp.array([0.3, 0.1, -0.2, 0.0, 0.1, -0.1, 0.0, 0.5])
        full = analytical_fk_jacobians_jax(q, _PARAMS_JAX)
        J_expected = full["lh"]

        cos_hub, sin_hub = jnp.cos(q[0]), jnp.sin(q[0])
        cos_ls = jnp.cos(q[0] + q[4])
        sin_ls = jnp.sin(q[0] + q[4])
        cos_le = jnp.cos(q[0] + q[4] + q[5])
        sin_le = jnp.sin(q[0] + q[4] + q[5])
        J_helper = _left_arm_base_jacobian(
            _PARAMS_JAX, cos_hub, sin_hub, cos_ls, sin_ls, cos_le, sin_le
        )
        assert jnp.allclose(J_helper, J_expected, atol=1e-6)

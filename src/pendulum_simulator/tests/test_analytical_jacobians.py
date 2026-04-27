"""Tests for analytical FK Jacobians and derived quantities.

Validates analytical implementations against existing numerical references.
Follows TDD principles: write tests first, then implement.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
from double_pendulum_golf.constraint_solver import (
    _constraint_acceleration_bias as numerical_bias,
    project_to_constraints,
)
from double_pendulum_golf.physics_golfer import (
    GolferParams,
    N_DOF,
    constraint_jacobian as numerical_constraint_jac,
    coriolis_matrix as numerical_coriolis,
    forward_kinematics,
    gravity_vector as numerical_gravity,
    mass_matrix as numerical_mass_matrix,
)

_logger = logging.getLogger(__name__)

# Test parameters
_PARAMS = GolferParams(
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


def _random_configs(n: int = 20, seed: int = 42) -> list[np.ndarray]:
    """Generate random configs on the constraint manifold."""
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        q = project_to_constraints(rng.uniform(-0.5, 0.5, N_DOF), _PARAMS)
        configs.append(q)
    return configs


def _numerical_jacobian_point(pos_func, q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute 2×8 Jacobian numerically using finite differences."""
    J = np.zeros((2, N_DOF))
    pos_0 = np.array(pos_func(q))
    for j in range(N_DOF):
        q_plus = q.copy()
        q_plus[j] += eps
        pos_j = np.array(pos_func(q_plus))
        J[:, j] = (pos_j - pos_0) / eps
    return J


@pytest.fixture
def test_configs() -> list[np.ndarray]:
    """20 random constrained configurations for testing."""
    return _random_configs(n=20, seed=42)


class TestAnalyticalFKJacobians:
    """FK Jacobians must match numerical computation."""

    def test_module_exports_analytical_jacobians(self) -> None:
        """Check that the analytical Jacobian functions are exported."""
        from double_pendulum_golf import physics_golfer

        # These functions should be available
        assert hasattr(physics_golfer, "analytical_fk_jacobians")
        assert callable(physics_golfer.analytical_fk_jacobians)

    def test_hub_jacobian_vs_numerical(self, test_configs: list[np.ndarray]) -> None:
        """Hub Jacobian (depends on q[0] only)."""
        from double_pendulum_golf.physics_golfer import analytical_fk_jacobians

        for q in test_configs:
            J_analytical = analytical_fk_jacobians(q, _PARAMS)["hub"]

            def hub_pos(qq):
                fk = forward_kinematics(qq, _PARAMS)
                return fk["hub"]

            J_numerical = _numerical_jacobian_point(hub_pos, q)

            assert np.allclose(J_analytical, J_numerical, atol=1e-5, rtol=1e-4), (
                f"Hub Jacobian mismatch at q={q}"
            )

    def test_re_jacobian_vs_numerical(self, test_configs: list[np.ndarray]) -> None:
        """RE Jacobian (depends on q[0], q[1])."""
        from double_pendulum_golf.physics_golfer import analytical_fk_jacobians

        for q in test_configs:
            J_analytical = analytical_fk_jacobians(q, _PARAMS)["re"]

            def re_pos(qq):
                fk = forward_kinematics(qq, _PARAMS)
                return fk["re"]

            J_numerical = _numerical_jacobian_point(re_pos, q)

            assert np.allclose(J_analytical, J_numerical, atol=1e-5, rtol=1e-4), (
                f"RE Jacobian mismatch at q={q}"
            )

    def test_rh_jacobian_vs_numerical(self, test_configs: list[np.ndarray]) -> None:
        """RH Jacobian (depends on q[0], q[1], q[2])."""
        from double_pendulum_golf.physics_golfer import analytical_fk_jacobians

        for q in test_configs:
            J_analytical = analytical_fk_jacobians(q, _PARAMS)["rh"]

            def rh_pos(qq):
                fk = forward_kinematics(qq, _PARAMS)
                return fk["rh"]

            J_numerical = _numerical_jacobian_point(rh_pos, q)

            assert np.allclose(J_analytical, J_numerical, atol=1e-5, rtol=1e-4), (
                f"RH Jacobian mismatch at q={q}"
            )

    def test_le_jacobian_vs_numerical(self, test_configs: list[np.ndarray]) -> None:
        """LE Jacobian (depends on q[0], q[4])."""
        from double_pendulum_golf.physics_golfer import analytical_fk_jacobians

        for q in test_configs:
            J_analytical = analytical_fk_jacobians(q, _PARAMS)["le"]

            def le_pos(qq):
                fk = forward_kinematics(qq, _PARAMS)
                return fk["le"]

            J_numerical = _numerical_jacobian_point(le_pos, q)

            assert np.allclose(J_analytical, J_numerical, atol=1e-5, rtol=1e-4), (
                f"LE Jacobian mismatch at q={q}"
            )

    def test_lh_jacobian_vs_numerical(self, test_configs: list[np.ndarray]) -> None:
        """LH Jacobian (depends on q[0], q[4], q[5])."""
        from double_pendulum_golf.physics_golfer import analytical_fk_jacobians

        for q in test_configs:
            J_analytical = analytical_fk_jacobians(q, _PARAMS)["lh"]

            def lh_pos(qq):
                fk = forward_kinematics(qq, _PARAMS)
                return fk["lh"]

            J_numerical = _numerical_jacobian_point(lh_pos, q)

            assert np.allclose(J_analytical, J_numerical, atol=1e-5, rtol=1e-4), (
                f"LH Jacobian mismatch at q={q}"
            )

    def test_club_com_jacobian_vs_numerical(self, test_configs: list[np.ndarray]) -> None:
        """Club COM Jacobian (depends on q[0], q[1], q[2], q[3], q[7])."""
        from double_pendulum_golf.physics_golfer import analytical_fk_jacobians

        for q in test_configs:
            J_analytical = analytical_fk_jacobians(q, _PARAMS)["club_com"]

            def club_com_pos(qq):
                fk = forward_kinematics(qq, _PARAMS)
                base = fk["club_base"]
                tip = fk["club_tip"]
                return (0.5 * (base[0] + tip[0]), 0.5 * (base[1] + tip[1]))

            J_numerical = _numerical_jacobian_point(club_com_pos, q)

            assert np.allclose(J_analytical, J_numerical, atol=1e-5, rtol=1e-4), (
                f"Club COM Jacobian mismatch at q={q}"
            )

    def test_club_tip_jacobian_vs_numerical(self, test_configs: list[np.ndarray]) -> None:
        """Club tip Jacobian (depends on q[0], q[1], q[2], q[3], q[7])."""
        from double_pendulum_golf.physics_golfer import analytical_fk_jacobians

        for q in test_configs:
            J_analytical = analytical_fk_jacobians(q, _PARAMS)["club_tip"]

            def club_tip_pos(qq):
                fk = forward_kinematics(qq, _PARAMS)
                return fk["club_tip"]

            J_numerical = _numerical_jacobian_point(club_tip_pos, q)

            assert np.allclose(J_analytical, J_numerical, atol=1e-5, rtol=1e-4), (
                f"Club tip Jacobian mismatch at q={q}"
            )


class TestAnalyticalMassMatrix:
    """Analytical mass matrix must match numerical computation."""

    def test_module_exports_analytical_mass_matrix(self) -> None:
        """Check that analytical_mass_matrix is exported."""
        from double_pendulum_golf import physics_golfer

        assert hasattr(physics_golfer, "analytical_mass_matrix")
        assert callable(physics_golfer.analytical_mass_matrix)

    def test_analytical_mass_matrix_parity(self, test_configs: list[np.ndarray]) -> None:
        """Analytical mass matrix matches numerical at 20 configs."""
        from double_pendulum_golf.physics_golfer import analytical_mass_matrix

        for q in test_configs:
            M_analytical = analytical_mass_matrix(q, _PARAMS)
            M_numerical = numerical_mass_matrix(q, _PARAMS)

            assert np.allclose(M_analytical, M_numerical, atol=1e-6, rtol=1e-4), (
                f"Mass matrix mismatch at q={q}"
            )

    def test_mass_matrix_symmetric(self, test_configs: list[np.ndarray]) -> None:
        """Analytical mass matrix is symmetric."""
        from double_pendulum_golf.physics_golfer import analytical_mass_matrix

        for q in test_configs:
            M = analytical_mass_matrix(q, _PARAMS)
            assert np.allclose(M, M.T, atol=1e-10)

    def test_mass_matrix_psd(self, test_configs: list[np.ndarray]) -> None:
        """Analytical mass matrix is positive semi-definite."""
        from double_pendulum_golf.physics_golfer import analytical_mass_matrix

        for q in test_configs:
            M = analytical_mass_matrix(q, _PARAMS)
            eigenvalues = np.linalg.eigvalsh(M)
            assert np.all(eigenvalues >= -1e-10)


class TestAnalyticalCoriolis:
    """Analytical Coriolis must match numerical computation."""

    def test_module_exports_analytical_coriolis(self) -> None:
        """Check that analytical_coriolis is exported."""
        from double_pendulum_golf import physics_golfer

        assert hasattr(physics_golfer, "analytical_coriolis")
        assert callable(physics_golfer.analytical_coriolis)

    def test_analytical_coriolis_parity(self, test_configs: list[np.ndarray]) -> None:
        """Analytical Coriolis matches numerical at 20 configs."""
        from double_pendulum_golf.physics_golfer import analytical_coriolis

        rng = np.random.default_rng(43)
        for q in test_configs:
            qdot = rng.uniform(-1, 1, N_DOF)
            C_analytical = analytical_coriolis(q, qdot, _PARAMS)
            C_numerical = numerical_coriolis(q, qdot, _PARAMS)

            assert np.allclose(C_analytical, C_numerical, atol=1e-5, rtol=1e-3), (
                f"Coriolis mismatch at q={q}, qdot={qdot}"
            )

    def test_coriolis_zero_at_zero_velocity(self, test_configs: list[np.ndarray]) -> None:
        """Coriolis is zero when velocity is zero."""
        from double_pendulum_golf.physics_golfer import analytical_coriolis

        for q in test_configs:
            qdot = np.zeros(N_DOF)
            C = analytical_coriolis(q, qdot, _PARAMS)
            assert np.allclose(C, 0.0, atol=1e-12)


class TestAnalyticalGravity:
    """Analytical gravity must match numerical computation."""

    def test_module_exports_analytical_gravity(self) -> None:
        """Check that analytical_gravity_vector is exported."""
        from double_pendulum_golf import physics_golfer

        assert hasattr(physics_golfer, "analytical_gravity_vector")
        assert callable(physics_golfer.analytical_gravity_vector)

    def test_analytical_gravity_parity(self, test_configs: list[np.ndarray]) -> None:
        """Analytical gravity matches numerical at 20 configs."""
        from double_pendulum_golf.physics_golfer import analytical_gravity_vector

        for q in test_configs:
            G_analytical = analytical_gravity_vector(q, _PARAMS)
            G_numerical = numerical_gravity(q, _PARAMS)

            assert np.allclose(G_analytical, G_numerical, atol=1e-5, rtol=1e-4), (
                f"Gravity mismatch at q={q}"
            )


class TestAnalyticalConstraintJacobian:
    """Analytical constraint Jacobian must match numerical computation."""

    def test_module_exports_analytical_constraint_jac(self) -> None:
        """Check that analytical_constraint_jacobian is exported."""
        from double_pendulum_golf import physics_golfer

        assert hasattr(physics_golfer, "analytical_constraint_jacobian")
        assert callable(physics_golfer.analytical_constraint_jacobian)

    def test_analytical_constraint_jac_parity(self, test_configs: list[np.ndarray]) -> None:
        """Analytical constraint Jacobian matches numerical at 20 configs."""
        from double_pendulum_golf.physics_golfer import (
            analytical_constraint_jacobian,
        )

        for q in test_configs:
            Phi_q_analytical = analytical_constraint_jacobian(q, _PARAMS)
            Phi_q_numerical = numerical_constraint_jac(q, _PARAMS)

            assert np.allclose(Phi_q_analytical, Phi_q_numerical, atol=1e-5, rtol=1e-4), (
                f"Constraint Jacobian mismatch at q={q}"
            )

    def test_constraint_jac_shape(self) -> None:
        """Constraint Jacobian has shape (4, 8)."""
        from double_pendulum_golf.physics_golfer import (
            analytical_constraint_jacobian,
        )

        q = np.zeros(N_DOF)
        Phi_q = analytical_constraint_jacobian(q, _PARAMS)
        assert Phi_q.shape == (4, N_DOF)


class TestAnalyticalConstraintAccelerationBias:
    """Analytical constraint acceleration bias must match numerical."""

    def test_module_exports_analytical_bias(self) -> None:
        """Check that analytical_constraint_acceleration_bias is exported."""
        from double_pendulum_golf import constraint_solver

        assert hasattr(constraint_solver, "analytical_constraint_acceleration_bias")
        assert callable(constraint_solver.analytical_constraint_acceleration_bias)

    def test_analytical_bias_parity(self, test_configs: list[np.ndarray]) -> None:
        """Analytical bias matches numerical at 20 configs."""
        from double_pendulum_golf.constraint_solver import (
            analytical_constraint_acceleration_bias,
        )

        rng = np.random.default_rng(44)
        for q in test_configs:
            qdot = rng.uniform(-1, 1, N_DOF)
            gamma_analytical = analytical_constraint_acceleration_bias(q, qdot, _PARAMS)
            gamma_numerical = numerical_bias(q, qdot, _PARAMS)

            assert np.allclose(gamma_analytical, gamma_numerical, atol=1e-5, rtol=1e-3), (
                f"Bias mismatch at q={q}, qdot={qdot}"
            )

    def test_bias_zero_at_zero_velocity(self, test_configs: list[np.ndarray]) -> None:
        """Bias is zero when velocity is zero."""
        from double_pendulum_golf.constraint_solver import (
            analytical_constraint_acceleration_bias,
        )

        for q in test_configs:
            qdot = np.zeros(N_DOF)
            gamma = analytical_constraint_acceleration_bias(q, qdot, _PARAMS)
            assert np.allclose(gamma, 0.0, atol=1e-12)


class TestAnalyticalBenchmark:
    """Benchmark analytical vs numerical implementations."""

    def test_speed_comparison_mass_matrix(self, test_configs: list[np.ndarray]) -> None:
        """Print timing comparison for mass matrix."""
        from double_pendulum_golf.physics_golfer import analytical_mass_matrix

        # Warm up
        for q in test_configs[:2]:
            analytical_mass_matrix(q, _PARAMS)
            numerical_mass_matrix(q, _PARAMS)

        # Time analytical
        start = time.perf_counter()
        for _ in range(10):
            for q in test_configs:
                analytical_mass_matrix(q, _PARAMS)
        t_analytical = time.perf_counter() - start

        # Time numerical
        start = time.perf_counter()
        for _ in range(10):
            for q in test_configs:
                numerical_mass_matrix(q, _PARAMS)
        t_numerical = time.perf_counter() - start

        speedup = t_numerical / t_analytical
        _logger.debug("\nMass matrix speedup: %.1fx", speedup)
        _logger.debug("  Analytical: %.3fs", t_analytical)
        _logger.debug("  Numerical:  %.3fs", t_numerical)

    def test_speed_comparison_coriolis(self, test_configs: list[np.ndarray]) -> None:
        """Print timing comparison for Coriolis."""
        from double_pendulum_golf.physics_golfer import analytical_coriolis

        rng = np.random.default_rng(45)
        qdots = [rng.uniform(-1, 1, N_DOF) for _ in test_configs]

        # Warm up
        for q, qdot in zip(test_configs[:2], qdots[:2]):
            analytical_coriolis(q, qdot, _PARAMS)
            numerical_coriolis(q, qdot, _PARAMS)

        # Time analytical
        start = time.perf_counter()
        for _ in range(10):
            for q, qdot in zip(test_configs, qdots):
                analytical_coriolis(q, qdot, _PARAMS)
        t_analytical = time.perf_counter() - start

        # Time numerical
        start = time.perf_counter()
        for _ in range(10):
            for q, qdot in zip(test_configs, qdots):
                numerical_coriolis(q, qdot, _PARAMS)
        t_numerical = time.perf_counter() - start

        speedup = t_numerical / t_analytical
        _logger.debug("\nCoriolis speedup: %.1fx", speedup)
        _logger.debug("  Analytical: %.3fs", t_analytical)
        _logger.debug("  Numerical:  %.3fs", t_numerical)

    def test_speed_comparison_gravity(self, test_configs: list[np.ndarray]) -> None:
        """Print timing comparison for gravity."""
        from double_pendulum_golf.physics_golfer import analytical_gravity_vector

        # Warm up
        for q in test_configs[:2]:
            analytical_gravity_vector(q, _PARAMS)
            numerical_gravity(q, _PARAMS)

        # Time analytical
        start = time.perf_counter()
        for _ in range(10):
            for q in test_configs:
                analytical_gravity_vector(q, _PARAMS)
        t_analytical = time.perf_counter() - start

        # Time numerical
        start = time.perf_counter()
        for _ in range(10):
            for q in test_configs:
                numerical_gravity(q, _PARAMS)
        t_numerical = time.perf_counter() - start

        speedup = t_numerical / t_analytical
        _logger.debug("\nGravity speedup: %.1fx", speedup)
        _logger.debug("  Analytical: %.3fs", t_analytical)
        _logger.debug("  Numerical:  %.3fs", t_numerical)


# ===========================================================================
# GH1691: Tests for private JAX Jacobian helpers extracted from
# analytical_fk_jacobians_jax
# ===========================================================================

_JAX_PARAMS_AVAILABLE = False
try:
    import jax.numpy as jnp
    from double_pendulum_golf.physics_golfer_jax import (
        GolferParamsJAX,
        _club_jacobians_jax,
        _left_arm_jacobians_jax,
        _right_arm_jacobians_jax,
        analytical_fk_jacobians_jax,
    )

    _JAX_PARAMS_AVAILABLE = True
except ImportError:
    pass

_JAX_PARAMS = (
    GolferParamsJAX(
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
    )
    if _JAX_PARAMS_AVAILABLE
    else None
)


def _assert_jac_keys_match(
    helpers: dict, full: dict, keys: tuple, *, atol: float = 1e-10
) -> None:
    """Assert that helper Jacobian arrays match the full analytical Jacobians.

    Extracted from three identical per-key assertion blocks in TestJacobianHelpers
    to satisfy DRY.
    """
    import numpy as np

    for key in keys:
        np.testing.assert_allclose(
            np.array(helpers[key]),
            np.array(full[key]),
            atol=atol,
            err_msg=f"Mismatch in key '{key}'",
        )


@pytest.mark.skipif(not _JAX_PARAMS_AVAILABLE, reason="JAX not available")
class TestJacobianHelpers:
    """GH1691: Private JAX Jacobian helper functions must produce results
    consistent with analytical_fk_jacobians_jax."""

    def _make_trig(self, q):
        """Precompute sin/cos values for a given configuration array."""
        th_hub = float(q[0])
        alpha_rs, alpha_re = float(q[1]), float(q[2])
        alpha_ls, alpha_le = float(q[4]), float(q[5])
        th_club = float(q[7])
        sin_hub, cos_hub = jnp.sin(th_hub), jnp.cos(th_hub)
        th_rs = th_hub + alpha_rs
        th_re = th_hub + alpha_rs + alpha_re
        sin_rs, cos_rs = jnp.sin(th_rs), jnp.cos(th_rs)
        sin_re, cos_re = jnp.sin(th_re), jnp.cos(th_re)
        th_ls = th_hub + alpha_ls
        th_le = th_hub + alpha_ls + alpha_le
        sin_ls, cos_ls = jnp.sin(th_ls), jnp.cos(th_ls)
        sin_le, cos_le = jnp.sin(th_le), jnp.cos(th_le)
        sin_club, cos_club = jnp.sin(th_club), jnp.cos(th_club)
        return dict(
            sin_hub=sin_hub,
            cos_hub=cos_hub,
            sin_rs=sin_rs,
            cos_rs=cos_rs,
            sin_re=sin_re,
            cos_re=cos_re,
            sin_ls=sin_ls,
            cos_ls=cos_ls,
            sin_le=sin_le,
            cos_le=cos_le,
            sin_club=sin_club,
            cos_club=cos_club,
        )

    def test_right_arm_jacobians_match_full(self) -> None:
        """_right_arm_jacobians_jax output must match keys from full function."""
        import numpy as np

        rng = np.random.default_rng(0)
        q = jnp.array(rng.uniform(-0.5, 0.5, 8))
        tr = self._make_trig(q)
        full = analytical_fk_jacobians_jax(q, _JAX_PARAMS)
        helpers = _right_arm_jacobians_jax(
            _JAX_PARAMS,
            tr["sin_hub"],
            tr["cos_hub"],
            tr["sin_rs"],
            tr["cos_rs"],
            tr["sin_re"],
            tr["cos_re"],
        )
        _assert_jac_keys_match(helpers, full, ("hub", "rs", "re", "rh"))

    def test_left_arm_jacobians_match_full(self) -> None:
        """_left_arm_jacobians_jax output must match keys from full function."""
        import numpy as np

        rng = np.random.default_rng(1)
        q = jnp.array(rng.uniform(-0.5, 0.5, 8))
        tr = self._make_trig(q)
        full = analytical_fk_jacobians_jax(q, _JAX_PARAMS)
        helpers = _left_arm_jacobians_jax(
            _JAX_PARAMS,
            tr["sin_hub"],
            tr["cos_hub"],
            tr["sin_ls"],
            tr["cos_ls"],
            tr["sin_le"],
            tr["cos_le"],
        )
        _assert_jac_keys_match(helpers, full, ("ls", "le", "lh"))

    def test_club_jacobians_match_full(self) -> None:
        """_club_jacobians_jax output must match keys from full function."""
        import numpy as np

        rng = np.random.default_rng(2)
        q = jnp.array(rng.uniform(-0.5, 0.5, 8))
        tr = self._make_trig(q)
        full = analytical_fk_jacobians_jax(q, _JAX_PARAMS)
        helpers = _club_jacobians_jax(
            _JAX_PARAMS,
            tr["sin_hub"],
            tr["cos_hub"],
            tr["sin_rs"],
            tr["cos_rs"],
            tr["sin_re"],
            tr["cos_re"],
            tr["sin_club"],
            tr["cos_club"],
        )
        _assert_jac_keys_match(helpers, full, ("club_com", "club_tip"))

    def test_helpers_return_correct_shapes(self) -> None:
        """Each Jacobian helper must return 2×8 arrays for all keys."""
        import numpy as np

        rng = np.random.default_rng(3)
        q = jnp.array(rng.uniform(-0.5, 0.5, 8))
        tr = self._make_trig(q)
        right = _right_arm_jacobians_jax(
            _JAX_PARAMS,
            tr["sin_hub"],
            tr["cos_hub"],
            tr["sin_rs"],
            tr["cos_rs"],
            tr["sin_re"],
            tr["cos_re"],
        )
        for key, val in right.items():
            arr = np.array(val)
            assert arr.shape == (2, 8), f"key '{key}': expected (2,8) got {arr.shape}"

    def test_helpers_precondition_none_p(self) -> None:
        """_right_arm_jacobians_jax must raise AssertionError if p is None."""
        import numpy as np

        rng = np.random.default_rng(4)
        q = jnp.array(rng.uniform(-0.5, 0.5, 8))
        tr = self._make_trig(q)
        with pytest.raises(AssertionError):
            _right_arm_jacobians_jax(
                None,  # type: ignore[arg-type]
                tr["sin_hub"],
                tr["cos_hub"],
                tr["sin_rs"],
                tr["cos_rs"],
                tr["sin_re"],
                tr["cos_re"],
            )

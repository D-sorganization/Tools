"""Tests for simulation_golfer_gpu.py utility functions.

Since JAX and diffrax are optional dependencies that may not be installed,
these tests patch the imports at the sys.modules level before importing
the module under test. This allows us to test the extract_final_state,
extract_trajectory, extract_times, and constrained_eom_jax helper
functions without requiring an actual GPU or JAX installation.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock
import numpy as np

# ---------------------------------------------------------------------------
# Build fake jax / diffrax modules before importing the GPU module
# ---------------------------------------------------------------------------


def _make_fake_jax() -> types.ModuleType:
    """Return a minimal fake jax module tree."""
    jax = types.ModuleType("jax")
    jnp = types.ModuleType("jax.numpy")

    # Delegate numeric operations to numpy so arithmetic still works
    jnp.zeros = np.zeros
    jnp.array = np.array
    jnp.concatenate = np.concatenate
    jnp.arange = np.arange
    jnp.ndarray = np.ndarray

    def jnp_linalg_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(A, b)

    jnp_linalg = types.ModuleType("jax.numpy.linalg")
    jnp_linalg.solve = jnp_linalg_solve
    jnp.linalg = jnp_linalg

    # zeros_like for constructing arrays from existing arrays
    jnp.zeros_like = np.zeros_like

    # .at[...].set(...) / .at[...].add(...)  — emulate with a helper class
    class _AtIndexer:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr.copy()

        def set(self, val: np.ndarray) -> np.ndarray:
            result = self._arr.copy()
            # Called as arr.at[:n].set(val) — figure out slice from context
            # We monkey-patch at the array level below
            return result

    # Fake jnp.ndarray with .at property
    # We don't need full at[...].set() fidelity here, just the shape.

    jax.jit = lambda f: f  # passthrough decorator
    jax.numpy = jnp
    jax.vmap = MagicMock(side_effect=lambda f, **kw: f)

    return jax, jnp


def _make_fake_diffrax() -> types.ModuleType:
    diffrax = types.ModuleType("diffrax")
    diffrax.diffeqsolve = MagicMock()
    diffrax.Dopri5 = MagicMock(return_value=MagicMock())
    diffrax.ODETerm = MagicMock(side_effect=lambda f: MagicMock())
    diffrax.SaveAt = MagicMock(return_value=MagicMock())
    diffrax.PIDController = MagicMock(return_value=MagicMock())
    return diffrax


# We insert the mocks BEFORE importing the module under test.
_fake_jax, _fake_jnp = _make_fake_jax()
_fake_diffrax = _make_fake_diffrax()
sys.modules.setdefault("jax", _fake_jax)
sys.modules.setdefault("jax.numpy", _fake_jnp)
sys.modules.setdefault("diffrax", _fake_diffrax)

# Also mock the JAX physics module to avoid recursive import issues
_fake_physics_jax = types.ModuleType("double_pendulum_golf.physics_golfer_jax")
_fake_physics_jax.GolferParamsJAX = MagicMock()
_fake_physics_jax.N_CONSTRAINTS = 4
_fake_physics_jax.N_DOF = 8
_fake_physics_jax.constraint_jacobian_jax = MagicMock(return_value=np.zeros((4, 8)))
_fake_physics_jax.constraint_vector_jax = MagicMock(return_value=np.zeros(4))
_fake_physics_jax.coriolis_jax = MagicMock(return_value=np.zeros(8))
_fake_physics_jax.gravity_vector_jax = MagicMock(return_value=np.zeros(8))
_fake_physics_jax.mass_matrix_jax = MagicMock(return_value=np.eye(8))
sys.modules.setdefault("double_pendulum_golf.physics_golfer_jax", _fake_physics_jax)

# Now import the module under test
from double_pendulum_golf import simulation_golfer_gpu as _gpu_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Tests for extract_final_state
# ---------------------------------------------------------------------------


class TestExtractFinalState:
    def _make_sol(self, n: int = 5) -> MagicMock:
        sol = MagicMock()
        sol.ys = np.random.default_rng(1).random((n, 16))
        return sol

    def test_returns_last_row(self) -> None:
        sol = self._make_sol(n=7)
        result = _gpu_mod.extract_final_state(sol)
        np.testing.assert_array_equal(result, sol.ys[-1])

    def test_shape_is_16(self) -> None:
        sol = self._make_sol(n=3)
        result = _gpu_mod.extract_final_state(sol)
        assert result.shape == (16,)

    def test_single_step(self) -> None:
        sol = MagicMock()
        sol.ys = np.array([[1.0] * 16])
        result = _gpu_mod.extract_final_state(sol)
        np.testing.assert_array_equal(result, sol.ys[0])

    def test_result_is_last_not_first(self) -> None:
        sol = MagicMock()
        sol.ys = np.zeros((5, 16))
        sol.ys[-1] = np.ones(16)
        result = _gpu_mod.extract_final_state(sol)
        np.testing.assert_array_equal(result, np.ones(16))


# ---------------------------------------------------------------------------
# Tests for extract_trajectory
# ---------------------------------------------------------------------------


class TestExtractTrajectory:
    def test_returns_ys(self) -> None:
        sol = MagicMock()
        sol.ys = np.arange(30).reshape(5, 6)
        result = _gpu_mod.extract_trajectory(sol)
        np.testing.assert_array_equal(result, sol.ys)

    def test_shape_preserved(self) -> None:
        sol = MagicMock()
        sol.ys = np.zeros((100, 16))
        result = _gpu_mod.extract_trajectory(sol)
        assert result.shape == (100, 16)


# ---------------------------------------------------------------------------
# Tests for extract_times
# ---------------------------------------------------------------------------


class TestExtractTimes:
    def test_returns_ts(self) -> None:
        sol = MagicMock()
        sol.ts = np.linspace(0.0, 1.0, 20)
        result = _gpu_mod.extract_times(sol)
        np.testing.assert_array_equal(result, sol.ts)

    def test_monotonic(self) -> None:
        sol = MagicMock()
        sol.ts = np.linspace(0.0, 2.0, 100)
        ts = _gpu_mod.extract_times(sol)
        assert np.all(np.diff(ts) > 0)


# ---------------------------------------------------------------------------
# Tests for constants / defaults
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_alpha_positive(self) -> None:
        assert _gpu_mod.DEFAULT_ALPHA > 0

    def test_default_beta_positive(self) -> None:
        assert _gpu_mod.DEFAULT_BETA > 0

    def test_default_gains_match_expected(self) -> None:
        assert _gpu_mod.DEFAULT_ALPHA == 5.0
        assert _gpu_mod.DEFAULT_BETA == 5.0


# ---------------------------------------------------------------------------
# Tests for run_single_simulation_jax (mocked diffrax call)
# ---------------------------------------------------------------------------


class TestRunSingleSimulation:
    def test_calls_diffeqsolve(self) -> None:
        params = MagicMock()
        initial_state = np.zeros(16)
        torque_coeffs = np.zeros(7)
        _fake_diffrax.diffeqsolve.return_value = MagicMock(
            ys=np.zeros((10, 16)), ts=np.linspace(0, 1, 10)
        )

        _gpu_mod.run_single_simulation_jax(
            params=params,
            initial_state=initial_state,
            t_end=0.1,
            torque_coeffs=torque_coeffs,
        )
        _fake_diffrax.diffeqsolve.assert_called_once()

    def test_returns_solution_object(self) -> None:
        params = MagicMock()
        mock_sol = MagicMock(ys=np.zeros((5, 16)), ts=np.linspace(0, 0.1, 5))
        _fake_diffrax.diffeqsolve.return_value = mock_sol

        result = _gpu_mod.run_single_simulation_jax(
            params=params,
            initial_state=np.zeros(16),
            t_end=0.1,
            torque_coeffs=np.zeros(7),
        )
        assert result is mock_sol

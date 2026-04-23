"""Tests for the advanced optimizer (CMA-ES, warm-start, batch eval)."""

from __future__ import annotations

import numpy as np
import pytest


def _has_optimizer() -> bool:
    """Check if optimizer can be imported (False in headless CI environments)."""
    try:
        from double_pendulum_golf.gui.optimization_widget import (  # noqa: F401
            CMAESState,
            _cmaes_step,
        )

        return True
    except (ImportError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_optimizer(), reason="PyQt6/optimizer not available"
)


class TestCMAESStep:
    """Verify CMA-ES produces improving solutions over generations."""

    @staticmethod
    def _sphere(x: np.ndarray) -> float:
        """Simple sphere function: global minimum at origin."""
        return float(np.sum(x**2))

    def test_basic_improvement(self) -> None:
        """CMA-ES should improve fitness over multiple generations."""
        from double_pendulum_golf.gui.optimization_widget import (
            CMAESState,
            _cmaes_step,
        )

        rng = np.random.default_rng(42)
        n = 4

        state = CMAESState(
            mean=np.ones(n) * 5.0,
            sigma=2.0,
            C=np.eye(n),
            p_sigma=np.zeros(n),
            p_c=np.zeros(n),
        )

        initial_fitness = self._sphere(state.mean)
        for _ in range(30):
            state, _ = _cmaes_step(state, self._sphere, pop_size=10, rng=rng)

        assert state.best_fitness < initial_fitness
        assert state.generation == 30

    def test_finds_optimum(self) -> None:
        """CMA-ES should converge near the origin for the sphere fn."""
        from double_pendulum_golf.gui.optimization_widget import (
            CMAESState,
            _cmaes_step,
        )

        rng = np.random.default_rng(0)
        n = 2
        state = CMAESState(
            mean=np.array([3.0, -4.0]),
            sigma=1.0,
            C=np.eye(n),
            p_sigma=np.zeros(n),
            p_c=np.zeros(n),
        )

        for _ in range(100):
            state, _ = _cmaes_step(state, self._sphere, pop_size=8, rng=rng)

        assert state.best_fitness < 0.1
        assert state.best_solution is not None
        assert np.linalg.norm(state.best_solution) < 1.0

    def test_warm_start_advantage(self) -> None:
        """Warm-starting near the optimum should converge faster."""
        from double_pendulum_golf.gui.optimization_widget import (
            CMAESState,
            _cmaes_step,
        )

        rng = np.random.default_rng(42)
        n = 4

        # Cold start far from optimum
        state_cold = CMAESState(
            mean=np.ones(n) * 10.0,
            sigma=5.0,
            C=np.eye(n),
            p_sigma=np.zeros(n),
            p_c=np.zeros(n),
        )

        # Warm start near optimum
        state_warm = CMAESState(
            mean=np.ones(n) * 0.5,
            sigma=1.0,
            C=np.eye(n),
            p_sigma=np.zeros(n),
            p_c=np.zeros(n),
        )

        for _ in range(20):
            state_cold, _ = _cmaes_step(state_cold, self._sphere, pop_size=10, rng=rng)
        rng_w = np.random.default_rng(42)
        for _ in range(20):
            state_warm, _ = _cmaes_step(
                state_warm, self._sphere, pop_size=10, rng=rng_w
            )

        assert state_warm.best_fitness < state_cold.best_fitness

    def test_stall_detection(self) -> None:
        """CMA-ES should detect stalling when fitness plateaus."""
        from double_pendulum_golf.gui.optimization_widget import (
            CMAESState,
            _cmaes_step,
        )

        rng = np.random.default_rng(42)
        n = 2

        # Already at optimum — should stall
        state = CMAESState(
            mean=np.zeros(n),
            sigma=0.001,
            C=np.eye(n) * 1e-8,
            p_sigma=np.zeros(n),
            p_c=np.zeros(n),
        )

        for _ in range(25):
            state, _ = _cmaes_step(state, self._sphere, pop_size=6, rng=rng)

        assert state.stall_count > 0


class TestNativeBackendAutoMode:
    """Verify the auto backend mode logic."""

    def test_backend_defaults_to_auto(self) -> None:
        """Backend mode should default to auto (rust when available)."""
        import os

        from double_pendulum_golf.native_backend import _backend_mode

        # Remove env vars to test default
        for key in [
            "PENDULUM_DOUBLE_BACKEND",
            "PENDULUM_TRIPLE_BACKEND",
            "PENDULUM_GOLFER_BACKEND",
        ]:
            os.environ.pop(key, None)

        mode = _backend_mode("PENDULUM_DOUBLE_BACKEND")
        assert mode in {"python", "rust"}

    def test_explicit_python_override(self) -> None:
        """Setting env to 'python' should force Python backend."""
        import os

        from double_pendulum_golf.native_backend import _backend_mode

        os.environ["PENDULUM_DOUBLE_BACKEND"] = "python"
        try:
            mode = _backend_mode("PENDULUM_DOUBLE_BACKEND")
            assert mode == "python"
        finally:
            del os.environ["PENDULUM_DOUBLE_BACKEND"]

    def test_explicit_auto_mode(self) -> None:
        """Setting env to 'auto' should auto-detect."""
        import os

        from double_pendulum_golf.native_backend import _backend_mode

        os.environ["PENDULUM_DOUBLE_BACKEND"] = "auto"
        try:
            mode = _backend_mode("PENDULUM_DOUBLE_BACKEND")
            assert mode in {"python", "rust"}
        finally:
            del os.environ["PENDULUM_DOUBLE_BACKEND"]

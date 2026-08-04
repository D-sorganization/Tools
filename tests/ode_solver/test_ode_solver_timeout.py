"""Tests for ODE solver timeout utility and load performance.

Covers:
- ``with_timeout``: DbC guards (TypeError / ValueError), happy path,
  timeout firing, exception propagation from the worker thread
- ``SolverTimeoutError``: public exception is a subclass of TimeoutError
- Load/performance: wall-clock timing for standard ODE solve calls to
  confirm typical workloads complete well within budget

Design principles:
- TDD: tests written against the ``timeout.py`` interface
- DbC: each test documents the precondition it exercises
- DRY: common fixture creates a simple ODE that always finishes quickly
- Orthogonality: timeout unit tests are independent of performance tests
"""

from __future__ import annotations

import time

import pytest
from ode_solver.timeout import SolverTimeoutError, with_timeout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fast_identity(x: int) -> int:
    """Return *x* immediately — used as a fast callable in unit tests."""
    return x


def _slow_forever() -> None:
    """Block indefinitely — used to verify timeout fires."""
    while True:  # noqa: PERF203
        time.sleep(0.01)


def _raise_value_error() -> None:
    """Always raises ValueError — used to verify exception propagation."""
    raise ValueError("deliberate error from worker")


# ---------------------------------------------------------------------------
# Unit tests: with_timeout DbC and happy path
# ---------------------------------------------------------------------------


class TestWithTimeoutDbC:
    """Verify precondition guards on ``with_timeout``."""

    def test_type_error_for_non_numeric_seconds(self) -> None:
        """Non-numeric seconds raises TypeError (DbC)."""
        with pytest.raises(TypeError, match="seconds must be a numeric value"):
            with_timeout("30", _fast_identity, 1)  # type: ignore[arg-type]

    def test_type_error_for_none_seconds(self) -> None:
        """None as seconds raises TypeError (DbC)."""
        with pytest.raises(TypeError, match="seconds must be a numeric value"):
            with_timeout(None, _fast_identity, 1)  # type: ignore[arg-type]

    def test_value_error_for_zero_seconds(self) -> None:
        """Zero seconds raises ValueError (DbC)."""
        with pytest.raises(ValueError, match="seconds must be > 0"):
            with_timeout(0, _fast_identity, 1)

    def test_value_error_for_negative_seconds(self) -> None:
        """Negative seconds raises ValueError (DbC)."""
        with pytest.raises(ValueError, match="seconds must be > 0"):
            with_timeout(-5.0, _fast_identity, 1)

    def test_type_error_for_non_callable_func(self) -> None:
        """Non-callable func raises TypeError (DbC)."""
        with pytest.raises(TypeError, match="func must be callable"):
            with_timeout(1.0, "not_callable")  # type: ignore[arg-type]

    def test_type_error_for_none_func(self) -> None:
        """None func raises TypeError (DbC)."""
        with pytest.raises(TypeError, match="func must be callable"):
            with_timeout(1.0, None)  # type: ignore[arg-type]


class TestWithTimeoutHappyPath:
    """Verify correct return values and exception propagation."""

    def test_returns_function_result(self) -> None:
        """with_timeout returns the callable's return value on success."""
        result = with_timeout(5.0, _fast_identity, 42)
        assert result == 42

    def test_accepts_int_seconds(self) -> None:
        """Integer seconds are accepted (int is a numeric type)."""
        result = with_timeout(5, _fast_identity, 99)
        assert result == 99

    def test_kwargs_forwarded(self) -> None:
        """Keyword arguments are forwarded to the wrapped callable."""

        def add(a: int, b: int = 0) -> int:
            return a + b

        result = with_timeout(5.0, add, 3, b=7)
        assert result == 10

    def test_propagates_worker_exception(self) -> None:
        """Exceptions from the worker are re-raised in the caller thread."""
        with pytest.raises(ValueError, match="deliberate error from worker"):
            with_timeout(5.0, _raise_value_error)

    def test_solver_timeout_error_is_timeout_error(self) -> None:
        """SolverTimeoutError is a subclass of the built-in TimeoutError."""
        assert issubclass(SolverTimeoutError, TimeoutError)

    def test_timeout_fires_for_slow_callable(self) -> None:
        """with_timeout raises SolverTimeoutError when deadline is exceeded."""
        with pytest.raises(SolverTimeoutError, match="timed out"):
            with_timeout(0.1, _slow_forever)  # 100 ms budget — fires quickly


# ---------------------------------------------------------------------------
# Load / performance tests (wall-clock timing)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.performance
class TestODESolverLoadPerformance:
    """Wall-clock timing tests for typical ODE solve workloads.

    These tests confirm that common ODE problems complete well within the
    30 s application timeout.  If a test fails here it means the scipy
    integration layer is unexpectedly slow — not a timeout.py issue.

    Uses ``time.perf_counter()`` rather than pytest-benchmark, consistent
    with the ``test_syngas_water_overflow.py`` convention.
    """

    @pytest.fixture(scope="class")
    def ode_solver(self) -> object:
        """Return an ODESolver instance if upstream_drift_tools is available."""
        try:
            from upstream_drift_tools.process_calculators.ode_solver import ODESolver

            return ODESolver
        except ImportError:
            pytest.skip("upstream_drift_tools not available in test environment")

    def test_exponential_decay_completes_in_budget(self, ode_solver: type) -> None:
        """Exponential decay solve completes in < 5 s (typical: < 0.1 s).

        Preconditions:
            ODESolver imported and available.
        """
        import numpy as np

        solver = ode_solver({"y": "-k*y"}, {"k": 0.1})
        t_eval = np.linspace(0, 50, 500)

        start = time.perf_counter()
        sol = solver.solve((0, 50), [100.0], t_eval=t_eval)
        elapsed = time.perf_counter() - start

        assert sol is not None
        assert (
            elapsed < 5.0
        ), f"Exponential decay solve took {elapsed:.3f} s, expected < 5 s"

    def test_harmonic_oscillator_completes_in_budget(self, ode_solver: type) -> None:
        """Harmonic oscillator solve completes in < 5 s (typical: < 0.1 s).

        Preconditions:
            ODESolver imported and available.
        """
        import numpy as np

        solver = ode_solver({"x": "v", "v": "-omega**2*x"}, {"omega": 1.0})
        t_eval = np.linspace(0, 50, 1000)

        start = time.perf_counter()
        sol = solver.solve((0, 50), [1.0, 0.0], t_eval=t_eval)
        elapsed = time.perf_counter() - start

        assert sol is not None
        assert (
            elapsed < 5.0
        ), f"Harmonic oscillator solve took {elapsed:.3f} s, expected < 5 s"

    def test_lotka_volterra_completes_in_budget(self, ode_solver: type) -> None:
        """Lotka-Volterra (predator-prey) solve completes in < 5 s.

        Preconditions:
            ODESolver imported and available.
        """
        import numpy as np

        solver = ode_solver(
            {"x": "a*x - b*x*y", "y": "-c*y + d*x*y"},
            {"a": 1.0, "b": 0.1, "c": 1.5, "d": 0.075},
        )
        t_eval = np.linspace(0, 100, 1000)

        start = time.perf_counter()
        sol = solver.solve((0, 100), [10.0, 5.0], t_eval=t_eval)
        elapsed = time.perf_counter() - start

        assert sol is not None
        assert (
            elapsed < 5.0
        ), f"Lotka-Volterra solve took {elapsed:.3f} s, expected < 5 s"

    def test_with_timeout_overhead_is_negligible(self) -> None:
        """with_timeout wrapper adds < 100 ms overhead for fast operations.

        Validates that the threading overhead does not materially affect
        performance of quick solver calls.
        """

        def trivial() -> int:
            return 42

        start = time.perf_counter()
        for _ in range(100):
            result = with_timeout(5.0, trivial)
        elapsed = time.perf_counter() - start

        assert result == 42
        avg_ms = (elapsed / 100) * 1000
        assert (
            avg_ms < 100
        ), f"with_timeout average overhead {avg_ms:.1f} ms/call, expected < 100 ms"

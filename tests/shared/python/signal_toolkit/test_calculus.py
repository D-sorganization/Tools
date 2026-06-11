"""Regression tests for assert-based precondition removal in calculus.py (#3344).

Issue #3344: bare ``assert`` statements on public-API boundaries in
``signal_toolkit.calculus`` vanish under ``python -O``, silently accepting
``None`` and producing opaque ``AttributeError``\\s later.

These tests verify that:
1. Every converted entry-point raises ``ValueError`` (not ``AssertionError``)
   when given ``None``.
2. The guards survive ``python -O``  — because they are ``if``/``raise``, not
   bare ``assert``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from signal_toolkit.calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegrationMethod,
    Integrator,
    compute_all_tangent_lines,
    compute_arc_length,
    compute_curvature,
    compute_derivative,
    compute_tangent_line,
    find_extrema,
    find_inflection_points,
)
from signal_toolkit.core import Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal() -> Signal:
    """Return a simple test signal."""
    t = np.linspace(0.0, 10.0, 201)
    return Signal(t, np.sin(t))


# ---------------------------------------------------------------------------
# Differentiator.__init__ — method=None
# ---------------------------------------------------------------------------


class TestDifferentiatorInit:
    """Differentiator() raises ValueError when method is None (#3344)."""

    def test_none_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="method must be provided"):
            Differentiator(method=None)  # type: ignore[arg-type]

    def test_valid_method_accepted(self) -> None:
        d = Differentiator(method=DifferentiationMethod.SAVGOL)
        assert d.method == DifferentiationMethod.SAVGOL


# ---------------------------------------------------------------------------
# Differentiator.differentiate — signal=None
# ---------------------------------------------------------------------------


class TestDifferentiatePreconditions:
    """Differentiator.differentiate raises ValueError when signal is None (#3344)."""

    def test_none_signal_raises_value_error(self) -> None:
        d = Differentiator()
        with pytest.raises(ValueError, match="signal must be provided"):
            d.differentiate(None)  # type: ignore[arg-type]

    def test_valid_signal_returns_signal(self) -> None:
        result = Differentiator().differentiate(_make_signal())
        assert result is not None


# ---------------------------------------------------------------------------
# Differentiator.compute_at_point — signal=None
# ---------------------------------------------------------------------------


class TestComputeAtPointPreconditions:
    """Differentiator.compute_at_point raises ValueError when signal is None (#3344)."""

    def test_none_signal_raises_value_error(self) -> None:
        d = Differentiator()
        with pytest.raises(ValueError, match="signal must be provided"):
            d.compute_at_point(None, t_point=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integrator.__init__ — method=None
# ---------------------------------------------------------------------------


class TestIntegratorInit:
    """Integrator() raises ValueError when method is None (#3344)."""

    def test_none_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="method must be provided"):
            Integrator(method=None)  # type: ignore[arg-type]

    def test_valid_method_accepted(self) -> None:
        i = Integrator(method=IntegrationMethod.TRAPEZOID)
        assert i.method == IntegrationMethod.TRAPEZOID


# ---------------------------------------------------------------------------
# Integrator.cumulative_integral — signal=None
# ---------------------------------------------------------------------------


class TestCumulativeIntegralPreconditions:
    """Integrator.cumulative_integral raises ValueError when signal is None (#3344)."""

    def test_none_signal_raises_value_error(self) -> None:
        integrator = Integrator()
        with pytest.raises(ValueError, match="signal must be provided"):
            integrator.cumulative_integral(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Module-level functions — signal=None
# ---------------------------------------------------------------------------


class TestModuleFunctionPreconditions:
    """Module-level funcs raise ValueError when signal is None (#3344)."""

    def test_compute_derivative_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_derivative(None)  # type: ignore[arg-type]

    def test_compute_tangent_line_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_tangent_line(None, t_point=1.0)  # type: ignore[arg-type]

    def test_compute_all_tangent_lines_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_all_tangent_lines(None)  # type: ignore[arg-type]

    def test_compute_curvature_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_curvature(None)  # type: ignore[arg-type]

    def test_compute_arc_length_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_arc_length(None)  # type: ignore[arg-type]

    def test_find_extrema_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            find_extrema(None)  # type: ignore[arg-type]

    def test_find_inflection_points_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            find_inflection_points(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# python -O guard survival test
# ---------------------------------------------------------------------------


class TestGuardsSurviveOptimizedMode:
    """Guards are ``if``/``raise`` (not bare ``assert``) — survive ``-O`` (#3344)."""

    def test_compute_derivative_raises_under_minus_o(self) -> None:
        """Run ``python -O`` in a subprocess; expect ValueError, not AttributeError."""
        code = (
            "from signal_toolkit.calculus import compute_derivative; "
            "compute_derivative(None)"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Expected non-zero exit (exception raised)"
        assert "ValueError" in result.stderr, (
            f"Expected ValueError in stderr, got:\n{result.stderr}"
        )
        assert "AttributeError" not in result.stderr, (
            "Got AttributeError — guard is still a bare assert (issue #3344)"
        )

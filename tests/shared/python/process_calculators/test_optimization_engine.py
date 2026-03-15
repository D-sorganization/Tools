"""Tests for the optimization.py helpers used by the advanced plots tab.

Tests cover:
- Surface optimisation (grid search, L-BFGS-B, DE) on synthetic numpy surfaces
- Adam parameter-update internals (_adam_update, _init_adam_state)
- _build_override_mapping filtering logic
- Edge cases: empty configs, unknown methods, callback invocation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.optimization import (
    OptimizationHistoryEntry,
    _adam_update,
    _AdamState,
    _build_override_mapping,
    _init_adam_state,
    find_optimal_on_surface,
)

# ============================================================================
# FIXTURES
# ============================================================================


def _make_surface(
    n_x: int = 20, n_y: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple quadratic surface: z = -(x-3)^2 - (y-4)^2, max at (3,4)."""
    x = np.linspace(0, 6, n_x)
    y = np.linspace(0, 8, n_y)
    xx, yy = np.meshgrid(x, y)  # shape (n_y, n_x)
    zz = -((xx - 3) ** 2) - (yy - 4) ** 2
    return x, y, zz  # z is (n_y, n_x)


# ============================================================================
# _build_override_mapping
# ============================================================================


class TestBuildOverrideMapping:
    def test_only_known_params_included(self) -> None:
        names = ["Temperature", "O2/Feed Ratio", "SomethingElse"]
        values = [800.0, 0.25, 99.0]
        mapping = _build_override_mapping(names, values)
        assert "Temperature" in mapping
        assert "O2/Feed Ratio" in mapping
        assert "SomethingElse" not in mapping

    def test_known_params_have_correct_values(self) -> None:
        names = ["Temperature", "Pressure", "Steam/Feed Ratio"]
        values = [1000.0, 5.0, 0.5]
        mapping = _build_override_mapping(names, values)
        assert mapping["Temperature"] == pytest.approx(1000.0)
        assert mapping["Pressure"] == pytest.approx(5.0)
        assert mapping["Steam/Feed Ratio"] == pytest.approx(0.5)

    def test_empty_inputs_empty_output(self) -> None:
        assert _build_override_mapping([], []) == {}

    def test_no_known_params_returns_empty(self) -> None:
        mapping = _build_override_mapping(["alpha", "beta"], [1.0, 2.0])
        assert mapping == {}


# ============================================================================
# Surface optimisation: find_optimal_on_surface
# ============================================================================


class TestFindOptimalOnSurfaceGridSearch:
    def test_finds_known_maximum(self) -> None:
        x, y, z = _make_surface(25, 25)
        result = find_optimal_on_surface(x, y, z, method="Grid Search")

        assert result["success"] is True
        # optimal should be close to (3, 4) with 25 pts over [0,6] and [0,8]
        assert result["optimal_x"] == pytest.approx(3.0, abs=0.5)
        assert result["optimal_y"] == pytest.approx(4.0, abs=0.5)
        assert result["optimal_z"] > -1.0  # near zero at peak

    def test_evaluations_count_positive(self) -> None:
        x, y, z = _make_surface(10, 10)
        result = find_optimal_on_surface(x, y, z, method="Grid Search")
        assert result["evaluations"] > 0

    def test_callback_is_invoked(self) -> None:
        x, y, z = _make_surface(25, 25)
        counts: list[int] = []

        def _cb(done: int, total: int) -> None:
            counts.append(done)

        find_optimal_on_surface(x, y, z, method="Grid Search", callback=_cb)
        assert len(counts) > 0  # callback called at least once

    def test_custom_bounds(self) -> None:
        x, y, z = _make_surface(25, 25)
        bounds = ((0.0, 2.0), (0.0, 2.0))  # region away from true optimum
        result = find_optimal_on_surface(x, y, z, method="Grid Search", bounds=bounds)
        assert result["success"] is True
        # optimal within supplied bounds
        assert 0.0 <= result["optimal_x"] <= 2.0
        assert 0.0 <= result["optimal_y"] <= 2.0

    def test_unknown_method_raises(self) -> None:
        x, y, z = _make_surface(10, 10)
        with pytest.raises(ValueError, match="Unknown optimization method"):
            find_optimal_on_surface(x, y, z, method="magic_algo")

    def test_mismatched_z_shape_raises(self) -> None:
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        z_bad = np.zeros((3, 4))  # wrong shape
        with pytest.raises(ValueError, match="does not match"):
            find_optimal_on_surface(x, y, z_bad, method="Grid Search")


class TestFindOptimalOnSurfaceLBFGSB:
    def test_finds_reasonable_maximum(self) -> None:
        x, y, z = _make_surface(20, 20)
        result = find_optimal_on_surface(x, y, z, method="L-BFGS-B")
        assert "optimal_x" in result
        assert "optimal_y" in result
        # z should be near maximum (which is 0 at center)
        assert result["optimal_z"] > -2.0  # reasonably close

    def test_evaluations_nonzero(self) -> None:
        x, y, z = _make_surface(15, 15)
        result = find_optimal_on_surface(x, y, z, method="L-BFGS-B")
        assert result["evaluations"] > 0


class TestFindOptimalOnSurfaceDifferentialEvolution:
    def test_finds_reasonable_maximum(self) -> None:
        x, y, z = _make_surface(20, 20)
        result = find_optimal_on_surface(x, y, z, method="Differential Evolution")
        assert "optimal_x" in result
        assert "optimal_y" in result
        assert result["optimal_z"] > -3.0

    def test_reports_evaluations(self) -> None:
        x, y, z = _make_surface(10, 10)
        result = find_optimal_on_surface(x, y, z, method="Differential Evolution")
        assert result["evaluations"] > 0


# ============================================================================
# Adam internals
# ============================================================================


def _make_adam_state(
    names: list[str],
    lows: list[float],
    highs: list[float],
    vals: list[float],
    *,
    maximize: bool = True,
) -> _AdamState:
    """Helper to build a minimal _AdamState for testing."""
    arr = np.array(vals, dtype=float)
    lo = np.array(lows, dtype=float)
    hi = np.array(highs, dtype=float)
    return _AdamState(
        parameter_names=names,
        lower_bounds=lo,
        upper_bounds=hi,
        values=arr.copy(),
        m=np.zeros_like(arr),
        v=np.zeros_like(arr),
        best_output=-np.inf if maximize else np.inf,
        best_parameters={},
        best_state={},
        best_composition={},
        history=[],
        previous_values=arr.copy(),
        base_params={},
        output_name="syngas_efficiency",
    )


class TestAdamUpdate:
    def test_positive_gradient_increases_value_when_maximizing(self) -> None:
        """With a positive gradient and maximize=True, value should increase."""
        st = _make_adam_state(
            ["Temperature"], [500.0], [2000.0], [1000.0], maximize=True
        )
        grad = np.array([1.0])
        _adam_update(
            st,
            grad,
            1,
            maximize=True,
            learning_rate=10.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        assert st.values[0] > 1000.0

    def test_values_clipped_to_bounds(self) -> None:
        """Adam must not push values outside [lower, upper]."""
        st = _make_adam_state(
            ["Temperature"], [500.0], [1000.0], [999.0], maximize=True
        )
        # Massive gradient to push beyond upper bound
        grad = np.array([1e6])
        _adam_update(
            st,
            grad,
            1,
            maximize=True,
            learning_rate=1e5,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        assert st.values[0] <= 1000.0

    def test_minimize_moves_away_from_gradient(self) -> None:
        """When minimize=True, positive gradient → value decreases."""
        st = _make_adam_state(
            ["Temperature"], [0.0], [2000.0], [1000.0], maximize=False
        )
        grad = np.array([1.0])
        _adam_update(
            st,
            grad,
            1,
            maximize=False,
            learning_rate=10.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )
        assert st.values[0] < 1000.0


class TestInitAdamState:
    def test_initialises_vectors(self) -> None:
        analysis_params: dict[str, Any] = {
            "base_params": {"Temperature": 800.0},
            "output_variable": "syn_efficiency",
        }
        configs = [
            {"name": "Temperature", "min": 500.0, "max": 1500.0, "initial": 800.0},
            {"name": "O2/Feed Ratio", "min": 0.2, "max": 0.6, "initial": 0.4},
        ]
        st = _init_adam_state(analysis_params, configs, maximize=True)

        assert len(st.parameter_names) == 2
        assert st.values.shape == (2,)
        assert st.m.shape == (2,)
        assert st.v.shape == (2,)
        assert st.best_output == -np.inf  # maximize mode

    def test_minimize_mode_sets_inf_lower(self) -> None:
        analysis_params: dict[str, Any] = {
            "base_params": {},
            "output_variable": "cost",
        }
        configs = [
            {"name": "Pressure", "min": 1.0, "max": 10.0, "initial": 5.0},
        ]
        st = _init_adam_state(analysis_params, configs, maximize=False)
        assert st.best_output == np.inf  # minimize mode


class TestOptimizationHistoryEntry:
    def test_dataclass_fields(self) -> None:
        entry = OptimizationHistoryEntry(
            iteration=3,
            objective=0.87,
            parameters={"Temperature": 1000.0},
        )
        assert entry.iteration == 3
        assert entry.objective == pytest.approx(0.87)
        assert entry.parameters["Temperature"] == pytest.approx(1000.0)

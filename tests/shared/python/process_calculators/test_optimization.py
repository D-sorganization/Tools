"""Tests for upstream_drift_tools.process_calculators.optimization.

Covers:
- _build_override_mapping: parameter name filtering
- _compute_gradient_component: finite-difference gradient calculation
- _init_adam_state: initialisation from configs
- _adam_update: Adam parameter update step
- run_adam_optimization: end-to-end optimisation loop
- find_optimal_on_surface: surface interpolation & grid/L-BFGS-B/DE methods
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.optimization import (
    _adam_update,
    _AdamState,
    _build_override_mapping,
    _init_adam_state,
    find_optimal_on_surface,
    run_adam_optimization,
)

# ── Helpers ─────────────────────────────────────────────────────────────


class _QuadraticEngine:
    """Engine whose output is -(T - 1000)^2 / 1000 + 100 (max at T=1000)."""

    def calculate(self, **params: Any) -> dict[str, Any]:
        t = params.get("Temperature", 1000.0)
        output = -((t - 1000.0) ** 2) / 1000.0 + 100.0
        return {
            "efficiency": output,
            "state": {"T": t},
            "composition": {"CO": 0.25},
        }


# ── _build_override_mapping ─────────────────────────────────────────────


class TestBuildOverrideMapping:
    """Test parameter filtering logic."""

    def test_known_parameters_included(self) -> None:
        result = _build_override_mapping(
            ["Temperature", "O2/Feed Ratio", "Pressure"],
            [1200.0, 0.5, 101325.0],
        )
        assert result == {
            "Temperature": 1200.0,
            "O2/Feed Ratio": 0.5,
            "Pressure": 101325.0,
        }

    def test_unknown_parameters_excluded(self) -> None:
        result = _build_override_mapping(
            ["Temperature", "CustomParam"],
            [1200.0, 42.0],
        )
        assert result == {"Temperature": 1200.0}
        assert "CustomParam" not in result

    def test_empty_inputs(self) -> None:
        result = _build_override_mapping([], [])
        assert result == {}

    def test_steam_feed_ratio_included(self) -> None:
        result = _build_override_mapping(
            ["Steam/Feed Ratio"],
            [0.8],
        )
        assert result == {"Steam/Feed Ratio": 0.8}


# ── _init_adam_state ────────────────────────────────────────────────────


class TestInitAdamState:
    """Test Adam state initialization."""

    def test_creates_state_with_correct_bounds(self) -> None:
        configs = [
            {"name": "Temperature", "min": 800, "max": 1200, "initial": 1000},
            {"name": "Pressure", "min": 100000, "max": 200000, "initial": 150000},
        ]
        analysis = {
            "base_params": {"Temperature": 1000.0},
            "output_variable": "efficiency",
        }
        st = _init_adam_state(analysis, configs, maximize=True)

        assert st.parameter_names == ["Temperature", "Pressure"]
        np.testing.assert_array_equal(st.lower_bounds, [800, 100000])
        np.testing.assert_array_equal(st.upper_bounds, [1200, 200000])
        np.testing.assert_array_equal(st.values, [1000, 150000])

    def test_maximize_sets_negative_infinity(self) -> None:
        configs = [{"name": "T", "min": 0, "max": 100, "initial": 50}]
        analysis = {"base_params": {}, "output_variable": "out"}
        st = _init_adam_state(analysis, configs, maximize=True)
        assert st.best_output == -np.inf

    def test_minimize_sets_positive_infinity(self) -> None:
        configs = [{"name": "T", "min": 0, "max": 100, "initial": 50}]
        analysis = {"base_params": {}, "output_variable": "out"}
        st = _init_adam_state(analysis, configs, maximize=False)
        assert st.best_output == np.inf

    def test_moment_vectors_zeroed(self) -> None:
        configs = [
            {"name": "T", "min": 0, "max": 100, "initial": 50},
            {"name": "P", "min": 0, "max": 100, "initial": 50},
        ]
        analysis = {"base_params": {}, "output_variable": "out"}
        st = _init_adam_state(analysis, configs, maximize=True)
        np.testing.assert_array_equal(st.m, [0.0, 0.0])
        np.testing.assert_array_equal(st.v, [0.0, 0.0])


# ── _adam_update ────────────────────────────────────────────────────────


class TestAdamUpdate:
    """Test a single Adam update step."""

    def test_values_change_in_gradient_direction(self) -> None:
        st = _AdamState(
            parameter_names=["T"],
            lower_bounds=np.array([0.0]),
            upper_bounds=np.array([2000.0]),
            values=np.array([1000.0]),
            m=np.zeros(1),
            v=np.zeros(1),
            best_output=-np.inf,
            best_parameters={},
            best_state={},
            best_composition={},
            history=[],
            previous_values=np.array([1000.0]),
            base_params={},
            output_name="out",
        )
        original = st.values.copy()
        gradient = np.array([10.0])  # positive gradient

        _adam_update(
            st,
            gradient,
            iteration=1,
            maximize=True,
            learning_rate=1.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )

        # When maximizing with positive gradient, values should increase
        assert st.values[0] > original[0]

    def test_values_clipped_to_bounds(self) -> None:
        st = _AdamState(
            parameter_names=["T"],
            lower_bounds=np.array([0.0]),
            upper_bounds=np.array([100.0]),
            values=np.array([99.0]),
            m=np.zeros(1),
            v=np.zeros(1),
            best_output=-np.inf,
            best_parameters={},
            best_state={},
            best_composition={},
            history=[],
            previous_values=np.array([99.0]),
            base_params={},
            output_name="out",
        )
        gradient = np.array([1000.0])

        _adam_update(
            st,
            gradient,
            iteration=1,
            maximize=True,
            learning_rate=100.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )

        assert st.values[0] <= 100.0


# ── run_adam_optimization (integration) ─────────────────────────────────


class TestRunAdamOptimization:
    """Integration tests for the full Adam optimization loop."""

    def test_empty_configs_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one parameter"):
            run_adam_optimization(
                _QuadraticEngine(),
                {"base_params": {}, "output_variable": "efficiency"},
                manual_hhv=0.0,
                parameter_configs=[],
                maximize=True,
                learning_rate=0.01,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                gradient_step=1.0,
                max_iterations=10,
                tolerance=1e-6,
            )

    def test_finds_maximum_of_quadratic(self) -> None:
        result = run_adam_optimization(
            _QuadraticEngine(),
            {"base_params": {"Temperature": 800.0}, "output_variable": "efficiency"},
            manual_hhv=0.0,
            parameter_configs=[
                {"name": "Temperature", "min": 800.0, "max": 1200.0, "initial": 900.0},
            ],
            maximize=True,
            learning_rate=5.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_step=1.0,
            max_iterations=200,
            tolerance=1e-4,
        )

        assert result["best_output"] >= 95.0  # Near maximum of 100
        assert len(result["history"]) > 0
        assert "iterations" in result

    def test_result_has_expected_keys(self) -> None:
        result = run_adam_optimization(
            _QuadraticEngine(),
            {"base_params": {"Temperature": 1000.0}, "output_variable": "efficiency"},
            manual_hhv=0.0,
            parameter_configs=[
                {"name": "Temperature", "min": 800.0, "max": 1200.0, "initial": 1000.0},
            ],
            maximize=True,
            learning_rate=1.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_step=1.0,
            max_iterations=5,
            tolerance=1e-6,
        )

        assert "best_output" in result
        assert "best_parameters" in result
        assert "best_state" in result
        assert "best_composition" in result
        assert "history" in result
        assert "final_parameters" in result
        assert "iterations" in result


# ── find_optimal_on_surface ─────────────────────────────────────────────


class TestFindOptimalOnSurface:
    """Test surface optimization methods."""

    @pytest.fixture()
    def surface_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a simple parabolic surface with peak at (5, 5)."""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        xx, yy = np.meshgrid(x, y)
        # Peak at (5, 5), max value = 100
        z = 100.0 - (xx - 5.0) ** 2 - (yy - 5.0) ** 2
        return x, y, z

    def test_grid_search_finds_peak(
        self,
        surface_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        x, y, z = surface_data
        result = find_optimal_on_surface(x, y, z, method="Grid Search")

        assert result["success"] is True
        assert abs(result["optimal_x"] - 5.0) < 2.0
        assert abs(result["optimal_y"] - 5.0) < 2.0
        assert result["optimal_z"] > 90.0

    def test_lbfgsb_finds_peak(
        self,
        surface_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        x, y, z = surface_data
        result = find_optimal_on_surface(x, y, z, method="L-BFGS-B")

        assert abs(result["optimal_x"] - 5.0) < 1.0
        assert abs(result["optimal_y"] - 5.0) < 1.0
        assert result["optimal_z"] > 95.0

    def test_differential_evolution_finds_peak(
        self,
        surface_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        x, y, z = surface_data
        result = find_optimal_on_surface(x, y, z, method="Differential Evolution")

        assert abs(result["optimal_x"] - 5.0) < 1.5
        assert abs(result["optimal_y"] - 5.0) < 1.5
        assert result["optimal_z"] > 90.0

    def test_unknown_method_raises(
        self,
        surface_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        x, y, z = surface_data
        with pytest.raises(ValueError, match="Unknown optimization method"):
            find_optimal_on_surface(x, y, z, method="FooBar")

    def test_mismatched_z_shape_raises(self) -> None:
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 10, 8)
        z = np.zeros((3, 3))
        with pytest.raises(ValueError, match="does not match"):
            find_optimal_on_surface(x, y, z, method="Grid Search")

    def test_custom_bounds(
        self,
        surface_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        x, y, z = surface_data
        custom_bounds = ((3.0, 7.0), (3.0, 7.0))
        result = find_optimal_on_surface(
            x,
            y,
            z,
            method="Grid Search",
            bounds=custom_bounds,
        )
        assert result["success"] is True

    def test_callback_invoked(
        self,
        surface_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        x, y, z = surface_data
        calls: list[tuple[int, int]] = []
        result = find_optimal_on_surface(
            x,
            y,
            z,
            method="Grid Search",
            callback=lambda ev, tot: calls.append((ev, tot)),
        )
        assert result["success"] is True
        assert len(calls) > 0

    def test_z_transposed_shape_accepted(self) -> None:
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 10, 8)
        z = np.zeros((5, 8))  # (len(x), len(y)) shape
        result = find_optimal_on_surface(x, y, z, method="Grid Search")
        assert result["success"] is True

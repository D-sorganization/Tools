"""Tests for upstream_drift_tools.process_calculators.multi_param_analysis.

Covers:
- run_multi_parameter_analysis: grid sweep, result shape, parameter names
- _evaluate_single_point: individual point evaluation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.multi_param_analysis import (
    _evaluate_single_point,
    run_multi_parameter_analysis,
)

# ── Fixtures ────────────────────────────────────────────────────────────


class _GridEngine:
    """Engine that returns a deterministic output based on parameters."""

    def calculate(self, **params: Any) -> dict[str, Any]:
        t = params.get("temperature", 0.0)
        p = params.get("pressure", 0.0)
        # Simple function: output = T + 2*P
        return {"efficiency": t + 2.0 * p}


@pytest.fixture()
def engine() -> _GridEngine:
    return _GridEngine()


@pytest.fixture()
def analysis_params() -> dict[str, Any]:
    return {
        "base_params": {"temperature": 500.0, "pressure": 101325.0},
        "param1_name": "temperature",
        "param2_name": "pressure",
        "output_variable": "efficiency",
    }


# ── Sequential analysis ────────────────────────────────────────────────


class TestRunMultiParameterAnalysis:
    """Test the sequential multi-parameter sweep."""

    def test_result_shape(self, engine: _GridEngine, analysis_params: dict[str, Any]) -> None:
        p1 = np.array([400.0, 500.0, 600.0])
        p2 = np.array([100000.0, 200000.0])

        result = run_multi_parameter_analysis(
            engine,
            analysis_params,
            manual_hhv=0.0,
            param1_values=p1,
            param2_values=p2,
        )

        assert result["output_values"].shape == (3, 2)

    def test_output_values_correct(
        self, engine: _GridEngine, analysis_params: dict[str, Any]
    ) -> None:
        p1 = np.array([100.0, 200.0])
        p2 = np.array([10.0, 20.0])

        result = run_multi_parameter_analysis(
            engine,
            analysis_params,
            manual_hhv=0.0,
            param1_values=p1,
            param2_values=p2,
        )

        # output = T + 2*P
        expected = np.array(
            [
                [100.0 + 20.0, 100.0 + 40.0],
                [200.0 + 20.0, 200.0 + 40.0],
            ]
        )
        np.testing.assert_array_almost_equal(result["output_values"], expected)

    def test_result_keys_present(
        self, engine: _GridEngine, analysis_params: dict[str, Any]
    ) -> None:
        p1 = np.array([500.0])
        p2 = np.array([100000.0])

        result = run_multi_parameter_analysis(
            engine,
            analysis_params,
            manual_hhv=0.0,
            param1_values=p1,
            param2_values=p2,
        )

        assert "param1_values" in result
        assert "param2_values" in result
        assert "output_values" in result
        assert "param1_name" in result
        assert "param2_name" in result
        assert "output_name" in result
        assert "output_data" in result
        assert "convergence_map" in result

    def test_param_names_in_result(
        self, engine: _GridEngine, analysis_params: dict[str, Any]
    ) -> None:
        result = run_multi_parameter_analysis(
            engine,
            analysis_params,
            manual_hhv=0.0,
            param1_values=np.array([500.0]),
            param2_values=np.array([100000.0]),
        )

        assert result["param1_name"] == "temperature"
        assert result["param2_name"] == "pressure"
        assert result["output_name"] == "efficiency"

    def test_convergence_map_all_ones(
        self, engine: _GridEngine, analysis_params: dict[str, Any]
    ) -> None:
        p1 = np.array([400.0, 500.0])
        p2 = np.array([100000.0, 200000.0])

        result = run_multi_parameter_analysis(
            engine,
            analysis_params,
            manual_hhv=0.0,
            param1_values=p1,
            param2_values=p2,
        )

        np.testing.assert_array_equal(result["convergence_map"], np.ones((2, 2)))

    def test_single_point_grid(self, engine: _GridEngine, analysis_params: dict[str, Any]) -> None:
        result = run_multi_parameter_analysis(
            engine,
            analysis_params,
            manual_hhv=0.0,
            param1_values=np.array([100.0]),
            param2_values=np.array([50.0]),
        )

        assert result["output_values"].shape == (1, 1)
        assert result["output_values"][0, 0] == pytest.approx(200.0)  # 100 + 2*50


# ── Single point evaluation ────────────────────────────────────────────


class TestEvaluateSinglePoint:
    """Test the helper used by the parallel version."""

    def test_returns_indices_and_value(self, engine: _GridEngine) -> None:
        i, j, output = _evaluate_single_point(
            i=2,
            j=3,
            p1=100.0,
            p2=50.0,
            engine=engine,
            base={"temperature": 0.0, "pressure": 0.0},
            manual_hhv=0.0,
            param1_name="temperature",
            param2_name="pressure",
            output_variable="efficiency",
        )

        assert i == 2
        assert j == 3
        assert output == pytest.approx(200.0)  # 100 + 2*50

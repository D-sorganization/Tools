"""Tests for upstream_drift_tools.process_calculators.analysis_utils.

Covers:
- evaluate_output: parameter merging, output extraction, error handling
- engine calculation delegation
- HHV injection
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
from upstream_drift_tools.process_calculators.analysis_utils import evaluate_output

# ── Fixtures ────────────────────────────────────────────────────────────


class _StubEngine:
    """Minimal engine that records calls and returns configurable results."""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {}
        self.last_params: dict[str, Any] = {}

    def calculate(self, **params: Any) -> dict[str, Any]:
        self.last_params = params
        return self.result


class _ExplodingEngine:
    """Engine that always raises."""

    def calculate(self, **params: Any) -> dict[str, Any]:
        msg = "boom"
        raise ValueError(msg)


@pytest.fixture()
def engine() -> _StubEngine:
    return _StubEngine(
        result={
            "temperature": 1200.0,
            "pressure": 101325.0,
            "efficiency": 0.87,
            "state": {"T": 1200.0, "P": 101325.0},
            "composition": {"CO": 0.25, "H2": 0.35},
        }
    )


# ── Basic extraction ────────────────────────────────────────────────────


class TestEvaluateOutputBasic:
    """Test the happy-path behaviour of evaluate_output."""

    def test_extracts_named_output(self, engine: _StubEngine) -> None:
        value, state, comp = evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="efficiency",
        )
        assert value == pytest.approx(0.87)

    def test_returns_state_and_composition(self, engine: _StubEngine) -> None:
        _, state, comp = evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="efficiency",
        )
        assert state == {"T": 1200.0, "P": 101325.0}
        assert comp == {"CO": 0.25, "H2": 0.35}

    def test_missing_output_variable_returns_nan(self, engine: _StubEngine) -> None:
        """A missing output key is a failure, not an objective of 0.0 (#3976).

        This test previously asserted ``value == 0.0`` and so *cemented* the
        defect: a typo'd ``output_variable`` produced a plausible objective of
        zero, the gradient estimator saw 0.0 at both perturbed points, computed
        an exactly-zero gradient, and reported convergence on garbage.
        """
        value, state, comp = evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="nonexistent",
        )
        assert math.isnan(value)
        assert state == {}
        assert comp == {}

    def test_non_numeric_output_returns_nan(self) -> None:
        """A non-numeric value under the requested key is a failure (#3976)."""
        engine = _StubEngine(result={"efficiency": "not-a-number"})
        value, state, comp = evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="efficiency",
        )
        assert math.isnan(value)
        assert state == {}
        assert comp == {}

    def test_legitimate_zero_output_is_preserved(self) -> None:
        """0.0 must still round-trip: it is a valid objective, not a sentinel."""
        engine = _StubEngine(result={"efficiency": 0.0})
        value, _, _ = evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="efficiency",
        )
        assert value == 0.0
        assert not math.isnan(value)


# ── Parameter merging ───────────────────────────────────────────────────


class TestParameterHandling:
    """Test parameter merging and HHV injection."""

    def test_overrides_merge_with_base(self, engine: _StubEngine) -> None:
        evaluate_output(
            engine,
            {"T_in": 500.0, "P_in": 101325.0},
            manual_hhv=0.0,
            output_variable="temperature",
            overrides={"T_in": 600.0},
        )
        assert engine.last_params["T_in"] == 600.0
        assert engine.last_params["P_in"] == 101325.0

    def test_base_params_not_mutated(self, engine: _StubEngine) -> None:
        base = {"T_in": 500.0}
        evaluate_output(
            engine,
            base,
            manual_hhv=0.0,
            output_variable="temperature",
            overrides={"T_in": 600.0},
        )
        assert base["T_in"] == 500.0  # original unchanged

    def test_hhv_injected_when_positive(self, engine: _StubEngine) -> None:
        evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=12.5,
            output_variable="temperature",
        )
        assert engine.last_params["manual_hhv"] == 12.5

    def test_hhv_not_injected_when_zero(self, engine: _StubEngine) -> None:
        evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="temperature",
        )
        assert "manual_hhv" not in engine.last_params

    def test_no_overrides_passes_base_only(self, engine: _StubEngine) -> None:
        evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="temperature",
        )
        assert engine.last_params == {"T_in": 500.0}


# ── Error handling ──────────────────────────────────────────────────────


class TestErrorHandling:
    """Test that evaluate_output gracefully handles failures.

    Regression coverage for issue #3976: a failed evaluation must return
    NaN, not 0.0. Both callers (optimization.py's gradient estimator and
    objective evaluator, multi_param_analysis's grid sweep) already check
    ``np.isfinite(...)`` to detect a failed point and apply their own
    fallback/penalty — a silent 0.0 masqueraded a failure as a real answer
    and bypassed that handling entirely.
    """

    def test_engine_raises_returns_nan(self) -> None:
        value, state, comp = evaluate_output(
            _ExplodingEngine(),
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="efficiency",
        )
        assert math.isnan(value)
        assert state == {}
        assert comp == {}

    def test_engine_returns_non_dict(self) -> None:
        engine = MagicMock()
        engine.calculate.return_value = "not a dict"
        value, state, comp = evaluate_output(
            engine,
            {"T_in": 500.0},
            manual_hhv=0.0,
            output_variable="efficiency",
        )
        assert math.isnan(value)
        assert state == {}

    def test_missing_state_returns_empty_dict(self) -> None:
        engine = _StubEngine(result={"efficiency": 0.5})
        _, state, comp = evaluate_output(
            engine,
            {},
            manual_hhv=0.0,
            output_variable="efficiency",
        )
        assert state == {}
        assert comp == {}

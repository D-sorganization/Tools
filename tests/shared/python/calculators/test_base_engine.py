"""Tests for upstream_drift_tools.calculators.base.

Covers:
- BaseCalculationEngine cannot be instantiated directly
- Subclasses must implement calculate()
- Subclass calculate() returns expected dict
"""

from __future__ import annotations

from typing import Any

import pytest
from upstream_drift_tools.calculators.base import BaseCalculationEngine

# ── Concrete subclass for testing ───────────────────────────────────────


class _ConcreteEngine(BaseCalculationEngine):
    """Minimal working implementation."""

    def calculate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"result": sum(args) + sum(kwargs.values())}


class _IncompleteEngine(BaseCalculationEngine):
    """Subclass that does NOT implement calculate()."""


# ── Tests ───────────────────────────────────────────────────────────────


class TestBaseCalculationEngine:
    """Test the abstract base class contract."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseCalculationEngine()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            _IncompleteEngine()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        engine = _ConcreteEngine()
        result = engine.calculate(1, 2, 3, bonus=4)
        assert result == {"result": 10}

    def test_calculate_returns_dict(self) -> None:
        engine = _ConcreteEngine()
        result = engine.calculate()
        assert isinstance(result, dict)

    def test_is_subclass(self) -> None:
        assert issubclass(_ConcreteEngine, BaseCalculationEngine)

    def test_isinstance_check(self) -> None:
        engine = _ConcreteEngine()
        assert isinstance(engine, BaseCalculationEngine)

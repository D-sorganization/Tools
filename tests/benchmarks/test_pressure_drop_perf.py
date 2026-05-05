"""Performance benchmarks for pressure drop calculator.

Measures the performance of the PressureDropCalculator hot path under
various conditions. SLA target: < 100ms per calculation.

Uses upstream_drift_tools.process_calculators.pressure_drop_calculator
which is the shared library used by the calc_backend and downstream repos.

Issue #2413 — systematic performance benchmarking.
"""

from __future__ import annotations

import pytest

try:
    from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
        PressureDropCalculator,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip(
        "upstream_drift_tools.process_calculators not available",
        allow_module_level=True,
    )


pytestmark = pytest.mark.benchmark

# Shared calculator instance — reuse across benchmarks (matches production usage).
_CALCULATOR = PressureDropCalculator()

# Baseline parameters for a 4-inch gas line, typical process conditions.
_BASE_PARAMS: dict = {
    "pipe_diameter_m": 0.1,
    "pipe_length_m": 100.0,
    "roughness_m": 0.000045,
    "flow_rate_kg_s": 1.0,
    "temperature_k": 400.0,
    "pressure_pa": 500_000.0,
    "molecular_weight_kg_mol": 0.029,
}


@pytest.mark.performance
class TestPressureDropCalculations:
    """Performance benchmarks for pressure drop calculations."""

    def test_pressure_drop_single_calculation(self, benchmark) -> None:
        """Benchmark single pressure drop calculation.

        SLA: < 100ms
        """
        result = benchmark(_CALCULATOR.calculate_pressure_drop, **_BASE_PARAMS)
        assert result.pressure_drop_pa > 0, "Pressure drop should be positive"

    def test_pressure_drop_engine_initialization(self, benchmark) -> None:
        """Benchmark calculator object creation overhead.

        SLA: < 50ms
        """
        calculator = benchmark(PressureDropCalculator)
        assert calculator is not None

    def test_pressure_drop_with_varying_density(self, benchmark) -> None:
        """Benchmark calculation across a range of molecular weights (proxy for
        gas density).

        SLA: < 100ms per call (5 calls total inside benchmark loop)
        """
        mol_weights = [0.002, 0.016, 0.029, 0.044, 0.028]  # H2, CH4, air, CO2, CO

        def calculate_variable() -> list:
            return [
                _CALCULATOR.calculate_pressure_drop(
                    **{**_BASE_PARAMS, "molecular_weight_kg_mol": mw}
                )
                for mw in mol_weights
            ]

        results = benchmark(calculate_variable)
        assert len(results) == 5
        assert all(r.pressure_drop_pa > 0 for r in results)

    def test_pressure_drop_with_varying_length(self, benchmark) -> None:
        """Benchmark calculation across a range of pipe lengths.

        SLA: < 100ms per call (5 calls total)
        """
        lengths_m = [10.0, 25.0, 50.0, 75.0, 100.0]

        def calculate_variable() -> list:
            return [
                _CALCULATOR.calculate_pressure_drop(
                    **{**_BASE_PARAMS, "pipe_length_m": length}
                )
                for length in lengths_m
            ]

        results = benchmark(calculate_variable)
        assert len(results) == 5
        # Pressure drop should increase with pipe length
        drops = [r.pressure_drop_pa for r in results]
        assert drops == sorted(drops), "Pressure drop must increase with pipe length"

    def test_pressure_drop_repeated_calls(self, benchmark) -> None:
        """Benchmark 100 repeated pressure drop calculations (throughput test).

        SLA: < 100ms per operation amortized
        """

        def calculate_repeated() -> list:
            return [
                _CALCULATOR.calculate_pressure_drop(**_BASE_PARAMS) for _ in range(100)
            ]

        results = benchmark(calculate_repeated)
        assert len(results) == 100
        assert all(r.pressure_drop_pa > 0 for r in results)

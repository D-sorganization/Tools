"""Performance benchmarks for pressure drop calculator.

Measures the performance of pressure drop calculation under various load
conditions. SLA target: < 100ms per calculation.
"""

from __future__ import annotations

import pytest

# Import from the actual pressure drop calculator
try:
    from pressure_drop_calculator import (
        calculate_pressure_drop,
        PressureDropInputs,
        PressureDropCalculationEngine,
    )
except (ImportError, NameError):
    # Fallback for testing without the full dependency (GUI/EGL issues)
    pytest.skip(
        "pressure_drop_calculator not available",
        allow_module_level=True,
    )


pytestmark = pytest.mark.benchmark


@pytest.mark.performance
class TestPressureDropCalculations:
    """Performance benchmarks for pressure drop calculations."""

    def test_pressure_drop_single_calculation(
        self, benchmark, pipe_parameters
    ):
        """Benchmark single pressure drop calculation.

        SLA: < 100ms
        Tests: Basic calculation with standard parameters
        """
        result = benchmark(
            calculate_pressure_drop,
            pipe_parameters["inlet_pressure"],
            pipe_parameters["fluid_density"],
            pipe_parameters["pipe_length"],
        )
        assert result > 0, "Pressure drop should be positive"

    def test_pressure_drop_engine_initialization(self, benchmark):
        """Benchmark pressure drop calculator engine initialization.

        SLA: < 50ms
        Tests: Engine object creation overhead
        """
        def create_engine():
            return PressureDropCalculationEngine()

        engine = benchmark(create_engine)
        assert engine is not None

    def test_pressure_drop_with_varying_density(
        self, benchmark, pipe_parameters
    ):
        """Benchmark calculation with varying fluid density.

        SLA: < 100ms
        Tests: Sensitivity to density parameter
        """
        def calculate_variable():
            densities = [0.1, 0.2, 0.3, 0.4, 0.5]
            results = []
            for rho in densities:
                result = calculate_pressure_drop(
                    pipe_parameters["inlet_pressure"],
                    rho,
                    pipe_parameters["pipe_length"],
                )
                results.append(result)
            return results

        results = benchmark(calculate_variable)
        assert len(results) == 5
        assert all(r > 0 for r in results)

    def test_pressure_drop_with_varying_length(
        self, benchmark, pipe_parameters
    ):
        """Benchmark calculation with varying pipe length.

        SLA: < 100ms
        Tests: Sensitivity to length parameter
        """
        def calculate_variable():
            lengths = [0.01, 0.025, 0.05, 0.075, 0.1]
            results = []
            for length in lengths:
                result = calculate_pressure_drop(
                    pipe_parameters["inlet_pressure"],
                    pipe_parameters["fluid_density"],
                    length,
                )
                results.append(result)
            return results

        results = benchmark(calculate_variable)
        assert len(results) == 5
        assert all(r > 0 for r in results)

    def test_pressure_drop_repeated_calls(self, benchmark, pipe_parameters):
        """Benchmark 100 repeated pressure drop calculations.

        SLA: < 100ms (per operation, amortized)
        Tests: Throughput and consistency
        """
        def calculate_repeated():
            results = []
            for _ in range(100):
                result = calculate_pressure_drop(
                    pipe_parameters["inlet_pressure"],
                    pipe_parameters["fluid_density"],
                    pipe_parameters["pipe_length"],
                )
                results.append(result)
            return results

        results = benchmark(calculate_repeated)
        assert len(results) == 100
        assert all(r > 0 for r in results)

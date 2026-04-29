"""Unit Conversion Benchmark Suite.

Measures performance of core unit conversion operations across
temperature, pressure, length, mass, and flow-rate categories.

Run with: python -m unit_conversion_benchmark
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from web_applications.unit_converter.converter import UnitConverter

logger = logging.getLogger(__name__)


class UnitConversionBenchmark:
    """Benchmark suite for unit conversion performance."""

    ITERATIONS = 100_000

    def __init__(self) -> None:
        self.converter = UnitConverter()

    @staticmethod
    def _benchmark(label: str, func: Callable[[], float]) -> dict[str, Any]:
        start = time.perf_counter()
        for _ in range(UnitConversionBenchmark.ITERATIONS):
            func()
        elapsed = time.perf_counter() - start
        return {
            "label": label,
            "iterations": UnitConversionBenchmark.ITERATIONS,
            "total_seconds": elapsed,
            "ops_per_second": UnitConversionBenchmark.ITERATIONS / elapsed,
        }

    def benchmark_temperature(self) -> dict[str, Any]:
        return self._benchmark(
            "temperature: celsius_to_fahrenheit",
            lambda: self.converter.convert(100.0, "C", "F"),
        )

    def benchmark_pressure(self) -> dict[str, Any]:
        return self._benchmark(
            "pressure: psi_to_bar",
            lambda: self.converter.convert(14.7, "psi", "bar"),
        )

    def benchmark_length(self) -> dict[str, Any]:
        return self._benchmark(
            "length: meters_to_feet",
            lambda: self.converter.convert(1.0, "m", "ft"),
        )

    def benchmark_mass(self) -> dict[str, Any]:
        return self._benchmark(
            "mass: kg_to_lb",
            lambda: self.converter.convert(1.0, "kg", "lb"),
        )

    def benchmark_flow_rate(self) -> dict[str, Any]:
        return self._benchmark(
            "flow_rate: scfm_to_m3h",
            lambda: self.converter.convert(1000.0, "SCFM_60F", "m3/h"),
        )

    def run_all(self) -> list[dict[str, Any]]:
        """Run all benchmarks and return results."""
        results = [
            self.benchmark_temperature(),
            self.benchmark_pressure(),
            self.benchmark_length(),
            self.benchmark_mass(),
            self.benchmark_flow_rate(),
        ]
        for r in results:
            label = r["label"]
            ops = r["ops_per_second"]
            secs = r["total_seconds"]
            logger.info("%s: %.0f ops/sec (%.3fs)", label, ops, secs)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    UnitConversionBenchmark().run_all()

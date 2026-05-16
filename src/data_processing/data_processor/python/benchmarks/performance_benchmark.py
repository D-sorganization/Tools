"""Performance Benchmark Suite for Data Processor.

This script measures the performance characteristics of the refactored
Data Processor application across various operations and dataset sizes.

Metrics measured:
- File loading speed
- Signal processing speed (filtering, integration, differentiation)
- Memory usage
- Scalability with dataset size
- End-to-end workflow performance

Run with: python performance_benchmark.py
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from data_processor.core.data_loader import DataLoader
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import (
    DifferentiationConfig,
    FilterConfig,
    IntegrationConfig,
)

# Try to import memory profiler
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import from centralized utilities
from utils.csv_utils import safe_write_csv
from utils.file_utils import safe_write_json


class PerformanceBenchmark:
    """Performance benchmark suite for Data Processor."""

    def __init__(self) -> None:
        """Initialize benchmark suite."""
        self.results: dict[str, dict[str, Any]] = {}
        self.loader = DataLoader(use_high_performance=True)
        self.processor = SignalProcessor()

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return float(process.memory_info().rss / 1024 / 1024)
        return 0.0

    def create_test_data(
        self,
        n_rows: int,
        n_signals: int,
        tmp_path: Path,
        suffix: str = "",
    ) -> str:
        """Create test CSV file with specified dimensions."""
        if n_rows is None:
            raise ValueError("n_rows must be provided")
        np.random.seed(42)

        # Generate test data
        data: dict[str, Any] = {
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="1s"),
        }

        # Add signal columns
        for i in range(n_signals):
            # Mix of different signal types
            if i % 3 == 0:
                # Sine wave with noise
                data[f"signal_{i}"] = (
                    10
                    + 5 * np.sin(np.linspace(0, 10, n_rows))
                    + np.random.randn(n_rows) * 0.5
                )
            elif i % 3 == 1:
                # Linear trend with noise
                data[f"signal_{i}"] = (
                    np.linspace(0, 100, n_rows) + np.random.randn(n_rows) * 2
                )
            else:
                # Random walk
                data[f"signal_{i}"] = np.cumsum(np.random.randn(n_rows))

        df = pd.DataFrame(data)

        # Save to CSV
        csv_file = tmp_path / f"benchmark_data_{n_rows}x{n_signals}{suffix}.csv"
        safe_write_csv(df, csv_file, index=False)

        return str(csv_file)

    def benchmark_file_loading(self) -> dict[str, dict[str, float | int]]:
        """Benchmark file loading performance."""

        results: dict[str, dict[str, float | int]] = {}

        # Use benchmarks directory for test data (security-approved location)
        tmp_path = Path(__file__).parent / "test_data"
        tmp_path.mkdir(exist_ok=True)

        try:
            # Test different file sizes
            test_sizes = [
                (1_000, 5, "1K rows, 5 signals"),
                (10_000, 10, "10K rows, 10 signals"),
                (100_000, 20, "100K rows, 20 signals"),
            ]

            for n_rows, n_signals, label in test_sizes:
                csv_file = self.create_test_data(n_rows, n_signals, tmp_path)

                # Benchmark loading
                start = time.perf_counter()
                df = self.loader.load_csv_file(csv_file, validate_security=False)
                elapsed = time.perf_counter() - start

                # Validate the load was successful
                if not (df is not None and len(df) == n_rows):
                    raise ValueError(f"Load failed for {label}")

                throughput = n_rows / elapsed
                results[f"load_{label}"] = {
                    "time": elapsed,
                    "throughput": throughput,
                    "rows": n_rows,
                }

            # Test multiple file loading
            files = [
                self.create_test_data(5_000, 5, tmp_path, suffix=f"_{i}")
                for i in range(5)
            ]

            start = time.perf_counter()
            dataframes = self.loader.load_multiple_files(files)
            elapsed = time.perf_counter() - start

            # Validate all files loaded successfully
            assert len(dataframes) == len(files), (
                f"Expected {len(files)} dataframes, got {len(dataframes)}"
            )

            results["load_multiple_5_files"] = {
                "time": elapsed,
                "files": len(files),
            }

        finally:
            # Clean up test data
            if tmp_path.exists():
                shutil.rmtree(tmp_path)

        return results

    def benchmark_filtering(self) -> dict[str, dict[str, float]]:
        """Benchmark signal filtering performance."""

        results = {}

        # Create test data
        n_rows = 50_000
        df = pd.DataFrame(
            {
                "signal1": np.sin(np.linspace(0, 10, n_rows))
                + np.random.randn(n_rows) * 0.1,
                "signal2": np.cos(np.linspace(0, 10, n_rows))
                + np.random.randn(n_rows) * 0.1,
                "signal3": np.random.randn(n_rows),
            },
        )

        # Test different filter types
        filter_tests = [
            (
                "Moving Average",
                FilterConfig(
                    filter_type="Moving Average", parameters={"ma_window": 10}
                ),
            ),
            (
                "Butterworth Low-pass",
                FilterConfig(
                    filter_type="Butterworth Low-pass",
                    parameters={"bw_order": 3, "bw_cutoff": 0.1},
                ),
            ),
            (
                "Median Filter",
                FilterConfig(
                    filter_type="Median Filter", parameters={"median_kernel": 5}
                ),
            ),
            (
                "Gaussian Filter",
                FilterConfig(
                    filter_type="Gaussian Filter", parameters={"gaussian_sigma": 2.0}
                ),
            ),
            (
                "Savitzky-Golay",
                FilterConfig(
                    filter_type="Savitzky-Golay",
                    parameters={"savgol_window": 11, "savgol_polyorder": 3},
                ),
            ),
        ]

        for filter_name, config in filter_tests:
            start = time.perf_counter()
            filtered_df = self.processor.apply_filter(df, config)
            elapsed = time.perf_counter() - start

            # Validate filter output
            assert filtered_df is not None and len(filtered_df) == n_rows, (
                f"Filter {filter_name} failed"
            )

            throughput = n_rows / elapsed
            results[f"filter_{filter_name}"] = {
                "time": elapsed,
                "throughput": throughput,
            }

        return results

    def benchmark_integration_differentiation(self) -> dict[str, dict[str, float]]:
        """Benchmark integration and differentiation operations."""

        results = {}

        # Create test data
        n_rows = 50_000
        df = pd.DataFrame(
            {
                "signal1": np.sin(np.linspace(0, 10, n_rows)),
                "signal2": np.cos(np.linspace(0, 10, n_rows)),
            },
            index=pd.date_range("2024-01-01", periods=n_rows, freq="1s"),
        )

        # Integration benchmark
        int_config = IntegrationConfig(
            signals=["signal1", "signal2"],
            method="cumulative",
        )

        start = time.perf_counter()
        int_df = self.processor.integrate_signals(df, int_config)
        elapsed = time.perf_counter() - start

        # Validate integration output
        if not (int_df is not None and len(int_df) == n_rows):
            raise ValueError("Integration failed")

        results["integration"] = {
            "time": elapsed,
            "throughput": n_rows / elapsed,
        }

        # Differentiation benchmark
        diff_config = DifferentiationConfig(
            signals=["signal1", "signal2"],
            order=1,
            method="central",
        )

        start = time.perf_counter()
        diff_df = self.processor.differentiate_signals(df, diff_config)
        elapsed = time.perf_counter() - start

        # Validate differentiation output
        if not (diff_df is not None and len(diff_df) == n_rows):
            raise ValueError("Differentiation failed")

        results["differentiation"] = {
            "time": elapsed,
            "throughput": n_rows / elapsed,
        }

        return results

    def benchmark_custom_formulas(self) -> dict[str, dict[str, float]]:
        """Benchmark custom formula evaluation."""

        results = {}

        # Create test data
        n_rows = 50_000
        df = pd.DataFrame(
            {
                "signal1": np.random.randn(n_rows),
                "signal2": np.random.randn(n_rows),
                "signal3": np.random.randn(n_rows),
            },
        )

        # Test different formulas
        formulas = [
            ("simple_add", "signal1 + signal2"),
            ("complex_expr", "(signal1 * signal2) + signal3 / 2"),
            ("trigonometric", "sin(signal1) + cos(signal2)"),
        ]

        for name, formula in formulas:
            start = time.perf_counter()
            _result_df, success = self.processor.apply_custom_formula(
                df,
                f"result_{name}",
                formula,
            )
            elapsed = time.perf_counter() - start

            if success:
                results[f"formula_{name}"] = {
                    "time": elapsed,
                    "throughput": n_rows / elapsed,
                }

        return results

    def benchmark_end_to_end_workflow(self) -> dict[str, dict[str, float]]:
        """Benchmark complete end-to-end workflow."""

        results = {}

        # Use benchmarks directory for test data
        tmp_path = Path(__file__).parent / "test_data_workflow"
        tmp_path.mkdir(exist_ok=True)

        try:
            # Create test file
            csv_file = self.create_test_data(50_000, 10, tmp_path)

            start_total = time.perf_counter()

            # Step 1: Load data
            start = time.perf_counter()
            df = self.loader.load_csv_file(csv_file, validate_security=False)
            load_time = time.perf_counter() - start

            # Step 2: Detect and convert time column
            start = time.perf_counter()
            time_col = self.loader.detect_time_column(df)
            if time_col is not None:
                df = self.loader.convert_time_column(df, time_col)
            time_convert_time = time.perf_counter() - start

            # Step 3: Apply filtering
            start = time.perf_counter()
            filter_config = FilterConfig(
                filter_type="Moving Average", parameters={"ma_window": 10}
            )
            df = self.processor.apply_filter(df, filter_config)
            filter_time = time.perf_counter() - start

            # Step 4: Integration
            start = time.perf_counter()
            signals = self.loader.get_numeric_signals(df)[:5]  # First 5 signals
            int_config = IntegrationConfig(
                signals=signals,
                method="cumulative",
            )
            df = self.processor.integrate_signals(df, int_config)
            integration_time = time.perf_counter() - start

            # Step 5: Statistics
            start = time.perf_counter()
            # detect_signal_statistics does not accept a signal argument
            stats = self.processor.detect_signal_statistics(df)
            stats_time = time.perf_counter() - start

            # Validate statistics output
            assert stats is not None and "mean" in stats, (
                "Statistics calculation failed"
            )

            # Step 6: Save
            start = time.perf_counter()
            output_file = tmp_path / "output.csv"
            self.loader.save_dataframe(df, str(output_file))
            save_time = time.perf_counter() - start

            total_time = time.perf_counter() - start_total

            results["workflow_complete"] = {
                "total_time": total_time,
                "load_time": load_time,
                "time_convert_time": time_convert_time,
                "filter_time": filter_time,
                "integration_time": integration_time,
                "stats_time": stats_time,
                "save_time": save_time,
            }

        finally:
            # Clean up test data
            if tmp_path.exists():
                shutil.rmtree(tmp_path)

        return results

    def benchmark_scalability(self) -> dict[str, dict[str, float]]:
        """Benchmark performance scaling with dataset size."""

        results = {}

        dataset_sizes = [1_000, 5_000, 10_000, 50_000, 100_000]

        for n_rows in dataset_sizes:
            # Create test data - optimized using direct NumPy operations
            data = {}
            for i in range(5):
                data[f"signal_{i}"] = np.random.randn(n_rows)
            df = pd.DataFrame(data)

            # Apply moving average filter
            config = FilterConfig(
                filter_type="Moving Average", parameters={"ma_window": 10}
            )

            start = time.perf_counter()
            filtered = self.processor.apply_filter(df, config)
            elapsed = time.perf_counter() - start

            # Validate filter output
            assert filtered is not None and len(filtered) == n_rows, (
                f"Scalability test failed for {n_rows} rows"
            )

            throughput = n_rows / elapsed

            results[f"scale_{n_rows}"] = {
                "time": elapsed,
                "throughput": throughput,
                "rows": n_rows,
            }

        return results

    def benchmark_memory_usage(self) -> dict[str, dict[str, float]]:
        """Benchmark memory usage during operations."""
        if not PSUTIL_AVAILABLE:
            return {}

        results = {}

        # Get baseline memory
        baseline_memory = self.get_memory_usage_mb()

        # Test memory usage with large dataset
        n_rows = 100_000
        df = pd.DataFrame({f"signal_{i}": np.random.randn(n_rows) for i in range(20)})

        memory_before = self.get_memory_usage_mb()

        # Apply filter
        config = FilterConfig(
            filter_type="Moving Average", parameters={"ma_window": 10}
        )
        filtered = self.processor.apply_filter(df, config)

        # Validate filter was applied
        assert filtered is not None and len(filtered) == n_rows, (
            "Memory benchmark filter failed"
        )

        memory_after = self.get_memory_usage_mb()

        memory_used = memory_after - baseline_memory

        results["memory_100k_20signals"] = {
            "baseline_mb": baseline_memory,
            "before_mb": memory_before,
            "after_mb": memory_after,
            "used_mb": memory_used,
        }

        return results

    def run_all_benchmarks(self) -> dict[str, dict[str, Any]]:
        """Run all benchmarks and return results."""

        self.results["file_loading"] = self.benchmark_file_loading()
        self.results["filtering"] = self.benchmark_filtering()
        self.results["integration_differentiation"] = (
            self.benchmark_integration_differentiation()
        )
        self.results["custom_formulas"] = self.benchmark_custom_formulas()
        self.results["end_to_end"] = self.benchmark_end_to_end_workflow()
        self.results["scalability"] = self.benchmark_scalability()
        self.results["memory"] = self.benchmark_memory_usage()

        return self.results

    def save_results(self, output_file: str) -> None:
        """Save benchmark results to JSON file."""
        safe_write_json(output_file, self.results, indent=2)

    def print_summary(self) -> None:
        """Print benchmark summary."""

        # File loading summary
        if "file_loading" in self.results:
            for value in self.results["file_loading"].values():
                if "throughput" in value:
                    pass  # Throughput data found, no action needed

        # Filtering summary
        if "filtering" in self.results:
            throughputs = [
                v["throughput"]
                for v in self.results["filtering"].values()
                if "throughput" in v
            ]
            if throughputs:
                np.mean(throughputs)

        # End-to-end workflow
        if (
            "end_to_end" in self.results
            and "workflow_complete" in self.results["end_to_end"]
        ):
            self.results["end_to_end"]["workflow_complete"]

        # Memory usage
        if (
            "memory" in self.results
            and "memory_100k_20signals" in self.results["memory"]
        ):
            self.results["memory"]["memory_100k_20signals"]


def main() -> None:
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()

    # Run all benchmarks
    benchmark.run_all_benchmarks()

    # Print summary
    benchmark.print_summary()

    # Save results
    output_file = Path(__file__).parent / "benchmark_results.json"
    benchmark.save_results(str(output_file))


if __name__ == "__main__":
    main()

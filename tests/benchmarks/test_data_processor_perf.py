"""Performance benchmarks for data processor.

Measures the performance of data filtering and signal processing operations.
SLA target: < 500ms for 10K rows.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from data_processor.fft_filter_ops import (
        design_frequency_window,
        apply_window_function,
        apply_fft_filter_core,
    )
except (ImportError, NameError):
    pytest.skip(
        "data_processor not available",
        allow_module_level=True,
    )


pytestmark = pytest.mark.benchmark


@pytest.mark.performance
class TestDataProcessorFilterOperations:
    """Performance benchmarks for FFT-based filter operations."""

    def test_design_frequency_window_small(self, benchmark):
        """Benchmark frequency window design for small signal (100 samples).

        SLA: < 100ms
        Tests: Low-pass filter window design
        """
        def design_window():
            return design_frequency_window(
                filter_type="FFT Low-pass",
                freq_low=0.2,
                freq_high=0.4,
                window_shape="hamming",
                n_samples=100,
                transition_bw=0.05,
            )

        result = benchmark(design_window)
        assert len(result) == 100
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_design_frequency_window_medium(self, benchmark):
        """Benchmark frequency window design for medium signal (1000 samples).

        SLA: < 100ms
        Tests: Band-pass filter window design
        """
        def design_window():
            return design_frequency_window(
                filter_type="FFT Band-pass",
                freq_low=0.1,
                freq_high=0.3,
                window_shape="hann",
                n_samples=1000,
                transition_bw=0.05,
            )

        result = benchmark(design_window)
        assert len(result) == 1000
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_design_frequency_window_large(self, benchmark):
        """Benchmark frequency window design for large signal (10K samples).

        SLA: < 500ms
        Tests: High-pass filter window design at scale
        """
        def design_window():
            return design_frequency_window(
                filter_type="FFT High-pass",
                freq_low=0.05,
                freq_high=0.5,
                window_shape="blackman",
                n_samples=10000,
                transition_bw=0.01,
            )

        result = benchmark(design_window)
        assert len(result) == 10000
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_apply_window_function_small(
        self, benchmark, sample_small_array
    ):
        """Benchmark window function application to small signal (100 samples).

        SLA: < 50ms
        Tests: Hamming window application
        """
        result = benchmark(
            apply_window_function, sample_small_array, "Hamming"
        )
        assert len(result) == len(sample_small_array)

    def test_apply_window_function_medium(
        self, benchmark, sample_medium_array
    ):
        """Benchmark window function application to medium signal (1000 samples).

        SLA: < 100ms
        Tests: Hann window application
        """
        result = benchmark(
            apply_window_function, sample_medium_array, "Hann"
        )
        assert len(result) == len(sample_medium_array)

    def test_apply_window_function_large(
        self, benchmark, sample_large_array
    ):
        """Benchmark window function application to large signal (10K samples).

        SLA: < 500ms
        Tests: Blackman window application at scale
        """
        result = benchmark(
            apply_window_function, sample_large_array, "Blackman"
        )
        assert len(result) == len(sample_large_array)

    def test_fft_filter_core_lowpass(
        self, benchmark, sample_time_series_data
    ):
        """Benchmark FFT low-pass filter on 1000-sample signal.

        SLA: < 100ms
        Tests: Single FFT filter application
        """
        signal = sample_time_series_data["signal"]
        window = np.hamming(len(signal))

        def apply_filter():
            return apply_fft_filter_core(
                signal,
                window,
                zero_phase=False,
            )

        result = benchmark(apply_filter)
        assert len(result) == len(signal)

    def test_fft_filter_core_bandpass(
        self, benchmark, sample_time_series_data
    ):
        """Benchmark FFT band-pass filter on 1000-sample signal.

        SLA: < 100ms
        Tests: Band-pass filtering
        """
        signal = sample_time_series_data["signal"]
        window = np.hanning(len(signal))

        def apply_filter():
            return apply_fft_filter_core(
                signal,
                window,
                zero_phase=False,
            )

        result = benchmark(apply_filter)
        assert len(result) == len(signal)

    def test_fft_filter_core_large_signal(
        self, benchmark, sample_large_time_series_data
    ):
        """Benchmark FFT filter on large signal (10K samples).

        SLA: < 500ms
        Tests: Filtering performance at scale
        """
        signal = sample_large_time_series_data["signal"]
        window = np.blackman(len(signal))

        def apply_filter():
            return apply_fft_filter_core(
                signal,
                window,
                zero_phase=False,
            )

        result = benchmark(apply_filter)
        assert len(result) == len(signal)


@pytest.mark.performance
class TestDataProcessorChainedOperations:
    """Performance benchmarks for chained filter operations."""

    def test_window_design_and_application(
        self, benchmark, sample_time_series_data
    ):
        """Benchmark combined window design and application.

        SLA: < 200ms
        Tests: Window design + application on 1000-sample signal
        """
        signal = sample_time_series_data["signal"]

        def design_and_apply():
            window = design_frequency_window(
                filter_type="FFT Low-pass",
                freq_low=0.1,
                freq_high=0.3,
                window_shape="hamming",
                n_samples=len(signal),
                transition_bw=0.05,
            )
            filtered = apply_window_function(signal, "Hamming")
            return filtered

        result = benchmark(design_and_apply)
        assert len(result) == len(signal)

    def test_complete_filter_pipeline_small(
        self, benchmark, sample_time_series_data
    ):
        """Benchmark complete filter pipeline on 1000-sample signal.

        SLA: < 200ms
        Tests: Window design + application + FFT filtering
        """
        signal = sample_time_series_data["signal"]

        def complete_pipeline():
            window = design_frequency_window(
                filter_type="FFT Band-pass",
                freq_low=0.02,
                freq_high=0.1,
                window_shape="hann",
                n_samples=len(signal),
                transition_bw=0.01,
            )
            windowed = apply_window_function(signal, "Hann")
            filtered = apply_fft_filter_core(
                windowed, window, zero_phase=False
            )
            return filtered

        result = benchmark(complete_pipeline)
        assert len(result) == len(signal)

    def test_complete_filter_pipeline_large(
        self, benchmark, sample_large_time_series_data
    ):
        """Benchmark complete filter pipeline on 10K-sample signal.

        SLA: < 500ms
        Tests: Full pipeline at scale
        """
        signal = sample_large_time_series_data["signal"]

        def complete_pipeline():
            window = design_frequency_window(
                filter_type="FFT Low-pass",
                freq_low=0.05,
                freq_high=0.2,
                window_shape="blackman",
                n_samples=len(signal),
                transition_bw=0.01,
            )
            windowed = apply_window_function(signal, "Blackman")
            filtered = apply_fft_filter_core(
                windowed, window, zero_phase=False
            )
            return filtered

        result = benchmark(complete_pipeline)
        assert len(result) == len(signal)

    def test_repeated_filter_operations(
        self, benchmark, sample_time_series_data
    ):
        """Benchmark 10 repeated complete filter operations.

        SLA: < 500ms (amortized)
        Tests: Throughput with repeated filtering
        """
        signal = sample_time_series_data["signal"]

        def repeated_filtering():
            results = []
            for _ in range(10):
                window = design_frequency_window(
                    filter_type="FFT Band-pass",
                    freq_low=0.02,
                    freq_high=0.1,
                    window_shape="hann",
                    n_samples=len(signal),
                    transition_bw=0.01,
                )
                filtered = apply_fft_filter_core(
                    signal, window, zero_phase=False
                )
                results.append(filtered)
            return results

        results = benchmark(repeated_filtering)
        assert len(results) == 10

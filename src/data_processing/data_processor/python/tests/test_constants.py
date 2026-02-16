"""Tests for data_processor.constants module."""

from __future__ import annotations

import pytest

from data_processor.constants import (
    CHUNK_SIZE,
    DEFAULT_BW_CUTOFF,
    DEFAULT_BW_ORDER,
    DEFAULT_DPI,
    DEFAULT_ENCODING,
    DEFAULT_FFT_WINDOW_SHAPE,
    DEFAULT_GAUSSIAN_MODE,
    DEFAULT_HAMPEL_THRESHOLD,
    DEFAULT_HAMPEL_WINDOW,
    DEFAULT_MA_WINDOW,
    DEFAULT_MEDIAN_KERNEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW,
    DEFAULT_ZSCORE_THRESHOLD,
    ERROR_MSG_EMPTY_FILE,
    ERROR_MSG_NO_FILES,
    ERROR_MSG_NO_PLOTS,
    LARGE_FILE_THRESHOLD,
    LOG_LEVEL,
    MAX_DERIVATIVE_ORDER,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_PLOT_POINTS,
    MIN_BUTTERWORTH_DATA_MULTIPLIER,
    MIN_KERNEL_SIZE,
    MIN_SIGNAL_DATA_POINTS,
    SUPPORTED_FORMATS,
    TIME_COLUMN_KEYWORDS,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
)


class TestFileConstants:
    """Tests for file processing constants."""

    def test_max_file_size_relationship(self) -> None:
        assert MAX_FILE_SIZE_BYTES == MAX_FILE_SIZE_MB * 1024 * 1024

    def test_chunk_size_positive(self) -> None:
        assert CHUNK_SIZE > 0

    def test_default_encoding(self) -> None:
        assert DEFAULT_ENCODING == "utf-8"


class TestProcessingConstants:
    """Tests for processing constants."""

    def test_sample_rate(self) -> None:
        assert DEFAULT_SAMPLE_RATE == 1000

    def test_max_plot_points(self) -> None:
        assert MAX_PLOT_POINTS > 0

    def test_min_signal_data_points(self) -> None:
        assert MIN_SIGNAL_DATA_POINTS > 0

    def test_large_file_threshold(self) -> None:
        assert LARGE_FILE_THRESHOLD > 0


class TestFilterConstants:
    """Tests for filter engine constants."""

    def test_butterworth_defaults(self) -> None:
        assert 0 < DEFAULT_BW_CUTOFF < 1
        assert DEFAULT_BW_ORDER > 0
        assert MIN_BUTTERWORTH_DATA_MULTIPLIER >= 3

    def test_moving_average_defaults(self) -> None:
        assert DEFAULT_MA_WINDOW > 0

    def test_median_defaults(self) -> None:
        assert DEFAULT_MEDIAN_KERNEL >= MIN_KERNEL_SIZE

    def test_savgol_defaults(self) -> None:
        assert DEFAULT_SAVGOL_WINDOW > DEFAULT_SAVGOL_POLYORDER
        assert DEFAULT_SAVGOL_POLYORDER > 0

    def test_max_derivative(self) -> None:
        assert MAX_DERIVATIVE_ORDER >= 1

    def test_hampel_defaults(self) -> None:
        assert DEFAULT_HAMPEL_WINDOW > 0
        assert DEFAULT_HAMPEL_THRESHOLD > 0

    def test_zscore_threshold(self) -> None:
        assert DEFAULT_ZSCORE_THRESHOLD > 0

    def test_gaussian_defaults(self) -> None:
        assert DEFAULT_GAUSSIAN_MODE in ("reflect", "nearest", "constant", "wrap")

    def test_fft_defaults(self) -> None:
        assert DEFAULT_FFT_WINDOW_SHAPE == "hann"


class TestExportConstants:
    """Tests for export constants."""

    def test_dpi(self) -> None:
        assert DEFAULT_DPI == 300

    def test_supported_formats(self) -> None:
        assert ".csv" in SUPPORTED_FORMATS
        assert ".json" in SUPPORTED_FORMATS

    def test_log_level(self) -> None:
        assert LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR")


class TestPlotConstants:
    """Tests for plot constants."""

    def test_zoom_factors(self) -> None:
        assert ZOOM_OUT_FACTOR < 1.0
        assert ZOOM_IN_FACTOR > 1.0

    def test_time_column_keywords(self) -> None:
        assert "time" in TIME_COLUMN_KEYWORDS
        assert "timestamp" in TIME_COLUMN_KEYWORDS


class TestErrorMessages:
    """Tests for error message constants."""

    def test_error_messages_non_empty(self) -> None:
        assert len(ERROR_MSG_NO_FILES) > 0
        assert len(ERROR_MSG_EMPTY_FILE) > 0
        assert len(ERROR_MSG_NO_PLOTS) > 0

"""Tests for the SignalProcessor wrapper around the filter engine."""

from __future__ import annotations

import pandas as pd
import pytest
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models import FilterConfig


def test_signal_processor_applies_moving_average() -> None:
    """Moving average filter should smooth numeric signals."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=12, freq="s"),
            "signal": [float(i) for i in range(12)],
        }
    )
    processor = SignalProcessor()
    filter_config = FilterConfig.from_mapping({"filter_type": "Moving Average", "ma_window": 3})

    result = processor.apply_filter(df, filter_config)
    expected_signal = df["signal"].rolling(window=3, min_periods=1, center=True).mean()

    pd.testing.assert_series_equal(result["signal"], expected_signal)
    pd.testing.assert_series_equal(result["time"], df["time"])


def test_signal_processor_rejects_empty_frame() -> None:
    """Empty dataframes should raise a friendly error."""
    processor = SignalProcessor()
    filter_config = FilterConfig.from_mapping({"filter_type": "Moving Average", "ma_window": 3})

    with pytest.raises(ValueError):
        processor.apply_filter(pd.DataFrame(), filter_config)


def test_signal_processor_rejects_unknown_filter() -> None:
    """Unsupported filter types should fail fast."""
    processor = SignalProcessor()

    class DummyFilterConfig:
        filter_type = "Unknown"

        def to_engine_parameters(self) -> dict[str, str]:
            return {"filter_type": self.filter_type}

    with pytest.raises(ValueError):
        processor.apply_filter(
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}), DummyFilterConfig()
        )

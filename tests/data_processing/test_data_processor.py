"""Tests for the shared DataProcessor facade.

See issue #407.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.python.data_processing import DataProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small sample DataFrame with a time column."""
    np.random.seed(42)
    n = 100
    t = np.linspace(0, 10, n)
    return pd.DataFrame(
        {
            "time": t,
            "sensor1": np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.1, n),
            "sensor2": np.cos(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.1, n),
            "temperature": 25 + 0.5 * t,
        }
    )


@pytest.fixture
def csv_path(sample_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Write sample data to a CSV file."""
    p = tmp_path / "test_data.csv"
    sample_df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------


class TestLoad:
    """Test data loading."""

    def test_load_csv(self, csv_path: Path) -> None:
        dp = DataProcessor()
        dp.load(csv_path)
        assert dp.info.num_rows == 100
        assert dp.info.num_columns == 4
        assert "sensor1" in dp.info.columns

    def test_load_dataframe(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df, name="test")
        assert dp.info.num_rows == 100

    def test_load_unsupported_format(self) -> None:
        dp = DataProcessor()
        with pytest.raises(ValueError, match="Unsupported"):
            dp.load("data.xyz")

    def test_no_data_raises(self) -> None:
        dp = DataProcessor()
        with pytest.raises(RuntimeError, match="No data loaded"):
            _ = dp.dataframe

    def test_info_empty(self) -> None:
        dp = DataProcessor()
        info = dp.info
        assert info.num_rows == 0
        assert info.columns == []


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransform:
    """Test data transformations."""

    def test_trim_time(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        dp.trim_time(2.0, 8.0)
        assert dp.dataframe["time"].min() >= 2.0
        assert dp.dataframe["time"].max() <= 8.0

    def test_apply_formula(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        dp.apply_formula("sum_sensors", "sensor1 + sensor2")
        assert "sum_sensors" in dp.dataframe.columns

    def test_drop_columns(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        dp.drop_columns(["sensor2"])
        assert "sensor2" not in dp.dataframe.columns
        assert "sensor1" in dp.dataframe.columns

    def test_rename_columns(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        dp.rename_columns({"sensor1": "pressure"})
        assert "pressure" in dp.dataframe.columns
        assert "sensor1" not in dp.dataframe.columns

    def test_sort(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        dp.sort("temperature", ascending=False)
        temps = dp.dataframe["temperature"].values
        assert temps[0] >= temps[-1]

    def test_dropna(self) -> None:
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [None, 2, 3, 4]})
        dp = DataProcessor()
        dp.load_dataframe(df)
        dp.dropna()
        assert dp.info.num_rows == 2

    def test_method_chaining(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        result = (
            dp.load_dataframe(sample_df)
            .trim_time(1, 9)
            .apply_formula("double", "sensor1 * 2")
            .drop_columns(["sensor2"])
        )
        assert isinstance(result, DataProcessor)
        assert "double" in dp.dataframe.columns
        assert "sensor2" not in dp.dataframe.columns

    def test_apply_filter_unknown_type_raises(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        with pytest.raises(ValueError, match="Unknown filter type"):
            dp.apply_filter("unknown_filter", columns=["sensor1"])

    def test_apply_filter_nonpositive_window_raises(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        with pytest.raises(ValueError, match="window_size must be positive"):
            dp.apply_filter("moving_average", window_size=0, columns=["sensor1"])

    def test_apply_filter_no_matching_columns_raises(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        with pytest.raises(ValueError, match="No valid columns to filter"):
            dp.apply_filter("moving_average", columns=["missing_col"])


# ---------------------------------------------------------------------------
# Analyze tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Test analysis methods."""

    def test_describe(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        stats = dp.describe()
        assert stats["shape"] == [100, 4]
        assert "statistics" in stats
        assert "sensor1" in stats["statistics"]

    def test_correlate(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        corr = dp.correlate()
        assert corr.shape == (4, 4)
        assert corr.loc["sensor1", "sensor1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExport:
    """Test data export."""

    def test_export_csv(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        out = dp.export(tmp_path / "out.csv")
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 100

    def test_export_json(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        out = dp.export(tmp_path / "out.json")
        assert out.exists()

    def test_export_parquet(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow", reason="pyarrow not installed")
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        out = dp.export(tmp_path / "out.parquet")
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 100

    def test_export_unsupported(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        with pytest.raises(ValueError, match="Unsupported"):
            dp.export(tmp_path / "out.xyz")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistory:
    """Test processing history tracking."""

    def test_history_records_operations(self, sample_df: pd.DataFrame) -> None:
        dp = DataProcessor()
        dp.load_dataframe(sample_df)
        dp.trim_time(1, 9)
        dp.apply_formula("x", "sensor1 * 2")

        history = dp.history
        assert len(history) == 3
        assert "Loaded" in history[0]
        assert "Trimmed" in history[1]
        assert "Created" in history[2]

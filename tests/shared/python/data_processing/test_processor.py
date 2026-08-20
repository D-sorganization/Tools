"""Tests for data_processing.processor DataProcessor facade.

Covers:
- DataProcessor construction
- load_dataframe / load from file
- trim_time, resample, apply_filter, apply_formula
- drop_columns, rename_columns, sort, dropna
- describe, correlate
- detect_outliers
- export to CSV/JSON
- DatasetInfo metadata
- Method chaining
- Edge cases
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("pandas")
import pandas as pd
from data_processing.processor import DataProcessor, DatasetInfo

# ── Construction & Loading ───────────────────────────────────────────────


class TestDataProcessorConstruction:
    """Test DataProcessor construction and loading."""

    def test_construction(self) -> None:
        dp = DataProcessor()
        assert dp is not None

    def test_no_data_raises(self) -> None:
        dp = DataProcessor()
        with pytest.raises(RuntimeError):
            _ = dp.dataframe

    def test_load_dataframe(self) -> None:
        dp = DataProcessor()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        dp.load_dataframe(df, name="test")
        pd.testing.assert_frame_equal(dp.dataframe, df)

    def test_load_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_path, index=False)
        dp = DataProcessor()
        dp.load(csv_path)
        assert len(dp.dataframe) == 2


# ── DatasetInfo ──────────────────────────────────────────────────────────


class TestDatasetInfo:
    """Test DatasetInfo metadata."""

    def test_info_after_load(self) -> None:
        dp = DataProcessor()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        dp.load_dataframe(df)
        info = dp.info
        assert isinstance(info, DatasetInfo)
        assert info.num_rows == 3
        assert info.num_columns == 2
        assert set(info.columns) == {"x", "y"}

    def test_info_memory_positive(self) -> None:
        dp = DataProcessor()
        df = pd.DataFrame({"x": range(1000)})
        dp.load_dataframe(df)
        assert dp.info.memory_mb > 0


# ── Transformations ──────────────────────────────────────────────────────


class TestDataProcessorTransformations:
    """Test data transformation methods."""

    @pytest.fixture()
    def dp(self) -> DataProcessor:
        dp = DataProcessor()
        df = pd.DataFrame(
            {
                "time": np.linspace(0, 10, 100),
                "signal": np.sin(np.linspace(0, 10, 100)),
                "noise": np.random.default_rng(42).normal(0, 0.1, 100),
            }
        )
        dp.load_dataframe(df, name="test")
        return dp

    def test_trim_time(self, dp: DataProcessor) -> None:
        dp.trim_time(2.0, 8.0)
        assert dp.dataframe["time"].min() >= 2.0
        assert dp.dataframe["time"].max() <= 8.0

    def test_drop_columns(self, dp: DataProcessor) -> None:
        dp.drop_columns(["noise"])
        assert "noise" not in dp.dataframe.columns
        assert "signal" in dp.dataframe.columns

    def test_rename_columns(self, dp: DataProcessor) -> None:
        dp.rename_columns({"signal": "amplitude"})
        assert "amplitude" in dp.dataframe.columns
        assert "signal" not in dp.dataframe.columns

    def test_sort(self, dp: DataProcessor) -> None:
        dp.sort("signal", ascending=True)
        values = dp.dataframe["signal"].tolist()
        assert values == sorted(values)

    def test_sort_descending(self, dp: DataProcessor) -> None:
        dp.sort("signal", ascending=False)
        values = dp.dataframe["signal"].tolist()
        assert values == sorted(values, reverse=True)

    def test_apply_formula(self, dp: DataProcessor) -> None:
        dp.apply_formula("double_signal", "signal * 2")
        assert "double_signal" in dp.dataframe.columns
        expected = dp.dataframe["signal"] * 2
        pd.testing.assert_series_equal(
            dp.dataframe["double_signal"], expected, check_names=False
        )

    def test_apply_formula_falls_back_without_numexpr(
        self, dp: DataProcessor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_eval(
            frame: pd.DataFrame, expression: str, engine: str | None = None, **_: object
        ) -> pd.Series:
            if engine == "numexpr":
                raise ImportError("No module named 'numexpr'")
            assert engine == "python"
            return frame["signal"] * 2

        monkeypatch.setattr(pd.DataFrame, "eval", fake_eval)

        dp.apply_formula("double_signal", "signal * 2")

        pd.testing.assert_series_equal(
            dp.dataframe["double_signal"],
            dp.dataframe["signal"] * 2,
            check_names=False,
        )

    def test_apply_formula_rejects_unknown_name(self, dp: DataProcessor) -> None:
        with pytest.raises(ValueError, match="Unknown formula column"):
            dp.apply_formula("bad", "signal + missing")

    def test_apply_formula_rejects_function_calls(self, dp: DataProcessor) -> None:
        with pytest.raises(
            ValueError, match="Unsupported formula syntax|contains forbidden pattern"
        ):
            dp.apply_formula("bad", "__import__('os')")

    def test_dropna(self) -> None:
        dp = DataProcessor()
        df = pd.DataFrame(
            {
                "a": [1, np.nan, 3],
                "b": [4, 5, np.nan],
            }
        )
        dp.load_dataframe(df)
        dp.dropna()
        assert len(dp.dataframe) == 1  # Only first row has no NaN

    def test_apply_filter_unknown_type_raises(self, dp: DataProcessor) -> None:
        with pytest.raises(ValueError, match="Unknown filter type"):
            dp.apply_filter("unknown_filter", columns=["signal"])

    def test_apply_filter_nonpositive_window_raises(self, dp: DataProcessor) -> None:
        with pytest.raises(ValueError, match="window_size must be positive"):
            dp.apply_filter("moving_average", columns=["signal"], window_size=0)

    def test_apply_filter_no_matching_columns_raises(self, dp: DataProcessor) -> None:
        with pytest.raises(ValueError, match="No valid columns to filter"):
            dp.apply_filter("moving_average", columns=["missing_col"])

    def test_butterworth_filter_uses_detected_sample_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, float] = {}

        fake_scipy = types.ModuleType("scipy")
        fake_signal = types.ModuleType("scipy.signal")

        def fake_butter(
            order: int, cutoff: float, *, btype: str, fs: float
        ) -> tuple[list[float], list[float]]:
            captured["fs"] = fs
            return [1.0], [1.0]

        fake_signal.butter = fake_butter
        fake_signal.filtfilt = lambda b, a, values: values
        fake_signal.medfilt = lambda values, kernel_size: values
        fake_signal.savgol_filter = lambda values, window_size, polyorder: values
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
        monkeypatch.setitem(sys.modules, "scipy.signal", fake_signal)

        dp = DataProcessor()
        dp.load_dataframe(
            pd.DataFrame(
                {
                    "time": np.arange(30, dtype=float) * 0.01,
                    "signal": np.sin(np.arange(30, dtype=float)),
                }
            )
        )

        dp.apply_filter("butterworth", columns=["signal"], cutoff=10.0, order=2)

        assert captured["fs"] == pytest.approx(100.0)


# ── Analysis Methods ─────────────────────────────────────────────────────


class TestDataProcessorAnalysis:
    """Test analysis methods."""

    @pytest.fixture()
    def dp(self) -> DataProcessor:
        dp = DataProcessor()
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, 100),
                "y": rng.normal(0, 1, 100),
            }
        )
        dp.load_dataframe(df)
        return dp

    def test_describe(self, dp: DataProcessor) -> None:
        stats = dp.describe()
        assert isinstance(stats, dict)
        assert "statistics" in stats
        assert "columns" in stats

    def test_correlate(self, dp: DataProcessor) -> None:
        corr = dp.correlate()
        assert isinstance(corr, pd.DataFrame)
        # Diagonal should be 1
        assert abs(corr.loc["x", "x"] - 1.0) < 1e-10

    def test_detect_outliers(self, dp: DataProcessor) -> None:
        outlier_mask = dp.detect_outliers(method="zscore", threshold=3.0)
        assert isinstance(outlier_mask, pd.DataFrame)
        # The mask should have the same number of rows as the data
        assert outlier_mask.shape[0] <= len(dp.dataframe)


# ── Export ───────────────────────────────────────────────────────────────


class TestDataProcessorExport:
    """Test data export functionality."""

    @pytest.fixture()
    def dp(self) -> DataProcessor:
        dp = DataProcessor()
        dp.load_dataframe(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        return dp

    def test_export_csv(self, dp: DataProcessor, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        dp.export(out)
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 3

    def test_export_json(self, dp: DataProcessor, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        dp.export(out)
        assert out.exists()


# ── History ──────────────────────────────────────────────────────────────


class TestDataProcessorHistory:
    """Test processing history tracking."""

    def test_history_empty_initially(self) -> None:
        dp = DataProcessor()
        assert dp.history == []

    def test_history_records_operations(self) -> None:
        dp = DataProcessor()
        df = pd.DataFrame({"time": [0, 1, 2], "val": [3, 4, 5]})
        dp.load_dataframe(df)
        dp.drop_columns(["val"])
        assert len(dp.history) > 0


# ── Method Chaining ──────────────────────────────────────────────────────


class TestMethodChaining:
    """Test fluent API method chaining."""

    def test_load_returns_self(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": [1]}).to_csv(csv_path, index=False)
        dp = DataProcessor()
        result = dp.load(csv_path)
        assert result is dp


# ── DbC Contract Violations ──────────────────────────────────────────────


class TestDataProcessorContracts:
    """Tests that verify DbC preconditions raise PreconditionError."""

    @pytest.fixture()
    def dp_loaded(self) -> DataProcessor:
        dp = DataProcessor()
        df = pd.DataFrame(
            {
                "time": [0.0, 0.1, 0.2, 0.3, 0.4],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        dp.load_dataframe(df, name="test")
        return dp

    # load_dataframe contracts

    def test_load_dataframe_requires_dataframe(self) -> None:
        from contracts import PreconditionError

        dp = DataProcessor()
        with pytest.raises(PreconditionError):
            dp.load_dataframe({"not": "a dataframe"})

    def test_load_dataframe_requires_non_empty_name(self) -> None:
        from contracts import PreconditionError

        dp = DataProcessor()
        with pytest.raises(PreconditionError):
            dp.load_dataframe(pd.DataFrame({"x": [1]}), name="")

    def test_load_dataframe_requires_string_name(self) -> None:
        from contracts import PreconditionError

        dp = DataProcessor()
        with pytest.raises(PreconditionError):
            dp.load_dataframe(pd.DataFrame({"x": [1]}), name=123)

    # trim_time contracts

    def test_trim_time_requires_end_gte_start(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.trim_time(0.5, 0.1)  # end < start

    def test_trim_time_requires_numeric_start(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.trim_time("start", 0.5)

    def test_trim_time_requires_numeric_end(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.trim_time(0.0, "end")

    # resample contracts

    def test_resample_zero_rate_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.resample(0)

    def test_resample_negative_rate_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.resample(-100)

    def test_resample_string_rate_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.resample("100hz")

    # apply_formula contracts

    def test_apply_formula_empty_column_rejected(
        self, dp_loaded: DataProcessor
    ) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.apply_formula("", "x + y")

    def test_apply_formula_empty_expression_rejected(
        self, dp_loaded: DataProcessor
    ) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.apply_formula("result", "")

    # drop_columns contracts

    def test_drop_columns_empty_list_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.drop_columns([])

    def test_drop_columns_non_list_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.drop_columns("x")

    # rename_columns contracts

    def test_rename_columns_empty_dict_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.rename_columns({})

    def test_rename_columns_non_dict_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.rename_columns([("x", "val")])

    # sort contracts

    def test_sort_empty_column_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.sort("")

    def test_sort_non_bool_ascending_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.sort("x", ascending=1)

    # correlate contracts

    def test_correlate_empty_method_rejected(self, dp_loaded: DataProcessor) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.correlate(method="")

    def test_correlate_non_string_method_rejected(
        self, dp_loaded: DataProcessor
    ) -> None:
        from contracts import PreconditionError

        with pytest.raises(PreconditionError):
            dp_loaded.correlate(method=None)

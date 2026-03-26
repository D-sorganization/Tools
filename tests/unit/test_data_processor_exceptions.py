"""Tests for typed exceptions in DataProcessorEngine.

Verifies that the engine raises specific exception types instead of
broad ``Exception`` catches, enabling callers to handle errors precisely.

Addresses #830 (typed errors) and #826 (DbC coverage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from upstream_drift_tools.data_processing.core import (
    DataProcessorEngine,
    FitType,
)
from upstream_drift_tools.data_processing.exceptions import (
    ColumnNotFoundError,
    DataNotLoadedError,
    DataProcessingError,
    FileIOError,
    FilterError,
    FitError,
    TransformationError,
    UnsupportedOperationError,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def engine() -> DataProcessorEngine:
    """Create a fresh DataProcessorEngine."""
    return DataProcessorEngine()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Dana", "Eve"],
            "age": [30, 25, 35, 28, 40],
            "salary": [70000.0, 55000.0, 90000.0, 65000.0, 85000.0],
            "department": ["Eng", "Eng", "Sales", "Sales", "Eng"],
        }
    )


@pytest.fixture
def loaded_engine(
    engine: DataProcessorEngine, sample_df: pd.DataFrame
) -> DataProcessorEngine:
    """Return an engine pre-loaded with sample data."""
    result = engine.load_dataframe(sample_df)
    assert result.success
    return engine


# ── Exception hierarchy tests ─────────────────────────────────────


class TestExceptionHierarchy:
    """All custom exceptions must derive from DataProcessingError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            DataNotLoadedError,
            ColumnNotFoundError,
            FileIOError,
            FilterError,
            FitError,
            TransformationError,
            UnsupportedOperationError,
        ],
    )
    def test_inherits_from_base(self, exc_class: type) -> None:
        assert issubclass(exc_class, DataProcessingError)

    def test_column_not_found_stores_metadata(self) -> None:
        exc = ColumnNotFoundError("foo", ["bar", "baz"])
        assert exc.column == "foo"
        assert "foo" in str(exc)
        assert exc.available == ["bar", "baz"]


# ── DataNotLoadedError tests ──────────────────────────────────────


class TestDataNotLoadedErrors:
    """Operations on an empty engine must raise DataNotLoadedError."""

    def test_export_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.export_data("output.csv")

    def test_add_column_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.add_calculated_column("x", "1 + 1")

    def test_rename_column_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.rename_column("old", "new")

    def test_drop_columns_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.drop_columns(["col"])

    def test_transform_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.transform_column("col", "log")

    def test_smooth_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.smooth_column("col", "moving_average")

    def test_aggregate_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            from upstream_drift_tools.data_processing.core import AggregationType

            engine.aggregate(None, None, AggregationType.MEAN)

    def test_filter_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.filter_data("col", "==", 1)

    def test_query_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.query("col > 0")

    def test_fit_curve_without_data(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(DataNotLoadedError):
            engine.fit_curve("x", "y", FitType.LINEAR)


# ── ColumnNotFoundError tests ─────────────────────────────────────


class TestColumnNotFoundErrors:
    """Operations targeting a missing column must raise ColumnNotFoundError."""

    def test_rename_missing_column(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(ColumnNotFoundError):
            loaded_engine.rename_column("nonexistent", "new")

    def test_drop_missing_column(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(ColumnNotFoundError):
            loaded_engine.drop_columns(["nonexistent"])

    def test_transform_missing_column(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(ColumnNotFoundError):
            loaded_engine.transform_column("nonexistent", "log")

    def test_smooth_missing_column(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(ColumnNotFoundError):
            loaded_engine.smooth_column("nonexistent", "moving_average")

    def test_filter_missing_column(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(ColumnNotFoundError):
            loaded_engine.filter_data("nonexistent", "==", 1)

    def test_fit_curve_missing_x_column(
        self, loaded_engine: DataProcessorEngine
    ) -> None:
        with pytest.raises(ColumnNotFoundError):
            loaded_engine.fit_curve("nonexistent", "salary", FitType.LINEAR)

    def test_fit_curve_missing_y_column(
        self, loaded_engine: DataProcessorEngine
    ) -> None:
        with pytest.raises(ColumnNotFoundError):
            loaded_engine.fit_curve("age", "nonexistent", FitType.LINEAR)


# ── UnsupportedOperationError tests ──────────────────────────────


class TestUnsupportedOperationErrors:
    def test_unknown_transform(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(UnsupportedOperationError):
            loaded_engine.transform_column("age", "quantum_transform")

    def test_unknown_smooth_method(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(UnsupportedOperationError):
            loaded_engine.smooth_column("salary", "quantum_filter")

    def test_unsupported_fit_type(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(UnsupportedOperationError):
            loaded_engine.fit_curve("age", "salary", FitType.EXPONENTIAL)


# ── FitError tests ───────────────────────────────────────────────


class TestFitErrors:
    def test_fit_insufficient_data(self) -> None:
        engine = DataProcessorEngine()
        engine.load_dataframe(pd.DataFrame({"x": [1.0], "y": [2.0]}))
        with pytest.raises(FitError, match="Need >= 2"):
            engine.fit_curve("x", "y", FitType.LINEAR)

    def test_fit_all_nan_data(self) -> None:
        engine = DataProcessorEngine()
        engine.load_dataframe(
            pd.DataFrame({"x": [np.nan, np.nan], "y": [np.nan, np.nan]})
        )
        with pytest.raises(FitError, match="Need >= 2"):
            engine.fit_curve("x", "y", FitType.LINEAR)


# ── FilterError tests ────────────────────────────────────────────


class TestFilterErrors:
    def test_empty_query_expression(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(FilterError):
            loaded_engine.query("")


# ── FileIOError tests ────────────────────────────────────────────


class TestFileIOErrors:
    def test_load_empty_path(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(FileIOError):
            engine.load_file("")

    def test_load_nonexistent_file(self, engine: DataProcessorEngine) -> None:
        with pytest.raises(FileIOError):
            engine.load_file("/this/path/does/not/exist.csv")


# ── TransformationError tests ────────────────────────────────────


class TestTransformationErrors:
    def test_add_column_bad_expression(
        self, loaded_engine: DataProcessorEngine
    ) -> None:
        with pytest.raises(TransformationError):
            loaded_engine.add_calculated_column("bad", "??? invalid +++")

    def test_add_column_empty_name(self, loaded_engine: DataProcessorEngine) -> None:
        with pytest.raises(TransformationError):
            loaded_engine.add_calculated_column("", "salary / 1000")


# ── Data integrity after errors ──────────────────────────────────


class TestDataIntegrityAfterErrors:
    """After a failed operation, data must remain unchanged (undo on error)."""

    def test_data_unchanged_after_bad_transform(
        self, loaded_engine: DataProcessorEngine
    ) -> None:
        original = loaded_engine.data.copy()
        with pytest.raises(UnsupportedOperationError):
            loaded_engine.transform_column("age", "quantum_transform")
        pd.testing.assert_frame_equal(loaded_engine.data, original)

    def test_data_unchanged_after_bad_filter(
        self, loaded_engine: DataProcessorEngine
    ) -> None:
        original = loaded_engine.data.copy()
        with pytest.raises(FilterError):
            loaded_engine.filter_data("age", "INVALID_OP", 30)
        pd.testing.assert_frame_equal(loaded_engine.data, original)

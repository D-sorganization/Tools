"""Tests for the DataProcessorEngine in upstream_drift_tools.

Ported from Gasification Model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from upstream_drift_tools.data_processing.core import (
    AggregationType,
    ColumnStats,
    DataProcessorEngine,
    FitType,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


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
            "salary": [70000, 55000, 90000, 65000, 85000],
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


class TestDataProcessorInit:
    def test_init_creates_empty_engine(self, engine: DataProcessorEngine) -> None:
        assert engine.data is None
        assert not engine.data is not None

    def test_load_dataframe_success(
        self, engine: DataProcessorEngine, sample_df: pd.DataFrame
    ) -> None:
        result = engine.load_dataframe(sample_df)
        assert result.success
        assert engine.data is not None
        assert len(engine.data) == 5


class TestDataProcessorStatistics:
    def test_basic_stats(self, loaded_engine: DataProcessorEngine) -> None:
        stats = loaded_engine._get_basic_stats()
        assert stats["rows"] == 5
        assert stats["columns"] == 4

    def test_get_statistics_returns_column_stats(
        self, loaded_engine: DataProcessorEngine
    ) -> None:
        stats = loaded_engine.get_statistics()
        assert "age" in stats
        age_stats = stats["age"]
        assert isinstance(age_stats, ColumnStats)
        assert age_stats.count == 5
        assert age_stats.mean == pytest.approx(31.6, abs=0.1)


class TestColumnOperations:
    def test_add_calculated_column(self, loaded_engine: DataProcessorEngine) -> None:
        result = loaded_engine.add_calculated_column("salary_k", "salary / 1000")
        assert result.success
        assert "salary_k" in loaded_engine.data.columns
        assert loaded_engine.data["salary_k"].iloc[0] == pytest.approx(70.0)

    def test_rename_column(self, loaded_engine: DataProcessorEngine) -> None:
        result = loaded_engine.rename_column("age", "years")
        assert result.success
        assert "years" in loaded_engine.data.columns
        assert "age" not in loaded_engine.data.columns

    def test_drop_column(self, loaded_engine: DataProcessorEngine) -> None:
        result = loaded_engine.drop_columns(["department"])
        assert result.success
        assert "department" not in loaded_engine.data.columns

    def test_transform_column_log(self, loaded_engine: DataProcessorEngine) -> None:
        result = loaded_engine.transform_column("salary", "log")
        assert result.success
        assert loaded_engine.data["salary"].iloc[0] == pytest.approx(np.log(70000), abs=0.01)


class TestFiltering:
    def test_filter_equals(self, loaded_engine: DataProcessorEngine) -> None:
        result = loaded_engine.filter_data("department", "==", "Eng")
        assert result.success
        assert len(loaded_engine.data) == 3

    def test_query_expression(self, loaded_engine: DataProcessorEngine) -> None:
        result = loaded_engine.query("age > 30 and salary > 80000")
        assert result.success
        assert len(loaded_engine.data) == 2


class TestCurveFitting:
    def test_linear_fit(self, loaded_engine: DataProcessorEngine) -> None:
        result = loaded_engine.fit_curve("age", "salary", FitType.LINEAR)
        assert result is not None
        assert result.r_squared >= 0.0
        assert len(result.coefficients) == 2


class TestUndoRedo:
    def test_undo_after_filter(self, loaded_engine: DataProcessorEngine) -> None:
        original_len = len(loaded_engine.data)
        loaded_engine.filter_data("age", ">", 30)
        assert len(loaded_engine.data) < original_len

        result = loaded_engine.reset()
        assert result.success
        assert len(loaded_engine.data) == original_len

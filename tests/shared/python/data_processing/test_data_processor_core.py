"""Tests for the core DataProcessorEngine.

Focuses on strict adherence to the new shared-component testing quality standard.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np

pytest.importorskip("pandas")
import pandas as pd
from upstream_drift_tools.data_processing.core import (
    AggregationType,
    DataProcessorEngine,
    FitType,
)
from upstream_drift_tools.data_processing.exceptions import (
    ColumnNotFoundError,
    DataNotLoadedError,
    FilterError,
    TransformationError,
)


@pytest.fixture
def empty_engine() -> DataProcessorEngine:
    """Return a fresh DataProcessorEngine."""
    return DataProcessorEngine()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a mock pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": ["cat", "dog", "cat", "bird", "dog"],
        }
    )


@pytest.fixture
def eng(sample_df: pd.DataFrame) -> DataProcessorEngine:
    """Return an engine pre-loaded with mock data."""
    engine = DataProcessorEngine()
    engine.load_dataframe(sample_df)
    return engine


class TestDataProcessorEngineSetup:
    """Tests for fundamental state logic and loading."""

    def test_initial_state_is_empty(self, empty_engine: DataProcessorEngine) -> None:
        """Engine should initialize with no data."""
        assert empty_engine.data is None
        assert empty_engine.original_data is None
        assert not empty_engine.has_data()
        assert empty_engine.get_column_names() == []
        assert empty_engine.get_numeric_columns() == []

    def test_load_dataframe_sets_state(self, eng: DataProcessorEngine) -> None:
        """Loading a dataframe should properly initialize internals."""
        assert eng.has_data()
        assert eng.get_column_names() == ["A", "B", "C"]
        assert eng.get_numeric_columns() == ["A", "B"]

    def test_reset_restores_original_data(self, eng: DataProcessorEngine) -> None:
        """Reset should revert data back to exactly what was loaded."""
        assert eng.data is not None
        eng.drop_columns(["A"])
        assert "A" not in eng.data.columns
        res = eng.reset()
        assert res.success
        assert eng.data is not None
        assert "A" in eng.data.columns


class TestDataProcessorEngineColumns:
    """Tests for column manipulation methods."""

    def test_drop_columns_success(self, eng: DataProcessorEngine) -> None:
        """Dropping a column should remove it."""
        eng.drop_columns(["C"])
        assert eng.data is not None
        assert "C" not in eng.data.columns

    def test_drop_columns_raises_not_found(self, eng: DataProcessorEngine) -> None:
        """Dropping non-existent column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError):
            eng.drop_columns(["Z_MISSING"])

    def test_rename_column_success(self, eng: DataProcessorEngine) -> None:
        """Renaming a column alters the registry appropriately."""
        eng.rename_column("A", "Alpha")
        assert eng.data is not None
        assert "Alpha" in eng.data.columns
        assert "A" not in eng.data.columns

    def test_add_calculated_column(self, eng: DataProcessorEngine) -> None:
        """Adding calculated column evaluates expressions correctly."""
        eng.add_calculated_column("A_plus_B", "A + B")
        assert eng.data is not None
        assert "A_plus_B" in eng.data.columns
        assert eng.data["A_plus_B"].iloc[0] == 11.0

    def test_add_calculated_column_falls_back_without_numexpr(
        self, eng: DataProcessorEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing numexpr should fall back to the python eval engine."""

        def fake_eval(
            frame: pd.DataFrame, expression: str, engine: str | None = None, **_: object
        ) -> pd.Series:
            if engine == "numexpr":
                raise ImportError("No module named 'numexpr'")
            assert engine == "python"
            return frame["A"] + frame["B"]

        monkeypatch.setattr(pd.DataFrame, "eval", fake_eval)

        eng.add_calculated_column("A_plus_B", "A + B")

        assert eng.data is not None
        pd.testing.assert_series_equal(
            eng.data["A_plus_B"],
            eng.data["A"] + eng.data["B"],
            check_names=False,
        )

    def test_add_calculated_column_rejects_function_calls(
        self, eng: DataProcessorEngine
    ) -> None:
        """Formula validation rejects calls before pandas eval."""
        with pytest.raises(
            TransformationError,
            match="Unsupported formula syntax|contains forbidden pattern",
        ):
            eng.add_calculated_column("bad", "__import__('os')")

    def test_add_calculated_column_rolls_back_after_rejected_formula(
        self, eng: DataProcessorEngine
    ) -> None:
        """Rejected formulas must not leave partial columns behind."""
        with pytest.raises(TransformationError):
            eng.add_calculated_column("bad", "A.__class__")

        assert eng.data is not None
        assert "bad" not in eng.data.columns

    def test_transform_column_log(self, eng: DataProcessorEngine) -> None:
        """Log transformation applies correctly to numeric columns."""
        eng.transform_column("A", "log")
        assert eng.data is not None
        assert np.isclose(eng.data["A"].iloc[0], 0.0)  # log(1.0) == 0.0

    def test_transform_column_raises_missing(self, eng: DataProcessorEngine) -> None:
        """Transformation on a missing column raises ColumnNotFoundError."""
        with pytest.raises(ColumnNotFoundError):
            eng.transform_column("GHOST", "log")

    def test_transform_column_raises_not_loaded(
        self, empty_engine: DataProcessorEngine
    ) -> None:
        """Transformation without data loaded raises DataNotLoadedError."""
        with pytest.raises(DataNotLoadedError):
            empty_engine.transform_column("A", "log")

    def test_transform_column_zero_division(self) -> None:
        """Test transformation failure."""
        engine = DataProcessorEngine()
        engine.load_dataframe(pd.DataFrame({"A": ["string", "bad"]}))
        with pytest.raises(TransformationError):
            engine.transform_column("A", "log")


class TestDataProcessorEngineAnalytics:
    """Tests for aggregating and fitting."""

    def test_aggregate_sum(self, eng: DataProcessorEngine) -> None:
        """Aggregation sum produces correct math over groups."""
        # Cat has rows [1.0, 10.0] and [3.0, 30.0]
        res = eng.aggregate(["C"], "A", AggregationType.SUM)
        assert res.success
        assert eng.data is not None
        cat_sum = eng.data.loc[eng.data["C"] == "cat", "A"].values[0]
        assert cat_sum == 4.0

    def test_aggregate_mean_all(self, eng: DataProcessorEngine) -> None:
        """Aggregation across the entire frame without group_by."""
        eng.aggregate(None, "A", AggregationType.MEAN)
        assert eng.data is not None
        assert float(eng.data["A"].iloc[0]) == 3.0

    def test_fit_curve_linear(self, eng: DataProcessorEngine) -> None:
        """Linear curve fitting produces coefficients for y = 10x."""
        res = eng.fit_curve("A", "B", FitType.LINEAR)
        # B = 10 * A, so coeff should be [10.0, 0.0] roughly
        assert np.isclose(res.coefficients[0], 10.0)
        assert res.r_squared > 0.99

    def test_fit_curve_raises_missing_column(self, eng: DataProcessorEngine) -> None:
        """Curve fitting raises when missing a column."""
        with pytest.raises(ColumnNotFoundError):
            eng.fit_curve("A", "MISSING", FitType.LINEAR)


class TestDataProcessorEngineQueries:
    """Tests for query and filter functionality."""

    def test_filter_data_operator_in(self, eng: DataProcessorEngine) -> None:
        """Filters with string operator works."""
        # Bypassing the filter by patching is not strictly necessary for unit tests of the tool itself.
        # It's an internal test of the tool.
        res = eng.query("C == 'cat'")
        assert res.success
        assert eng.data is not None
        assert len(eng.data) == 2

    def test_query_invalid_syntax_raises(self, eng: DataProcessorEngine) -> None:
        """Invalid pandas query string raises FilterError."""
        with pytest.raises(FilterError):
            eng.query("Z >>>> 10")

    def test_query_empty_string_raises(self, eng: DataProcessorEngine) -> None:
        """Empty query raises FilterError."""
        with pytest.raises(FilterError):
            eng.query("")

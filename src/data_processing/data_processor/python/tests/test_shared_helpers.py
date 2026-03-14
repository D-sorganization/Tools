"""Tests for the shared DRY helper functions in signal_processing.

TDD contract tests for compute_r_squared() and time_to_numeric().
Covers both normal operation and DbC boundary enforcement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from data_processor.contracts import PreconditionError
from data_processor.core.signal_processing import compute_r_squared, time_to_numeric

# ── time_to_numeric ─────────────────────────────────────────────────────────


class TestTimeToNumeric:
    """Tests for time_to_numeric helper."""

    def test_datetime_series_returns_seconds(self) -> None:
        """Datetime series should be converted to seconds from start."""
        ts = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:05"])
        result = time_to_numeric(pd.Series(ts))
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 5.0

    def test_numeric_series_passes_through(self) -> None:
        """Already-numeric series should pass through unchanged."""
        series = pd.Series([0.0, 1.0, 2.0, 3.0])
        result = time_to_numeric(series)
        np.testing.assert_array_almost_equal(result.values, [0.0, 1.0, 2.0, 3.0])

    def test_output_length_matches_input(self) -> None:
        """Postcondition: output length must match input."""
        series = pd.Series([10.0, 20.0, 30.0])
        result = time_to_numeric(series)
        assert len(result) == len(series)

    def test_empty_series_raises_precondition(self) -> None:
        """Precondition: empty series must raise PreconditionError."""
        with pytest.raises((PreconditionError, AssertionError)):
            time_to_numeric(pd.Series([], dtype=float))

    def test_string_timestamps_handled(self) -> None:
        """String-formatted timestamps should be coerced to numeric."""
        series = pd.Series(["1.0", "2.0", "3.0"])
        result = time_to_numeric(series)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result.values, [1.0, 2.0, 3.0])


# ── compute_r_squared ───────────────────────────────────────────────────────


class TestComputeRSquared:
    """Tests for compute_r_squared helper."""

    def test_perfect_fit_returns_one(self) -> None:
        """Perfect prediction should yield R² = 1.0."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        r2 = compute_r_squared(y, y)
        assert r2 == pytest.approx(1.0)

    def test_mean_prediction_returns_zero(self) -> None:
        """Predicting the mean yields R² ≈ 0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y, np.mean(y))
        r2 = compute_r_squared(y, y_pred)
        assert r2 == pytest.approx(0.0)

    def test_returns_float(self) -> None:
        """R² must always be a Python float."""
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        r2 = compute_r_squared(y, y_pred)
        assert isinstance(r2, float)

    def test_good_fit_between_zero_and_one(self) -> None:
        """A reasonable fit should give 0 < R² < 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        r2 = compute_r_squared(y, y_pred)
        assert 0.0 < r2 < 1.0

    def test_mismatched_lengths_raises(self) -> None:
        """Precondition: y_true and y_pred must have same length."""
        with pytest.raises((PreconditionError, AssertionError)):
            compute_r_squared(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_few_points_raises(self) -> None:
        """Precondition: need at least 2 data points."""
        with pytest.raises((PreconditionError, AssertionError)):
            compute_r_squared(np.array([1.0]), np.array([1.0]))

    def test_constant_y_returns_zero(self) -> None:
        """All-constant y should return R² = 0 (ss_tot = 0)."""
        y = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        r2 = compute_r_squared(y, y_pred)
        assert r2 == pytest.approx(0.0)

    def test_negative_r_squared_for_terrible_fit(self) -> None:
        """Very bad predictions can produce negative R²."""
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        r2 = compute_r_squared(y, y_pred)
        assert r2 < 0.0

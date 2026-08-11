# ruff: noqa: E501
"""TDD / DbC tests for VectorizedFilterEngine — issue #929.

Tests cover:
  1. DbC pre-conditions on __init__ and apply_filter_batch
  2. Correct output shapes (DRY: all filters share the same shape contract)
  3. NaN-preservation invariant for all filter types
  4. Edge cases: too-short signals, unknown filter type
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# conftest.py ensures data_processor is on sys.path via utils.path_helpers.
from data_processor.contracts import PreconditionError
from data_processor.vectorized_filter_engine import VectorizedFilterEngine

# ── Helpers ───────────────────────────────────────────────────────────────────


def _clean_signal(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n), name="signal")


def _signal_with_nans(n: int = 200, nan_frac: float = 0.1, seed: int = 42) -> pd.Series:
    s = _clean_signal(n, seed)
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(n) < nan_frac
    s[mask] = np.nan
    return s


def _make_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({"A": _clean_signal(n), "B": _clean_signal(n, seed=99)})


# ── __init__ DbC ─────────────────────────────────────────────────────────────


class TestVectorizedFilterEngineInit:
    def test_default_construction(self) -> None:
        engine = VectorizedFilterEngine()
        assert engine is not None
        assert engine.n_jobs >= 1

    def test_n_jobs_minus_one_uses_all_cores(self) -> None:
        import multiprocessing as mp

        engine = VectorizedFilterEngine(n_jobs=-1)
        assert engine.n_jobs == mp.cpu_count()

    def test_n_jobs_positive_integer_accepted(self) -> None:
        engine = VectorizedFilterEngine(n_jobs=2)
        assert engine.n_jobs == 2

    def test_n_jobs_zero_raises(self) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            VectorizedFilterEngine(n_jobs=0)

    def test_n_jobs_negative_non_minus_one_raises(self) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            VectorizedFilterEngine(n_jobs=-2)

    def test_custom_logger_accepted(self) -> None:
        messages: list[str] = []
        log_fn = messages.append  # capture stable reference
        engine = VectorizedFilterEngine(logger=log_fn, n_jobs=1)
        assert engine.logger is log_fn


# ── apply_filter_batch DbC ────────────────────────────────────────────────────


class TestApplyFilterBatchContracts:
    @pytest.fixture
    def engine(self):
        return VectorizedFilterEngine(n_jobs=1)

    def test_rejects_empty_dataframe(self, engine) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            engine.apply_filter_batch(pd.DataFrame(), "Moving Average", {})

    def test_rejects_empty_filter_type(self, engine) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            engine.apply_filter_batch(_make_df(), "", {})

    def test_rejects_whitespace_filter_type(self, engine) -> None:
        with pytest.raises((PreconditionError, ValueError)):
            engine.apply_filter_batch(_make_df(), "   ", {})

    def test_rejects_non_dict_params(self, engine) -> None:
        with pytest.raises((PreconditionError, ValueError, TypeError)):
            engine.apply_filter_batch(_make_df(), "Moving Average", None)  # type: ignore[arg-type]

    def test_unknown_filter_type_returns_original(self, engine) -> None:
        df = _make_df()
        result = engine.apply_filter_batch(df, "Ghost Filter", {})
        pd.testing.assert_frame_equal(result, df)

    def test_returns_dataframe(self, engine) -> None:
        result = engine.apply_filter_batch(_make_df(), "Moving Average", {})
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_matches_input(self, engine) -> None:
        df = _make_df()
        result = engine.apply_filter_batch(df, "Moving Average", {})
        assert result.shape == df.shape


# ── Filter correctness ────────────────────────────────────────────────────────

# DRY: parametrize over all filter types — same shape contract for every filter
FILTER_TYPES = [
    ("Moving Average", {}),
    ("Butterworth Low-pass", {"bw_order": 2, "bw_cutoff": 0.2}),
    (
        "Butterworth High-pass",
        {"bw_order": 2, "bw_cutoff": 0.2, "filter_type": "Butterworth High-pass"},
    ),
    ("Median Filter", {"median_kernel": 5}),
    ("Hampel Filter", {"hampel_window": 7, "hampel_threshold": 3.0}),
    ("Z-Score Filter", {"zscore_threshold": 3.0}),
    ("Savitzky-Golay", {"savgol_window": 11, "savgol_polyorder": 2}),
    ("Gaussian Filter", {"gaussian_sigma": 2.0}),
]


class TestFilterOutputShapes:
    """All filters must return a DataFrame with the same columns as input."""

    @pytest.fixture
    def engine(self):
        return VectorizedFilterEngine(n_jobs=1)

    @pytest.mark.parametrize("filter_type,params", FILTER_TYPES)
    def test_output_columns_preserved(
        self, engine, filter_type: str, params: dict
    ) -> None:
        df = _make_df(n=300)
        result = engine.apply_filter_batch(df, filter_type, params)
        assert list(result.columns) == list(
            df.columns
        ), f"{filter_type}: columns changed"

    @pytest.mark.parametrize("filter_type,params", FILTER_TYPES)
    def test_output_row_count_preserved(
        self, engine, filter_type: str, params: dict
    ) -> None:
        df = _make_df(n=300)
        result = engine.apply_filter_batch(df, filter_type, params)
        assert len(result) == len(
            df
        ), f"{filter_type}: row count changed {len(result)} != {len(df)}"


class TestMovingAverageCorrectness:
    """Targeted tests for Moving Average — the most DRY-critical filter."""

    @pytest.fixture
    def engine(self):
        return VectorizedFilterEngine(n_jobs=1)

    def test_constant_signal_unchanged(self, engine) -> None:
        df = pd.DataFrame({"x": np.ones(100)})
        result = engine.apply_filter_batch(df, "Moving Average", {"ma_window": 11})
        np.testing.assert_allclose(result["x"].dropna(), 1.0, rtol=1e-10)

    def test_reduces_noise(self, engine) -> None:
        rng = np.random.default_rng(0)
        clean = np.sin(np.linspace(0, 4 * np.pi, 500))
        noisy = clean + 0.5 * rng.standard_normal(500)
        df = pd.DataFrame({"x": noisy})
        result = engine.apply_filter_batch(df, "Moving Average", {"ma_window": 21})
        noise_in = float(np.std(noisy - clean))
        noise_out = float(
            np.std(result["x"].dropna().values - clean[: len(result["x"].dropna())])
        )
        assert noise_out < noise_in, "Moving average should reduce noise"

    def test_too_short_signal_returned_unchanged(self, engine) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = engine.apply_filter_batch(df, "Moving Average", {"ma_window": 11})
        pd.testing.assert_series_equal(result["x"], df["x"])


class TestNaNPreservation:
    """Invariant: NaN positions in the input must be preserved in the output."""

    @pytest.fixture
    def engine(self):
        return VectorizedFilterEngine(n_jobs=1)

    @pytest.mark.parametrize(
        "filter_type,params",
        [
            ("Moving Average", {}),
            ("Hampel Filter", {}),
            ("Z-Score Filter", {}),
        ],
    )
    def test_nan_rows_remain_nan(self, engine, filter_type: str, params: dict) -> None:
        s = _signal_with_nans(n=200, nan_frac=0.05)
        nan_idx = s.index[s.isna()]
        df = pd.DataFrame({"x": s})
        result = engine.apply_filter_batch(df, filter_type, params)
        nan_after = result["x"].index[result["x"].isna()]
        # All original NaN positions should still be NaN
        for idx in nan_idx:
            assert (
                idx in nan_after
            ), f"{filter_type}: NaN at index {idx} was filled unexpectedly"


class TestParallelVsSequentialConsistency:
    """DRY: parallel and sequential processing must give the same results."""

    def test_moving_average_parallel_matches_sequential(self) -> None:
        df = _make_df(n=500)
        engine_seq = VectorizedFilterEngine(n_jobs=1)
        engine_par = VectorizedFilterEngine(n_jobs=2)
        result_seq = engine_seq.apply_filter_batch(
            df, "Moving Average", {"ma_window": 11}
        )
        result_par = engine_par.apply_filter_batch(
            df, "Moving Average", {"ma_window": 11}
        )
        pd.testing.assert_frame_equal(
            result_seq, result_par, check_exact=False, rtol=1e-10
        )

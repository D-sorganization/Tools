"""Tests for vectorized filter engine."""

import unittest

import numpy as np
import pandas as pd
from data_processor.vectorized_filter_engine import VectorizedFilterEngine


class TestVectorizedFilterEngine(unittest.TestCase):
    @staticmethod
    def _roughness(series: pd.Series) -> float:
        return float(np.nanstd(np.diff(series.to_numpy())))

    def setUp(self) -> None:
        """Initialize the test environment with a fresh engine and synthetic data."""
        self.engine = VectorizedFilterEngine()
        self.rng = np.random.default_rng(42)

        # Create synthetic data
        self.t = np.linspace(0, 10, 100)
        self.signal = np.sin(2 * np.pi * 1.0 * self.t)  # 1 Hz sine wave
        self.noisy_signal = self.signal + self.rng.normal(0, 0.1, 100)
        self.index = pd.date_range("2024-01-01", periods=100, freq="100ms")

        self.df = pd.DataFrame(
            {"Time": self.t, "Clean": self.signal, "Noisy": self.noisy_signal},
            index=self.index,
        )

    def test_apply_moving_average_vectorized(self) -> None:
        """Test the vectorized moving average filter."""
        params = {"ma_window": 5}
        result = self.engine._apply_moving_average_vectorized(self.df["Noisy"], params)

        # Check output type and shape
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 100)
        self.assertTrue(result.index.equals(self.df.index))

        # MA should smooth the data roughly
        self.assertFalse(result.isna().all())
        self.assertLess(self._roughness(result), self._roughness(self.df["Noisy"]))

    def test_apply_butterworth_vectorized(self) -> None:
        """Test the vectorized Butterworth filter."""
        params = {
            "bw_order": 2,
            "bw_cutoff": 0.99,
            "filter_type": "Butterworth Low-pass",
        }

        result = self.engine._apply_butterworth_vectorized(self.df["Noisy"], params)
        self.assertEqual(len(result), 100)
        self.assertTrue(result.index.equals(self.df.index))
        self.assertLess(self._roughness(result), self._roughness(self.df["Noisy"]))

    def test_apply_median_vectorized(self) -> None:
        """Test the vectorized median filter."""
        params = {"median_kernel": 5}
        noisy_with_outlier = self.df["Noisy"].copy()
        noisy_with_outlier.iloc[50] = 100.0
        result = self.engine._apply_median_vectorized(noisy_with_outlier, params)
        self.assertEqual(len(result), 100)
        self.assertNotEqual(result.iloc[50], 100.0)
        self.assertLess(result.iloc[50], 2.0)

    def test_apply_hampel_vectorized(self) -> None:
        """Test the vectorized Hampel filter (outlier removal)."""
        # Add an outlier
        outlier_sig = self.df["Clean"].copy()
        outlier_sig.iloc[50] = 100.0

        params = {"hampel_window": 5, "hampel_threshold": 3.0}
        result = self.engine._apply_hampel_vectorized(outlier_sig, params)

        # Outlier should be removed (replaced)
        self.assertNotEqual(result.iloc[50], 100.0)
        self.assertLess(result.iloc[50], 2.0)

    def test_apply_zscore_vectorized(self) -> None:
        """Test the vectorized Z-score filter (outlier removal)."""
        # Add an outlier
        outlier_sig = self.df["Clean"].copy()
        outlier_sig.iloc[50] = 100.0

        params = {"zscore_threshold": 3.0, "zscore_method": "Replace with Median"}
        result = self.engine._apply_zscore_vectorized(outlier_sig, params)

        # Outlier should be removed
        self.assertNotEqual(result.iloc[50], 100.0)

    def test_apply_savgol_vectorized(self) -> None:
        """Test the vectorized Savitzky-Golay filter."""
        params = {"savgol_window": 11, "savgol_polyorder": 2}
        result = self.engine._apply_savgol_vectorized(self.df["Noisy"], params)
        self.assertEqual(len(result), 100)
        self.assertTrue(result.index.equals(self.df.index))
        self.assertLess(self._roughness(result), self._roughness(self.df["Noisy"]))

    def test_batch_processing(self) -> None:
        """Test applying filters in batch to multiple columns."""
        params = {"ma_window": 5}
        result_df = self.engine.apply_filter_batch(
            self.df, "Moving Average", params, signal_names=["Clean", "Noisy"]
        )

        self.assertEqual(result_df.shape, self.df.shape)
        self.assertTrue("Clean" in result_df.columns)
        self.assertTrue("Noisy" in result_df.columns)

        # Ensure it actually changed the noisy data (smoothing)
        self.assertFalse(result_df["Noisy"].equals(self.df["Noisy"]))
        self.assertLess(
            self._roughness(result_df["Noisy"]), self._roughness(self.df["Noisy"])
        )


if __name__ == "__main__":
    unittest.main()

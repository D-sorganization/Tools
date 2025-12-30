
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import sys
import os

# Add the project root to path to verify imports
project_root = r"c:\Users\diete\Repositories\Tools\data_processing\data_processor\python\data_processor"
sys.path.insert(0, project_root)

# Mock constants if import fails
try:
    from vectorized_filter_engine import VectorizedFilterEngine
except ImportError:
    # If standard import fails, we might need to adjust path or mock modules
    pass

class TestVectorizedFilterEngine(unittest.TestCase):
    def setUp(self):
        self.engine = VectorizedFilterEngine()
        
        # Create synthetic data
        self.t = np.linspace(0, 10, 100)
        self.signal = np.sin(2 * np.pi * 1.0 * self.t)  # 1 Hz sine wave
        self.noisy_signal = self.signal + np.random.normal(0, 0.1, 100)
        
        self.df = pd.DataFrame({
            'Time': self.t,
            'Clean': self.signal,
            'Noisy': self.noisy_signal
        })
        
    def test_apply_moving_average_vectorized(self):
        params = {"ma_window": 5}
        result = self.engine._apply_moving_average_vectorized(self.df['Clean'], params)
        
        # Check output type and shape
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 100)
        
        # MA should smooth the data roughly
        # First few and last few might be NaN depending on implementation, 
        # but the engine handles it preserving length.
        self.assertFalse(result.isna().all())
        
    def test_apply_butterworth_vectorized(self):
        params = {
            "bw_order": 2, 
            "bw_cutoff": 0.5,
            "filter_type": "Butterworth Low-pass"
        }
        # Butterworth requires calculating sampling rate or assumes one
        # The engine assumes normalized if it can't calc SR from index.
        # Let's clean up index to be datetime or basic integers.
        # Default index is RangeIndex (integers).
        
        result = self.engine._apply_butterworth_vectorized(self.df['Clean'], params)
        self.assertEqual(len(result), 100)
        
    def test_apply_median_vectorized(self):
        params = {"median_kernel": 5}
        result = self.engine._apply_median_vectorized(self.df['Clean'], params)
        self.assertEqual(len(result), 100)
        
    def test_apply_hampel_vectorized(self):
        # Add an outlier
        outlier_sig = self.df['Clean'].copy()
        outlier_sig.iloc[50] = 100.0 
        
        params = {"hampel_window": 5, "hampel_threshold": 3.0}
        result = self.engine._apply_hampel_vectorized(outlier_sig, params)
        
        # Outlier should be removed (replaced)
        self.assertNotEqual(result.iloc[50], 100.0)
        self.assertLess(result.iloc[50], 2.0)
        
    def test_apply_zscore_vectorized(self):
         # Add an outlier
        outlier_sig = self.df['Clean'].copy()
        outlier_sig.iloc[50] = 100.0 
        
        params = {
            "zscore_threshold": 3.0,
            "zscore_method": "Replace with Median"
        }
        result = self.engine._apply_zscore_vectorized(outlier_sig, params)
        
        # Outlier should be removed
        self.assertNotEqual(result.iloc[50], 100.0)
        
    def test_apply_savgol_vectorized(self):
        params = {"savgol_window": 11, "savgol_polyorder": 2}
        result = self.engine._apply_savgol_vectorized(self.df['Clean'], params)
        self.assertEqual(len(result), 100)
        
    def test_batch_processing(self):
        params = {"ma_window": 5}
        result_df = self.engine.apply_filter_batch(
            self.df, 
            "Moving Average", 
            params, 
            signal_names=['Clean', 'Noisy']
        )
        
        self.assertEqual(result_df.shape, self.df.shape)
        self.assertTrue('Clean' in result_df.columns)
        self.assertTrue('Noisy' in result_df.columns)
        
        # Ensure it actually changed the noisy data (smoothing)
        # Not guaranteed for clean sine wave to be identical, but should be close
        self.assertFalse(result_df['Noisy'].equals(self.df['Noisy']))

if __name__ == '__main__':
    unittest.main()

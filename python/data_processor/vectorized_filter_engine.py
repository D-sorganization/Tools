"""
High-Performance Vectorized Filter Engine

Optimized for chemical plant data processing with:
- Vectorized operations using NumPy/SciPy
- Batch processing of multiple signals
- Memory-efficient operations
- Parallel processing support
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Import constants
from constants import (
    DEFAULT_MA_WINDOW,
    DEFAULT_BW_ORDER,
    DEFAULT_BW_CUTOFF,
    DEFAULT_MEDIAN_KERNEL,
    DEFAULT_SAVGOL_WINDOW,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_HAMPEL_WINDOW,
    DEFAULT_HAMPEL_THRESHOLD,
    DEFAULT_ZSCORE_THRESHOLD,
    DEFAULT_ZSCORE_METHOD,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_GAUSSIAN_MODE,
    MIN_SIGNAL_DATA_POINTS,
    MIN_BUTTERWORTH_DATA_MULTIPLIER,
    NORMAL_DISTRIBUTION_CONSTANT,
)

# Optional Savitzky-Golay import with guard
try:
    from scipy.signal import savgol_filter as _savgol_filter
except ImportError:
    _savgol_filter = None


class VectorizedFilterEngine:
    """
    High-performance vectorized filter engine optimized for chemical plant data.
    
    Features:
    - Vectorized operations using NumPy/SciPy
    - Batch processing of multiple signals
    - Memory-efficient operations
    - Parallel processing support
    - Optimized for large datasets (1M+ points)
    """
    
    def __init__(self, logger: Optional[Callable] = None, n_jobs: int = -1):
        """
        Initialize the vectorized filter engine.
        
        Args:
            logger: Optional logging function. If None, uses print.
            n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        """
        self.logger = logger or print
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.filters = {
            "Moving Average": self._apply_moving_average_vectorized,
            "Butterworth Low-pass": self._apply_butterworth_vectorized,
            "Butterworth High-pass": self._apply_butterworth_vectorized,
            "Median Filter": self._apply_median_vectorized,
            "Hampel Filter": self._apply_hampel_vectorized,
            "Z-Score Filter": self._apply_zscore_vectorized,
            "Savitzky-Golay": self._apply_savgol_vectorized,
            "Gaussian Filter": self._apply_gaussian_vectorized,
        }
    
    def apply_filter_batch(self, df: pd.DataFrame, filter_type: str, 
                          params: Dict[str, Any], signal_names: List[str] = None) -> pd.DataFrame:
        """
        Apply filter to multiple signals in batch for maximum performance.
        
        Args:
            df: DataFrame containing signals
            filter_type: Type of filter to apply
            params: Filter parameters dictionary
            signal_names: List of signal names to process (None = all numeric columns)
            
        Returns:
            DataFrame with filtered signals
        """
        if filter_type not in self.filters:
            self.logger(f"Warning: Unknown filter type '{filter_type}'")
            return df
        
        # Determine signals to process
        if signal_names is None:
            signal_names = df.select_dtypes(include=np.number).columns.tolist()
        
        if not signal_names:
            return df
        
        # Create copy for results
        result_df = df.copy()
        
        # Apply filter to each signal
        if self.n_jobs == 1:
            # Sequential processing
            for signal_name in signal_names:
                if signal_name in df.columns:
                    result_df[signal_name] = self._apply_single_filter(
                        df[signal_name], filter_type, params, signal_name
                    )
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all filter tasks
                future_to_signal = {
                    executor.submit(
                        self._apply_single_filter, 
                        df[signal_name], 
                        filter_type, 
                        params, 
                        signal_name
                    ): signal_name 
                    for signal_name in signal_names 
                    if signal_name in df.columns
                }
                
                # Collect results
                for future in as_completed(future_to_signal):
                    signal_name = future_to_signal[future]
                    try:
                        result_df[signal_name] = future.result()
                    except Exception as e:
                        self.logger(f"Error processing {signal_name}: {e}")
                        result_df[signal_name] = df[signal_name]  # Keep original
        
        return result_df
    
    def _apply_single_filter(self, signal: pd.Series, filter_type: str, 
                           params: Dict[str, Any], signal_name: str = "") -> pd.Series:
        """Apply filter to a single signal."""
        # Validate signal
        clean_signal = signal.dropna()
        if len(clean_signal) < MIN_SIGNAL_DATA_POINTS:
            self.logger(f"Warning: {signal_name} too short for filtering ({len(clean_signal)} points)")
            return signal
        
        # Apply filter
        try:
            filtered = self.filters[filter_type](signal, params)
            return filtered
        except Exception as e:
            self.logger(f"Error applying {filter_type} to {signal_name}: {e}")
            return signal  # Return original on error
    
    def _apply_moving_average_vectorized(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Vectorized moving average filter using scipy.ndimage.uniform_filter1d."""
        window = self._safe_get_param(params, 'ma_window', DEFAULT_MA_WINDOW, min_val=3, max_val=1000)
        
        # Use scipy's optimized uniform filter (much faster than pandas rolling)
        clean_data = signal.dropna()
        if len(clean_data) < window:
            return signal
        
        try:
            # Vectorized operation - much faster than pandas rolling
            filtered_data = uniform_filter1d(clean_data.values, size=window, mode='nearest')
            return pd.Series(filtered_data, index=clean_data.index)
        except Exception as e:
            # Fallback to pandas rolling
            return signal.rolling(window=window, min_periods=1, center=True).mean()
    
    def _apply_butterworth_vectorized(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Vectorized Butterworth filter."""
        order = self._safe_get_param(params, 'bw_order', DEFAULT_BW_ORDER, min_val=1, max_val=10)
        cutoff = self._safe_get_param(params, 'bw_cutoff', DEFAULT_BW_CUTOFF, min_val=0.01, max_val=0.99)
        
        # Determine filter type from params
        filter_type = params.get('filter_type', 'Butterworth Low-pass')
        btype = "low" if "Low-pass" in filter_type else "high"
        
        # Calculate sampling rate
        sr = self._calculate_sampling_rate(signal)
        if sr is None or len(signal.dropna()) <= order * MIN_BUTTERWORTH_DATA_MULTIPLIER:
            self.logger(f"Warning: Insufficient data for Butterworth filter")
            return signal
        
        try:
            b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sr)
            clean_data = signal.dropna()
            filtered_data = filtfilt(b, a, clean_data.values)
            return pd.Series(filtered_data, index=clean_data.index)
        except Exception as e:
            self.logger(f"Butterworth filter failed: {e}")
            return signal
    
    def _apply_median_vectorized(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Vectorized median filter using scipy.ndimage."""
        kernel = self._safe_get_param(params, 'median_kernel', DEFAULT_MEDIAN_KERNEL, min_val=3, max_val=101)
        
        # Ensure odd kernel size
        if kernel % 2 == 0:
            kernel += 1
        
        clean_data = signal.dropna()
        if len(clean_data) <= kernel:
            self.logger(f"Warning: Signal too short for median filter (kernel={kernel})")
            return signal
        
        try:
            # Vectorized operation using scipy
            filtered_data = medfilt(clean_data.values, kernel_size=kernel)
            return pd.Series(filtered_data, index=clean_data.index)
        except Exception as e:
            self.logger(f"Median filter failed: {e}")
            return signal
    
    def _apply_hampel_vectorized(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """
        Highly optimized vectorized Hampel filter.
        
        Uses sliding window vectorized operations instead of loops.
        Performance: O(n) instead of O(n×w) for large datasets.
        """
        window = self._safe_get_param(params, 'hampel_window', DEFAULT_HAMPEL_WINDOW, min_val=3, max_val=100)
        threshold = self._safe_get_param(params, 'hampel_threshold', DEFAULT_HAMPEL_THRESHOLD, min_val=1.0, max_val=10.0)
        
        clean_data = signal.dropna()
        if len(clean_data) < window:
            self.logger(f"Warning: Signal too short for Hampel filter (window={window})")
            return signal
        
        try:
            # Use pandas rolling for exact median calculation (still much faster than loop)
            # This is a compromise between exactness and performance
            rolling_median = clean_data.rolling(window=window, center=True).median()
            rolling_mad = (clean_data - rolling_median).abs().rolling(window=window, center=True).median()
            threshold_values = threshold * NORMAL_DISTRIBUTION_CONSTANT * rolling_mad
            
            # Vectorized outlier detection and replacement
            outlier_mask = (clean_data - rolling_median).abs() > threshold_values
            
            # Create filtered signal
            filtered_signal = signal.copy()
            filtered_signal.iloc[clean_data.index[outlier_mask]] = rolling_median[outlier_mask]
            
            return filtered_signal
            
        except Exception as e:
            self.logger(f"Vectorized Hampel filter failed, using fallback: {e}")
            # Fallback to simpler approach
            return self._apply_hampel_fallback(signal, params)
    
    def _apply_hampel_fallback(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Simplified Hampel filter fallback."""
        window = self._safe_get_param(params, 'hampel_window', DEFAULT_HAMPEL_WINDOW, min_val=3, max_val=100)
        threshold = self._safe_get_param(params, 'hampel_threshold', DEFAULT_HAMPEL_THRESHOLD, min_val=1.0, max_val=10.0)
        
        clean_data = signal.dropna()
        filtered_signal = signal.copy()
        
        # Simplified approach using pandas rolling
        rolling_median = clean_data.rolling(window=window, center=True).median()
        rolling_mad = (clean_data - rolling_median).abs().rolling(window=window, center=True).median()
        threshold_values = threshold * NORMAL_DISTRIBUTION_CONSTANT * rolling_mad
        
        outlier_mask = (clean_data - rolling_median).abs() > threshold_values
        filtered_signal.iloc[clean_data.index[outlier_mask]] = rolling_median[outlier_mask]
        
        return filtered_signal
    
    def _apply_zscore_vectorized(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Vectorized Z-score filter."""
        threshold = self._safe_get_param(params, 'zscore_threshold', DEFAULT_ZSCORE_THRESHOLD, min_val=1.0, max_val=10.0)
        method = params.get('zscore_method', DEFAULT_ZSCORE_METHOD)
        
        clean_data = signal.dropna()
        if len(clean_data) < 3:
            self.logger(f"Warning: Signal too short for Z-score filter")
            return signal
        
        try:
            if method == "modified":
                # Vectorized modified Z-score using MAD
                median = np.median(clean_data.values)
                mad = np.median(np.abs(clean_data.values - median))
                z_scores = np.abs((clean_data.values - median) / (NORMAL_DISTRIBUTION_CONSTANT * mad))
            else:
                # Vectorized standard Z-score
                mean = np.mean(clean_data.values)
                std = np.std(clean_data.values)
                if std == 0:
                    return signal
                z_scores = np.abs((clean_data.values - mean) / std)
            
            # Vectorized outlier removal
            filtered_signal = signal.copy()
            outlier_mask = z_scores > threshold
            filtered_signal.iloc[clean_data.index[outlier_mask]] = np.nan
            
            return filtered_signal
        except Exception as e:
            self.logger(f"Z-score filter failed: {e}")
            return signal
    
    def _apply_savgol_vectorized(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Vectorized Savitzky-Golay filter."""
        window = self._safe_get_param(params, 'savgol_window', DEFAULT_SAVGOL_WINDOW, min_val=3, max_val=101)
        polyorder = self._safe_get_param(params, 'savgol_polyorder', DEFAULT_SAVGOL_POLYORDER, min_val=1, max_val=6)
        
        # Ensure odd window size
        if window % 2 == 0:
            window += 1
        
        # Ensure polyorder < window
        if polyorder >= window:
            polyorder = window - 1
        
        clean_data = signal.dropna()
        if len(clean_data) <= window:
            self.logger(f"Warning: Signal too short for Savitzky-Golay filter (window={window})")
            return signal
        
        if _savgol_filter is None:
            self.logger(f"Warning: scipy.signal.savgol_filter unavailable")
            return signal
        
        try:
            # Vectorized operation
            filtered_data = _savgol_filter(clean_data.values, window, polyorder)
            return pd.Series(filtered_data, index=clean_data.index)
        except Exception as e:
            self.logger(f"Savitzky-Golay filter failed: {e}")
            return signal
    
    def _apply_gaussian_vectorized(self, signal: pd.Series, params: Dict[str, Any]) -> pd.Series:
        """Vectorized Gaussian filter."""
        sigma = self._safe_get_param(params, 'gaussian_sigma', DEFAULT_GAUSSIAN_SIGMA, min_val=0.1, max_val=100.0)
        mode = params.get('gaussian_mode', DEFAULT_GAUSSIAN_MODE)
        
        clean_data = signal.dropna()
        if len(clean_data) < 2:
            self.logger(f"Warning: Signal too short for Gaussian filter")
            return signal
        
        try:
            # Vectorized operation using scipy.ndimage
            filtered_data = gaussian_filter1d(clean_data.values, sigma=sigma, mode=mode)
            return pd.Series(filtered_data, index=clean_data.index)
        except Exception as e:
            self.logger(f"Gaussian filter failed, using moving average fallback: {e}")
            # Fallback to moving average
            return signal.rolling(window=min(10, len(signal)), min_periods=1, center=True).mean()
    
    def _safe_get_param(self, params: Dict[str, Any], key: str, default: Any, 
                       min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        """Safely extract and validate parameter."""
        value = params.get(key, default)
        
        # Convert string to float if possible
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                self.logger(f"Warning: Invalid {key} value '{value}', using default {default}")
                return default
        
        # Validate numeric bounds
        if isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                self.logger(f"Warning: {key} too small ({value}), clamping to {min_val}")
                value = min_val
            if max_val is not None and value > max_val:
                self.logger(f"Warning: {key} too large ({value}), clamping to {max_val}")
                value = max_val
        
        return value
    
    def _calculate_sampling_rate(self, signal: pd.Series) -> Optional[float]:
        """Calculate sampling rate from signal index."""
        try:
            if isinstance(signal.index, pd.DatetimeIndex):
                time_diffs = signal.index.to_series().diff().dt.total_seconds()
                mean_diff = time_diffs.mean()
                if pd.notna(mean_diff) and mean_diff > 0:
                    return 1.0 / mean_diff
            return None
        except Exception:
            return None


# Backward compatibility wrapper
class FilterEngine(VectorizedFilterEngine):
    """Backward compatibility wrapper for the original FilterEngine."""
    
    def apply_filter(self, signal: pd.Series, filter_type: str, 
                     params: Dict[str, Any], signal_name: str = "") -> pd.Series:
        """Apply filter to a single signal (backward compatibility)."""
        return self._apply_single_filter(signal, filter_type, params, signal_name)


# Convenience function for backward compatibility
def apply_filter(signal: pd.Series, filter_type: str, params: Dict[str, Any], 
                signal_name: str = "", logger: Optional[Callable] = None) -> pd.Series:
    """Convenience function to apply a filter to a signal."""
    engine = VectorizedFilterEngine(logger)
    return engine._apply_single_filter(signal, filter_type, params, signal_name)

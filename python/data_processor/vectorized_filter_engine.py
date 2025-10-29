"""
High-Performance Vectorized Filter Engine

Optimized for chemical plant data processing with:
- Vectorized operations using NumPy/SciPy
- Batch processing of multiple signals
- Memory-efficient operations
- Parallel processing support
"""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, windows
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
    DEFAULT_FFT_FREQ_LOW,
    DEFAULT_FFT_FREQ_HIGH,
    DEFAULT_FFT_TRANSITION_BW,
    DEFAULT_FFT_WINDOW_SHAPE,
    DEFAULT_FFT_ZERO_PHASE,
    DEFAULT_FFT_FREQ_UNIT,
    MIN_FFT_FREQUENCY,
    MAX_FFT_FREQUENCY,
    MIN_FFT_TRANSITION_BW,
    MAX_FFT_TRANSITION_BW,
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
            "FFT Low-pass": self._apply_fft_filter_vectorized,
            "FFT High-pass": self._apply_fft_filter_vectorized,
            "FFT Band-pass": self._apply_fft_filter_vectorized,
            "FFT Band-stop": self._apply_fft_filter_vectorized,
        }

    def apply_filter_batch(
        self,
        df: pd.DataFrame,
        filter_type: str,
        params: Dict[str, Any],
        signal_names: List[str] = None,
    ) -> pd.DataFrame:
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
                        signal_name,
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

    def _apply_single_filter(
        self,
        signal: pd.Series,
        filter_type: str,
        params: Dict[str, Any],
        signal_name: str = "",
    ) -> pd.Series:
        """Apply filter to a single signal."""
        # Validate signal
        clean_signal = signal.dropna()
        if len(clean_signal) < MIN_SIGNAL_DATA_POINTS:
            self.logger(
                f"Warning: {signal_name} too short for filtering ({len(clean_signal)} points)"
            )
            return signal

        # Apply filter
        try:
            filtered = self.filters[filter_type](signal, params)
            return filtered
        except Exception as e:
            self.logger(f"Error applying {filter_type} to {signal_name}: {e}")
            return signal  # Return original on error

    def _apply_moving_average_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Vectorized moving average filter using scipy.ndimage.uniform_filter1d."""
        window = self._safe_get_param(
            params, "ma_window", DEFAULT_MA_WINDOW, min_val=3, max_val=1000
        )

        # Use scipy's optimized uniform filter (much faster than pandas rolling)
        clean_data = signal.dropna()
        if len(clean_data) < window:
            return signal

        try:
            # Vectorized operation - much faster than pandas rolling
            filtered_data = uniform_filter1d(
                clean_data.values, size=window, mode="nearest"
            )
            return pd.Series(filtered_data, index=clean_data.index)
        except Exception as e:
            # Fallback to pandas rolling
            return signal.rolling(window=window, min_periods=1, center=True).mean()

    def _apply_butterworth_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Vectorized Butterworth filter."""
        order = self._safe_get_param(
            params, "bw_order", DEFAULT_BW_ORDER, min_val=1, max_val=10
        )
        cutoff = self._safe_get_param(
            params, "bw_cutoff", DEFAULT_BW_CUTOFF, min_val=0.01, max_val=0.99
        )

        # Determine filter type from params
        filter_type = params.get("filter_type", "Butterworth Low-pass")
        btype = "low" if "Low-pass" in filter_type else "high"

        # Calculate sampling rate
        sr = self._calculate_sampling_rate(signal)
        if (
            sr is None
            or len(signal.dropna()) <= order * MIN_BUTTERWORTH_DATA_MULTIPLIER
        ):
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

    def _apply_median_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Vectorized median filter using scipy.ndimage."""
        kernel = self._safe_get_param(
            params, "median_kernel", DEFAULT_MEDIAN_KERNEL, min_val=3, max_val=101
        )

        # Ensure odd kernel size
        if kernel % 2 == 0:
            kernel += 1

        clean_data = signal.dropna()
        if len(clean_data) <= kernel:
            self.logger(
                f"Warning: Signal too short for median filter (kernel={kernel})"
            )
            return signal

        try:
            # Vectorized operation using scipy
            filtered_data = medfilt(clean_data.values, kernel_size=kernel)
            return pd.Series(filtered_data, index=clean_data.index)
        except Exception as e:
            self.logger(f"Median filter failed: {e}")
            return signal

    def _apply_hampel_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """
        Highly optimized vectorized Hampel filter.

        Uses sliding window vectorized operations instead of loops.
        Performance: O(n) instead of O(n×w) for large datasets.
        """
        window = self._safe_get_param(
            params, "hampel_window", DEFAULT_HAMPEL_WINDOW, min_val=3, max_val=100
        )
        threshold = self._safe_get_param(
            params,
            "hampel_threshold",
            DEFAULT_HAMPEL_THRESHOLD,
            min_val=1.0,
            max_val=10.0,
        )

        clean_data = signal.dropna()
        if len(clean_data) < window:
            self.logger(
                f"Warning: Signal too short for Hampel filter (window={window})"
            )
            return signal

        try:
            # Use pandas rolling for exact median calculation (still much faster than loop)
            # This is a compromise between exactness and performance
            rolling_median = clean_data.rolling(window=window, center=True).median()
            rolling_mad = (
                (clean_data - rolling_median)
                .abs()
                .rolling(window=window, center=True)
                .median()
            )
            threshold_values = threshold * NORMAL_DISTRIBUTION_CONSTANT * rolling_mad

            # Vectorized outlier detection and replacement
            outlier_mask = (clean_data - rolling_median).abs() > threshold_values

            # Create filtered signal
            filtered_signal = signal.copy()
            filtered_signal.iloc[clean_data.index[outlier_mask]] = rolling_median[
                outlier_mask
            ]

            return filtered_signal

        except Exception as e:
            self.logger(f"Vectorized Hampel filter failed, using fallback: {e}")
            # Fallback to simpler approach
            return self._apply_hampel_fallback(signal, params)

    def _apply_hampel_fallback(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Simplified Hampel filter fallback."""
        window = self._safe_get_param(
            params, "hampel_window", DEFAULT_HAMPEL_WINDOW, min_val=3, max_val=100
        )
        threshold = self._safe_get_param(
            params,
            "hampel_threshold",
            DEFAULT_HAMPEL_THRESHOLD,
            min_val=1.0,
            max_val=10.0,
        )

        clean_data = signal.dropna()
        filtered_signal = signal.copy()

        # Simplified approach using pandas rolling
        rolling_median = clean_data.rolling(window=window, center=True).median()
        rolling_mad = (
            (clean_data - rolling_median)
            .abs()
            .rolling(window=window, center=True)
            .median()
        )
        threshold_values = threshold * NORMAL_DISTRIBUTION_CONSTANT * rolling_mad

        outlier_mask = (clean_data - rolling_median).abs() > threshold_values
        filtered_signal.iloc[clean_data.index[outlier_mask]] = rolling_median[
            outlier_mask
        ]

        return filtered_signal

    def _apply_zscore_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Vectorized Z-score filter."""
        threshold = self._safe_get_param(
            params,
            "zscore_threshold",
            DEFAULT_ZSCORE_THRESHOLD,
            min_val=1.0,
            max_val=10.0,
        )
        method = params.get("zscore_method", DEFAULT_ZSCORE_METHOD)

        clean_data = signal.dropna()
        if len(clean_data) < 3:
            self.logger(f"Warning: Signal too short for Z-score filter")
            return signal

        try:
            if method == "modified":
                # Vectorized modified Z-score using MAD
                median = np.median(clean_data.values)
                mad = np.median(np.abs(clean_data.values - median))
                z_scores = np.abs(
                    (clean_data.values - median) / (NORMAL_DISTRIBUTION_CONSTANT * mad)
                )
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

    def _apply_savgol_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Vectorized Savitzky-Golay filter."""
        window = self._safe_get_param(
            params, "savgol_window", DEFAULT_SAVGOL_WINDOW, min_val=3, max_val=101
        )
        polyorder = self._safe_get_param(
            params, "savgol_polyorder", DEFAULT_SAVGOL_POLYORDER, min_val=1, max_val=6
        )

        # Ensure odd window size
        if window % 2 == 0:
            window += 1

        # Ensure polyorder < window
        if polyorder >= window:
            polyorder = window - 1

        clean_data = signal.dropna()
        if len(clean_data) <= window:
            self.logger(
                f"Warning: Signal too short for Savitzky-Golay filter (window={window})"
            )
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

    def _apply_gaussian_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """Vectorized Gaussian filter."""
        sigma = self._safe_get_param(
            params, "gaussian_sigma", DEFAULT_GAUSSIAN_SIGMA, min_val=0.1, max_val=100.0
        )
        mode = params.get("gaussian_mode", DEFAULT_GAUSSIAN_MODE)

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
            return signal.rolling(
                window=min(10, len(signal)), min_periods=1, center=True
            ).mean()

    def _safe_get_param(
        self,
        params: Dict[str, Any],
        key: str,
        default: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Any:
        """Safely extract and validate parameter."""
        value = params.get(key, default)

        # Convert string to float if possible
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                self.logger(
                    f"Warning: Invalid {key} value '{value}', using default {default}"
                )
                return default

        # Validate numeric bounds
        if isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                self.logger(
                    f"Warning: {key} too small ({value}), clamping to {min_val}"
                )
                value = min_val
            if max_val is not None and value > max_val:
                self.logger(
                    f"Warning: {key} too large ({value}), clamping to {max_val}"
                )
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

    def _apply_fft_filter_vectorized(
        self, signal: pd.Series, params: Dict[str, Any]
    ) -> pd.Series:
        """
        Comprehensive FFT-based frequency domain filtering.

        Supports:
        - Filter types: Low-pass, High-pass, Band-pass, Band-stop
        - Window functions: Gaussian, Rectangular, Hamming, Hann, Blackman, Kaiser, Tukey, Bartlett
        - Custom frequency bands with adjustable transition bandwidth
        - Phase-preserving option (zero-phase filtering)
        """
        # Extract parameters
        filter_type = params.get("filter_type", "FFT Low-pass")
        window_shape = params.get("fft_window_shape", DEFAULT_FFT_WINDOW_SHAPE)
        freq_low = self._safe_get_param(
            params,
            "fft_freq_low",
            DEFAULT_FFT_FREQ_LOW,
            min_val=MIN_FFT_FREQUENCY,
            max_val=MAX_FFT_FREQUENCY,
        )
        freq_high = self._safe_get_param(
            params,
            "fft_freq_high",
            DEFAULT_FFT_FREQ_HIGH,
            min_val=MIN_FFT_FREQUENCY,
            max_val=MAX_FFT_FREQUENCY,
        )
        transition_bw = self._safe_get_param(
            params,
            "fft_transition_bw",
            DEFAULT_FFT_TRANSITION_BW,
            min_val=MIN_FFT_TRANSITION_BW,
            max_val=MAX_FFT_TRANSITION_BW,
        )
        zero_phase = params.get("fft_zero_phase", DEFAULT_FFT_ZERO_PHASE)
        freq_unit = params.get("fft_freq_unit", DEFAULT_FFT_FREQ_UNIT)

        clean_data = signal.dropna()
        if len(clean_data) < 4:
            self.logger(f"Warning: Signal too short for FFT filter")
            return signal

        try:
            # Calculate sampling rate if needed
            sample_rate = None
            if freq_unit == "Hz":
                sample_rate = self._calculate_sampling_rate(signal)
                if sample_rate is None:
                    self.logger(
                        f"Warning: Cannot determine sample rate, using normalized frequencies"
                    )
                    freq_unit = "normalized"

            # Convert Hz to normalized if needed
            if freq_unit == "Hz" and sample_rate is not None:
                freq_low = freq_low / (sample_rate / 2)
                freq_high = freq_high / (sample_rate / 2)
                transition_bw = transition_bw / (sample_rate / 2)

            # Ensure frequency bounds
            freq_low = max(0.0, min(freq_low, 0.5))
            freq_high = max(freq_low, min(freq_high, 0.5))

            # Design frequency window
            filter_coeffs = self._design_frequency_window(
                filter_type,
                freq_low,
                freq_high,
                window_shape,
                len(clean_data),
                transition_bw,
            )

            # Apply FFT filter
            filtered_data = self._apply_fft_filter_core(
                clean_data.values, filter_coeffs, zero_phase
            )

            return pd.Series(filtered_data, index=clean_data.index)

        except Exception as e:
            self.logger(f"FFT filter failed: {e}")
            return signal

    def _design_frequency_window(
        self,
        filter_type: str,
        freq_low: float,
        freq_high: float,
        window_shape: str,
        n_samples: int,
        transition_bw: float,
    ) -> np.ndarray:
        """
        Design frequency domain window for FFT filtering.

        Args:
            filter_type: Type of filter (Low-pass, High-pass, Band-pass, Band-stop)
            freq_low: Lower cutoff frequency (normalized)
            freq_high: Upper cutoff frequency (normalized)
            window_shape: Window function type
            n_samples: Number of samples in signal
            transition_bw: Transition bandwidth (normalized)

        Returns:
            Frequency domain filter coefficients
        """
        # Create frequency array
        freqs = np.fft.fftfreq(n_samples)
        freqs = np.abs(freqs)  # Use positive frequencies only

        # Initialize filter response
        filter_response = np.zeros_like(freqs)

        # Design ideal filter response
        if filter_type == "FFT Low-pass":
            filter_response[freqs <= freq_low] = 1.0
            # Add transition band
            transition_mask = (freqs > freq_low) & (freqs <= freq_low + transition_bw)
            filter_response[transition_mask] = 0.5 * (
                1 + np.cos(np.pi * (freqs[transition_mask] - freq_low) / transition_bw)
            )

        elif filter_type == "FFT High-pass":
            filter_response[freqs >= freq_high] = 1.0
            # Add transition band
            transition_mask = (freqs >= freq_high - transition_bw) & (freqs < freq_high)
            filter_response[transition_mask] = 0.5 * (
                1
                - np.cos(
                    np.pi
                    * (freqs[transition_mask] - freq_high + transition_bw)
                    / transition_bw
                )
            )

        elif filter_type == "FFT Band-pass":
            filter_response[(freqs >= freq_low) & (freqs <= freq_high)] = 1.0
            # Add transition bands
            low_transition = (freqs > freq_low - transition_bw) & (freqs <= freq_low)
            high_transition = (freqs >= freq_high) & (freqs < freq_high + transition_bw)
            filter_response[low_transition] = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (freqs[low_transition] - freq_low + transition_bw)
                    / transition_bw
                )
            )
            filter_response[high_transition] = 0.5 * (
                1 - np.cos(np.pi * (freqs[high_transition] - freq_high) / transition_bw)
            )

        elif filter_type == "FFT Band-stop":
            filter_response[(freqs < freq_low) | (freqs > freq_high)] = 1.0
            # Add transition bands
            low_transition = (freqs >= freq_low) & (freqs < freq_low + transition_bw)
            high_transition = (freqs > freq_high - transition_bw) & (freqs <= freq_high)
            filter_response[low_transition] = 0.5 * (
                1 - np.cos(np.pi * (freqs[low_transition] - freq_low) / transition_bw)
            )
            filter_response[high_transition] = 0.5 * (
                1
                + np.cos(
                    np.pi
                    * (freqs[high_transition] - freq_high + transition_bw)
                    / transition_bw
                )
            )

        # Apply window function to smooth the response
        if window_shape != "Rectangular":
            filter_response = self._apply_window_function(filter_response, window_shape)

        return filter_response

    def _apply_window_function(
        self, filter_response: np.ndarray, window_shape: str
    ) -> np.ndarray:
        """Apply window function to smooth frequency response."""
        n = len(filter_response)

        if window_shape == "Gaussian":
            # Gaussian window
            sigma = n / 8  # Adjust sigma for smoothness
            window = np.exp(-0.5 * ((np.arange(n) - n / 2) / sigma) ** 2)

        elif window_shape == "Hamming":
            window = windows.hamming(n)

        elif window_shape == "Hann":
            window = windows.hann(n)

        elif window_shape == "Blackman":
            window = windows.blackman(n)

        elif window_shape == "Kaiser":
            window = windows.kaiser(n, beta=8.6)  # Beta for good stopband attenuation

        elif window_shape == "Tukey":
            window = windows.tukey(n, alpha=0.5)

        elif window_shape == "Bartlett":
            window = windows.bartlett(n)

        else:  # Rectangular or unknown
            return filter_response

        # Apply window smoothing (convolve with window)
        # Use FFT-based convolution for efficiency
        window_fft = np.fft.fft(window)
        response_fft = np.fft.fft(filter_response)
        smoothed_fft = response_fft * window_fft
        smoothed_response = np.real(np.fft.ifft(smoothed_fft))

        # Normalize to maintain magnitude
        smoothed_response = smoothed_response / np.max(smoothed_response)

        return smoothed_response

    def _apply_fft_filter_core(
        self, signal_data: np.ndarray, filter_coeffs: np.ndarray, zero_phase: bool
    ) -> np.ndarray:
        """
        Core FFT filtering implementation.

        Args:
            signal_data: Input signal data
            filter_coeffs: Frequency domain filter coefficients
            zero_phase: Whether to use zero-phase filtering

        Returns:
            Filtered signal data
        """
        # Ensure filter coefficients match signal length
        if len(filter_coeffs) != len(signal_data):
            # Interpolate filter coefficients to match signal length
            old_indices = np.linspace(0, len(filter_coeffs) - 1, len(filter_coeffs))
            new_indices = np.linspace(0, len(filter_coeffs) - 1, len(signal_data))
            filter_coeffs = np.interp(new_indices, old_indices, filter_coeffs)

        # Apply filter in frequency domain
        signal_fft = np.fft.fft(signal_data)
        filtered_fft = signal_fft * filter_coeffs

        if zero_phase:
            # Zero-phase filtering: apply filter forward and backward
            filtered_signal = np.real(np.fft.ifft(filtered_fft))
            # Apply filter again in reverse direction
            filtered_fft_rev = np.fft.fft(filtered_signal[::-1])
            filtered_fft_rev = filtered_fft_rev * filter_coeffs
            filtered_signal_rev = np.real(np.fft.ifft(filtered_fft_rev))
            filtered_signal = filtered_signal_rev[::-1]
        else:
            # Linear phase filtering
            filtered_signal = np.real(np.fft.ifft(filtered_fft))

        return filtered_signal

    def calculate_frequency_response(
        self, filter_type: str, params: Dict[str, Any], n_freqs: int = 1024
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response of FFT filter for preview.

        Args:
            filter_type: Type of filter
            params: Filter parameters
            n_freqs: Number of frequency points

        Returns:
            Tuple of (frequencies, magnitude_response)
        """
        try:
            # Extract parameters
            window_shape = params.get("fft_window_shape", DEFAULT_FFT_WINDOW_SHAPE)
            freq_low = self._safe_get_param(
                params,
                "fft_freq_low",
                DEFAULT_FFT_FREQ_LOW,
                min_val=MIN_FFT_FREQUENCY,
                max_val=MAX_FFT_FREQUENCY,
            )
            freq_high = self._safe_get_param(
                params,
                "fft_freq_high",
                DEFAULT_FFT_FREQ_HIGH,
                min_val=MIN_FFT_FREQUENCY,
                max_val=MAX_FFT_FREQUENCY,
            )
            transition_bw = self._safe_get_param(
                params,
                "fft_transition_bw",
                DEFAULT_FFT_TRANSITION_BW,
                min_val=MIN_FFT_TRANSITION_BW,
                max_val=MAX_FFT_TRANSITION_BW,
            )

            # Design filter
            filter_coeffs = self._design_frequency_window(
                filter_type, freq_low, freq_high, window_shape, n_freqs, transition_bw
            )

            # Create frequency array
            freqs = np.fft.fftfreq(n_freqs)
            freqs = freqs[: n_freqs // 2]  # Positive frequencies only

            # Get magnitude response
            magnitude = np.abs(filter_coeffs[: n_freqs // 2])

            return freqs, magnitude

        except Exception as e:
            self.logger(f"Frequency response calculation failed: {e}")
            return np.array([]), np.array([])


# Backward compatibility wrapper
class FilterEngine(VectorizedFilterEngine):
    """Backward compatibility wrapper for the original FilterEngine."""

    def apply_filter(
        self,
        signal: pd.Series,
        filter_type: str,
        params: Dict[str, Any],
        signal_name: str = "",
    ) -> pd.Series:
        """Apply filter to a single signal (backward compatibility)."""
        return self._apply_single_filter(signal, filter_type, params, signal_name)


# Convenience function for backward compatibility
def apply_filter(
    signal: pd.Series,
    filter_type: str,
    params: Dict[str, Any],
    signal_name: str = "",
    logger: Optional[Callable] = None,
) -> pd.Series:
    """Convenience function to apply a filter to a signal."""
    engine = VectorizedFilterEngine(logger)
    return engine._apply_single_filter(signal, filter_type, params, signal_name)

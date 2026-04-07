"""Facade for the vectorized filter engine and its extracted filter modules."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

try:
    from .constants import MIN_SIGNAL_DATA_POINTS
    from .vectorized_filter_frequency_domain import (
        apply_fft_filter_core,
        apply_fft_filter_vectorized,
        apply_window_function,
        calculate_frequency_response,
        design_frequency_window,
    )
    from .vectorized_filter_time_domain import (
        apply_butterworth_vectorized,
        apply_gaussian_vectorized,
        apply_hampel_fallback,
        apply_hampel_vectorized,
        apply_median_vectorized,
        apply_moving_average_vectorized,
        apply_savgol_vectorized,
        apply_zscore_vectorized,
    )
except ImportError:
    from constants import MIN_SIGNAL_DATA_POINTS  # type: ignore[no-redef]
    from vectorized_filter_frequency_domain import (  # type: ignore[no-redef]
        apply_fft_filter_core,
        apply_fft_filter_vectorized,
        apply_window_function,
        calculate_frequency_response,
        design_frequency_window,
    )
    from vectorized_filter_time_domain import (  # type: ignore[no-redef]
        apply_butterworth_vectorized,
        apply_gaussian_vectorized,
        apply_hampel_fallback,
        apply_hampel_vectorized,
        apply_median_vectorized,
        apply_moving_average_vectorized,
        apply_savgol_vectorized,
        apply_zscore_vectorized,
    )

try:
    from data_processor.contracts import require
except ImportError:
    try:
        from contracts import require  # type: ignore[no-redef]
    except ImportError:

        def require(
            condition: bool,
            message: str,
            value: object = None,
        ) -> None:  # type: ignore[misc]
            if not condition:
                raise ValueError(
                    f"[DbC pre-condition] {message}"
                    + (f" (got: {value!r})" if value is not None else "")
                )


class VectorizedFilterEngine:
    """High-performance batch filter facade with DbC guards at the API boundary."""

    def __init__(
        self,
        logger: Callable[..., Any] | None = None,
        n_jobs: int = -1,
    ) -> None:
        """Initialize the engine with a logger and worker count."""
        if n_jobs is None:
            raise ValueError("n_jobs must be provided")
        require(
            n_jobs == -1 or (isinstance(n_jobs, int) and n_jobs >= 1),
            "n_jobs must be -1 (auto) or a positive integer",
            n_jobs,
        )
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
        params: dict[str, Any],
        signal_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Apply a filter to each selected numeric column."""
        if df is None:
            raise ValueError("df must be provided")
        require(
            isinstance(df, pd.DataFrame) and not df.empty,
            "df must be a non-empty DataFrame",
        )
        require(
            isinstance(filter_type, str) and bool(filter_type.strip()),
            "filter_type must be a non-empty string",
            filter_type,
        )
        require(isinstance(params, dict), "params must be a dict", type(params))
        if filter_type not in self.filters:
            self.logger(f"Warning: Unknown filter type '{filter_type}'")
            return df

        if signal_names is None:
            signal_names = df.select_dtypes(include=np.number).columns.tolist()
        if not signal_names:
            return df

        result_df = df.copy()
        if self.n_jobs == 1:
            for signal_name in signal_names:
                if signal_name in df.columns:
                    result_df[signal_name] = self._apply_single_filter(
                        df[signal_name],
                        filter_type,
                        params,
                        signal_name,
                    )
            return result_df

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
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
            for future in as_completed(future_to_signal):
                signal_name = future_to_signal[future]
                try:
                    result_df[signal_name] = future.result()
                except (ValueError, TypeError, RuntimeError, KeyError) as error:
                    self.logger(f"Error processing {signal_name}: {error}")
                    result_df[signal_name] = df[signal_name]
        return result_df

    def _apply_single_filter(
        self,
        signal: pd.Series,
        filter_type: str,
        params: dict[str, Any],
        signal_name: str = "",
    ) -> pd.Series:
        """Apply one configured filter to a single signal."""
        if signal is None:
            raise ValueError("signal must be provided")
        clean_signal = signal.dropna()
        if len(clean_signal) < MIN_SIGNAL_DATA_POINTS:
            self.logger(
                f"Warning: {signal_name} too short for filtering "
                f"({len(clean_signal)} points)",
            )
            return signal
        try:
            return self.filters[filter_type](signal, params)
        except (ValueError, TypeError, RuntimeError, KeyError) as error:
            self.logger(f"Error applying {filter_type} to {signal_name}: {error}")
            return signal

    def _apply_moving_average_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_moving_average_vectorized(
            signal,
            params,
            self._safe_get_param,
            self.logger,
        )

    def _apply_butterworth_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_butterworth_vectorized(
            signal,
            params,
            self._safe_get_param,
            self._calculate_sampling_rate,
            self.logger,
        )

    def _apply_median_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_median_vectorized(
            signal, params, self._safe_get_param, self.logger
        )

    def _apply_hampel_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_hampel_vectorized(
            signal, params, self._safe_get_param, self.logger
        )

    def _apply_hampel_fallback(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_hampel_fallback(signal, params, self._safe_get_param)

    def _apply_zscore_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_zscore_vectorized(
            signal, params, self._safe_get_param, self.logger
        )

    def _apply_savgol_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_savgol_vectorized(
            signal, params, self._safe_get_param, self.logger
        )

    def _apply_gaussian_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_gaussian_vectorized(
            signal, params, self._safe_get_param, self.logger
        )

    def _safe_get_param(
        self,
        params: dict[str, Any],
        key: str,
        default: Any,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> Any:
        """Safely extract and clamp numeric parameters."""
        if params is None:
            raise ValueError("params must be provided")
        value = params.get(key, default)

        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                self.logger(
                    f"Warning: Invalid {key} value '{value}', using default {default}",
                )
                return default

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

    def _calculate_sampling_rate(self, signal: pd.Series) -> float | None:
        """Calculate a sample rate from a datetime index when available."""
        try:
            if isinstance(signal.index, pd.DatetimeIndex):
                time_diffs = signal.index.to_series().diff().dt.total_seconds()
                mean_diff = time_diffs.mean()
                if pd.notna(mean_diff) and mean_diff > 0:
                    return float(1.0 / mean_diff)
            return None
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    def _apply_fft_filter_vectorized(
        self,
        signal: pd.Series,
        params: dict[str, Any],
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return apply_fft_filter_vectorized(
            signal,
            params,
            self._safe_get_param,
            self._calculate_sampling_rate,
            self.logger,
        )

    def _design_frequency_window(
        self,
        filter_type: str,
        freq_low: float,
        freq_high: float,
        window_shape: str,
        n_samples: int,
        transition_bw: float,
    ) -> np.ndarray[Any, Any]:
        if filter_type is None:
            raise ValueError("filter_type must be provided")
        return design_frequency_window(
            filter_type,
            freq_low,
            freq_high,
            window_shape,
            n_samples,
            transition_bw,
        )

    def _apply_window_function(
        self,
        filter_response: np.ndarray[Any, Any],
        window_shape: str,
    ) -> np.ndarray[Any, Any]:
        if filter_response is None:
            raise ValueError("filter_response must be provided")
        return apply_window_function(filter_response, window_shape)

    def _apply_fft_filter_core(
        self,
        signal_data: np.ndarray[Any, Any],
        filter_coeffs: np.ndarray[Any, Any],
        zero_phase: bool,
    ) -> np.ndarray[Any, Any]:
        if signal_data is None:
            raise ValueError("signal_data must be provided")
        return apply_fft_filter_core(signal_data, filter_coeffs, zero_phase)

    def calculate_frequency_response(
        self,
        filter_type: str,
        params: dict[str, Any],
        n_freqs: int = 1024,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate a preview response for an FFT filter."""
        return calculate_frequency_response(
            filter_type,
            params,
            self._safe_get_param,
            self.logger,
            n_freqs=n_freqs,
        )


class FilterEngine(VectorizedFilterEngine):
    """Backward compatibility wrapper for the original filter engine name."""

    def apply_filter(
        self,
        signal: pd.Series,
        filter_type: str,
        params: dict[str, Any],
        signal_name: str = "",
    ) -> pd.Series:
        if signal is None:
            raise ValueError("signal must be provided")
        return self._apply_single_filter(signal, filter_type, params, signal_name)


def apply_filter(
    signal: pd.Series,
    filter_type: str,
    params: dict[str, Any],
    signal_name: str = "",
    logger: Callable[..., Any] | None = None,
) -> pd.Series:
    """Convenience function for compatibility with older call sites."""
    if signal is None:
        raise ValueError("signal must be provided")
    engine = VectorizedFilterEngine(logger)
    return engine._apply_single_filter(signal, filter_type, params, signal_name)

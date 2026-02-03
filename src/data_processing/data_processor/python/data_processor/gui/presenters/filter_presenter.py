"""Filter presenter - handles signal filtering operations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import FilterConfig
from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Available filter types
FILTER_TYPES = [
    "Moving Average",
    "Butterworth Low-pass",
    "Butterworth High-pass",
    "Median Filter",
    "Hampel Filter",
    "Z-Score Filter",
    "Savitzky-Golay",
    "Gaussian Filter",
    "FFT Low-pass",
    "FFT High-pass",
]


class FilterPresenter(QObject):
    """Presenter for signal filtering operations."""

    # Signals
    filter_applied = pyqtSignal(object)  # Filtered DataFrame
    filter_failed = pyqtSignal(str)  # Error message
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the filter presenter."""
        super().__init__(parent)
        self.signal_processor = SignalProcessor()

    def apply_filter(
        self,
        df: pd.DataFrame,
        signals: list[str],
        config: dict[str, Any],
    ) -> None:
        """Apply filter to signals in DataFrame."""
        self.processing_started.emit()

        try:
            filtered_df = self._apply_filter(df, signals, config)
            self.filter_applied.emit(filtered_df)
        except Exception as e:
            logger.exception("Error applying filter")
            self.filter_failed.emit(str(e))
        finally:
            self.processing_finished.emit()

    def _apply_filter(
        self,
        df: pd.DataFrame,
        signals: list[str],
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Apply filter to signals."""
        filter_config = self._build_filter_config(config)
        subset = self._select_signals(df, signals)
        filtered = self.signal_processor.apply_filter(subset, filter_config)
        return self._merge_filtered(df, filtered, signals)

    def _build_filter_config(self, config: dict[str, Any]) -> FilterConfig:
        """Build FilterConfig from dictionary."""
        return FilterConfig(
            filter_type=config["filter_type"],
            parameters=config.get("parameters", {}),
        )

    def _select_signals(self, df: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
        """Select signals from DataFrame."""
        valid_signals = [s for s in signals if s in df.columns]
        if not valid_signals:
            valid_signals = df.select_dtypes(include="number").columns.tolist()
        return df[valid_signals].copy()

    def _merge_filtered(
        self,
        original: pd.DataFrame,
        filtered: pd.DataFrame,
        signals: list[str],
    ) -> pd.DataFrame:
        """Merge filtered signals back into original DataFrame."""
        result = original.copy()
        for signal in signals:
            if signal in filtered.columns:
                result[signal] = filtered[signal]
        return result

    def get_filter_types(self) -> list[str]:
        """Get list of available filter types."""
        return FILTER_TYPES.copy()

    def get_default_params(self, filter_type: str) -> dict[str, Any]:
        """Get default parameters for a filter type."""
        defaults = {
            "Moving Average": {"ma_window": 10},
            "Butterworth Low-pass": {"bw_order": 3, "bw_cutoff": 0.1},
            "Butterworth High-pass": {"bw_order": 3, "bw_cutoff": 0.1},
            "Median Filter": {"median_kernel": 5},
            "Hampel Filter": {"hampel_window": 5, "hampel_threshold": 3.0},
            "Z-Score Filter": {"zscore_threshold": 3.0},
            "Savitzky-Golay": {"savgol_window": 5, "savgol_polyorder": 2},
            "Gaussian Filter": {"gaussian_sigma": 1.0},
        }
        return defaults.get(filter_type, {})

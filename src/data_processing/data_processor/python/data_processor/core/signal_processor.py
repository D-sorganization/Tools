"""Signal processing orchestration around the vectorized filter engine.

Design by Contract (DbC) guards on all public API boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data_processor.contracts import require
from data_processor.core.signal_processing import (
    apply_custom_variable,
    differentiate_signals,
    integrate_signals,
)
from data_processor.logging_config import get_logger
from data_processor.models import FilterConfig
from data_processor.vectorized_filter_engine import VectorizedFilterEngine


@dataclass
class SignalProcessor:
    """Apply declarative filters to signal data frames."""

    filter_engine: VectorizedFilterEngine | None = None

    def __post_init__(self) -> None:
        """Initialize the logger and filter engine after dataclass initialization."""
        self.logger = get_logger(__name__)
        if self.filter_engine is None:
            # Default to sequential processing for predictable resource usage.
            self.filter_engine = VectorizedFilterEngine(
                logger=self.logger.warning, n_jobs=1
            )

    def apply_filter(self, df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Apply the requested filter to the provided dataframe.

        **Pre-conditions** (DbC):
          - ``df`` must not be empty.
          - ``config.filter_type`` must be supported by the engine.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        require(not df.empty, "Cannot filter an empty dataframe")
        engine_params = config.to_engine_parameters()

        # Ensure filter_engine is not None before accessing attributes
        if self.filter_engine is None:
            self.filter_engine = VectorizedFilterEngine(
                logger=self.logger.warning, n_jobs=1
            )

        require(
            config.filter_type in self.filter_engine.filters,
            f"Filter '{config.filter_type}' is not supported by the filter engine",
            config.filter_type,
        )

        # Ensure the filter type is part of the payload so downstream logic can
        # differentiate between high/low pass behaviors.
        engine_params.setdefault("filter_type", config.filter_type)

        self.logger.info(
            "Applying filter",
            extra={
                "filter_type": config.filter_type,
                "parameters": engine_params,
            },
        )

        result: pd.DataFrame = self.filter_engine.apply_filter_batch(
            df,
            config.filter_type,
            engine_params,
        )
        return result

    def validate_signals_exist(self, df: pd.DataFrame, signals: list[str]) -> None:
        """Raise a descriptive error when required signals are missing."""
        missing = [signal for signal in signals if signal not in df.columns]
        if missing:
            msg = f"Missing required signals: {', '.join(sorted(missing))}"
            raise ValueError(msg)

    def integrate(
        self,
        df: pd.DataFrame,
        time_col: str,
        signals: list[str],
        method: str = "trapezoidal",
    ) -> pd.DataFrame:
        """Integrate specified signals over time.

        **Pre-conditions** (DbC):
          - ``df`` must not be empty.
          - ``time_col`` must exist in ``df``.

        Delegates to ``signal_processing.integrate_signals``.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        require(not df.empty, "Cannot integrate an empty dataframe")
        require(time_col in df.columns, f"time_col '{time_col}' not in df", time_col)
        return integrate_signals(df, time_col, signals, method)

    def differentiate(
        self,
        df: pd.DataFrame,
        time_col: str,
        signals: list[str],
        method: str = "spline",
        orders: list[int] | None = None,
        window_size: int = 11,
        poly_order: int = 3,
    ) -> pd.DataFrame:
        """Differentiate specified signals.

        **Pre-conditions** (DbC):
          - ``df`` must not be empty.
          - ``time_col`` must exist in ``df``.

        Delegates to ``signal_processing.differentiate_signals``.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        require(not df.empty, "Cannot differentiate an empty dataframe")
        require(time_col in df.columns, f"time_col '{time_col}' not in df", time_col)
        return differentiate_signals(
            df, time_col, signals, method, orders, window_size, poly_order
        )

    def apply_formula(
        self, df: pd.DataFrame, formula: str, new_column: str
    ) -> pd.DataFrame:
        """Apply a custom formula to the dataframe.

        **Pre-conditions** (DbC):
          - ``df`` must not be empty.
          - ``formula`` must be a non-empty string.
          - ``new_column`` must be a non-empty string.

        Delegates to ``signal_processing.apply_custom_variable``.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        require(not df.empty, "Cannot apply formula to an empty dataframe")
        require(bool(formula.strip()), "formula must be a non-empty string", formula)
        require(
            bool(new_column.strip()),
            "new_column must be a non-empty string",
            new_column,
        )
        return apply_custom_variable(df, new_column, formula)

    def detect_signal_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute descriptive statistics for all numeric signals.

        **Pre-conditions** (DbC):
          - ``df`` must not be empty.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        require(not df.empty, "Cannot compute statistics on an empty dataframe")
        numeric_df = df.select_dtypes(include=[np.number])
        stats: dict[str, Any] = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            stats[col] = {
                "count": int(len(series)),
                "mean": float(series.mean()) if len(series) > 0 else 0.0,
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()) if len(series) > 0 else 0.0,
                "max": float(series.max()) if len(series) > 0 else 0.0,
                "median": float(series.median()) if len(series) > 0 else 0.0,
            }
        return stats

    # -- Backward-compatible aliases for existing GUI callers --

    def integrate_signals(self, df: pd.DataFrame, config: Any) -> pd.DataFrame:
        """Legacy alias for ``integrate()``.

        Accepts an IntegrationConfig (or any object with ``signals``
        and ``method`` attributes) and delegates to the real implementation.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        signals = getattr(config, "signals", [])
        method = getattr(config, "method", "trapezoidal")
        # Try to detect a time column from the df index or first column
        time_col = df.index.name or df.columns[0]
        return self.integrate(df, str(time_col), signals, method)

    def differentiate_signals(self, df: pd.DataFrame, config: Any) -> pd.DataFrame:
        """Legacy alias for ``differentiate()``.

        Accepts a DifferentiationConfig (or any object with ``signals``
        and ``method``/``order`` attributes) and delegates.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        signals = getattr(config, "signals", [])
        method = getattr(config, "method", "spline")
        order = getattr(config, "order", 1)
        time_col = df.index.name or df.columns[0]
        return self.differentiate(df, str(time_col), signals, method, orders=[order])

    def apply_custom_formula(
        self, df: pd.DataFrame, name: str, formula: str
    ) -> tuple[pd.DataFrame, bool]:
        """Legacy alias for ``apply_formula()``.

        Returns ``(result_df, success)`` tuple for backward compat.
        """
        try:
            result = self.apply_formula(df, formula, name)
            return result, True
        except (ValueError, KeyError, TypeError):
            return df.copy(), False


__all__ = ["SignalProcessor"]

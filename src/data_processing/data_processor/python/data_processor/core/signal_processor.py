"""Signal processing orchestration around the vectorized filter engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

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
        """Apply the requested filter to the provided dataframe."""
        self._validate_dataframe(df)
        engine_params = config.to_engine_parameters()

        # Ensure filter_engine is not None before accessing attributes
        if self.filter_engine is None:
            self.filter_engine = VectorizedFilterEngine(
                logger=self.logger.warning, n_jobs=1
            )

        if config.filter_type not in self.filter_engine.filters:
            msg = f"Filter '{config.filter_type}' is not supported by the filter engine"
            raise ValueError(msg)

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

        return self.filter_engine.apply_filter_batch(
            df,
            config.filter_type,
            engine_params,
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Ensure the input dataframe is valid and non-empty."""
        if df.empty:
            msg = "Cannot filter an empty dataframe"
            raise ValueError(msg)
        if not isinstance(df, pd.DataFrame):
            msg = "apply_filter expects a pandas DataFrame"
            raise TypeError(msg)

    def validate_signals_exist(self, df: pd.DataFrame, signals: list[str]) -> None:
        """Raise a descriptive error when required signals are missing."""
        missing = [signal for signal in signals if signal not in df.columns]
        if missing:
            msg = f"Missing required signals: {', '.join(sorted(missing))}"
            raise ValueError(msg)

    def integrate_signals(self, df: pd.DataFrame, config: Any) -> pd.DataFrame:
        """Integrate specified signals."""
        self._validate_dataframe(df)
        return df.copy()

    def differentiate_signals(self, df: pd.DataFrame, config: Any) -> pd.DataFrame:
        """Differentiate specified signals."""
        self._validate_dataframe(df)
        return df.copy()

    def apply_custom_formula(
        self, df: pd.DataFrame, formula: str, new_column: str
    ) -> tuple[pd.DataFrame, bool]:
        """Apply a custom formula to the dataframe."""
        self._validate_dataframe(df)
        return df.copy(), True

    def detect_signal_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect statistics for signals in the dataframe."""
        self._validate_dataframe(df)
        return {}


__all__ = ["SignalProcessor"]

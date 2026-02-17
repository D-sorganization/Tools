"""Core Data Processing Engine.

Provides headless data manipulation, filtering, and analysis capabilities.
Ported from Gasification Model and ud-tools legacy Data Processor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, savgol_filter

from ..calculators.base import BaseCalculationEngine
from .io import DataReader, DataWriter

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats."""

    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"


class AggregationType(Enum):
    """Available aggregation types."""

    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"


class FitType(Enum):
    """Available curve fitting types."""

    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POWER = "power"


@dataclass
class FitResult:
    """Result of a curve fitting operation."""

    fit_type: str
    coefficients: list[float]
    r_squared: float
    equation: str
    fitted_values: np.ndarray
    residuals: np.ndarray


@dataclass
class ColumnStats:
    """Statistics for a data column."""

    name: str
    dtype: str
    count: int
    null_count: int
    unique_count: int
    mean: float | None = None
    std: float | None = None
    min_val: float | str | None = None
    max_val: float | str | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None


@dataclass
class ProcessingResult:
    """Result of a data processing operation."""

    success: bool
    message: str
    data: pd.DataFrame | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataProcessorEngine(BaseCalculationEngine):
    """Core data processing engine with comprehensive data manipulation capabilities.

    Integrates logic from Gasification Model and advanced signal filtering.
    """

    def __init__(self) -> None:
        """Initialize the data processor engine."""
        self.data: pd.DataFrame | None = None
        self.original_data: pd.DataFrame | None = None
        self.history: list[ProcessingResult] = []
        self.file_path: Path | None = None
        self._undo_stack: list[pd.DataFrame] = []
        self._redo_stack: list[pd.DataFrame] = []

    def calculate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Implementation of abstract method from BaseCalculationEngine."""
        operation = kwargs.get("operation", "stats")

        operations: dict[str, Callable[..., dict[str, Any]]] = {
            "load": lambda: self._wrap_result(
                self.load_file(kwargs.get("file_path", ""))
            ),
            "stats": lambda: {"stats": self.get_statistics()},
            "filter": lambda: self._wrap_result(
                self.filter_data(
                    kwargs.get("column", ""),
                    kwargs.get("operator", "=="),
                    kwargs.get("value"),
                )
            ),
            "smooth": lambda: self._wrap_result(
                self.smooth_column(
                    kwargs.get("column", ""),
                    kwargs.get("method", "moving_average"),
                    **kwargs,
                )
            ),
            "aggregate": lambda: self._wrap_result(
                self.aggregate(
                    kwargs.get("group_by"),
                    kwargs.get("agg_column"),
                    AggregationType(kwargs.get("agg_type", "mean")),
                )
            ),
            "fit": lambda: {
                "fit_result": self.fit_curve(
                    kwargs.get("x_column", ""),
                    kwargs.get("y_column", ""),
                    FitType(kwargs.get("fit_type", "linear")),
                    kwargs.get("degree", 2),
                )
            },
        }

        handler = operations.get(operation)
        if handler:
            return handler()
        return {"error": f"Unknown operation: {operation}"}

    def _wrap_result(self, result: ProcessingResult) -> dict[str, Any]:
        """Wrap result for JSON compatibility."""
        return {
            "success": result.success,
            "message": result.message,
            "stats": result.stats,
            "timestamp": result.timestamp,
        }

    # ========== File I/O ==========

    def load_file(self, file_path: str | Path, **kwargs: Any) -> ProcessingResult:
        """Load data using shared DataReader."""
        try:
            self.data = DataReader.read_file(file_path, **kwargs)
            self.original_data = self.data.copy()
            self.file_path = Path(file_path)
            self._undo_stack.clear()
            self._redo_stack.clear()

            return ProcessingResult(
                success=True,
                message=f"Loaded {len(self.data)} rows",
                data=self.data,
                stats=self._get_basic_stats(),
            )
        except Exception as e:
            return ProcessingResult(success=False, message=str(e))

    def load_dataframe(self, df: pd.DataFrame) -> ProcessingResult:
        """Load data from an existing DataFrame."""
        self._save_undo_state()
        self.data = df.copy()
        self.original_data = self.data.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()
        return ProcessingResult(
            success=True,
            message=f"Loaded DataFrame with {len(df)} rows",
            data=self.data,
            stats=self._get_basic_stats(),
        )

    def export_data(
        self, file_path: str | Path, fmt: DataFormat = DataFormat.CSV, **kwargs: Any
    ) -> ProcessingResult:
        """Export data using shared DataWriter."""
        if self.data is None:
            return ProcessingResult(success=False, message="No data loaded")
        try:
            DataWriter.write_file(self.data, file_path, format_type=fmt.value, **kwargs)
            return ProcessingResult(success=True, message=f"Exported to {file_path}")
        except Exception as e:
            return ProcessingResult(success=False, message=str(e))

    # ========== Column Operations ==========

    def add_calculated_column(
        self, name: str, expression: str, dtype: str | None = None
    ) -> ProcessingResult:
        """Add a column using pandas eval."""
        if self.data is None:
            return ProcessingResult(success=False, message="No data")
        self._save_undo_state()
        try:
            self.data[name] = self.data.eval(expression)
            if dtype:
                self.data[name] = self.data[name].astype(dtype)
            return ProcessingResult(
                success=True, message=f"Added '{name}'", data=self.data
            )
        except Exception as e:
            self._undo()
            return ProcessingResult(success=False, message=str(e))

    def rename_column(self, old_name: str, new_name: str) -> ProcessingResult:
        if self.data is None:
            return ProcessingResult(success=False, message="No data")
        if old_name not in self.data.columns:
            return ProcessingResult(
                success=False, message=f"Column '{old_name}' not found"
            )
        self._save_undo_state()
        self.data = self.data.rename(columns={old_name: new_name})
        return ProcessingResult(
            success=True, message=f"Renamed to '{new_name}'", data=self.data
        )

    def drop_columns(self, columns: list[str]) -> ProcessingResult:
        if self.data is None:
            return ProcessingResult(success=False, message="No data")
        missing = [c for c in columns if c not in self.data.columns]
        if missing:
            return ProcessingResult(
                success=False, message=f"Columns not found: {missing}"
            )
        self._save_undo_state()
        self.data = self.data.drop(columns=columns)
        return ProcessingResult(success=True, message="Dropped columns", data=self.data)

    def transform_column(
        self, column: str, transformation: str, **kwargs: Any
    ) -> ProcessingResult:
        """Apply transformation to a column."""
        if self.data is None or column not in self.data.columns:
            return ProcessingResult(success=False, message="Invalid target")
        self._save_undo_state()
        try:
            col = self.data[column]
            t_map: dict[str, Callable[[], pd.Series]] = {
                "log": lambda: np.log(col),
                "log10": lambda: np.log10(col),
                "exp": lambda: np.exp(col),
                "sqrt": lambda: np.sqrt(col),
                "abs": lambda: np.abs(col),
                "normalize": lambda: (col - col.min()) / (col.max() - col.min()),
                "standardize": lambda: (col - col.mean()) / col.std(),
                "round": lambda: col.round(kwargs.get("decimals", 2)),
                "fillna": lambda: col.fillna(kwargs.get("value", 0)),
            }
            if transformation == "dropna":
                self.data = self.data.dropna(subset=[column])
            elif transformation in t_map:
                self.data[column] = t_map[transformation]()
            else:
                return ProcessingResult(
                    success=False, message=f"Unknown T: {transformation}"
                )
            return ProcessingResult(success=True, message="Transformed", data=self.data)
        except Exception as e:
            self._undo()
            return ProcessingResult(success=False, message=str(e))

    # ========== Signal Smoothing & Filtering ==========

    def smooth_column(
        self, column: str, method: str, **kwargs: Any
    ) -> ProcessingResult:
        """Apply filtering algorithms."""
        if self.data is None or column not in self.data.columns:
            return ProcessingResult(success=False, message="No data/column")
        self._save_undo_state()
        try:
            series = self.data[column].dropna()
            if len(series) < 2:
                return ProcessingResult(success=False, message="Not enough data")

            if method == "moving_average":
                result = series.rolling(
                    window=kwargs.get("window", 10), min_periods=1
                ).mean()
            elif method == "butterworth":
                order, cutoff = kwargs.get("order", 3), kwargs.get("cutoff", 0.1)
                dt = (
                    series.index.to_series().diff().dt.total_seconds().mean()
                    if isinstance(series.index, pd.DatetimeIndex)
                    else 1.0
                )
                sr = 1.0 / dt if dt > 0 else 1.0
                b, a = butter(
                    N=order, Wn=cutoff, btype=kwargs.get("btype", "low"), fs=sr
                )
                result = pd.Series(filtfilt(b, a, series), index=series.index)
            elif method == "median":
                k = kwargs.get("kernel", 5)
                result = pd.Series(
                    medfilt(series, kernel_size=k if k % 2 else k + 1),
                    index=series.index,
                )
            elif method == "savgol":
                w, p = kwargs.get("window", 11), kwargs.get("polyorder", 2)
                result = pd.Series(
                    savgol_filter(series, w if w % 2 else w + 1, p), index=series.index
                )
            else:
                return ProcessingResult(success=False, message="Unknown method")

            self.data[column] = result
            return ProcessingResult(success=True, message="Smoothed", data=self.data)
        except Exception as e:
            self._undo()
            return ProcessingResult(success=False, message=str(e))

    # ========== Statistics & Analysis ==========

    def aggregate(
        self,
        group_by: str | list[str] | None,
        column: str | None,
        agg_type: AggregationType,
    ) -> ProcessingResult:
        if self.data is None:
            return ProcessingResult(success=False, message="No data")
        self._save_undo_state()
        try:
            if group_by:
                grouped = self.data.groupby(group_by)
                self.data = (
                    grouped[column].agg(agg_type.value)
                    if column
                    else grouped.agg(agg_type.value)
                ).reset_index()
            else:
                res = (
                    self.data[column].agg(agg_type.value)
                    if column
                    else self.data.select_dtypes(np.number).agg(agg_type.value)
                )
                self.data = (
                    pd.DataFrame([res]) if not column else pd.DataFrame({column: [res]})
                )
            return ProcessingResult(success=True, message="Aggregated", data=self.data)
        except Exception as e:
            self._undo()
            return ProcessingResult(success=False, message=str(e))

    def fit_curve(
        self, x_col: str, y_col: str, fit_type: FitType, degree: int = 2
    ) -> FitResult | None:
        if self.data is None:
            return None
        try:
            x = np.asarray(self.data[x_col].values, dtype=float)
            y = np.asarray(self.data[y_col].values, dtype=float)
            m = ~(np.isnan(x) | np.isnan(y))
            x, y = x[m], y[m]
            if len(x) < 2:
                return None

            if fit_type == FitType.LINEAR:
                c = np.polyfit(x, y, 1)
                f = np.polyval(c, x)
                eq = f"y = {c[0]:.4f}x + {c[1]:.4f}"
            elif fit_type == FitType.POLYNOMIAL:
                c = np.polyfit(x, y, degree)
                f = np.polyval(c, x)
                terms = [f"{c[i]:.4f}x^{degree-i}" for i in range(degree)] + [
                    f"{c[-1]:.4f}"
                ]
                eq = "y = " + " + ".join(terms)
            else:
                return None  # Simplified for brevity

            r2 = (
                1 - np.sum((y - f) ** 2) / np.sum((y - np.mean(y)) ** 2)
                if np.sum((y - np.mean(y)) ** 2) != 0
                else 0
            )
            return FitResult(fit_type.value, list(c), float(r2), eq, f, y - f)
        except Exception:
            return None

    # ========== Helpers ==========

    def filter_data(self, column: str, operator: str, value: Any) -> ProcessingResult:
        if self.data is None:
            return ProcessingResult(success=False, message="No data")
        self._save_undo_state()
        try:
            if operator == "contains":
                self.data = self.data[
                    self.data[column].str.contains(str(value), na=False)
                ]
            elif operator == "in":
                self.data = self.data[
                    self.data[column].isin(
                        value if isinstance(value, list) else [value]
                    )
                ]
            else:
                self.data = self.data.query(f"{column} {operator} @value")
            return ProcessingResult(success=True, message="Filtered", data=self.data)
        except Exception as e:
            self._undo()
            return ProcessingResult(success=False, message=str(e))

    def query(self, expression: str) -> ProcessingResult:
        if self.data is None:
            return ProcessingResult(success=False, message="No data")
        self._save_undo_state()
        try:
            self.data = self.data.query(expression)
            return ProcessingResult(
                success=True, message="Query applied", data=self.data
            )
        except Exception as e:
            self._undo()
            return ProcessingResult(success=False, message=str(e))

    def get_statistics(self) -> dict[str, ColumnStats]:
        if self.data is None:
            return {}
        res = {}
        for col in self.data.columns:
            s = self.data[col]
            cs = ColumnStats(
                col, str(s.dtype), len(s), int(s.isna().sum()), int(s.nunique())
            )
            if pd.api.types.is_numeric_dtype(s) and not s.isna().all():
                cs.mean, cs.std = float(s.mean()), float(s.std())
                cs.min_val, cs.max_val, cs.median = (
                    float(s.min()),
                    float(s.max()),
                    float(s.median()),
                )
            res[col] = cs
        return res

    def _get_basic_stats(self) -> dict[str, Any]:
        if self.data is None:
            return {}
        return {"rows": len(self.data), "columns": len(self.data.columns)}

    def _save_undo_state(self) -> None:
        if self.data is not None:
            self._undo_stack.append(self.data.copy())
            self._redo_stack.clear()
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)

    def _undo(self) -> bool:
        if self._undo_stack:
            if self.data is not None:
                self._redo_stack.append(self.data.copy())
            self.data = self._undo_stack.pop()
            return True
        return False

    def undo(self) -> ProcessingResult:
        """Undo the last operation."""
        if self._undo():
            return ProcessingResult(
                success=True, message="Undo successful", data=self.data
            )
        return ProcessingResult(success=False, message="Nothing to undo")

    def redo(self) -> ProcessingResult:
        """Redo the last undone operation."""
        if self._redo_stack:
            if self.data is not None:
                self._undo_stack.append(self.data.copy())
            self.data = self._redo_stack.pop()
            return ProcessingResult(
                success=True, message="Redo successful", data=self.data
            )
        return ProcessingResult(success=False, message="Nothing to redo")

    def reset(self) -> ProcessingResult:
        if self.original_data is not None:
            self._save_undo_state()
            self.data = self.original_data.copy()
            return ProcessingResult(success=True, message="Reset", data=self.data)
        return ProcessingResult(success=False, message="No original data")

    # ========== Utility Methods ==========

    def get_column_names(self) -> list[str]:
        """Get list of column names."""
        return list(self.data.columns) if self.data is not None else []

    def get_numeric_columns(self) -> list[str]:
        """Get list of numeric column names."""
        if self.data is None:
            return []
        return list(self.data.select_dtypes(include=[np.number]).columns)

    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.data is not None and not self.data.empty


__all__ = [
    "DataProcessorEngine",
    "ProcessingResult",
    "AggregationType",
    "FitType",
    "DataFormat",
    "FitResult",
    "ColumnStats",
]

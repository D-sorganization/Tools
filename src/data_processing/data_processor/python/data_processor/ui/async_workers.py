"""Qt workers for Data Processor operations that must not block the UI thread."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    import pandas as pd

    from data_processor.core.data_loader import DataLoader
    from data_processor.core.signal_processor import SignalProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataLoadResult:
    """Data and metadata prepared off the Qt main thread."""

    data: pd.DataFrame
    available_signals: list[str]
    time_column: str | None


class DataLoadWorker(QThread):
    """Load CSV files and derive metadata in a worker thread."""

    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        file_paths: list[str],
        loader: DataLoader | None = None,
        *,
        convert_time_column: bool = False,
    ) -> None:
        if not file_paths:
            raise ValueError("file_paths must contain at least one file")
        super().__init__()
        if loader is None:
            from data_processor.core.data_loader import DataLoader

            loader = DataLoader(use_high_performance=True)
        self.file_paths = file_paths.copy()
        self.loader = loader
        self.convert_time_column = convert_time_column

    def run(self) -> None:
        try:
            self.progress.emit(5)
            if len(self.file_paths) == 1:
                data = self.loader.load_csv_file(self.file_paths[0])
            else:
                dataframes = self.loader.load_multiple_files(
                    self.file_paths,
                    progress_callback=self._emit_file_progress,
                )
                data = self.loader.combine_dataframes(dataframes)

            if data is None:
                raise RuntimeError("No data could be loaded from the selected file(s).")

            self.progress.emit(80)
            time_column = self.loader.detect_time_column(data)
            if time_column and self.convert_time_column:
                data = self.loader.convert_time_column(data, time_column)

            available_signals = self.loader.get_numeric_signals(data)
            self.progress.emit(100)
            self.result_ready.emit(
                DataLoadResult(
                    data=data,
                    available_signals=available_signals,
                    time_column=time_column,
                )
            )
        except Exception as exc:  # noqa: BLE001 - prevent uncaught Qt thread failures
            logger.error("Data load failed in worker thread: %s", exc, exc_info=True)
            self.error.emit(str(exc))

    def _emit_file_progress(self, completed: int, total: int, _message: str) -> None:
        if total <= 0:
            return
        self.progress.emit(min(80, int((completed / total) * 75) + 5))


class ProcessingWorker(QThread):
    """Run signal processing operations in a worker thread."""

    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        operation: str,
        data: pd.DataFrame,
        processor: SignalProcessor | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if not operation:
            raise ValueError("operation must be provided")
        super().__init__()
        if processor is None:
            from data_processor.core.signal_processor import SignalProcessor

            processor = SignalProcessor()
        self.operation = operation
        self.data = data
        self.processor = processor
        self.config = config or {}

    def run(self) -> None:
        try:
            from data_processor.models.processing_config import (
                DifferentiationConfig,
                FilterConfig,
                IntegrationConfig,
            )

            self.progress.emit(10)
            if self.operation == "filter":
                filter_config = FilterConfig(
                    filter_type=self.config["filter_type"],
                    parameters=self.config["parameters"],
                )
                result = self.processor.apply_filter(self.data, filter_config)
            elif self.operation == "integrate":
                int_config = IntegrationConfig(
                    signals=self.config["signals"],
                    method=self.config.get("method", "cumulative"),
                )
                result = self.processor.integrate_signals(self.data, int_config)
            elif self.operation == "differentiate":
                diff_config = DifferentiationConfig(
                    signals=self.config["signals"],
                    order=self.config.get("order", 1),
                    method=self.config.get("method", "central"),
                )
                result = self.processor.differentiate_signals(self.data, diff_config)
            else:
                raise ValueError(f"Unknown processing operation: {self.operation}")

            self.progress.emit(100)
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001 - prevent uncaught Qt thread failures
            logger.error("Processing failed in worker thread: %s", exc, exc_info=True)
            self.error.emit(str(exc))

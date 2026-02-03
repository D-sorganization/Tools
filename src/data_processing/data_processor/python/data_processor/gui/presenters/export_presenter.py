"""Export presenter - handles data export operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from data_processor.core.data_loader import DataLoader
from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Available export formats
EXPORT_FORMATS = ["csv", "excel", "parquet", "hdf5", "feather"]


class ExportPresenter(QObject):
    """Presenter for data export operations."""

    # Signals
    export_completed = pyqtSignal(str)  # Output file path
    export_failed = pyqtSignal(str)  # Error message
    exporting_started = pyqtSignal()
    exporting_finished = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the export presenter."""
        super().__init__(parent)
        self.data_loader = DataLoader()

    def export_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        format_type: str,
        signals: list[str] | None = None,
    ) -> None:
        """Export data to file."""
        self.exporting_started.emit()

        try:
            export_df = self._prepare_export_data(df, signals)
            success = self._save_data(export_df, output_path, format_type)

            if success:
                self.export_completed.emit(output_path)
            else:
                self.export_failed.emit("Export failed")
        except Exception as e:
            logger.exception("Error exporting data")
            self.export_failed.emit(str(e))
        finally:
            self.exporting_finished.emit()

    def _prepare_export_data(
        self,
        df: pd.DataFrame,
        signals: list[str] | None,
    ) -> pd.DataFrame:
        """Prepare data for export."""
        if signals:
            valid_signals = [s for s in signals if s in df.columns]
            if valid_signals:
                return df[valid_signals].copy()
        return df.copy()

    def _save_data(
        self,
        df: pd.DataFrame,
        output_path: str,
        format_type: str,
    ) -> bool:
        """Save data to file."""
        return self.data_loader.save_dataframe(df, output_path, format_type=format_type)

    def get_export_formats(self) -> list[str]:
        """Get list of available export formats."""
        return EXPORT_FORMATS.copy()

    def get_file_extension(self, format_type: str) -> str:
        """Get file extension for format type."""
        extensions = {
            "csv": ".csv",
            "excel": ".xlsx",
            "parquet": ".parquet",
            "hdf5": ".h5",
            "feather": ".feather",
        }
        return extensions.get(format_type, ".csv")

    def get_file_filter(self, format_type: str) -> str:
        """Get file dialog filter for format type."""
        filters = {
            "csv": "CSV Files (*.csv)",
            "excel": "Excel Files (*.xlsx)",
            "parquet": "Parquet Files (*.parquet)",
            "hdf5": "HDF5 Files (*.h5)",
            "feather": "Feather Files (*.feather)",
        }
        return filters.get(format_type, "All Files (*)")

    def suggest_filename(
        self,
        original_path: str,
        format_type: str,
        suffix: str = "_processed",
    ) -> str:
        """Suggest output filename based on original."""
        original = Path(original_path)
        stem = original.stem
        extension = self.get_file_extension(format_type)
        return f"{stem}{suffix}{extension}"

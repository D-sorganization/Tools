"""Main window for the Data Processor PyQt6 GUI."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .presenters.data_presenter import DataPresenter
from .presenters.export_presenter import ExportPresenter
from .presenters.filter_presenter import FilterPresenter
from .styles.theme import apply_dark_theme
from .widgets.export_panel import ExportPanel
from .widgets.file_panel import FilePanel
from .widgets.filter_panel import FilterPanel
from .widgets.preview_table import PreviewTable
from .widgets.signal_panel import SignalPanel
from .widgets.statistics_panel import StatisticsPanel

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessorMainWindow(QMainWindow):
    """Main window for the Data Processor application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._file_paths: list[str] = []
        self._setup_presenters()
        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()

    def _setup_presenters(self) -> None:
        """Initialize presenters."""
        self.data_presenter = DataPresenter(self)
        self.filter_presenter = FilterPresenter(self)
        self.export_presenter = ExportPresenter(self)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Data Processor")
        self.setMinimumSize(1200, 800)

        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Create main splitter
        splitter = self._create_main_splitter()
        layout.addWidget(splitter)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._update_status("Ready")

    def _create_main_splitter(self) -> QSplitter:
        """Create the main horizontal splitter."""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Controls
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel: Data view
        right_panel = self._create_data_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 800])
        return splitter

    def _create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # File panel
        self.file_panel = FilePanel()
        layout.addWidget(self.file_panel)

        # Signal panel
        self.signal_panel = SignalPanel()
        layout.addWidget(self.signal_panel)

        # Tab widget for filter/export/stats
        tabs = QTabWidget()

        self.filter_panel = FilterPanel()
        tabs.addTab(self.filter_panel, "Filter")

        self.export_panel = ExportPanel()
        tabs.addTab(self.export_panel, "Export")

        self.statistics_panel = StatisticsPanel()
        tabs.addTab(self.statistics_panel, "Statistics")

        layout.addWidget(tabs)

        return widget

    def _create_data_panel(self) -> QWidget:
        """Create the right data panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.preview_table = PreviewTable()
        layout.addWidget(self.preview_table)

        return widget

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        # File panel signals
        self.file_panel.files_selected.connect(self._on_files_selected)
        self.file_panel.load_requested.connect(self._on_load_requested)
        self.file_panel.files_cleared.connect(self._on_files_cleared)

        # Signal panel signals
        self.signal_panel.selection_changed.connect(self._on_signal_selection_changed)

        # Filter panel signals
        self.filter_panel.filter_requested.connect(self._on_filter_requested)

        # Export panel signals
        self.export_panel.export_requested.connect(self._on_export_requested)

        # Statistics panel signals
        self.statistics_panel.calculate_requested.connect(
            self._on_statistics_requested
        )

        # Presenter signals
        self.data_presenter.data_loaded.connect(self._on_data_loaded)
        self.data_presenter.load_failed.connect(self._on_load_failed)
        self.data_presenter.signals_detected.connect(self._on_signals_detected)

        self.filter_presenter.filter_applied.connect(self._on_filter_applied)
        self.filter_presenter.filter_failed.connect(self._on_filter_failed)

        self.export_presenter.export_completed.connect(self._on_export_completed)
        self.export_presenter.export_failed.connect(self._on_export_failed)

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Shortcuts are handled by widgets or can be added here
        pass

    # Event handlers - kept short

    def _on_files_selected(self, files: list[str]) -> None:
        """Handle files selected event."""
        self._file_paths = files
        self._update_status(f"Selected {len(files)} files")

    def _on_load_requested(self) -> None:
        """Handle load request."""
        if self._file_paths:
            self._update_status("Loading files...")
            self.data_presenter.load_files(self._file_paths)

    def _on_files_cleared(self) -> None:
        """Handle files cleared event."""
        self._file_paths = []
        self.preview_table.clear()
        self.signal_panel.clear()
        self._update_status("Files cleared")

    def _on_signal_selection_changed(self, signals: list[str]) -> None:
        """Handle signal selection change."""
        count = len(signals)
        self._update_status(f"{count} signals selected")

    def _on_filter_requested(self, config: dict[str, Any]) -> None:
        """Handle filter request."""
        df = self.data_presenter.get_data()
        if df is None:
            self._show_warning("No data loaded", "Please load data first.")
            return

        signals = self.signal_panel.get_selected_signals()
        if not signals:
            signals = self.data_presenter.get_signals()

        self._update_status("Applying filter...")
        self.filter_presenter.apply_filter(df, signals, config)

    def _on_export_requested(self, format_type: str) -> None:
        """Handle export request."""
        df = self.data_presenter.get_data()
        if df is None:
            self._show_warning("No data loaded", "Please load data first.")
            return

        output_path = self._get_export_path(format_type)
        if output_path:
            signals = self.signal_panel.get_selected_signals()
            self._update_status("Exporting data...")
            self.export_presenter.export_data(df, output_path, format_type, signals)

    def _on_statistics_requested(self) -> None:
        """Handle statistics calculation request."""
        df = self.data_presenter.get_data()
        if df is None:
            self._show_warning("No data loaded", "Please load data first.")
            return

        signals = self.signal_panel.get_selected_signals()
        if not signals:
            signals = self.data_presenter.get_signals()

        stats = self._calculate_statistics(df, signals)
        self.statistics_panel.set_statistics(stats)
        self._update_status("Statistics calculated")

    def _on_data_loaded(self, df: pd.DataFrame) -> None:
        """Handle data loaded event."""
        self.preview_table.set_data(df)
        self._update_status(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    def _on_load_failed(self, error: str) -> None:
        """Handle load failure."""
        self._show_error("Load Failed", error)
        self._update_status("Load failed")

    def _on_signals_detected(self, signals: list[str]) -> None:
        """Handle signals detected event."""
        self.signal_panel.set_signals(signals)

    def _on_filter_applied(self, df: pd.DataFrame) -> None:
        """Handle filter applied event."""
        self.data_presenter.set_data(df)
        self.preview_table.set_data(df)
        self._update_status("Filter applied successfully")
        self._show_info("Success", "Filter applied successfully")

    def _on_filter_failed(self, error: str) -> None:
        """Handle filter failure."""
        self._show_error("Filter Failed", error)
        self._update_status("Filter failed")

    def _on_export_completed(self, path: str) -> None:
        """Handle export completion."""
        self._update_status(f"Exported to {path}")
        self._show_info("Export Complete", f"Data exported to:\n{path}")

    def _on_export_failed(self, error: str) -> None:
        """Handle export failure."""
        self._show_error("Export Failed", error)
        self._update_status("Export failed")

    # Helper methods

    def _update_status(self, message: str) -> None:
        """Update status bar message."""
        self.status_bar.showMessage(message)

    def _show_info(self, title: str, message: str) -> None:
        """Show information dialog."""
        QMessageBox.information(self, title, message)

    def _show_warning(self, title: str, message: str) -> None:
        """Show warning dialog."""
        QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str) -> None:
        """Show error dialog."""
        QMessageBox.critical(self, title, message)

    def _get_export_path(self, format_type: str) -> str | None:
        """Get export file path from user."""
        file_filter = self.export_presenter.get_file_filter(format_type)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", file_filter
        )
        return path if path else None

    def _calculate_statistics(
        self, df: pd.DataFrame, signals: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Calculate statistics for signals."""
        stats = {}
        for signal in signals:
            if signal in df.columns:
                col = df[signal]
                stats[signal] = {
                    "count": int(col.count()),
                    "mean": float(col.mean()) if col.count() > 0 else None,
                    "std": float(col.std()) if col.count() > 0 else None,
                    "min": float(col.min()) if col.count() > 0 else None,
                    "max": float(col.max()) if col.count() > 0 else None,
                    "median": float(col.median()) if col.count() > 0 else None,
                }
        return stats


def main() -> None:
    """Run the Data Processor application."""
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = DataProcessorMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

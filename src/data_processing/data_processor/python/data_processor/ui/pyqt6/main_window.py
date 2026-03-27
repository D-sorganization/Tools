# mypy: ignore-errors
"""Main window for PyQt6 Data Processor GUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from data_processor.core.config_manager import ConfigManager
from data_processor.core.data_loader import DataLoader
from data_processor.core.plot_config_manager import PlotConfigManager
from data_processor.core.signal_list_manager import SignalListManager
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import (
    DifferentiationConfig,
    FilterConfig,
    IntegrationConfig,
)

from .analysis_widgets import (
    AnalysisPanel,
    ContourPlotDialog,
)
from .main_window_analysis import AnalysisMixin
from .main_window_config import ConfigMixin
from .main_window_dat_import import DatImportMixin
from .main_window_data_ops import DataOperationsMixin
from .main_window_plot_config import PlotConfigMixin
from .main_window_tabs import TabCreationMixin
from .main_window_time_ops import TimeOpsMixin
from .widgets import (
    SignalListWidget,
    StatusBar,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# Dark theme stylesheet
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', sans-serif;
}

QTabWidget::pane {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    background-color: #252526;
}

QTabBar::tab {
    background-color: #2d2d2d;
    color: #d4d4d4;
    padding: 8px 16px;
    border: 1px solid #3c3c3c;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    border-bottom: 2px solid #007acc;
}

QTabBar::tab:hover:!selected {
    background-color: #383838;
}

QPushButton {
    background-color: #0e639c;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1177bb;
}

QPushButton:pressed {
    background-color: #094771;
}

QPushButton:disabled {
    background-color: #3c3c3c;
    color: #6c6c6c;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #5c5c5c;
    border-radius: 4px;
    padding: 6px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #007acc;
}

QListWidget, QTableWidget, QTextEdit {
    background-color: #252526;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    alternate-background-color: #2d2d2d;
}

QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #094771;
}

QListWidget::item:hover, QTableWidget::item:hover {
    background-color: #2a2d2e;
}

QHeaderView::section {
    background-color: #3c3c3c;
    color: #d4d4d4;
    padding: 6px;
    border: none;
    border-right: 1px solid #5c5c5c;
}

QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #007acc;
}

QProgressBar {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    text-align: center;
    background-color: #252526;
}

QProgressBar::chunk {
    background-color: #007acc;
    border-radius: 3px;
}

QScrollBar:vertical {
    background-color: #1e1e1e;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #5c5c5c;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #6c6c6c;
}

QSplitter::handle {
    background-color: #3c3c3c;
}

QMenuBar {
    background-color: #2d2d2d;
    color: #d4d4d4;
}

QMenuBar::item:selected {
    background-color: #3c3c3c;
}

QMenu {
    background-color: #252526;
    color: #d4d4d4;
    border: 1px solid #3c3c3c;
}

QMenu::item:selected {
    background-color: #094771;
}
"""


class ProcessingWorker(QThread):
    """Worker thread for processing operations."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        operation: str,
        data: pd.DataFrame,
        processor: SignalProcessor,
        config: dict,
    ) -> None:
        if not (operation is not None):
            raise ValueError("operation must be provided")
        super().__init__()
        self.operation = operation
        self.data = data
        self.processor = processor
        self.config = config

    def run(self) -> None:
        try:
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
                result = self.data

            self.finished.emit(result)
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            self.error.emit(str(e))


class DataProcessorMainWindow(
    AnalysisMixin,
    ConfigMixin,
    DatImportMixin,
    DataOperationsMixin,
    PlotConfigMixin,
    TabCreationMixin,
    TimeOpsMixin,
    QMainWindow,
):
    """Main window for the Data Processor application."""

    def __init__(self) -> None:
        super().__init__()

        # Core modules
        self.data_loader = DataLoader(use_high_performance=True)
        self.signal_processor = SignalProcessor()

        # Configuration managers (using new core library)
        self.config_manager = ConfigManager()
        self.signal_list_manager = SignalListManager()
        self.plot_config_manager = PlotConfigManager()

        # State
        self.current_data: pd.DataFrame | None = None
        self.filtered_data: pd.DataFrame | None = None
        self.selected_files: list[str] = []
        self.available_signals: list[str] = []
        self.time_column: str | None = None
        self.output_directory: str = str(Path.home() / "Documents")

        # Settings
        self.settings = QSettings("DataProcessor", "DataProcessorGUI")

        self._init_ui()
        self._setup_shortcuts()
        self._restore_state()
        self._refresh_saved_plots_list()

        logger.info("Data Processor GUI initialized")

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Data Processor")
        self.setMinimumSize(1200, 800)

        # Apply dark theme
        self.setStyleSheet(DARK_STYLESHEET)

        # Create menu bar
        self._create_menu_bar()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel("Data Processor")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #007acc; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Left panel - Signal selection
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)

        self.signal_list = SignalListWidget()
        self.signal_list.selectionChanged.connect(self._on_signal_selection_changed)
        left_layout.addWidget(self.signal_list)

        splitter.addWidget(left_panel)

        # Right panel - Tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self._create_tabs()
        right_layout.addWidget(self.tab_widget)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        # Status bar
        self.status_bar = StatusBar()
        main_layout.addWidget(self.status_bar)

    def _create_menu_bar(self) -> None:
        """Create menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Files...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_files)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_action = QAction("&Export...", self)
        export_action.setShortcut(QKeySequence.StandardKey.Save)
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Signal Set submenu
        signal_set_menu = file_menu.addMenu("Signal &Set")

        load_signal_set_action = QAction("&Load Signal Set...", self)
        load_signal_set_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
        load_signal_set_action.triggered.connect(self._load_signal_set)
        signal_set_menu.addAction(load_signal_set_action)

        save_signal_set_action = QAction("&Save Signal Set...", self)
        save_signal_set_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_signal_set_action.triggered.connect(self._save_signal_set)
        signal_set_menu.addAction(save_signal_set_action)

        file_menu.addSeparator()

        # App Configuration submenu
        config_menu = file_menu.addMenu("App &Configuration")

        save_config_action = QAction("&Save Configuration...", self)
        save_config_action.triggered.connect(self._save_app_config)
        config_menu.addAction(save_config_action)

        load_config_action = QAction("&Load Configuration...", self)
        load_config_action.triggered.connect(self._load_app_config)
        config_menu.addAction(load_config_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")

        clear_action = QAction("&Clear Data", self)
        clear_action.triggered.connect(self._clear_data)
        edit_menu.addAction(clear_action)

        # Analysis menu
        analysis_menu = menu_bar.addMenu("&Analysis")

        contour_action = QAction("&Contour Plot...", self)
        contour_action.triggered.connect(self._show_contour_plot)
        analysis_menu.addAction(contour_action)

        heatmap_action = QAction("&Heatmap...", self)
        heatmap_action.triggered.connect(self._show_heatmap)
        analysis_menu.addAction(heatmap_action)

        analysis_menu.addSeparator()

        filter_compare_action = QAction("Compare &Filters...", self)
        filter_compare_action.triggered.connect(self._show_filter_comparison)
        analysis_menu.addAction(filter_compare_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_tabs(self) -> None:
        """Create tab pages."""
        # File tab
        file_tab = self._create_file_tab()
        self.tab_widget.addTab(file_tab, "Files")

        # Filter tab
        filter_tab = self._create_filter_tab()
        self.tab_widget.addTab(filter_tab, "Filters")

        # Advanced tab (Integration/Differentiation/Formula)
        advanced_tab = self._create_advanced_tab()
        self.tab_widget.addTab(advanced_tab, "Advanced")

        # Resampling tab
        resample_tab = self._create_resample_tab()
        self.tab_widget.addTab(resample_tab, "Resample")

        # Time Range tab
        time_range_tab = self._create_time_range_tab()
        self.tab_widget.addTab(time_range_tab, "Time Range")

        # Preview tab
        preview_tab = self._create_preview_tab()
        self.tab_widget.addTab(preview_tab, "Preview")

        # Plot Config tab
        plot_config_tab = self._create_plot_config_tab()
        self.tab_widget.addTab(plot_config_tab, "Plot Config")

        # Statistics tab
        stats_tab = self._create_statistics_tab()
        self.tab_widget.addTab(stats_tab, "Statistics")

        # Analysis tab (PCA, ANOVA, Regression, Surface, Neural Network)
        self.analysis_panel = AnalysisPanel()
        self.analysis_panel.pca_widget.analysis_requested.connect(self._run_pca_analysis)
        self.analysis_panel.anova_widget.analysis_requested.connect(self._run_anova_analysis)
        self.analysis_panel.regression_widget.analysis_requested.connect(
            self._run_regression_analysis
        )
        self.analysis_panel.surface_widget.plot_requested.connect(self._run_surface_analysis)
        self.analysis_panel.nn_widget.train_requested.connect(self._run_nn_analysis)
        self.tab_widget.addTab(self.analysis_panel, "Analysis")

        # DAT Import tab
        dat_import_tab = self._create_dat_import_tab()
        self.tab_widget.addTab(dat_import_tab, "DAT Import")

        # Export tab
        export_tab = self._create_export_tab()
        self.tab_widget.addTab(export_tab, "Export")

        # Help tab
        help_tab = self._create_help_tab()
        self.tab_widget.addTab(help_tab, "Help")

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+L"), self, self._load_data)
        QShortcut(QKeySequence("Ctrl+F"), self, self._focus_search)

    def _restore_state(self) -> None:
        """Restore window state from settings."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def _save_state(self) -> None:
        """Save window state to settings."""
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event) -> None:
        """Handle window close."""
        if not (event is not None):
            raise ValueError("event must be provided")
        self._save_state()
        super().closeEvent(event)

    # Action handlers

    def _open_files(self) -> None:
        """Open file dialog to select CSV files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Open CSV Files",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if files:
            self.selected_files = files
            self._update_file_list()
            self.status_bar.set_status(f"Selected {len(files)} files")

    def _clear_files(self) -> None:
        """Clear file selection."""
        self.selected_files = []
        self._update_file_list()
        self.status_bar.set_status("File selection cleared")

    def _update_file_list(self) -> None:
        """Update file list display."""
        self.file_list.clear()
        self.file_list.extend([Path(f).name for f in self.selected_files])

    def _load_data(self) -> None:
        """Load data from selected files."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select files first.")
            return

        try:
            self.status_bar.set_status("Loading data...")
            self.status_bar.show_progress()
            QApplication.processEvents()

            if len(self.selected_files) == 1:
                file_path = self.selected_files[0]
                self.current_data = self.data_loader.load_csv_file(file_path)
            else:
                dataframes = self.data_loader.load_multiple_files(self.selected_files)
                self.current_data = self.data_loader.combine_dataframes(dataframes)

            if self.current_data is not None:
                # Detect time column
                time_col = self.data_loader.detect_time_column(self.current_data)
                if time_col:
                    self.current_data = self.data_loader.convert_time_column(
                        self.current_data, time_col
                    )
                    self.time_column = time_col

                # Get signals
                self.available_signals = self.data_loader.get_numeric_signals(self.current_data)

                # Update UI
                self._update_data_info()
                self.signal_list.set_signals(self.available_signals)
                self.preview_widget.update_preview(self.current_data)
                self.analysis_panel.set_dataframe(self.current_data)

                # Update time column combo boxes
                self._update_column_combos()

                # Update time range info
                self._update_time_range_info()

                self.status_bar.hide_progress()
                row_count = len(self.current_data)
                signal_count = len(self.available_signals)
                self.status_bar.set_status(f"Loaded {row_count} rows, {signal_count} signals")

                QMessageBox.information(
                    self,
                    "Success",
                    f"Loaded {len(self.current_data)} rows\n"
                    f"{len(self.available_signals)} signals detected",
                )
            else:
                self.status_bar.hide_progress()
                self.status_bar.set_status("Failed to load data")
                QMessageBox.warning(self, "Error", "Failed to load data")

        except (RuntimeError, AttributeError) as e:
            self.status_bar.hide_progress()
            self.status_bar.set_status("Error loading data")
            logger.error(f"Error loading data: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{e}")

    def _update_data_info(self) -> None:
        """Update data information display."""
        if self.current_data is not None:
            self.rows_label.setText(str(len(self.current_data)))
            self.cols_label.setText(str(len(self.current_data.columns)))
            self.signals_label.setText(str(len(self.available_signals)))
        else:
            self.rows_label.setText("-")
            self.cols_label.setText("-")
            self.signals_label.setText("-")

    def _update_column_combos(self) -> None:
        """Update column combo boxes with available columns."""
        if self.current_data is None:
            return

        columns = list(self.current_data.columns)

        # Update time column combo
        self.time_col_combo.clear()
        self.time_col_combo.addItems(columns)
        if self.time_column and self.time_column in columns:
            idx = columns.index(self.time_column)
            self.time_col_combo.setCurrentIndex(idx)

        # Update x-axis combo for plotting
        self.x_axis_combo.clear()
        self.x_axis_combo.addItems(columns)
        if self.time_column and self.time_column in columns:
            idx = columns.index(self.time_column)
            self.x_axis_combo.setCurrentIndex(idx)

    def _on_signal_selection_changed(self, signals: list[str]) -> None:
        """Handle signal selection change."""
        if not (signals is not None):
            raise ValueError("signals must be provided")
        logger.debug(f"Selected signals: {signals}")

        self.available_signals = []
        self.time_column = None
        self.signal_list.set_signals([])
        self.preview_widget.clear()
        self.stats_widget.clear()
        self._update_data_info()
        self.status_bar.set_status("Data cleared")

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Data Processor",
            "Data Processor v2.0\n\n"
            "A powerful tool for signal processing and data analysis.\n\n"
            "Features:\n"
            "- Multiple filter types\n"
            "- Integration and differentiation\n"
            "- Custom formulas\n"
            "- Multiple export formats\n"
            "- Load/Save signal sets\n"
            "- Signal search\n\n"
            "Built with PyQt6",
        )

    def _show_contour_plot(self) -> None:
        """Show contour plot dialog."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        dialog = ContourPlotDialog(self.current_data, self)
        dialog.exec()

    def _focus_search(self) -> None:
        """Focus the signal search field."""
        self.signal_list.focus_search()

    # Signal set and config methods -> ConfigMixin (main_window_config.py)
    # Time operations -> TimeOpsMixin (main_window_time_ops.py)
    # Plot config methods -> PlotConfigMixin (main_window_plot_config.py)
    # DAT import methods -> DatImportMixin (main_window_dat_import.py)


def main() -> None:
    """Run the Data Processor application."""
    import sys

    from plot_theme import setup_plot_theme_for_app
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DataProcessorMainWindow()
    setup_themed_app(app, window, settings_app="DataProcessor")
    setup_plot_theme_for_app(app, window, settings_app="DataProcessor")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# mypy: ignore-errors
"""Main window for PyQt6 Data Processor GUI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from data_processor.core.config_manager import ConfigManager
from data_processor.core.dat_importer import (
    export_dat_to_csv,
    read_dat_file,
)
from data_processor.core.data_loader import DataLoader
from data_processor.core.plot_config_manager import PlotConfigManager
from data_processor.core.signal_list_manager import SignalListManager
from data_processor.core.signal_processing import (
    calculate_trendline,
    resample_data,
    trim_time_range,
)
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
from .main_window_data_ops import DataOperationsMixin
from .main_window_tabs import TabCreationMixin
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
    DataOperationsMixin,
    TabCreationMixin,
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
        self.analysis_panel.pca_widget.analysis_requested.connect(
            self._run_pca_analysis
        )
        self.analysis_panel.anova_widget.analysis_requested.connect(
            self._run_anova_analysis
        )
        self.analysis_panel.regression_widget.analysis_requested.connect(
            self._run_regression_analysis
        )
        self.analysis_panel.surface_widget.plot_requested.connect(
            self._run_surface_analysis
        )
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
        for f in self.selected_files:
            self.file_list.append(Path(f).name)

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
                self.available_signals = self.data_loader.get_numeric_signals(
                    self.current_data
                )

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
                self.status_bar.set_status(
                    f"Loaded {row_count} rows, {signal_count} signals"
                )

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

    def _load_signal_set(self) -> None:
        """Load a signal set from JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Signal Set",
            "",
            "Signal Set Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        try:
            with open(filename, encoding="utf-8") as f:
                signal_set = json.load(f)

            if not isinstance(signal_set, dict):
                raise ValueError("Invalid signal set format")

            selected_signals = signal_set.get("selected_signals", [])
            if not isinstance(selected_signals, list):
                raise ValueError("Invalid selected_signals format")

            # Select the signals
            self.signal_list.select_signals(selected_signals)
            self.status_bar.set_status(
                f"Loaded signal set: {len(selected_signals)} signals"
            )

            QMessageBox.information(
                self,
                "Success",
                f"Loaded signal set from:\n{filename}\n\n"
                f"Selected {len(selected_signals)} signals",
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            QMessageBox.critical(self, "Error", f"Invalid JSON file:\n{e}")
        except (PermissionError, OSError) as e:
            logger.error(f"Load signal set error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load signal set:\n{e}")

    def _save_signal_set(self) -> None:
        """Save current signal selection to JSON file."""
        selected_signals = self.signal_list.get_selected_signals()

        if not selected_signals:
            QMessageBox.warning(self, "No Selection", "Please select signals to save.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Signal Set",
            "",
            "Signal Set Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        # Ensure .json extension
        if not filename.endswith(".json"):
            filename += ".json"

        try:
            signal_set = {
                "selected_signals": selected_signals,
                "total_available": len(self.available_signals),
                "source_files": [str(f) for f in self.selected_files],
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(signal_set, f, indent=2)

            self.status_bar.set_status(
                f"Saved signal set: {len(selected_signals)} signals"
            )

            QMessageBox.information(
                self,
                "Success",
                f"Signal set saved to:\n{filename}\n\n"
                f"Saved {len(selected_signals)} signals",
            )

        except (PermissionError, OSError) as e:
            logger.error(f"Save signal set error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save signal set:\n{e}")

    # App Configuration handlers

    def _save_app_config(self) -> None:
        """Save current app configuration."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Save Configuration",
            "Configuration name:",
            QLineEdit.EchoMode.Normal,
            "my_config",
        )

        if not ok or not name:
            return

        try:
            config = {
                "output_directory": self.output_directory,
                "export_format": self.export_format_combo.currentText(),
                "include_timestamp": self.include_timestamp_check.isChecked(),
                "include_filter": self.include_filter_check.isChecked(),
                "export_selected_only": self.export_selected_only_check.isChecked(),
                "resample_rule": self.resample_rule_combo.currentText(),
                "resample_method": self.resample_method_combo.currentText(),
                "interpolate": self.interpolate_check.isChecked(),
                "integration_method": self.int_method_combo.currentText(),
                "differentiation_method": self.diff_method_combo.currentText(),
                "diff_order": self.diff_order_spin.value(),
                "diff_window_size": self.diff_window_spin.value(),
                "diff_poly_order": self.diff_poly_order_spin.value(),
            }

            self.config_manager.save_config(name, config)
            self.status_bar.set_status(f"Saved configuration: {name}")
            QMessageBox.information(
                self,
                "Success",
                f"Configuration '{name}' saved successfully.",
            )

        except (RuntimeError, AttributeError) as e:
            logger.error(f"Save config error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{e}")

    def _load_app_config(self) -> None:
        """Load app configuration."""
        try:
            configs = self.config_manager.list_configs()
            if not configs:
                QMessageBox.information(
                    self,
                    "No Configurations",
                    "No saved configurations found.",
                )
                return

            from PyQt6.QtWidgets import QInputDialog

            name, ok = QInputDialog.getItem(
                self,
                "Load Configuration",
                "Select configuration:",
                configs,
                0,
                False,
            )

            if not ok or not name:
                return

            config = self.config_manager.load_config(name)
            if not config:
                QMessageBox.warning(self, "Error", f"Configuration '{name}' not found.")
                return

            # Apply configuration
            if "output_directory" in config:
                self.output_directory = config["output_directory"]
                self.output_folder_edit.setText(self.output_directory)

            if "export_format" in config:
                idx = self.export_format_combo.findText(config["export_format"])
                if idx >= 0:
                    self.export_format_combo.setCurrentIndex(idx)

            if "include_timestamp" in config:
                self.include_timestamp_check.setChecked(config["include_timestamp"])

            if "include_filter" in config:
                self.include_filter_check.setChecked(config["include_filter"])

            if "export_selected_only" in config:
                self.export_selected_only_check.setChecked(
                    config["export_selected_only"]
                )

            if "resample_rule" in config:
                idx = self.resample_rule_combo.findText(config["resample_rule"])
                if idx >= 0:
                    self.resample_rule_combo.setCurrentIndex(idx)

            if "resample_method" in config:
                idx = self.resample_method_combo.findText(config["resample_method"])
                if idx >= 0:
                    self.resample_method_combo.setCurrentIndex(idx)

            if "interpolate" in config:
                self.interpolate_check.setChecked(config["interpolate"])

            if "integration_method" in config:
                idx = self.int_method_combo.findText(config["integration_method"])
                if idx >= 0:
                    self.int_method_combo.setCurrentIndex(idx)

            if "differentiation_method" in config:
                idx = self.diff_method_combo.findText(config["differentiation_method"])
                if idx >= 0:
                    self.diff_method_combo.setCurrentIndex(idx)

            if "diff_order" in config:
                self.diff_order_spin.setValue(config["diff_order"])

            if "diff_window_size" in config:
                self.diff_window_spin.setValue(config["diff_window_size"])

            if "diff_poly_order" in config:
                self.diff_poly_order_spin.setValue(config["diff_poly_order"])

            self.status_bar.set_status(f"Loaded configuration: {name}")
            QMessageBox.information(
                self,
                "Success",
                f"Configuration '{name}' loaded successfully.",
            )

        except ImportError as e:
            logger.error(f"Load config error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")

    # Resampling handlers

    def _apply_resample(self) -> None:
        """Apply time resampling to data."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        try:
            # Get time column
            time_col = self.time_col_combo.currentText()
            if not time_col:
                time_col = self.time_column
            if not time_col:
                QMessageBox.warning(
                    self, "No Time Column", "Please select a time column."
                )
                return

            # Get resample rule
            rule = self.resample_rule_combo.currentText()

            # Get method
            method = self.resample_method_combo.currentText()

            # Get interpolate option
            interpolate = self.interpolate_check.isChecked()

            self.status_bar.set_status(f"Resampling to {rule}...")

            # Use core library resample function
            self.current_data = resample_data(
                self.current_data,
                time_col=time_col,
                rule=rule,
                method=method,
                interpolate=interpolate,
            )

            # Update preview
            self.preview_widget.update_preview(self.current_data)
            self._update_data_info()

            self.status_bar.set_status(f"Resampled to {rule}")
            QMessageBox.information(
                self,
                "Success",
                f"Data resampled to {rule}\n"
                f"Method: {method}\n"
                f"Rows: {len(self.current_data)}",
            )

        except (RuntimeError, AttributeError) as e:
            logger.error(f"Resample error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Resampling failed:\n{e}")

    # Time Range handlers

    def _update_time_range_info(self) -> None:
        """Update time range information display."""
        if self.current_data is None or not self.time_column:
            self.data_start_label.setText("-")
            self.data_end_label.setText("-")
            self.data_duration_label.setText("-")
            return

        try:
            time_data = self.current_data[self.time_column]
            start = time_data.min()
            end = time_data.max()

            self.data_start_label.setText(str(start))
            self.data_end_label.setText(str(end))

            # Calculate duration
            try:
                duration = end - start
                self.data_duration_label.setText(str(duration))
            except (RuntimeError, AttributeError):
                self.data_duration_label.setText("-")

        except (RuntimeError, AttributeError) as e:
            logger.error(f"Time range info error: {e}")

    def _trim_time_range(self) -> None:
        """Trim data to specified time range."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        try:
            time_col = self.time_column
            if not time_col:
                QMessageBox.warning(self, "No Time Column", "Time column not detected.")
                return

            start_str = self.start_time_edit.text().strip()
            end_str = self.end_time_edit.text().strip()
            date_str = self.date_filter_edit.text().strip()

            # Parse start/end times
            start_time = float(start_str) if start_str else None
            end_time = float(end_str) if end_str else None

            self.status_bar.set_status("Trimming time range...")

            # Use core library function
            self.current_data = trim_time_range(
                self.current_data,
                time_col=time_col,
                start_time=start_time,
                end_time=end_time,
                date=date_str if date_str else None,
            )

            # Update UI
            self.preview_widget.update_preview(self.current_data)
            self._update_data_info()
            self._update_time_range_info()

            self.status_bar.set_status("Time range trimmed")
            QMessageBox.information(
                self,
                "Success",
                f"Data trimmed to time range\n" f"Rows: {len(self.current_data)}",
            )

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numeric time values.",
            )
        except (ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Trim time range error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Time range trim failed:\n{e}")

    def _copy_time_range_to_preview(self) -> None:
        """Copy current time range to preview filter."""
        if self.current_data is None or not self.time_column:
            return

        try:
            time_data = self.current_data[self.time_column]
            self.start_time_edit.setText(str(time_data.min()))
            self.end_time_edit.setText(str(time_data.max()))
            self.status_bar.set_status("Time range copied")
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Copy time range error: {e}")

    # Plot Config handlers

    def _save_plot_config(self) -> None:
        """Save current plot configuration."""
        name = self.plot_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Please enter a configuration name.")
            return

        try:
            selected_signals = self.signal_list.get_selected_signals()
            x_axis = self.x_axis_combo.currentText()

            config = {
                "name": name,
                "x_axis": x_axis,
                "y_signals": selected_signals,
                "trendline_type": self.trendline_type_combo.currentText(),
                "poly_degree": self.poly_degree_spin.value(),
                "trend_x_min": self.trend_x_min_edit.text(),
                "trend_x_max": self.trend_x_max_edit.text(),
            }

            self.plot_config_manager.save_plot_config(name, config)
            self._refresh_saved_plots_list()

            self.status_bar.set_status(f"Saved plot config: {name}")

        except (RuntimeError, AttributeError) as e:
            logger.error(f"Save plot config error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save plot config:\n{e}")

    def _load_plot_config(self) -> None:
        """Load selected plot configuration."""
        item = self.saved_plots_list.currentItem()
        if not item:
            QMessageBox.warning(
                self, "No Selection", "Please select a configuration to load."
            )
            return

        try:
            name = item.text()
            config = self.plot_config_manager.load_plot_config(name)

            if config:
                self.plot_name_edit.setText(config.get("name", ""))

                # Set X-axis
                x_axis = config.get("x_axis", "")
                idx = self.x_axis_combo.findText(x_axis)
                if idx >= 0:
                    self.x_axis_combo.setCurrentIndex(idx)

                # Set trendline type
                trend_type = config.get("trendline_type", "None")
                idx = self.trendline_type_combo.findText(trend_type)
                if idx >= 0:
                    self.trendline_type_combo.setCurrentIndex(idx)

                # Set polynomial degree
                self.poly_degree_spin.setValue(config.get("poly_degree", 2))

                # Set time range
                self.trend_x_min_edit.setText(config.get("trend_x_min", ""))
                self.trend_x_max_edit.setText(config.get("trend_x_max", ""))

                # Select Y signals
                y_signals = config.get("y_signals", [])
                self.signal_list.select_signals(y_signals)

                self.status_bar.set_status(f"Loaded plot config: {name}")

        except (RuntimeError, AttributeError) as e:
            logger.error(f"Load plot config error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load plot config:\n{e}")

    def _delete_plot_config(self) -> None:
        """Delete selected plot configuration."""
        item = self.saved_plots_list.currentItem()
        if not item:
            QMessageBox.warning(
                self, "No Selection", "Please select a configuration to delete."
            )
            return

        name = item.text()
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete plot configuration '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.plot_config_manager.delete_plot_config(name)
                self._refresh_saved_plots_list()
                self.status_bar.set_status(f"Deleted plot config: {name}")
            except (RuntimeError, AttributeError) as e:
                logger.error(f"Delete plot config error: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete:\n{e}")

    def _refresh_saved_plots_list(self) -> None:
        """Refresh the saved plot configurations list."""
        self.saved_plots_list.clear()
        try:
            configs = self.plot_config_manager.list_plot_configs()
            for name in configs:
                self.saved_plots_list.addItem(name)
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Refresh plots list error: {e}")

    def _calculate_trendline(self) -> None:
        """Calculate trendline for selected signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        try:
            x_col = self.x_axis_combo.currentText()
            if not x_col:
                QMessageBox.warning(
                    self, "No X-Axis", "Please select an X-axis signal."
                )
                return

            selected = self.signal_list.get_selected_signals()
            if not selected:
                QMessageBox.warning(
                    self, "No Y-Signals", "Please select Y-axis signals."
                )
                return

            trend_type = self.trendline_type_combo.currentText()
            if trend_type == "None":
                QMessageBox.warning(
                    self, "No Trendline", "Please select a trendline type."
                )
                return

            degree = self.poly_degree_spin.value()

            # Parse time range
            x_min = None
            x_max = None
            try:
                if self.trend_x_min_edit.text().strip():
                    x_min = float(self.trend_x_min_edit.text().strip())
                if self.trend_x_max_edit.text().strip():
                    x_max = float(self.trend_x_max_edit.text().strip())
            except ValueError:
                pass

            # Calculate trendline for first selected signal
            y_col = selected[0]

            result = calculate_trendline(
                self.current_data,
                x_col=x_col,
                y_col=y_col,
                trend_type=trend_type,
                degree=degree,
                x_min=x_min,
                x_max=x_max,
            )

            # Display results
            result_text = f"Trendline: {trend_type}\n"
            result_text += f"Equation: {result.get('equation', 'N/A')}\n"
            result_text += f"R² = {result.get('r_squared', 0):.6f}\n"
            if "coefficients" in result:
                result_text += f"Coefficients: {result['coefficients']}\n"

            self.trendline_results.setText(result_text)
            self.status_bar.set_status("Trendline calculated")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Trendline error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Trendline calculation failed:\n{e}")

    # DAT Import handlers

    def _browse_dat_file(self) -> None:
        """Browse for DAT file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open DAT File",
            "",
            "DAT Files (*.dat);;DBF Files (*.dbf);;All Files (*)",
        )
        if filename:
            self.dat_file_edit.setText(filename)

    def _preview_dat_file(self) -> None:
        """Preview DAT file contents."""
        filename = self.dat_file_edit.text()
        if not filename:
            QMessageBox.warning(self, "No File", "Please select a DAT file first.")
            return

        try:
            # Read first few lines for preview
            with open(filename, encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(20)]

            preview_text = "".join(lines)
            self.dat_preview_text.setText(preview_text)

        except (PermissionError, OSError) as e:
            logger.error(f"DAT preview error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to preview file:\n{e}")

    def _import_dat_file(self) -> None:
        """Import DAT file as data."""
        filename = self.dat_file_edit.text()
        if not filename:
            QMessageBox.warning(self, "No File", "Please select a DAT file first.")
            return

        try:
            # Get delimiter
            delimiters = {"Tab": "\t", "Comma": ",", "Semicolon": ";", "Space": " "}
            delimiter = delimiters.get(self.dat_delimiter_combo.currentText(), "\t")

            self.status_bar.set_status("Importing DAT file...")

            # Use core library function
            self.current_data = read_dat_file(filename, delimiter=delimiter)

            if self.current_data is not None:
                # Update available signals
                self.available_signals = self.data_loader.get_numeric_signals(
                    self.current_data
                )

                # Update UI
                self._update_data_info()
                self.signal_list.set_signals(self.available_signals)
                self.preview_widget.update_preview(self.current_data)
                self.analysis_panel.set_dataframe(self.current_data)

                self.status_bar.set_status(
                    f"Imported DAT file: {len(self.current_data)} rows"
                )
                QMessageBox.information(
                    self,
                    "Success",
                    f"Imported DAT file\n"
                    f"Rows: {len(self.current_data)}\n"
                    f"Columns: {len(self.current_data.columns)}",
                )

        except (RuntimeError, AttributeError) as e:
            logger.error(f"DAT import error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to import DAT file:\n{e}")

    def _convert_dat_to_csv(self) -> None:
        """Convert DAT file to CSV."""
        dat_filename = self.dat_file_edit.text()
        if not dat_filename:
            QMessageBox.warning(self, "No File", "Please select a DAT file first.")
            return

        # Get output filename
        default_name = Path(dat_filename).stem + ".csv"
        output_filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV File",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not output_filename:
            return

        try:
            self.status_bar.set_status("Converting DAT to CSV...")

            # Use core library function
            output_path = export_dat_to_csv(dat_filename, output_filename)

            self.status_bar.set_status("DAT converted to CSV")
            QMessageBox.information(
                self,
                "Success",
                f"DAT file converted to CSV:\n{output_path}",
            )

        except (RuntimeError, AttributeError) as e:
            logger.error(f"DAT conversion error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to convert DAT file:\n{e}")


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

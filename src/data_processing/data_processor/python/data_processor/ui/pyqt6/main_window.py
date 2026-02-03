"""Main window for PyQt6 Data Processor GUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from data_processor.core.data_loader import DataLoader
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import (
    DifferentiationConfig,
    FilterConfig,
    IntegrationConfig,
)
from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .widgets import (
    DataPreviewWidget,
    FilterConfigWidget,
    SignalListWidget,
    StatisticsWidget,
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
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            self.error.emit(str(e))


class DataProcessorMainWindow(QMainWindow):
    """Main window for the Data Processor application."""

    def __init__(self) -> None:
        super().__init__()

        # Core modules
        self.data_loader = DataLoader(use_high_performance=True)
        self.signal_processor = SignalProcessor()

        # State
        self.current_data: pd.DataFrame | None = None
        self.selected_files: list[str] = []
        self.available_signals: list[str] = []

        # Settings
        self.settings = QSettings("DataProcessor", "DataProcessorGUI")

        self._init_ui()
        self._setup_shortcuts()
        self._restore_state()

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

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")

        clear_action = QAction("&Clear Data", self)
        clear_action.triggered.connect(self._clear_data)
        edit_menu.addAction(clear_action)

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

        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        self.tab_widget.addTab(advanced_tab, "Advanced")

        # Preview tab
        preview_tab = self._create_preview_tab()
        self.tab_widget.addTab(preview_tab, "Preview")

        # Statistics tab
        stats_tab = self._create_statistics_tab()
        self.tab_widget.addTab(stats_tab, "Statistics")

        # Export tab
        export_tab = self._create_export_tab()
        self.tab_widget.addTab(export_tab, "Export")

    def _create_file_tab(self) -> QWidget:
        """Create file selection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open Files (Ctrl+O)")
        self.open_btn.clicked.connect(self._open_files)
        btn_layout.addWidget(self.open_btn)

        self.clear_files_btn = QPushButton("Clear")
        self.clear_files_btn.clicked.connect(self._clear_files)
        btn_layout.addWidget(self.clear_files_btn)

        self.load_btn = QPushButton("Load Data (Ctrl+L)")
        self.load_btn.clicked.connect(self._load_data)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addStretch()
        file_layout.addLayout(btn_layout)

        # File list
        self.file_list = QTextEdit()
        self.file_list.setReadOnly(True)
        self.file_list.setMaximumHeight(150)
        self.file_list.setPlaceholderText("No files selected...")
        file_layout.addWidget(self.file_list)

        layout.addWidget(file_group)

        # Data info group
        info_group = QGroupBox("Data Information")
        info_layout = QFormLayout(info_group)

        self.rows_label = QLabel("-")
        self.cols_label = QLabel("-")
        self.signals_label = QLabel("-")

        info_layout.addRow("Rows:", self.rows_label)
        info_layout.addRow("Columns:", self.cols_label)
        info_layout.addRow("Numeric Signals:", self.signals_label)

        layout.addWidget(info_group)
        layout.addStretch()

        return widget

    def _create_filter_tab(self) -> QWidget:
        """Create filter configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Filter config widget
        self.filter_config = FilterConfigWidget()
        layout.addWidget(self.filter_config)

        # Apply button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.setMinimumWidth(150)
        self.apply_filter_btn.clicked.connect(self._apply_filter)
        btn_layout.addWidget(self.apply_filter_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def _create_advanced_tab(self) -> QWidget:
        """Create advanced operations tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Integration group
        int_group = QGroupBox("Integration")
        int_layout = QVBoxLayout(int_group)

        int_method_layout = QHBoxLayout()
        int_method_layout.addWidget(QLabel("Method:"))
        self.int_method_combo = QComboBox()
        self.int_method_combo.addItems(["cumulative", "trapezoidal", "simpson"])
        int_method_layout.addWidget(self.int_method_combo)
        int_method_layout.addStretch()
        int_layout.addLayout(int_method_layout)

        self.integrate_btn = QPushButton("Integrate Selected Signals")
        self.integrate_btn.clicked.connect(self._integrate_signals)
        int_layout.addWidget(self.integrate_btn)

        layout.addWidget(int_group)

        # Differentiation group
        diff_group = QGroupBox("Differentiation")
        diff_layout = QVBoxLayout(diff_group)

        diff_order_layout = QHBoxLayout()
        diff_order_layout.addWidget(QLabel("Order:"))
        self.diff_order_spin = QSpinBox()
        self.diff_order_spin.setRange(1, 3)
        self.diff_order_spin.setValue(1)
        diff_order_layout.addWidget(self.diff_order_spin)

        diff_order_layout.addWidget(QLabel("Method:"))
        self.diff_method_combo = QComboBox()
        self.diff_method_combo.addItems(["central", "forward", "backward"])
        diff_order_layout.addWidget(self.diff_method_combo)
        diff_order_layout.addStretch()
        diff_layout.addLayout(diff_order_layout)

        self.diff_btn = QPushButton("Differentiate Selected Signals")
        self.diff_btn.clicked.connect(self._differentiate_signals)
        diff_layout.addWidget(self.diff_btn)

        layout.addWidget(diff_group)

        # Custom formula group
        formula_group = QGroupBox("Custom Formula")
        formula_layout = QFormLayout(formula_group)

        self.formula_name_edit = QLineEdit()
        self.formula_name_edit.setPlaceholderText("e.g., velocity")
        formula_layout.addRow("New Signal Name:", self.formula_name_edit)

        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("e.g., signal1 + signal2 * 2")
        formula_layout.addRow("Formula:", self.formula_edit)

        self.apply_formula_btn = QPushButton("Apply Formula")
        self.apply_formula_btn.clicked.connect(self._apply_formula)
        formula_layout.addRow("", self.apply_formula_btn)

        layout.addWidget(formula_group)
        layout.addStretch()

        return widget

    def _create_preview_tab(self) -> QWidget:
        """Create data preview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.preview_widget = DataPreviewWidget()
        layout.addWidget(self.preview_widget)

        return widget

    def _create_statistics_tab(self) -> QWidget:
        """Create statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_layout = QHBoxLayout()
        self.calc_stats_btn = QPushButton("Calculate Statistics")
        self.calc_stats_btn.clicked.connect(self._calculate_statistics)
        btn_layout.addWidget(self.calc_stats_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.stats_widget = StatisticsWidget()
        layout.addWidget(self.stats_widget)

        return widget

    def _create_export_tab(self) -> QWidget:
        """Create export tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        export_group = QGroupBox("Export Options")
        export_layout = QFormLayout(export_group)

        self.export_format_combo = QComboBox()
        formats = ["csv", "excel", "parquet", "hdf5", "feather"]
        self.export_format_combo.addItems(formats)
        export_layout.addRow("Format:", self.export_format_combo)

        layout.addWidget(export_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.export_btn = QPushButton("Export Data (Ctrl+S)")
        self.export_btn.setMinimumWidth(150)
        self.export_btn.clicked.connect(self._export_data)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+L"), self, self._load_data)

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

                # Get signals
                self.available_signals = self.data_loader.get_numeric_signals(
                    self.current_data
                )

                # Update UI
                self._update_data_info()
                self.signal_list.set_signals(self.available_signals)
                self.preview_widget.update_preview(self.current_data)

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

        except Exception as e:
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

    def _on_signal_selection_changed(self, signals: list[str]) -> None:
        """Handle signal selection change."""
        logger.debug(f"Selected signals: {signals}")

    def _apply_filter(self) -> None:
        """Apply filter to current data."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        try:
            self.status_bar.set_status("Applying filter...")
            self.status_bar.show_progress()
            QApplication.processEvents()

            filter_type = self.filter_config.get_filter_type()
            params = self.filter_config.get_parameters()

            config = FilterConfig(filter_type=filter_type, parameters=params)
            self.current_data = self.signal_processor.apply_filter(
                self.current_data, config
            )

            self.preview_widget.update_preview(self.current_data)
            self.status_bar.hide_progress()
            self.status_bar.set_status(f"Applied {filter_type}")

            QMessageBox.information(
                self, "Success", f"{filter_type} applied successfully"
            )

        except Exception as e:
            self.status_bar.hide_progress()
            self.status_bar.set_status("Filter failed")
            logger.error(f"Filter error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Filter failed:\n{e}")

    def _integrate_signals(self) -> None:
        """Integrate selected signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        signals = self.signal_list.get_selected_signals()
        if not signals:
            signals = self.available_signals

        try:
            self.status_bar.set_status("Integrating...")
            config = IntegrationConfig(
                signals=signals, method=self.int_method_combo.currentText()
            )
            self.current_data = self.signal_processor.integrate_signals(
                self.current_data, config
            )
            self.preview_widget.update_preview(self.current_data)
            self.status_bar.set_status("Integration complete")
            QMessageBox.information(self, "Success", "Integration complete")
        except Exception as e:
            logger.error(f"Integration error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Integration failed:\n{e}")

    def _differentiate_signals(self) -> None:
        """Differentiate selected signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        signals = self.signal_list.get_selected_signals()
        if not signals:
            signals = self.available_signals

        try:
            self.status_bar.set_status("Differentiating...")
            config = DifferentiationConfig(
                signals=signals,
                order=self.diff_order_spin.value(),
                method=self.diff_method_combo.currentText(),
            )
            self.current_data = self.signal_processor.differentiate_signals(
                self.current_data, config
            )
            self.preview_widget.update_preview(self.current_data)
            self.status_bar.set_status("Differentiation complete")
            QMessageBox.information(self, "Success", "Differentiation complete")
        except Exception as e:
            logger.error(f"Differentiation error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Differentiation failed:\n{e}")

    def _apply_formula(self) -> None:
        """Apply custom formula."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        name = self.formula_name_edit.text().strip()
        formula = self.formula_edit.text().strip()

        if not name or not formula:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter both name and formula."
            )
            return

        try:
            self.status_bar.set_status(f"Applying formula: {name}...")
            self.current_data, success = self.signal_processor.apply_custom_formula(
                self.current_data, name, formula
            )

            if success:
                self.available_signals = self.data_loader.get_numeric_signals(
                    self.current_data
                )
                self.signal_list.set_signals(self.available_signals)
                self.preview_widget.update_preview(self.current_data)
                self.status_bar.set_status(f"Created signal: {name}")
                QMessageBox.information(
                    self, "Success", f"Signal '{name}' created successfully"
                )
            else:
                QMessageBox.warning(self, "Error", "Formula application failed")

        except Exception as e:
            logger.error(f"Formula error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Formula failed:\n{e}")

    def _calculate_statistics(self) -> None:
        """Calculate statistics for signals."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        signals = self.signal_list.get_selected_signals()
        if not signals:
            signals = self.available_signals[:20]  # Limit to 20

        self.stats_widget.update_statistics(self.current_data, signals)
        self.status_bar.set_status("Statistics calculated")

    def _export_data(self) -> None:
        """Export data to file."""
        if self.current_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        format_type = self.export_format_combo.currentText()
        extensions = {
            "csv": "CSV Files (*.csv)",
            "excel": "Excel Files (*.xlsx)",
            "parquet": "Parquet Files (*.parquet)",
            "hdf5": "HDF5 Files (*.h5)",
            "feather": "Feather Files (*.feather)",
        }

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            extensions.get(format_type, "All Files (*)"),
        )

        if not filename:
            return

        try:
            self.status_bar.set_status("Exporting...")
            success = self.data_loader.save_dataframe(
                self.current_data, filename, format_type=format_type
            )

            if success:
                self.status_bar.set_status(f"Exported to {Path(filename).name}")
                QMessageBox.information(
                    self, "Success", f"Data exported to:\n{filename}"
                )
            else:
                QMessageBox.warning(self, "Error", "Export failed")

        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    def _clear_data(self) -> None:
        """Clear all data."""
        self.current_data = None
        self.available_signals = []
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
            "- Multiple export formats\n\n"
            "Built with PyQt6",
        )


def main() -> None:
    """Run the Data Processor application."""
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DataProcessorMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

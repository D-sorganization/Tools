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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from data_processor.core.config_manager import ConfigManager
from data_processor.core.dat_importer import (
    export_dat_to_csv,
    read_dat_file,
)
from data_processor.core.data_loader import DataLoader
from data_processor.core.dataset_naming import (
    generate_dataset_name,
)
from data_processor.core.plot_config_manager import PlotConfigManager
from data_processor.core.signal_list_manager import SignalListManager
from data_processor.core.signal_processing import (
    apply_custom_variable,
    calculate_trendline,
    differentiate_signals,
    integrate_signals,
    resample_data,
    trim_time_range,
)
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import (
    DifferentiationConfig,
    FilterConfig,
    IntegrationConfig,
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

        # DAT Import tab
        dat_import_tab = self._create_dat_import_tab()
        self.tab_widget.addTab(dat_import_tab, "DAT Import")

        # Export tab
        export_tab = self._create_export_tab()
        self.tab_widget.addTab(export_tab, "Export")

        # Help tab
        help_tab = self._create_help_tab()
        self.tab_widget.addTab(help_tab, "Help")

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
        self.int_method_combo.addItems(["trapezoidal", "simpson", "rectangular"])
        self.int_method_combo.setToolTip(
            "Trapezoidal: Standard numerical integration\n"
            "Simpson: Higher accuracy for smooth signals\n"
            "Rectangular: Simple sum (left Riemann sum)"
        )
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

        diff_method_layout = QHBoxLayout()
        diff_method_layout.addWidget(QLabel("Method:"))
        self.diff_method_combo = QComboBox()
        self.diff_method_combo.addItems(["spline", "rolling_polynomial"])
        self.diff_method_combo.setToolTip(
            "Spline: Acausal smooth derivative using spline interpolation\n"
            "Rolling Polynomial: Causal derivative using Savitzky-Golay filter"
        )
        diff_method_layout.addWidget(self.diff_method_combo)
        diff_method_layout.addStretch()
        diff_layout.addLayout(diff_method_layout)

        diff_order_layout = QHBoxLayout()
        diff_order_layout.addWidget(QLabel("Order(s):"))
        self.diff_order_spin = QSpinBox()
        self.diff_order_spin.setRange(1, 3)
        self.diff_order_spin.setValue(1)
        self.diff_order_spin.setToolTip("Derivative order (1st, 2nd, or 3rd)")
        diff_order_layout.addWidget(self.diff_order_spin)

        diff_order_layout.addWidget(QLabel("Window Size:"))
        self.diff_window_spin = QSpinBox()
        self.diff_window_spin.setRange(3, 51)
        self.diff_window_spin.setValue(11)
        self.diff_window_spin.setSingleStep(2)
        self.diff_window_spin.setToolTip(
            "Window size for rolling polynomial (must be odd)"
        )
        diff_order_layout.addWidget(self.diff_window_spin)

        diff_order_layout.addWidget(QLabel("Poly Order:"))
        self.diff_poly_order_spin = QSpinBox()
        self.diff_poly_order_spin.setRange(2, 6)
        self.diff_poly_order_spin.setValue(3)
        self.diff_poly_order_spin.setToolTip(
            "Polynomial order for Savitzky-Golay filter"
        )
        diff_order_layout.addWidget(self.diff_poly_order_spin)

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

    def _create_resample_tab(self) -> QWidget:
        """Create time resampling tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Resampling group
        resample_group = QGroupBox("Time Resampling")
        resample_layout = QFormLayout(resample_group)

        # Resample rule (frequency)
        self.resample_rule_combo = QComboBox()
        self.resample_rule_combo.addItems(
            [
                "100ms",
                "250ms",
                "500ms",
                "1s",
                "2s",
                "5s",
                "10s",
                "30s",
                "1min",
                "5min",
                "10min",
                "15min",
                "30min",
                "1h",
            ]
        )
        self.resample_rule_combo.setCurrentText("1s")
        resample_layout.addRow("Target Frequency:", self.resample_rule_combo)

        # Custom frequency
        custom_freq_layout = QHBoxLayout()
        self.custom_freq_spin = QDoubleSpinBox()
        self.custom_freq_spin.setRange(0.001, 86400)
        self.custom_freq_spin.setValue(1.0)
        self.custom_freq_spin.setDecimals(3)
        custom_freq_layout.addWidget(self.custom_freq_spin)
        self.custom_freq_unit = QComboBox()
        self.custom_freq_unit.addItems(["seconds", "milliseconds", "minutes", "hours"])
        custom_freq_layout.addWidget(self.custom_freq_unit)
        resample_layout.addRow("Custom Frequency:", custom_freq_layout)

        # Aggregation method
        self.resample_method_combo = QComboBox()
        self.resample_method_combo.addItems(
            ["mean", "median", "first", "last", "min", "max", "sum"]
        )
        resample_layout.addRow("Aggregation Method:", self.resample_method_combo)

        # Interpolation option
        self.interpolate_check = QCheckBox("Interpolate missing values")
        self.interpolate_check.setChecked(True)
        resample_layout.addRow("", self.interpolate_check)

        layout.addWidget(resample_group)

        # Time column selection
        time_col_group = QGroupBox("Time Column")
        time_col_layout = QFormLayout(time_col_group)

        self.time_col_combo = QComboBox()
        time_col_layout.addRow("Time Column:", self.time_col_combo)

        layout.addWidget(time_col_group)

        # Apply button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.apply_resample_btn = QPushButton("Apply Resampling")
        self.apply_resample_btn.setMinimumWidth(150)
        self.apply_resample_btn.clicked.connect(self._apply_resample)
        btn_layout.addWidget(self.apply_resample_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def _create_time_range_tab(self) -> QWidget:
        """Create time range selection tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Time range group
        time_group = QGroupBox("Time Range Selection")
        time_layout = QFormLayout(time_group)

        # Start time
        self.start_time_edit = QLineEdit()
        self.start_time_edit.setPlaceholderText("e.g., 0.0 or 2024-01-01 10:00:00")
        time_layout.addRow("Start Time:", self.start_time_edit)

        # End time
        self.end_time_edit = QLineEdit()
        self.end_time_edit.setPlaceholderText("e.g., 100.0 or 2024-01-01 11:00:00")
        time_layout.addRow("End Time:", self.end_time_edit)

        # Date filter (optional)
        self.date_filter_edit = QLineEdit()
        self.date_filter_edit.setPlaceholderText("e.g., 2024-01-01 (optional)")
        time_layout.addRow("Filter by Date:", self.date_filter_edit)

        layout.addWidget(time_group)

        # Current data range info
        range_info_group = QGroupBox("Current Data Range")
        range_info_layout = QFormLayout(range_info_group)

        self.data_start_label = QLabel("-")
        self.data_end_label = QLabel("-")
        self.data_duration_label = QLabel("-")

        range_info_layout.addRow("Data Start:", self.data_start_label)
        range_info_layout.addRow("Data End:", self.data_end_label)
        range_info_layout.addRow("Duration:", self.data_duration_label)

        layout.addWidget(range_info_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.copy_to_preview_btn = QPushButton("Copy Range to Preview")
        self.copy_to_preview_btn.clicked.connect(self._copy_time_range_to_preview)
        btn_layout.addWidget(self.copy_to_preview_btn)

        self.trim_data_btn = QPushButton("Trim Data to Range")
        self.trim_data_btn.setMinimumWidth(150)
        self.trim_data_btn.clicked.connect(self._trim_time_range)
        btn_layout.addWidget(self.trim_data_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def _create_plot_config_tab(self) -> QWidget:
        """Create plot configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Plot settings group
        plot_settings_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout(plot_settings_group)

        # Plot name
        self.plot_name_edit = QLineEdit()
        self.plot_name_edit.setPlaceholderText("My Plot Configuration")
        plot_layout.addRow("Configuration Name:", self.plot_name_edit)

        # X-axis signal
        self.x_axis_combo = QComboBox()
        plot_layout.addRow("X-Axis Signal:", self.x_axis_combo)

        # Y-axis signals (selected from signal list)
        self.y_signals_label = QLabel("Use signal list to select Y-axis signals")
        plot_layout.addRow("Y-Axis Signals:", self.y_signals_label)

        layout.addWidget(plot_settings_group)

        # Trendline group
        trendline_group = QGroupBox("Trendline Analysis")
        trend_layout = QFormLayout(trendline_group)

        self.trendline_type_combo = QComboBox()
        self.trendline_type_combo.addItems(
            ["None", "linear", "polynomial", "exponential", "power"]
        )
        trend_layout.addRow("Trendline Type:", self.trendline_type_combo)

        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setRange(2, 10)
        self.poly_degree_spin.setValue(2)
        trend_layout.addRow("Polynomial Degree:", self.poly_degree_spin)

        # Trendline time range
        trend_range_layout = QHBoxLayout()
        self.trend_x_min_edit = QLineEdit()
        self.trend_x_min_edit.setPlaceholderText("X Min")
        trend_range_layout.addWidget(self.trend_x_min_edit)
        self.trend_x_max_edit = QLineEdit()
        self.trend_x_max_edit.setPlaceholderText("X Max")
        trend_range_layout.addWidget(self.trend_x_max_edit)
        trend_layout.addRow("Trendline Range:", trend_range_layout)

        layout.addWidget(trendline_group)

        # Saved configurations
        saved_group = QGroupBox("Saved Plot Configurations")
        saved_layout = QVBoxLayout(saved_group)

        self.saved_plots_list = QListWidget()
        self.saved_plots_list.setMaximumHeight(150)
        saved_layout.addWidget(self.saved_plots_list)

        saved_btn_layout = QHBoxLayout()
        self.save_plot_config_btn = QPushButton("Save Config")
        self.save_plot_config_btn.clicked.connect(self._save_plot_config)
        saved_btn_layout.addWidget(self.save_plot_config_btn)

        self.load_plot_config_btn = QPushButton("Load Config")
        self.load_plot_config_btn.clicked.connect(self._load_plot_config)
        saved_btn_layout.addWidget(self.load_plot_config_btn)

        self.delete_plot_config_btn = QPushButton("Delete Config")
        self.delete_plot_config_btn.clicked.connect(self._delete_plot_config)
        saved_btn_layout.addWidget(self.delete_plot_config_btn)

        saved_layout.addLayout(saved_btn_layout)
        layout.addWidget(saved_group)

        # Calculate trendline button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.calc_trendline_btn = QPushButton("Calculate Trendline")
        self.calc_trendline_btn.clicked.connect(self._calculate_trendline)
        btn_layout.addWidget(self.calc_trendline_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Trendline results
        self.trendline_results = QTextEdit()
        self.trendline_results.setReadOnly(True)
        self.trendline_results.setMaximumHeight(100)
        self.trendline_results.setPlaceholderText(
            "Trendline equation and R² will appear here..."
        )
        layout.addWidget(self.trendline_results)

        layout.addStretch()
        return widget

    def _create_dat_import_tab(self) -> QWidget:
        """Create DAT file import tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # DAT file selection
        dat_group = QGroupBox("DAT File Import")
        dat_layout = QVBoxLayout(dat_group)

        # File selection
        file_btn_layout = QHBoxLayout()
        self.dat_file_edit = QLineEdit()
        self.dat_file_edit.setReadOnly(True)
        self.dat_file_edit.setPlaceholderText("Select a DAT file...")
        file_btn_layout.addWidget(self.dat_file_edit)

        self.browse_dat_btn = QPushButton("Browse...")
        self.browse_dat_btn.clicked.connect(self._browse_dat_file)
        file_btn_layout.addWidget(self.browse_dat_btn)
        dat_layout.addLayout(file_btn_layout)

        # Delimiter selection
        delim_layout = QHBoxLayout()
        delim_layout.addWidget(QLabel("Delimiter:"))
        self.dat_delimiter_combo = QComboBox()
        self.dat_delimiter_combo.addItems(["Tab", "Comma", "Semicolon", "Space"])
        delim_layout.addWidget(self.dat_delimiter_combo)
        delim_layout.addStretch()
        dat_layout.addLayout(delim_layout)

        layout.addWidget(dat_group)

        # Preview group
        preview_group = QGroupBox("File Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.dat_preview_text = QTextEdit()
        self.dat_preview_text.setReadOnly(True)
        self.dat_preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self.dat_preview_text)

        preview_btn = QPushButton("Preview DAT File")
        preview_btn.clicked.connect(self._preview_dat_file)
        preview_layout.addWidget(preview_btn)

        layout.addWidget(preview_group)

        # Conversion buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.import_dat_btn = QPushButton("Import as Data")
        self.import_dat_btn.clicked.connect(self._import_dat_file)
        btn_layout.addWidget(self.import_dat_btn)

        self.convert_dat_btn = QPushButton("Convert to CSV")
        self.convert_dat_btn.clicked.connect(self._convert_dat_to_csv)
        btn_layout.addWidget(self.convert_dat_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def _create_help_tab(self) -> QWidget:
        """Create help/documentation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        help_browser = QTextBrowser()
        help_browser.setOpenExternalLinks(True)
        help_browser.setHtml(self._get_help_content())
        layout.addWidget(help_browser)

        return widget

    def _get_help_content(self) -> str:
        """Get help content HTML."""
        return """
        <h1 style="color: #007acc;">Data Processor Help</h1>

        <h2>Overview</h2>
        <p>Data Processor is a powerful tool for signal processing and
        data analysis. It supports loading CSV files, applying filters,
        integrating/differentiating signals, and exporting processed
        data in multiple formats.</p>

        <h2>Tabs</h2>

        <h3>Files</h3>
        <p>Load CSV data files. Multiple files can be selected and combined.</p>
        <ul>
            <li><b>Open Files (Ctrl+O)</b>: Select CSV files to load</li>
            <li><b>Load Data (Ctrl+L)</b>: Process selected files into memory</li>
            <li><b>Clear</b>: Clear current file selection</li>
        </ul>

        <h3>Filters</h3>
        <p>Apply digital filters to smooth or process signals:</p>
        <ul>
            <li><b>Low Pass</b>: Remove high-frequency noise</li>
            <li><b>High Pass</b>: Remove low-frequency drift</li>
            <li><b>Band Pass</b>: Keep frequencies within a range</li>
            <li><b>Moving Average</b>: Simple smoothing filter</li>
            <li><b>Savitzky-Golay</b>: Polynomial smoothing filter</li>
        </ul>

        <h3>Advanced</h3>
        <p>Mathematical operations on signals:</p>
        <ul>
            <li><b>Integration</b>: Trapezoidal, Simpson, rectangular</li>
            <li><b>Differentiation</b>: Spline or rolling polynomial</li>
            <li><b>Custom Formula</b>: Create new signals with math</li>
        </ul>

        <h3>Resample</h3>
        <p>Change the time resolution of your data:</p>
        <ul>
            <li>Select target frequency (1s, 100ms, etc.)</li>
            <li>Choose aggregation method (mean, median, max, etc.)</li>
            <li>Enable interpolation to fill gaps</li>
        </ul>

        <h3>Time Range</h3>
        <p>Trim data to a specific time window:</p>
        <ul>
            <li>Set start and end times</li>
            <li>Filter by specific date</li>
            <li>Copy range to preview plot</li>
        </ul>

        <h3>Preview</h3>
        <p>View your data in a table format before export.</p>

        <h3>Plot Config</h3>
        <p>Configure plots with trendlines:</p>
        <ul>
            <li><b>Trendline types</b>: Linear, polynomial, exponential, power</li>
            <li><b>Save/Load</b>: Store plot configurations for reuse</li>
            <li><b>Time windows</b>: Apply trendlines to specific ranges</li>
        </ul>

        <h3>Statistics</h3>
        <p>Calculate descriptive statistics for selected signals including mean, median,
        standard deviation, min, max, and more.</p>

        <h3>DAT Import</h3>
        <p>Import industrial DAT files:</p>
        <ul>
            <li>Preview file contents</li>
            <li>Select delimiter type</li>
            <li>Convert to CSV format</li>
        </ul>

        <h3>Export</h3>
        <p>Save processed data in various formats:</p>
        <ul>
            <li><b>CSV</b>: Universal compatibility</li>
            <li><b>Excel</b>: For spreadsheet analysis</li>
            <li><b>Parquet</b>: Efficient columnar storage</li>
            <li><b>HDF5</b>: Scientific data format</li>
            <li><b>Feather</b>: Fast binary format</li>
        </ul>

        <h2>Keyboard Shortcuts</h2>
        <table border="1" cellpadding="5">
            <tr><td><b>Ctrl+O</b></td><td>Open files</td></tr>
            <tr><td><b>Ctrl+L</b></td><td>Load data</td></tr>
            <tr><td><b>Ctrl+S</b></td><td>Export data</td></tr>
            <tr><td><b>Ctrl+F</b></td><td>Focus signal search</td></tr>
            <tr><td><b>Ctrl+Shift+L</b></td><td>Load signal set</td></tr>
            <tr><td><b>Ctrl+Shift+S</b></td><td>Save signal set</td></tr>
        </table>

        <h2>Signal List</h2>
        <p>The left panel shows available signals:</p>
        <ul>
            <li>Click to select individual signals</li>
            <li>Use Ctrl+Click for multiple selection</li>
            <li>Use the search box to filter signals</li>
            <li>Save/Load signal sets from the File menu</li>
        </ul>

        <h2>Custom Formulas</h2>
        <p>Create new signals using Python-like syntax:</p>
        <pre>
        velocity = signal1 * 2 + signal2
        power = voltage * current
        ratio = signal_a / signal_b
        scaled = sqrt(x**2 + y**2)
        </pre>
        <p>Available functions: sin, cos, tan, sqrt, abs, log, log10, exp, min, max</p>
        """

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

        # Output folder group
        folder_group = QGroupBox("Output Folder")
        folder_layout = QHBoxLayout(folder_group)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setText(self.output_directory)
        self.output_folder_edit.setReadOnly(True)
        folder_layout.addWidget(self.output_folder_edit)

        self.browse_folder_btn = QPushButton("Browse...")
        self.browse_folder_btn.clicked.connect(self._browse_output_folder)
        folder_layout.addWidget(self.browse_folder_btn)

        layout.addWidget(folder_group)

        # Dataset naming group
        naming_group = QGroupBox("Dataset Naming")
        naming_layout = QFormLayout(naming_group)

        self.dataset_name_edit = QLineEdit()
        self.dataset_name_edit.setPlaceholderText("Auto-generated or custom name")
        naming_layout.addRow("Dataset Name:", self.dataset_name_edit)

        self.include_timestamp_check = QCheckBox("Include timestamp")
        self.include_timestamp_check.setChecked(True)
        naming_layout.addRow("", self.include_timestamp_check)

        self.include_filter_check = QCheckBox("Include filter info in name")
        self.include_filter_check.setChecked(False)
        naming_layout.addRow("", self.include_filter_check)

        auto_name_btn = QPushButton("Auto-Generate Name")
        auto_name_btn.clicked.connect(self._auto_generate_name)
        naming_layout.addRow("", auto_name_btn)

        layout.addWidget(naming_group)

        # Export options group
        export_group = QGroupBox("Export Options")
        export_layout = QFormLayout(export_group)

        self.export_format_combo = QComboBox()
        formats = ["csv", "excel", "parquet", "hdf5", "feather"]
        self.export_format_combo.addItems(formats)
        export_layout.addRow("Format:", self.export_format_combo)

        self.export_selected_only_check = QCheckBox("Export selected signals only")
        self.export_selected_only_check.setChecked(False)
        export_layout.addRow("", self.export_selected_only_check)

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
            signals = self.available_signals[:10]  # Limit to first 10 if none selected

        if not self.time_column:
            QMessageBox.warning(self, "No Time Column", "Time column not detected.")
            return

        try:
            self.status_bar.set_status("Integrating...")
            method = self.int_method_combo.currentText()

            # Use core library function
            self.current_data = integrate_signals(
                self.current_data,
                time_col=self.time_column,
                signals=signals,
                method=method,
            )

            # Update available signals
            self.available_signals = self.data_loader.get_numeric_signals(
                self.current_data
            )
            self.signal_list.set_signals(self.available_signals)
            self.preview_widget.update_preview(self.current_data)

            self.status_bar.set_status(f"Integration complete ({method})")
            QMessageBox.information(
                self,
                "Success",
                f"Integration complete\n"
                f"Method: {method}\n"
                f"Signals: {len(signals)}",
            )
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
            signals = self.available_signals[:10]  # Limit to first 10 if none selected

        if not self.time_column:
            QMessageBox.warning(self, "No Time Column", "Time column not detected.")
            return

        try:
            self.status_bar.set_status("Differentiating...")
            method = self.diff_method_combo.currentText()
            order = self.diff_order_spin.value()
            window_size = self.diff_window_spin.value()
            poly_order = self.diff_poly_order_spin.value()

            # Ensure window size is odd
            if window_size % 2 == 0:
                window_size += 1

            # Use core library function
            self.current_data = differentiate_signals(
                self.current_data,
                time_col=self.time_column,
                signals=signals,
                method=method,
                orders=[order],
                window_size=window_size,
                poly_order=poly_order,
            )

            # Update available signals
            self.available_signals = self.data_loader.get_numeric_signals(
                self.current_data
            )
            self.signal_list.set_signals(self.available_signals)
            self.preview_widget.update_preview(self.current_data)

            self.status_bar.set_status(f"Differentiation complete ({method})")
            QMessageBox.information(
                self,
                "Success",
                f"Differentiation complete\n"
                f"Method: {method}\n"
                f"Order: {order}\n"
                f"Signals: {len(signals)}",
            )
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

            # Use core library function
            self.current_data = apply_custom_variable(
                self.current_data,
                name=name,
                formula=formula,
                time_col=self.time_column,
            )

            # Update available signals
            self.available_signals = self.data_loader.get_numeric_signals(
                self.current_data
            )
            self.signal_list.set_signals(self.available_signals)
            self.preview_widget.update_preview(self.current_data)
            self._update_column_combos()

            self.status_bar.set_status(f"Created signal: {name}")
            QMessageBox.information(
                self,
                "Success",
                f"Signal '{name}' created successfully\n" f"Formula: {formula}",
            )

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
            "csv": ("CSV Files (*.csv)", ".csv"),
            "excel": ("Excel Files (*.xlsx)", ".xlsx"),
            "parquet": ("Parquet Files (*.parquet)", ".parquet"),
            "hdf5": ("HDF5 Files (*.h5)", ".h5"),
            "feather": ("Feather Files (*.feather)", ".feather"),
        }

        filter_str, ext = extensions.get(format_type, ("All Files (*)", ""))

        # Get default filename from dataset name or generate one
        default_name = self.dataset_name_edit.text().strip()
        if not default_name:
            default_name = "processed_data"

        # Use output directory as starting point
        default_path = str(Path(self.output_directory) / (default_name + ext))

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            default_path,
            filter_str,
        )

        if not filename:
            return

        try:
            self.status_bar.set_status("Exporting...")

            # Prepare data for export
            export_data = self.current_data

            # If export selected only, filter columns
            if self.export_selected_only_check.isChecked():
                selected_signals = self.signal_list.get_selected_signals()
                if selected_signals:
                    # Keep time column and selected signals
                    columns_to_keep = []
                    if self.time_column:
                        columns_to_keep.append(self.time_column)
                    columns_to_keep.extend(
                        [s for s in selected_signals if s in export_data.columns]
                    )
                    export_data = export_data[columns_to_keep]

            success = self.data_loader.save_dataframe(
                export_data, filename, format_type=format_type
            )

            if success:
                self.status_bar.set_status(f"Exported to {Path(filename).name}")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Data exported to:\n{filename}\n\n"
                    f"Rows: {len(export_data)}\n"
                    f"Columns: {len(export_data.columns)}",
                )
            else:
                QMessageBox.warning(self, "Error", "Export failed")

        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    def _browse_output_folder(self) -> None:
        """Browse for output folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            self.output_directory,
        )
        if folder:
            self.output_directory = folder
            self.output_folder_edit.setText(folder)
            self.status_bar.set_status(f"Output folder: {folder}")

    def _auto_generate_name(self) -> None:
        """Auto-generate dataset name."""
        base_name = "processed_data"
        if self.selected_files:
            # Use first file name as base
            base_name = Path(self.selected_files[0]).stem

        include_timestamp = self.include_timestamp_check.isChecked()
        include_filter = self.include_filter_check.isChecked()

        # Get current filter type if any
        filter_type = None
        if include_filter:
            try:
                filter_type = self.filter_config.get_filter_type()
            except Exception:
                pass

        name = generate_dataset_name(
            base_name=base_name,
            include_timestamp=include_timestamp,
            include_filter=include_filter,
            filter_type=filter_type,
        )

        self.dataset_name_edit.setText(name)
        self.status_bar.set_status(f"Generated name: {name}")

    def _clear_data(self) -> None:
        """Clear all data."""
        self.current_data = None
        self.filtered_data = None
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
        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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
            except Exception:
                self.data_duration_label.setText("-")

        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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
            except Exception as e:
                logger.error(f"Delete plot config error: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete:\n{e}")

    def _refresh_saved_plots_list(self) -> None:
        """Refresh the saved plot configurations list."""
        self.saved_plots_list.clear()
        try:
            configs = self.plot_config_manager.list_plot_configs()
            for name in configs:
                self.saved_plots_list.addItem(name)
        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
            logger.error(f"DAT conversion error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to convert DAT file:\n{e}")


def main() -> None:
    """Run the Data Processor application."""
    import sys

    from shared.python.plot_theme import setup_plot_theme_for_app
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

# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""TabCreationMixin -- UI tab creation methods for DataProcessorMainWindow.

Creates all tab widgets: file, filter, advanced, resample, time range,
plot config, DAT import, help, preview, statistics, export.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .analysis_widgets import ChartStylePanel
from .widgets import DataPreviewWidget, FilterConfigWidget, StatisticsWidget

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TabCreationMixin:
    """Mixin providing tab creation methods for DataProcessorMainWindow."""

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

        self.compare_filters_btn = QPushButton("Compare Filters")
        self.compare_filters_btn.setMinimumWidth(150)
        self.compare_filters_btn.clicked.connect(self._show_filter_comparison)
        btn_layout.addWidget(self.compare_filters_btn)
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

        # Chart style controls (shared plot engine integration)
        self.chart_style_panel = ChartStylePanel()
        layout.addWidget(self.chart_style_panel)

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

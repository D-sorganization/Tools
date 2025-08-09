# =============================================================================
# Advanced CSV Time Series Processor & Analyzer - PyQt6 Version
#
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This version is built with PyQt6 for
# improved performance and modern UI capabilities.
#
# Dependencies for Python 3.8+:
# pip install PyQt6 pandas numpy scipy matplotlib openpyxl Pillow simpledbf pyarrow tables feather-format
#
# =============================================================================

import configparser
import io
import json
import logging
import math
import mimetypes
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from PyQt6.QtCore import (QEasingCurve, QMutex, QPoint, QPropertyAnimation,
                          QRect, QRunnable, QSize, Qt, QThread, QThreadPool,
                          QTimer, QWaitCondition, pyqtSignal)
from PyQt6.QtGui import (QAction, QBrush, QCloseEvent, QColor, QFont, QIcon,
                         QKeySequence, QPainter, QPalette, QPen, QPixmap,
                         QResizeEvent)
# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QMainWindow, QMenuBar,
                             QMessageBox, QProgressBar, QPushButton,
                             QRadioButton, QScrollArea, QSizePolicy, QSlider,
                             QSpacerItem, QSpinBox, QSplitter, QStatusBar,
                             QTableWidget, QTableWidgetItem, QTabWidget,
                             QTextEdit, QToolBar, QVBoxLayout, QWidget)
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from scipy.signal import butter, filtfilt, medfilt, savgol_filter
from scipy.stats import linregress
from simpledbf import Dbf5

# Additional ML-specific imports
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import scipy.io

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# PyArrow imports for parquet handling
try:
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Matplotlib imports for PyQt6
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from file_utils import DataReader, DataWriter, FileFormatDetector
from folder_tool_tab import FolderToolTab
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from threads import ConversionThread, FolderProcessingThread, ProcessingThread


# =============================================================================
# WORKER FUNCTION FOR PARALLEL PROCESSING
# =============================================================================
def process_single_csv_file(file_path, settings):
    """
    Processes a single CSV file based on a dictionary of settings.
    This function is designed to be run in a separate process.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)

        # Determine which signals to keep for this specific file
        signals_in_this_file = [
            s for s in settings["selected_signals"] if s in df.columns
        ]
        time_col = df.columns[0]
        if time_col not in signals_in_this_file:
            signals_in_this_file.insert(0, time_col)

        processed_df = df[signals_in_this_file].copy()

        # Data type conversion
        processed_df[time_col] = pd.to_datetime(processed_df[time_col], errors="coerce")
        processed_df.dropna(subset=[time_col], inplace=True)
        for col in processed_df.columns:
            if col != time_col:
                processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        if processed_df.empty:
            return None

        processed_df.set_index(time_col, inplace=True)

        # Apply Filtering
        filter_type = settings.get("filter_type")
        if filter_type and filter_type != "None":
            numeric_cols = processed_df.select_dtypes(
                include=np.number
            ).columns.tolist()
            for col in numeric_cols:
                signal_data = processed_df[col].dropna()
                if len(signal_data) < 2:
                    continue

                # Apply filtering based on type
                if filter_type == "Moving Average":
                    window_size = settings.get("ma_window", 10)
                    processed_df[col] = signal_data.rolling(
                        window=window_size, min_periods=1
                    ).mean()
                elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
                    order = settings.get("bw_order", 3)
                    cutoff = settings.get("bw_cutoff", 0.1)
                    sr = (
                        1.0
                        / pd.to_numeric(
                            signal_data.index.to_series().diff().dt.total_seconds()
                        ).mean()
                    )
                    if pd.notna(sr) and len(signal_data) > order * 3:
                        btype = (
                            "low" if filter_type == "Butterworth Low-pass" else "high"
                        )
                        b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sr)
                        processed_df[col] = pd.Series(
                            filtfilt(b, a, signal_data), index=signal_data.index
                        )
                elif filter_type == "Median Filter":
                    kernel = settings.get("median_kernel", 5)
                    if kernel % 2 == 0:
                        kernel += 1
                    if len(signal_data) > kernel:
                        processed_df[col] = pd.Series(
                            medfilt(signal_data, kernel_size=kernel),
                            index=signal_data.index,
                        )
                elif filter_type == "Savitzky-Golay":
                    window = settings.get("savgol_window", 11)
                    polyorder = settings.get("savgol_polyorder", 2)
                    if window % 2 == 0:
                        window += 1
                    if polyorder >= window:
                        polyorder = window - 1
                    if len(signal_data) > window:
                        processed_df[col] = pd.Series(
                            savgol_filter(signal_data, window, polyorder),
                            index=signal_data.index,
                        )
                elif filter_type == "Hampel Filter":
                    window = settings.get("hampel_window", 11)
                    if window % 2 == 0:
                        window += 1
                    if len(signal_data) > window:
                        processed_df[col] = pd.Series(
                            medfilt(signal_data, kernel_size=window),
                            index=signal_data.index,
                        )
                elif filter_type == "Z-Score Outlier Removal":
                    threshold = settings.get("zscore_threshold", 3.0)
                    z_scores = np.abs(
                        (signal_data - signal_data.mean()) / signal_data.std()
                    )
                    processed_df[col] = signal_data.mask(z_scores > threshold)

        # Apply Integration
        integration_signals = settings.get("integration_signals", [])
        integration_method = settings.get("integration_method", "Trapezoidal")
        if integration_signals:
            for signal in integration_signals:
                if signal in processed_df.columns:
                    if integration_method == "Trapezoidal":
                        processed_df[f"{signal}_integrated"] = processed_df[
                            signal
                        ].cumsum()
                    elif integration_method == "Simpson":
                        # Simple implementation - could be enhanced
                        processed_df[f"{signal}_integrated"] = processed_df[
                            signal
                        ].cumsum()

        # Apply Differentiation
        differentiation_signals = settings.get("differentiation_signals", [])
        differentiation_method = settings.get(
            "differentiation_method", "Spline (Acausal)"
        )
        if differentiation_signals:
            for signal in differentiation_signals:
                if signal in processed_df.columns:
                    if differentiation_method == "Simple Difference":
                        processed_df[f"{signal}_differentiated"] = processed_df[
                            signal
                        ].diff()
                    elif differentiation_method == "Spline (Acausal)":
                        signal_data = processed_df[signal].dropna()
                        if len(signal_data) > 3:
                            try:
                                spline = UnivariateSpline(
                                    range(len(signal_data)), signal_data, s=0
                                )
                                derivative = spline.derivative()
                                processed_df[f"{signal}_differentiated"] = pd.Series(
                                    derivative(range(len(signal_data))),
                                    index=signal_data.index,
                                )
                            except:
                                processed_df[f"{signal}_differentiated"] = processed_df[
                                    signal
                                ].diff()

        # Apply Custom Variables
        custom_vars = settings.get("custom_variables", {})
        for var_name, expression in custom_vars.items():
            try:
                # Create a safe evaluation environment
                safe_dict = {col: processed_df[col] for col in processed_df.columns}
                safe_dict.update(
                    {
                        "np": np,
                        "pd": pd,
                        "sin": np.sin,
                        "cos": np.cos,
                        "tan": np.tan,
                        "exp": np.exp,
                        "log": np.log,
                        "sqrt": np.sqrt,
                        "abs": np.abs,
                        "max": np.maximum,
                        "min": np.minimum,
                        "mean": np.mean,
                        "std": np.std,
                    }
                )
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                processed_df[var_name] = result
            except Exception as e:
                print(f"Error evaluating custom variable {var_name}: {e}")

        # Apply Resampling
        resample_rule = settings.get("resample_rule")
        if resample_rule and resample_rule != "None":
            try:
                processed_df = processed_df.resample(resample_rule).mean()
            except:
                pass

        # Apply Sorting
        sort_by = settings.get("sort_by")
        if sort_by and sort_by != "None":
            try:
                processed_df = processed_df.sort_values(by=sort_by)
            except:
                pass

        return processed_df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# =============================================================================
# COMPILER CONVERTER CLASSES
# =============================================================================


class SplitMethod(Enum):
    """Methods for splitting large files."""

    ROWS = "rows"
    SIZE = "size"
    TIME = "time"
    CUSTOM = "custom"


@dataclass
class SplitConfig:
    """Configuration for file splitting."""

    method: SplitMethod = SplitMethod.ROWS
    rows_per_file: int = 100000
    max_file_size_mb: float = 100.0
    time_column: str = ""
    time_interval_hours: float = 24.0
    custom_condition: str = ""
    output_directory: str = ""
    filename_pattern: str = "{base_name}_part_{part_number:04d}{extension}"
    compression: str = "snappy"  # For parquet files
    include_header: bool = True  # For CSV files

    def get_file_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return int(self.max_file_size_mb * 1024 * 1024)


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================


class CSVProcessorAppPyQt6(QMainWindow):
    """Main application class for the PyQt6 version of the CSV Processor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced CSV Time Series Processor & Analyzer - PyQt6")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize data storage
        self.data_files = []
        self.processed_data = {}
        self.signal_list = []
        self.selected_signals = []
        self.output_folder = ""

        # Initialize settings
        self.settings = {
            "filter_type": "None",
            "ma_window": 10,
            "bw_order": 3,
            "bw_cutoff": 0.1,
            "median_kernel": 5,
            "savgol_window": 11,
            "savgol_polyorder": 2,
            "hampel_window": 11,
            "zscore_threshold": 3.0,
            "integration_signals": [],
            "integration_method": "Trapezoidal",
            "differentiation_signals": [],
            "differentiation_method": "Spline (Acausal)",
            "custom_variables": {},
            "resample_rule": "None",
            "sort_by": "None",
        }

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create tabs
        self.create_processing_tab()
        self.create_plotting_tab()
        self.create_plots_list_tab()
        self.create_format_converter_tab()
        self.folder_tab = FolderToolTab(self.status_bar)
        self.tab_widget.addTab(self.folder_tab, "Folder Tool")
        self.create_dat_import_tab()

        self.create_help_tab()

        # Create menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Files", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.select_files)
        file_menu.addAction(open_action)

        save_action = QAction("Save Settings", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_settings)
        file_menu.addAction(save_action)

        load_action = QAction("Load Settings", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.load_settings)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_processing_tab(self):
        """Create the main processing tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        file_buttons_layout = QHBoxLayout()
        self.select_files_btn = QPushButton("Select Files")
        self.select_files_btn.clicked.connect(self.select_files)
        file_buttons_layout.addWidget(self.select_files_btn)

        self.select_output_btn = QPushButton("Select Output Folder")
        self.select_output_btn.clicked.connect(self.select_output_folder)
        file_buttons_layout.addWidget(self.select_output_btn)

        file_layout.addLayout(file_buttons_layout)

        # File list
        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)

        layout.addWidget(file_group)

        # Signal selection section
        signal_group = QGroupBox("Signal Selection")
        signal_layout = QVBoxLayout(signal_group)

        self.signal_list_widget = QListWidget()
        self.signal_list_widget.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        signal_layout.addWidget(self.signal_list_widget)

        signal_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_signals)
        signal_buttons_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_signals)
        signal_buttons_layout.addWidget(self.deselect_all_btn)

        signal_layout.addLayout(signal_buttons_layout)
        layout.addWidget(signal_group)

        # Processing options section
        options_group = QGroupBox("Processing Options")
        options_layout = QFormLayout(options_group)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            [
                "None",
                "Moving Average",
                "Butterworth Low-pass",
                "Butterworth High-pass",
                "Median Filter",
                "Savitzky-Golay",
                "Hampel Filter",
                "Z-Score Outlier Removal",
            ]
        )
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        options_layout.addRow("Filter Type:", self.filter_combo)

        layout.addWidget(options_group)

        # Process button
        self.process_btn = QPushButton("Process Files")
        self.process_btn.clicked.connect(self.process_files)
        layout.addWidget(self.process_btn)

        self.tab_widget.addTab(tab, "Processing")

    def create_plotting_tab(self):
        """Create the plotting tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Plot controls
        controls_layout = QHBoxLayout()

        self.plot_signals_combo = QComboBox()
        controls_layout.addWidget(QLabel("Signals:"))
        controls_layout.addWidget(self.plot_signals_combo)

        self.plot_btn = QPushButton("Plot")
        self.plot_btn.clicked.connect(self.create_plot)
        controls_layout.addWidget(self.plot_btn)

        layout.addLayout(controls_layout)

        # Plot area
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, tab)
        layout.addWidget(self.toolbar)

        self.tab_widget.addTab(tab, "Plotting & Analysis")

    def create_plots_list_tab(self):
        """Create the plots list tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Plots list
        self.plots_list = QListWidget()
        layout.addWidget(self.plots_list)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.add_plot_btn = QPushButton("Add Plot")
        self.add_plot_btn.clicked.connect(self.add_plot_to_list)
        buttons_layout.addWidget(self.add_plot_btn)

        self.load_plot_btn = QPushButton("Load Plot")
        self.load_plot_btn.clicked.connect(self.load_plot_from_list)
        buttons_layout.addWidget(self.load_plot_btn)

        self.delete_plot_btn = QPushButton("Delete Plot")
        self.delete_plot_btn.clicked.connect(self.delete_plot_from_list)
        buttons_layout.addWidget(self.delete_plot_btn)

        layout.addLayout(buttons_layout)

        self.tab_widget.addTab(tab, "Plots List")

    def create_format_converter_tab(self):
        """Create the format converter tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection
        file_group = QGroupBox("Files to Convert")
        file_layout = QVBoxLayout(file_group)

        file_buttons_layout = QHBoxLayout()
        self.converter_select_files_btn = QPushButton("Select Files")
        self.converter_select_files_btn.clicked.connect(self.converter_select_files)
        file_buttons_layout.addWidget(self.converter_select_files_btn)

        self.converter_select_folder_btn = QPushButton("Select Folder")
        self.converter_select_folder_btn.clicked.connect(self.converter_select_folder)
        file_buttons_layout.addWidget(self.converter_select_folder_btn)

        file_layout.addLayout(file_buttons_layout)

        self.converter_file_list = QListWidget()
        file_layout.addWidget(self.converter_file_list)

        layout.addWidget(file_group)

        # Conversion options
        options_group = QGroupBox("Conversion Options")
        options_layout = QFormLayout(options_group)

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(
            ["csv", "parquet", "excel", "json", "hdf5", "pickle"]
        )
        options_layout.addRow("Output Format:", self.output_format_combo)

        self.combine_files_check = QCheckBox("Combine all files into one")
        options_layout.addRow(self.combine_files_check)

        layout.addWidget(options_group)

        # Convert button
        self.convert_btn = QPushButton("Convert Files")
        self.convert_btn.clicked.connect(self.start_conversion)
        layout.addWidget(self.convert_btn)

        # Progress bar
        self.conversion_progress = QProgressBar()
        layout.addWidget(self.conversion_progress)

        # Log area
        self.conversion_log = QTextEdit()
        self.conversion_log.setMaximumHeight(150)
        layout.addWidget(self.conversion_log)

        self.tab_widget.addTab(tab, "Format Converter")

    def create_dat_import_tab(self):
        """Create the DAT file import tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection
        file_group = QGroupBox("DAT File Import")
        file_layout = QFormLayout(file_group)

        self.dat_file_edit = QLineEdit()
        self.dat_file_edit.setPlaceholderText("Select DAT file...")
        file_layout.addRow("DAT File:", self.dat_file_edit)

        self.dat_browse_btn = QPushButton("Browse")
        self.dat_browse_btn.clicked.connect(self.select_dat_file)
        file_layout.addRow("", self.dat_browse_btn)

        self.tag_file_edit = QLineEdit()
        self.tag_file_edit.setPlaceholderText("Select tag file (optional)...")
        file_layout.addRow("Tag File:", self.tag_file_edit)

        self.tag_browse_btn = QPushButton("Browse")
        self.tag_browse_btn.clicked.connect(self.select_tag_file)
        file_layout.addRow("", self.tag_browse_btn)

        layout.addWidget(file_group)

        # Import button
        self.import_dat_btn = QPushButton("Import DAT File")
        self.import_dat_btn.clicked.connect(self.import_dat_file)
        layout.addWidget(self.import_dat_btn)

        self.tab_widget.addTab(tab, "DAT File Import")

    def create_help_tab(self):
        """Create the help tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Help text
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml(
            """
        <h1>Advanced CSV Time Series Processor & Analyzer - PyQt6 Version</h1>
        
        <h2>Overview</h2>
        <p>This application provides comprehensive tools for processing, analyzing, and visualizing time series data from CSV files and other formats.</p>
        
        <h2>Features</h2>
        <ul>
            <li><strong>Processing:</strong> Load CSV files, select signals, apply filters, integration, differentiation, and custom variables</li>
            <li><strong>Plotting:</strong> Interactive plotting with matplotlib integration</li>
            <li><strong>Format Converter:</strong> Convert between various file formats (CSV, Parquet, Excel, JSON, etc.)</li>
            <li><strong>Folder Tool:</strong> Process and organize files in folders</li>
            <li><strong>DAT Import:</strong> Import DAT files with optional tag files</li>
        </ul>
        
        <h2>Getting Started</h2>
        <ol>
            <li>Go to the "Processing" tab</li>
            <li>Click "Select Files" to choose your CSV files</li>
            <li>Select the signals you want to process</li>
            <li>Configure processing options (filters, etc.)</li>
            <li>Click "Process Files" to start processing</li>
        </ol>
        
        <h2>File Formats Supported</h2>
        <ul>
            <li>CSV, TSV</li>
            <li>Parquet</li>
            <li>Excel (xlsx, xls)</li>
            <li>JSON</li>
            <li>HDF5</li>
            <li>Pickle</li>
            <li>NumPy arrays</li>
            <li>MATLAB files</li>
            <li>Feather</li>
            <li>Arrow</li>
            <li>SQLite</li>
        </ul>
        """
        )

        layout.addWidget(help_text)

        self.tab_widget.addTab(tab, "Help")

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def select_files(self):
        """Select input files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "All Files (*);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet)",
        )
        if files:
            self.data_files.extend(files)
            self.update_file_list()
            # Load signals from the first file
            if self.data_files and not self.signal_list:
                self.load_signals_from_first_file()

    def select_output_folder(self):
        """Select output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.status_bar.showMessage(f"Output folder: {folder}")

    def update_file_list(self):
        """Update the file list display."""
        self.file_list.clear()
        for file_path in self.data_files:
            self.file_list.addItem(os.path.basename(file_path))

    def load_signals_from_first_file(self):
        """Load signals from the first file in the list."""
        if not self.data_files:
            return

        try:
            file_path = self.data_files[0]
            # Detect file format
            format_type = FileFormatDetector.detect_format(file_path)

            if format_type == "csv":
                df = pd.read_csv(file_path, nrows=1)  # Read just header
            elif format_type == "excel":
                df = pd.read_excel(file_path, nrows=1)
            elif format_type == "parquet":
                df = pd.read_parquet(file_path)
            else:
                # Try CSV as default
                df = pd.read_csv(file_path, nrows=1)

            # Get column names
            self.signal_list = df.columns.tolist()
            self.update_signal_list()

        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Failed to load signals from {file_path}: {e}"
            )

    def update_signal_list(self):
        """Update the signal list widget."""
        self.signal_list_widget.clear()
        for signal in self.signal_list:
            self.signal_list_widget.addItem(signal)

    def select_all_signals(self):
        """Select all signals in the signal list."""
        for i in range(self.signal_list_widget.count()):
            self.signal_list_widget.item(i).setSelected(True)

    def deselect_all_signals(self):
        """Deselect all signals in the signal list."""
        self.signal_list_widget.clearSelection()

    def on_filter_changed(self, filter_type):
        """Handle filter type change."""
        self.settings["filter_type"] = filter_type

    def process_files(self):
        """Process the selected files."""
        if not self.data_files:
            QMessageBox.warning(self, "Warning", "Please select files to process.")
            return

        if not self.output_folder:
            QMessageBox.warning(self, "Warning", "Please select an output folder.")
            return

        # Get selected signals
        selected_items = self.signal_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select signals to process.")
            return

        self.selected_signals = [item.text() for item in selected_items]

        # Start processing in a separate thread
        self.processing_thread = ProcessingThread(
            self.data_files, self.settings, self.output_folder, process_single_csv_file
        )
        self.processing_thread.progress_updated.connect(self.update_processing_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()

    def update_processing_progress(self, value, message):
        """Update processing progress."""
        self.status_bar.showMessage(message)

    def processing_finished(self):
        """Handle processing completion."""
        QMessageBox.information(self, "Complete", "File processing completed!")
        self.status_bar.showMessage("Processing completed")

    def create_plot(self):
        """Create a plot of the selected signals."""
        if not self.data_files:
            QMessageBox.warning(self, "Warning", "Please select files to plot.")
            return

        # Get selected signals
        selected_items = self.signal_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select signals to plot.")
            return

        selected_signals = [item.text() for item in selected_items]

        try:
            # Load data from the first file
            file_path = self.data_files[0]
            format_type = FileFormatDetector.detect_format(file_path)

            if format_type == "csv":
                df = pd.read_csv(file_path)
            elif format_type == "excel":
                df = pd.read_excel(file_path)
            elif format_type == "parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path)

            # Convert time column to datetime
            time_col = df.columns[0]
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df.set_index(time_col, inplace=True)

            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Plot selected signals
            for signal in selected_signals:
                if signal in df.columns:
                    ax.plot(df.index, df[signal], label=signal)

            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_title("Time Series Plot")
            ax.legend()
            ax.grid(True)

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45)

            # Adjust layout and redraw
            self.figure.tight_layout()
            self.canvas.draw()

            self.status_bar.showMessage("Plot created successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create plot: {e}")

    def add_plot_to_list(self):
        """Add current plot to the plots list."""
        # Get current plot configuration
        selected_items = self.signal_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select signals for the plot.")
            return

        selected_signals = [item.text() for item in selected_items]

        # Create plot configuration
        plot_config = {
            "name": f"Plot_{len(self.plots_list) + 1}",
            "signals": selected_signals,
            "filter_type": self.settings.get("filter_type", "None"),
            "created": datetime.now().isoformat(),
        }

        # Add to plots list
        self.plots_list.addItem(plot_config["name"])

        # Store plot configuration (in a real implementation, you'd save this to a file)
        if not hasattr(self, "plot_configs"):
            self.plot_configs = {}
        self.plot_configs[plot_config["name"]] = plot_config

        QMessageBox.information(
            self, "Success", f"Plot '{plot_config['name']}' added to list!"
        )

    def load_plot_from_list(self):
        """Load a plot from the plots list."""
        current_item = self.plots_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a plot from the list.")
            return

        plot_name = current_item.text()
        if not hasattr(self, "plot_configs") or plot_name not in self.plot_configs:
            QMessageBox.warning(self, "Warning", "Plot configuration not found.")
            return

        plot_config = self.plot_configs[plot_name]

        # Select the signals in the signal list
        self.signal_list_widget.clearSelection()
        for i in range(self.signal_list_widget.count()):
            item = self.signal_list_widget.item(i)
            if item.text() in plot_config["signals"]:
                item.setSelected(True)

        # Apply filter settings
        self.settings["filter_type"] = plot_config.get("filter_type", "None")
        self.filter_combo.setCurrentText(self.settings["filter_type"])

        QMessageBox.information(self, "Success", f"Plot '{plot_name}' loaded!")

    def delete_plot_from_list(self):
        """Delete a plot from the plots list."""
        current_item = self.plots_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a plot from the list.")
            return

        plot_name = current_item.text()

        # Remove from plots list
        self.plots_list.takeItem(self.plots_list.row(current_item))

        # Remove from configurations
        if hasattr(self, "plot_configs") and plot_name in self.plot_configs:
            del self.plot_configs[plot_name]

        QMessageBox.information(self, "Success", f"Plot '{plot_name}' deleted!")

    def converter_select_files(self):
        """Select files for conversion."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files for Conversion",
            "",
            "All Files (*);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet)",
        )
        if files:
            # Add to converter file list
            for file_path in files:
                self.converter_file_list.addItem(file_path)

    def converter_select_folder(self):
        """Select folder for conversion."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder for Conversion")
        if folder:
            # Add all files from folder
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.converter_file_list.addItem(file_path)

    def start_conversion(self):
        """Start the file conversion process."""
        if self.converter_file_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Please select files to convert.")
            return

        # Get conversion options
        output_format = self.output_format_combo.currentText()
        combine_files = self.combine_files_check.isChecked()

        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        # Start conversion in a separate thread
        self.conversion_thread = ConversionThread(
            self.get_converter_file_list(), output_format, combine_files, output_dir
        )
        self.conversion_thread.progress_updated.connect(self.update_conversion_progress)
        self.conversion_thread.log_updated.connect(self.update_conversion_log)
        self.conversion_thread.finished.connect(self.conversion_finished)
        self.conversion_thread.start()

    def get_converter_file_list(self):
        """Get list of files from converter file list widget."""
        files = []
        for i in range(self.converter_file_list.count()):
            files.append(self.converter_file_list.item(i).text())
        return files

    def update_conversion_progress(self, value):
        """Update conversion progress bar."""
        self.conversion_progress.setValue(value)

    def update_conversion_log(self, message):
        """Update conversion log."""
        self.conversion_log.append(message)

    def conversion_finished(self):
        """Handle conversion completion."""
        QMessageBox.information(self, "Complete", "File conversion completed!")
        self.status_bar.showMessage("Conversion completed")

    def select_dat_file(self):
        """Select DAT file for import."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DAT File", "", "DAT Files (*.dat);;All Files (*)"
        )
        if file_path:
            self.dat_file_edit.setText(file_path)

    def select_tag_file(self):
        """Select tag file for DAT import."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Tag File", "", "Tag Files (*.tag);;All Files (*)"
        )
        if file_path:
            self.tag_file_edit.setText(file_path)

    def import_dat_file(self):
        """Import the selected DAT file."""
        dat_file = self.dat_file_edit.text()
        tag_file = self.tag_file_edit.text()

        if not dat_file:
            QMessageBox.warning(self, "Warning", "Please select a DAT file.")
            return

        if not os.path.exists(dat_file):
            QMessageBox.warning(self, "Warning", "Selected DAT file does not exist.")
            return

        try:
            # Read DAT file
            data = self.read_dat_file(dat_file)

            if data is None:
                QMessageBox.warning(self, "Warning", "Failed to read DAT file.")
                return

            # Read tag file if provided
            tags = None
            if tag_file and os.path.exists(tag_file):
                tags = self.read_tag_file(tag_file)

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Add column names if tags are available
            if tags:
                df.columns = tags
            else:
                # Use generic column names
                df.columns = [f"Column_{i}" for i in range(len(df.columns))]

            # Save as CSV
            output_file = os.path.splitext(dat_file)[0] + "_imported.csv"
            df.to_csv(output_file, index=False)

            QMessageBox.information(
                self,
                "Success",
                f"DAT file imported successfully!\nSaved as: {output_file}",
            )

            # Add to data files list
            self.data_files.append(output_file)
            self.update_file_list()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import DAT file: {e}")

    def read_dat_file(self, file_path):
        """Read DAT file and return data."""
        try:
            # Try to read as binary file
            with open(file_path, "rb") as f:
                data = f.read()

            # Try to interpret as structured data
            # This is a simplified implementation - real DAT files may have specific formats
            import struct

            # Try different interpretations
            interpretations = []

            # Try as 32-bit floats
            try:
                float_data = struct.unpack(
                    "f" * (len(data) // 4), data[: len(data) - (len(data) % 4)]
                )
                interpretations.append(("32-bit floats", float_data))
            except:
                pass

            # Try as 64-bit floats
            try:
                double_data = struct.unpack(
                    "d" * (len(data) // 8), data[: len(data) - (len(data) % 8)]
                )
                interpretations.append(("64-bit floats", double_data))
            except:
                pass

            # Try as 16-bit integers
            try:
                int16_data = struct.unpack(
                    "h" * (len(data) // 2), data[: len(data) - (len(data) % 2)]
                )
                interpretations.append(("16-bit integers", int16_data))
            except:
                pass

            # Try as 32-bit integers
            try:
                int32_data = struct.unpack(
                    "i" * (len(data) // 4), data[: len(data) - (len(data) % 4)]
                )
                interpretations.append(("32-bit integers", int32_data))
            except:
                pass

            if interpretations:
                # Use the first successful interpretation
                return interpretations[0][1]
            else:
                # Fallback: treat as text
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    data = []
                    for line in lines:
                        values = line.strip().split()
                        if values:
                            try:
                                data.append([float(v) for v in values])
                            except:
                                pass
                    return data

        except Exception as e:
            print(f"Error reading DAT file: {e}")
            return None

    def read_tag_file(self, file_path):
        """Read tag file and return column names."""
        try:
            with open(file_path, "r") as f:
                tags = [line.strip() for line in f.readlines() if line.strip()]
            return tags
        except Exception as e:
            print(f"Error reading tag file: {e}")
            return None

    def save_settings(self):
        """Save current settings to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(self.settings, f, indent=2)
                QMessageBox.information(self, "Success", "Settings saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")

    def load_settings(self):
        """Load settings from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    self.settings = json.load(f)
                QMessageBox.information(
                    self, "Success", "Settings loaded successfully!"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load settings: {e}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About",
            "Advanced CSV Time Series Processor & Analyzer\n"
            "PyQt6 Version\n\n"
            "A comprehensive tool for processing and analyzing time series data.",
        )


# =============================================================================
# WORKER THREADS
# =============================================================================


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show the main window
    window = CSVProcessorAppPyQt6()
    window.show()

    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

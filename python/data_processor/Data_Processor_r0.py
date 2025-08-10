# =============================================================================
# Advanced CSV Time Series Processor & Analyzer - Complete Version
#
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This version combines all advanced features
# from Rev2 with the UI fixes from Rev4_Claude, ensuring complete functionality.
#
# Dependencies for Python 3.8+:
# pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow
# simpledbf pyarrow tables feather-format
#
# =============================================================================

import io  # noqa: F401 (kept for possible runtime use)
import json
import os
import re  # noqa: F401
import tkinter as tk
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: F401
from tkinter import colorchooser, filedialog, messagebox, simpledialog
from typing import Any

import customtkinter as ctk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image  # noqa: F401
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import linregress  # noqa: F401
from simpledbf import Dbf5  # noqa: F401

# Optional Savitzky-Golay import with guard
try:
    from scipy.signal import savgol_filter as _savgol_filter
except Exception:  # pragma: no cover - optional dependency
    _savgol_filter = None

# Import constants
from .constants import (
    DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT, DEFAULT_PADDING, DEFAULT_BUTTON_HEIGHT,
    DEFAULT_TEXT_HEIGHT, DEFAULT_SEARCH_WIDTH, GRID_WEIGHT_MAIN,
    MIN_SIGNAL_DATA_POINTS, MIN_PERIODS_DEFAULT, DEFAULT_MA_WINDOW,
    DEFAULT_BW_ORDER, DEFAULT_BW_CUTOFF, DEFAULT_BW_NYQUIST,
    DEFAULT_MEDIAN_KERNEL, MIN_KERNEL_SIZE, DEFAULT_SAVGOL_WINDOW,
    DEFAULT_SAVGOL_POLYORDER, MAX_DERIVATIVE_ORDER,
    TIME_COLUMN_KEYWORDS, LARGE_SIGNAL_THRESHOLD, SIGNAL_BATCH_SIZE,
    BULK_SAMPLE_SIZE, LARGE_FILE_THRESHOLD, PLOT_UPDATE_DELAY_MS,
    ZOOM_OUT_FACTOR, ZOOM_IN_FACTOR, DEFAULT_LINE_WIDTH, DEFAULT_GRID_ALPHA,
    DEFAULT_GRID_LINESTYLE, ERROR_MSG_NO_FILES, ERROR_MSG_EMPTY_FILE,
    ERROR_MSG_NO_PLOTS, DEFAULT_PLOT_TITLE, DEFAULT_PLOT_XLABEL,
    DEFAULT_PLOT_YLABEL, DEFAULT_LEGEND_POSITION, DEFAULT_TIME_FORMAT
)


# =============================================================================
# WORKER FUNCTION FOR PARALLEL PROCESSING
# =============================================================================
def process_single_csv_file(
    file_path: str,
    settings: dict[str, Any],
) -> pd.DataFrame | None:
    """
    Processes a single CSV file based on a dictionary of settings.
    This function is designed to be run in a separate process.

    Args:
        file_path: Path to the CSV file to process
        settings: Dictionary containing processing settings

    Returns:
        Processed DataFrame or None if processing failed
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
                include=np.number,
            ).columns.tolist()
            for col in numeric_cols:
                signal_data = processed_df[col].dropna()
                if len(signal_data) < MIN_SIGNAL_DATA_POINTS:
                    continue

                # Apply filtering based on type
                if filter_type == "Moving Average":
                    window_size = settings.get("ma_window", 10)
                    processed_df[col] = signal_data.rolling(
                        window=window_size,
                        min_periods=1,
                    ).mean()
                elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
                    order = settings.get("bw_order", DEFAULT_BW_ORDER)
                    cutoff = settings.get("bw_cutoff", DEFAULT_BW_CUTOFF)
                    sr = (
                        1.0
                        / pd.to_numeric(
                            signal_data.index.to_series().diff().dt.total_seconds(),
                        ).mean()
                    )
                    if pd.notna(sr) and len(signal_data) > order * MIN_BUTTERWORTH_DATA_MULTIPLIER:
                        btype = (
                            "low" if filter_type == "Butterworth Low-pass" else "high"
                        )
                        b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sr)
                        processed_df[col] = pd.Series(
                            filtfilt(b, a, signal_data),
                            index=signal_data.index,
                        )
                elif filter_type == "Median Filter":
                    kernel = settings.get("median_kernel", DEFAULT_MEDIAN_KERNEL)
                    if kernel % 2 == 0:
                        kernel += 1
                    if len(signal_data) > kernel:
                        processed_df[col] = pd.Series(
                            medfilt(signal_data, kernel_size=kernel),
                            index=signal_data.index,
                        )
                elif filter_type == "Savitzky-Golay":
                    window = settings.get("savgol_window", DEFAULT_SAVGOL_WINDOW)
                    polyorder = settings.get("savgol_polyorder", DEFAULT_SAVGOL_POLYORDER)
                    if window % 2 == 0:
                        window += 1
                    if polyorder >= window:
                        polyorder = window - 1
                    if len(signal_data) > window:
                        if _savgol_filter is None:
                            raise RuntimeError(
                                "scipy.signal.savgol_filter unavailable. "
                                "Install SciPy or skip smoothing.",
                            )
                        processed_df[col] = pd.Series(
                            _savgol_filter(signal_data, window, polyorder),
                            index=signal_data.index,
                        )

        # Apply Resampling
        if settings.get("resample_enabled"):
            resample_rule = settings.get("resample_rule")
            if resample_rule:
                processed_df = (
                    processed_df.resample(resample_rule).mean().dropna(how="all")
                )

        if processed_df.empty:
            return None

        processed_df.reset_index(inplace=True)
        return processed_df
    except Exception as e:
        print(f"Error processing {file_path}: {e!s}")
        return None


# Helper function for causal derivative calculation
def _poly_derivative(
    series: pd.Series,
    window: int,
    poly_order: int,
    deriv_order: int,
    delta_x: float,
) -> pd.Series:
    """Calculates the derivative of a series using a rolling polynomial fit."""
    if poly_order < deriv_order:
        return pd.Series(np.nan, index=series.index)

    # Pad the series at the beginning to get derivatives for the initial points
    padded_series = pd.concat([pd.Series([series.iloc[0]] * (window - 1)), series])

    def get_deriv(w: pd.Series) -> float:
        """
        Calculate derivative for a window of data.

        Args:
            w: Series containing the data window

        Returns:
            Calculated derivative value or NaN if calculation fails
        """
        # Can't compute if the window is not full or has NaNs
        if len(w) < window or np.isnan(w).any():
            return np.nan
        x = np.arange(len(w)) * delta_x
        try:
            # Fit polynomial to the window's data
            coeffs = np.polyfit(x, w, poly_order)
            # Get the derivative of the polynomial
            deriv_coeffs = np.polyder(coeffs, deriv_order)
            # Evaluate the derivative at the last point of the window
            return np.polyval(deriv_coeffs, x[-1])
        except (np.linalg.LinAlgError, TypeError):
            # Handle cases where the fit fails
            return np.nan

    return (
        padded_series.rolling(window=window)
        .apply(get_deriv, raw=True)
        .iloc[window - 1 :]
    )


class CSVProcessorApp(ctk.CTk):
    """The main application class with all advanced features and UI fixes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the CSV Processor application with all UI components and state variables."""
        super().__init__(*args, **kwargs)

        # Layout persistence variables
        self.layout_config_file = os.path.join(
            os.path.expanduser("~"),
            ".csv_processor_layout.json",
        )
        self.splitters = {}
        self.layout_data = self._load_layout_config()

        self.title("Advanced CSV Processor & DAT Importer - Complete Version")

        # Force reasonable window size (ignore saved layout for size)
        window_width = DEFAULT_WINDOW_WIDTH
        window_height = DEFAULT_WINDOW_HEIGHT
        self.geometry(f"{window_width}x{window_height}")

        # Center the window on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.winfo_screenheight() // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Set up closing handler
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Set up window resize handler to save layout
        self.bind("<Configure>", self._on_window_configure)

        # App State Variables
        self.input_file_paths = []
        self.loaded_data_cache = {}
        self.processed_files = {}  # Store processed data for plotting
        self.output_directory = os.path.expanduser("~/Documents")
        self.signal_vars = {}
        self.plot_signal_vars = {}
        self.filter_names = [
            "None",
            "Moving Average",
            "Median Filter",
            "Hampel Filter",
            "Z-Score Filter",
            "Butterworth Low-pass",
            "Butterworth High-pass",
            "Savitzky-Golay",
        ]
        self.custom_vars_list = []
        self.reference_signal_widgets = {}
        self.dat_import_tag_file_path = None
        self.dat_import_data_file_path = None
        self.dat_tag_vars = {}
        self.tag_delimiter_var = tk.StringVar(value="newline")

        # Plots List variables
        self.plots_list = []
        self.current_plot_config = None

        # Signal List Management variables
        self.saved_signal_list = []
        self.saved_signal_list_name = ""

        # Integration and Differentiation variables
        self.integrator_signal_vars = {}
        self.deriv_signal_vars = {}
        self.derivative_vars = {}
        for i in range(1, MAX_DERIVATIVE_ORDER + 1):  # Support up to 5th order derivatives
            self.derivative_vars[i] = tk.BooleanVar(value=False)

        # Plot view state management
        self.saved_plot_view = None

        # Custom legend entries for plots
        self.custom_legend_entries = {}

        # Custom colors for plots
        self.custom_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Create Main UI
        self.main_tab_view = ctk.CTkTabview(self)
        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.main_tab_view.add("Processing")
        self.main_tab_view.add("Plotting & Analysis")
        self.main_tab_view.add("Plots List")
        self.main_tab_view.add("DAT File Import")
        self.main_tab_view.add("Help")

        self.create_setup_and_process_tab(self.main_tab_view.tab("Processing"))
        self.create_plotting_tab(self.main_tab_view.tab("Plotting & Analysis"))
        self.create_plots_list_tab(self.main_tab_view.tab("Plots List"))
        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))
        self.create_help_tab(self.main_tab_view.tab("Help"))

        self.create_status_bar()
        self.status_label.configure(
            text="Ready. Select input files or import a DAT file.",
        )

        # Load saved plots and other settings
        self._load_plots_from_file()

    def create_setup_and_process_tab(self, parent_tab: ctk.CTkFrame) -> None:
        """Fixed version with proper splitter implementation and all advanced features."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)

        def create_left_content(left_panel: ctk.CTkFrame) -> None:
            """Create the left panel content"""
            left_panel.grid_rowconfigure(0, weight=1)
            left_panel.grid_columnconfigure(0, weight=1)

            # Create a scrollable frame for the processing tab view
            processing_scrollable_frame = ctk.CTkScrollableFrame(left_panel)
            processing_scrollable_frame.grid(
                row=0,
                column=0,
                padx=10,
                pady=10,
                sticky="nsew",
            )
            processing_scrollable_frame.grid_columnconfigure(0, weight=1)

            processing_tab_view = ctk.CTkTabview(processing_scrollable_frame)
            processing_tab_view.grid(row=0, column=0, sticky="nsew")
            processing_tab_view.grid_columnconfigure(0, weight=1)
            processing_tab_view.grid_rowconfigure(0, weight=1)
            processing_tab_view.add("Setup")
            processing_tab_view.add("Processing")
            processing_tab_view.add("Custom Vars")

            # Configure individual tabs to expand properly
            setup_tab = processing_tab_view.tab("Setup")
            setup_tab.grid_columnconfigure(0, weight=1)
            setup_tab.grid_rowconfigure(0, weight=1)

            processing_tab = processing_tab_view.tab("Processing")
            processing_tab.grid_columnconfigure(0, weight=1)
            processing_tab.grid_rowconfigure(0, weight=1)

            custom_vars_tab = processing_tab_view.tab("Custom Vars")
            custom_vars_tab.grid_columnconfigure(0, weight=1)
            custom_vars_tab.grid_rowconfigure(0, weight=1)

            self.populate_setup_sub_tab(setup_tab)
            self.populate_processing_sub_tab(processing_tab)
            self.populate_custom_var_sub_tab(custom_vars_tab)

            self.process_button = ctk.CTkButton(
                left_panel,
                text="Process & Batch Export Files",
                height=40,
                command=self.process_files,
            )
            self.process_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        def create_right_content(right_panel: ctk.CTkFrame) -> None:
            """Create the right panel content"""
            right_panel.grid_rowconfigure(2, weight=1)
            right_panel.grid_columnconfigure(0, weight=1)

            # File list frame
            self.file_list_frame = ctk.CTkScrollableFrame(
                right_panel,
                label_text="Selected Input Files",
                height=120,
            )
            self.file_list_frame.grid(
                row=0,
                column=0,
                padx=10,
                pady=(0, 10),
                sticky="new",
            )
            self.initial_file_label = ctk.CTkLabel(
                self.file_list_frame,
                text="Files you select will be listed here.",
            )
            self.initial_file_label.pack(padx=5, pady=5)

            # Signal control frame
            signal_control_frame = ctk.CTkFrame(right_panel)
            signal_control_frame.grid(row=1, column=0, padx=10, pady=0, sticky="ew")
            signal_control_frame.grid_columnconfigure(0, weight=1)

            self.search_entry = ctk.CTkEntry(
                signal_control_frame,
                placeholder_text="Search for signals...",
            )
            self.search_entry.grid(row=0, column=0, padx=(0, 5), sticky="ew")
            self.search_entry.bind("<KeyRelease>", self._filter_signals)
            self.clear_search_button = ctk.CTkButton(
                signal_control_frame,
                text="X",
                width=28,
                command=self._clear_search,
            )
            self.clear_search_button.grid(row=0, column=1, padx=5)
            ctk.CTkButton(
                signal_control_frame,
                text="Select All",
                width=100,
                command=self.select_all,
            ).grid(row=0, column=2, padx=5)
            ctk.CTkButton(
                signal_control_frame,
                text="Deselect All",
                width=100,
                command=self.deselect_all,
            ).grid(row=0, column=3)

            # Signal list frame
            self.signal_list_frame = ctk.CTkScrollableFrame(
                right_panel,
                label_text="Available Signals to Process",
            )
            self.signal_list_frame.grid(
                row=2,
                column=0,
                padx=10,
                pady=(5, 10),
                sticky="nsew",
            )
            self.signal_list_frame.grid_columnconfigure(0, weight=1)

        # Create the splitter with the content creator functions
        splitter_frame = self._create_splitter(
            parent_tab,
            create_left_content,
            create_right_content,
            "setup_left_width",
            350,
        )
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def populate_setup_sub_tab(self, tab: ctk.CTkFrame) -> None:
        """Populate the setup sub-tab."""
        tab.grid_columnconfigure(0, weight=1)

        # File selection frame
        file_frame = ctk.CTkFrame(tab)
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        file_frame.grid_columnconfigure(0, weight=1)

        # Header with help button
        file_header_frame = ctk.CTkFrame(file_frame)
        file_header_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        file_header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            file_header_frame,
            text="Input File Selection",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
        ctk.CTkButton(
            file_header_frame,
            text="?",
            width=25,
            command=self._show_input_file_help,
            fg_color="gray",
            hover_color="darkgray",
        ).grid(row=0, column=1, padx=(0, 0), pady=5, sticky="e")

        # File selection buttons in a horizontal frame
        file_buttons_frame = ctk.CTkFrame(file_frame)
        file_buttons_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        file_buttons_frame.grid_columnconfigure(0, weight=1)
        file_buttons_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            file_buttons_frame,
            text="Select Input Files",
            command=self.select_files,
        ).grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        ctk.CTkButton(
            file_buttons_frame,
            text="Clear All Files",
            command=self._clear_all_files,
        ).grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")

        ctk.CTkButton(
            file_frame,
            text="Select Output Folder",
            command=self.select_output_folder,
        ).grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        # Bulk processing mode toggle
        bulk_frame = ctk.CTkFrame(file_frame)
        bulk_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        bulk_frame.grid_columnconfigure(0, weight=1)

        # Create a horizontal frame for checkbox
        bulk_content_frame = ctk.CTkFrame(bulk_frame)
        bulk_content_frame.pack(fill="x", padx=5, pady=5)
        bulk_content_frame.grid_columnconfigure(0, weight=1)

        self.bulk_mode_var = ctk.BooleanVar(value=False)  # Default to False per request
        bulk_checkbox = ctk.CTkCheckBox(
            bulk_content_frame,
            text="Bulk Processing Mode",
            variable=self.bulk_mode_var,
            command=self._on_bulk_mode_change,
        )
        bulk_checkbox.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")

        # First file only option will be moved to Signal List Management section
        self.first_file_only_var = ctk.BooleanVar(value=False)

        self.output_label = ctk.CTkLabel(
            file_frame,
            text=f"Output: {self.output_directory}",
            wraplength=300,
            justify="left",
            font=ctk.CTkFont(size=11),
        )
        self.output_label.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="w")

        # Signal List Management frame - MOVED TO TOP
        signal_list_frame = ctk.CTkFrame(tab)
        signal_list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        signal_list_frame.grid_columnconfigure(0, weight=1)
        signal_list_frame.grid_columnconfigure(1, weight=1)
        signal_list_frame.grid_columnconfigure(2, weight=1)

        # Header with help button
        header_frame = ctk.CTkFrame(signal_list_frame)
        header_frame.grid(
            row=0,
            column=0,
            columnspan=3,
            padx=10,
            pady=(10, 5),
            sticky="ew",
        )
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame,
            text="Signal List Management",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
        ctk.CTkButton(
            header_frame,
            text="?",
            width=25,
            command=self._show_signal_list_help,
            fg_color="gray",
            hover_color="darkgray",
        ).grid(row=0, column=1, padx=(0, 0), pady=5, sticky="e")

        # Buttons for signal list management - Row 1
        ctk.CTkButton(
            signal_list_frame,
            text="Save Signal List",
            command=self.save_signal_list,
        ).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            signal_list_frame,
            text="Load Signal List",
            command=self.load_signal_list,
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            signal_list_frame,
            text="Load from Files",
            command=self._manual_load_signals,
        ).grid(row=1, column=2, padx=10, pady=5, sticky="ew")

        # Buttons for signal list management - Row 2
        ctk.CTkButton(
            signal_list_frame,
            text="Create Signal List",
            command=self._create_signal_list,
        ).grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            signal_list_frame,
            text="Apply Signals",
            command=self.apply_saved_signals,
        ).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            signal_list_frame,
            text="Load from First File",
            command=self._load_signals_from_first_file,
        ).grid(row=2, column=2, padx=10, pady=5, sticky="ew")

        # Status label for signal list operations - Row 3
        self.signal_list_status_label = ctk.CTkLabel(
            signal_list_frame,
            text="No saved signal list loaded",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        )
        self.signal_list_status_label.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=10,
            pady=(5, 10),
            sticky="w",
        )

        # Settings frame
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(row=2, column=0, padx=10, pady=10, sticky="new")
        settings_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            settings_frame,
            text="Configuration Save and Load",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkButton(
            settings_frame,
            text="Save Settings",
            command=self.save_settings,
        ).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            settings_frame,
            text="Load Settings",
            command=self.load_settings,
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            settings_frame,
            text="Manage Configurations",
            command=self.manage_configurations,
        ).grid(row=1, column=2, padx=10, pady=5, sticky="ew")

        # Custom dataset name frame - MOVED BELOW CONFIGURATION
        dataset_frame = ctk.CTkFrame(tab)
        dataset_frame.grid(row=3, column=0, padx=10, pady=10, sticky="new")
        dataset_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            dataset_frame,
            text="Dataset Naming",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        # Default naming option
        self.dataset_naming_var = ctk.StringVar(value="auto")
        auto_radio = ctk.CTkRadioButton(
            dataset_frame,
            text="Auto-generate from file names",
            variable=self.dataset_naming_var,
            value="auto",
            command=self._on_dataset_naming_change,
        )
        auto_radio.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Custom naming option
        custom_radio = ctk.CTkRadioButton(
            dataset_frame,
            text="Custom dataset name:",
            variable=self.dataset_naming_var,
            value="custom",
            command=self._on_dataset_naming_change,
        )
        custom_radio.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.custom_dataset_entry = ctk.CTkEntry(
            dataset_frame,
            placeholder_text="Enter custom dataset name",
        )
        self.custom_dataset_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.custom_dataset_entry.configure(state="disabled")  # Initially disabled

        # Warning label for file overwriting
        self.overwrite_warning_label = ctk.CTkLabel(
            dataset_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="orange",
        )
        self.overwrite_warning_label.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=10,
            pady=5,
            sticky="w",
        )

        # Export options frame
        export_frame = ctk.CTkFrame(tab)
        export_frame.grid(row=4, column=0, padx=10, pady=10, sticky="new")
        export_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            export_frame,
            text="Export Options",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(export_frame, text="Format:").grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )

        self.export_type_var = ctk.StringVar(value="CSV (Separate Files)")
        ctk.CTkOptionMenu(
            export_frame,
            variable=self.export_type_var,
            values=[
                "CSV (Separate Files)",
                "CSV (Compiled)",
                "Excel (Multi-sheet)",
                "Excel (Separate Files)",
                "MAT (Separate Files)",
                "MAT (Compiled)",
                "Parquet (Single File)",
                "Parquet (Separate Files)",
                "HDF5 (Single File)",
                "HDF5 (Separate Files)",
                "Feather (Single File)",
                "Feather (Separate Files)",
                "Pickle (Single File)",
                "Pickle (Separate Files)",
            ],
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(export_frame, text="Sort By:").grid(
            row=2,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        self.sort_col_menu = ctk.CTkOptionMenu(export_frame, values=["No Sorting"])
        self.sort_col_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        self.sort_order_var = ctk.StringVar(value="Ascending")
        sort_asc = ctk.CTkRadioButton(
            export_frame,
            text="Ascending",
            variable=self.sort_order_var,
            value="Ascending",
        )
        sort_asc.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        sort_desc = ctk.CTkRadioButton(
            export_frame,
            text="Descending",
            variable=self.sort_order_var,
            value="Descending",
        )
        sort_desc.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    def populate_processing_sub_tab(self, tab: ctk.CTkFrame) -> None:
        """Populate the processing sub-tab with all advanced features."""
        tab.grid_columnconfigure(0, weight=1)
        time_units = ["ms", "s", "min", "hr"]

        # Time trimming frame - moved to top for better workflow
        trim_frame = ctk.CTkFrame(tab)
        trim_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="new")
        trim_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            trim_frame,
            text="Time Trimming",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(
            trim_frame,
            text="Trim data to specific time range before processing",
            justify="left",
        ).grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")

        ctk.CTkLabel(trim_frame, text="Date (YYYY-MM-DD):").grid(
            row=2,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        self.trim_date_entry = ctk.CTkEntry(
            trim_frame,
            placeholder_text="e.g., 2024-01-15",
        )
        self.trim_date_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(trim_frame, text="Start Time (HH:MM:SS):").grid(
            row=3,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        self.trim_start_entry = ctk.CTkEntry(
            trim_frame,
            placeholder_text="e.g., 09:30:00",
        )
        self.trim_start_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(trim_frame, text="End Time (HH:MM:SS):").grid(
            row=4,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        self.trim_end_entry = ctk.CTkEntry(
            trim_frame,
            placeholder_text="e.g., 17:00:00",
        )
        self.trim_end_entry.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkButton(
            trim_frame,
            text="Copy Times to Plot Range",
            command=self._copy_trim_to_plot_range,
        ).grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            trim_frame,
            text="Copy Plot Range to Times",
            command=self._copy_plot_range_to_trim,
        ).grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        # Filter frame
        filter_frame = ctk.CTkFrame(tab)
        filter_frame.grid(row=1, column=0, padx=10, pady=(5, 5), sticky="new")
        filter_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            filter_frame,
            text="Signal Filtering",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(filter_frame, text="Filter Type:").grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )

        self.filter_type_var = ctk.StringVar(value="None")
        self.filter_menu = ctk.CTkOptionMenu(
            filter_frame,
            variable=self.filter_type_var,
            values=self.filter_names,
            command=self._update_filter_ui,
        )
        self.filter_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Create filter parameter frames
        (self.ma_frame, self.ma_value_entry, self.ma_unit_menu) = (
            self._create_ma_param_frame(filter_frame, time_units)
        )
        (self.bw_frame, self.bw_order_entry, self.bw_cutoff_entry) = (
            self._create_bw_param_frame(filter_frame)
        )
        (self.median_frame, self.median_kernel_entry) = self._create_median_param_frame(
            filter_frame,
        )
        (self.hampel_frame, self.hampel_window_entry, self.hampel_threshold_entry) = (
            self._create_hampel_param_frame(filter_frame)
        )
        (self.zscore_frame, self.zscore_threshold_entry, self.zscore_method_menu) = (
            self._create_zscore_param_frame(filter_frame)
        )
        (self.savgol_frame, self.savgol_window_entry, self.savgol_polyorder_entry) = (
            self._create_savgol_param_frame(filter_frame)
        )
        self._update_filter_ui("None")

        # Resample frame
        resample_frame = ctk.CTkFrame(tab)
        resample_frame.grid(row=2, column=0, padx=10, pady=(5, 5), sticky="new")
        resample_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            resample_frame,
            text="Time Resampling",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        self.resample_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            resample_frame,
            text="Enable Resampling",
            variable=self.resample_var,
        ).grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(resample_frame, text="Time Gap:").grid(
            row=2,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )

        resample_time_frame = ctk.CTkFrame(resample_frame, fg_color="transparent")
        resample_time_frame.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        resample_time_frame.grid_columnconfigure(0, weight=2)
        resample_time_frame.grid_columnconfigure(1, weight=1)

        self.resample_value_entry = ctk.CTkEntry(
            resample_time_frame,
            placeholder_text="e.g., 10",
        )
        self.resample_value_entry.grid(row=0, column=0, sticky="ew")

        self.resample_unit_menu = ctk.CTkOptionMenu(
            resample_time_frame,
            values=time_units,
        )
        self.resample_unit_menu.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        # Integration frame
        integrator_frame = ctk.CTkFrame(tab)
        integrator_frame.grid(row=3, column=0, padx=10, pady=(5, 5), sticky="new")
        integrator_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            integrator_frame,
            text="Signal Integration",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(5, 3), sticky="w")
        ctk.CTkLabel(
            integrator_frame,
            text="Create cumulative columns for flow calculations",
            justify="left",
        ).grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")

        ctk.CTkLabel(integrator_frame, text="Integration Method:").grid(
            row=2,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        self.integrator_method_var = ctk.StringVar(value="Trapezoidal")
        ctk.CTkOptionMenu(
            integrator_frame,
            variable=self.integrator_method_var,
            values=["Trapezoidal", "Rectangular", "Simpson"],
        ).grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Integration signals selection frame
        integrator_signals_frame = ctk.CTkFrame(integrator_frame)
        integrator_signals_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=10,
            pady=5,
            sticky="ew",
        )
        integrator_signals_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            integrator_signals_frame,
            text="Signals to Integrate:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        self.integrator_search_entry = ctk.CTkEntry(
            integrator_signals_frame,
            placeholder_text="Search signals to integrate...",
        )
        self.integrator_search_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.integrator_search_entry.bind(
            "<KeyRelease>",
            self._filter_integrator_signals,
        )

        ctk.CTkButton(
            integrator_signals_frame,
            text="X",
            width=28,
            command=self._clear_integrator_search,
        ).grid(row=1, column=1, padx=5, pady=5)

        self.integrator_signals_frame = ctk.CTkScrollableFrame(
            integrator_signals_frame,
            height=100,
        )
        self.integrator_signals_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="ew",
        )

        integrator_buttons_frame = ctk.CTkFrame(
            integrator_frame,
            fg_color="transparent",
        )
        integrator_buttons_frame.grid(
            row=4,
            column=0,
            columnspan=2,
            padx=10,
            pady=5,
            sticky="ew",
        )

        ctk.CTkButton(
            integrator_buttons_frame,
            text="Select All",
            command=self._integrator_select_all,
        ).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkButton(
            integrator_buttons_frame,
            text="Deselect All",
            command=self._integrator_deselect_all,
        ).grid(row=0, column=1, padx=5, pady=5)

        # Differentiation Frame with searchable signals
        deriv_frame = ctk.CTkFrame(tab)
        deriv_frame.grid(row=4, column=0, padx=10, pady=10, sticky="new")
        deriv_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            deriv_frame,
            text="Signal Differentiation",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(5, 3), sticky="w")
        ctk.CTkLabel(
            deriv_frame,
            text="Create derivative columns for signal analysis",
            justify="left",
        ).grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")

        # Differentiation method selection
        ctk.CTkLabel(deriv_frame, text="Method:").grid(
            row=2,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        self.deriv_method_var = ctk.StringVar(value="Spline (Acausal)")
        ctk.CTkOptionMenu(
            deriv_frame,
            variable=self.deriv_method_var,
            values=["Spline (Acausal)", "Rolling Polynomial (Causal)"],
        ).grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Differentiation signals selection frame
        deriv_signals_frame = ctk.CTkFrame(deriv_frame)
        deriv_signals_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=10,
            pady=5,
            sticky="ew",
        )
        deriv_signals_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            deriv_signals_frame,
            text="Signals to Differentiate:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Search bar for differentiation signals
        self.deriv_search_entry = ctk.CTkEntry(
            deriv_signals_frame,
            placeholder_text="Search signals to differentiate...",
        )
        self.deriv_search_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.deriv_search_entry.bind("<KeyRelease>", self._filter_deriv_signals)

        ctk.CTkButton(
            deriv_signals_frame,
            text="X",
            width=28,
            command=self._clear_deriv_search,
        ).grid(row=1, column=1, padx=5, pady=5)

        # Scrollable frame for differentiation signal checkboxes
        self.deriv_signals_frame = ctk.CTkScrollableFrame(
            deriv_signals_frame,
            height=100,
        )
        self.deriv_signals_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=5,
            pady=5,
            sticky="ew",
        )
        self.deriv_signals_frame.grid_columnconfigure(0, weight=1)

        # Differentiation control buttons
        deriv_buttons_frame = ctk.CTkFrame(deriv_frame, fg_color="transparent")
        deriv_buttons_frame.grid(
            row=4,
            column=0,
            columnspan=2,
            padx=10,
            pady=5,
            sticky="ew",
        )

        ctk.CTkButton(
            deriv_buttons_frame,
            text="Select All",
            command=self._deriv_select_all,
        ).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkButton(
            deriv_buttons_frame,
            text="Deselect All",
            command=self._deriv_deselect_all,
        ).grid(row=0, column=1, padx=5, pady=5)

        # Derivative order selection (up to 5th order)
        deriv_order_frame = ctk.CTkFrame(deriv_frame)
        deriv_order_frame.grid(
            row=5,
            column=0,
            columnspan=2,
            padx=10,
            pady=5,
            sticky="ew",
        )
        ctk.CTkLabel(
            deriv_order_frame,
            text="Derivative Orders:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=5, padx=10, pady=5, sticky="w")

        for i in range(1, MAX_DERIVATIVE_ORDER + 1):  # Support up to 5th order
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(deriv_order_frame, text=f"Order {i}", variable=var)
            cb.grid(row=1, column=i - 1, padx=10, pady=2, sticky="w")
            self.derivative_vars[i] = var

    def _create_ma_param_frame(
        self,
        parent: ctk.CTkFrame,
        time_units: str,
    ) -> ctk.CTkFrame:
        """Create Moving Average parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Window Size:").grid(
            row=0,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        value_entry = ctk.CTkEntry(frame, placeholder_text="10")
        value_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        value_entry.insert(0, "10")  # Set default value

        ctk.CTkLabel(frame, text="Unit:").grid(
            row=0,
            column=2,
            padx=10,
            pady=5,
            sticky="w",
        )
        unit_menu = ctk.CTkOptionMenu(frame, values=time_units)
        unit_menu.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
        unit_menu.set("s")  # Set default unit

        return frame, value_entry, unit_menu

    def _create_bw_param_frame(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Create Butterworth filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Order:").grid(
            row=0,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        order_entry = ctk.CTkEntry(frame, placeholder_text="3")
        order_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(frame, text="Cutoff:").grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        cutoff_entry = ctk.CTkEntry(frame, placeholder_text="0.1")
        cutoff_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        return frame, order_entry, cutoff_entry

    def _create_median_param_frame(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Create Median filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Kernel Size:").grid(
            row=0,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        kernel_entry = ctk.CTkEntry(frame, placeholder_text="5")
        kernel_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        return frame, kernel_entry

    def _create_savgol_param_frame(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Create Savitzky-Golay filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Window Size:").grid(
            row=0,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        window_entry = ctk.CTkEntry(frame, placeholder_text="11")
        window_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(frame, text="Polynomial Order:").grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        polyorder_entry = ctk.CTkEntry(frame, placeholder_text="2")
        polyorder_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        return frame, window_entry, polyorder_entry

    def _create_hampel_param_frame(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Create Hampel filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Window Size:").grid(
            row=0,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        window_entry = ctk.CTkEntry(frame, placeholder_text="7")
        window_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(frame, text="Threshold (σ):").grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        threshold_entry = ctk.CTkEntry(frame, placeholder_text="3.0")
        threshold_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        return frame, window_entry, threshold_entry

    def _create_zscore_param_frame(self, parent: ctk.CTkFrame) -> ctk.CTkFrame:
        """Create Z-Score filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Threshold (σ):").grid(
            row=0,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        threshold_entry = ctk.CTkEntry(frame, placeholder_text="3.0")
        threshold_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(frame, text="Method:").grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="w",
        )
        method_menu = ctk.CTkOptionMenu(
            frame,
            values=["Remove Outliers", "Clip Outliers", "Replace with Median"],
        )
        method_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        return frame, threshold_entry, method_menu

    def _update_filter_ui(self, filter_type: str) -> None:
        """Update filter UI based on selected filter type."""
        # Hide all frames
        for frame in [
            self.ma_frame,
            self.bw_frame,
            self.median_frame,
            self.hampel_frame,
            self.zscore_frame,
            self.savgol_frame,
        ]:
            frame.grid_remove()

        # Show relevant frame
        if filter_type == "Moving Average":
            self.ma_frame.grid()
        elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.bw_frame.grid()
        elif filter_type == "Median Filter":
            self.median_frame.grid()
        elif filter_type == "Hampel Filter":
            self.hampel_frame.grid()
        elif filter_type == "Z-Score Filter":
            self.zscore_frame.grid()
        elif filter_type == "Savitzky-Golay":
            self.savgol_frame.grid()

    def _update_plot_filter_ui(self, filter_type: str) -> None:
        """Update plot filter UI based on selected filter type."""
        # Hide all frames
        for frame in [
            self.plot_ma_frame,
            self.plot_bw_frame,
            self.plot_median_frame,
            self.plot_hampel_frame,
            self.plot_zscore_frame,
            self.plot_savgol_frame,
        ]:
            frame.grid_remove()

        # Show relevant frame
        if filter_type == "Moving Average":
            self.plot_ma_frame.grid()
        elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.plot_bw_frame.grid()
        elif filter_type == "Median Filter":
            self.plot_median_frame.grid()
        elif filter_type == "Hampel Filter":
            self.plot_hampel_frame.grid()
        elif filter_type == "Z-Score Filter":
            self.plot_zscore_frame.grid()
        elif filter_type == "Savitzky-Golay":
            self.plot_savgol_frame.grid()

    def _update_compare_filter_ui(self, filter_type: str) -> None:
        """Update comparison filter UI based on selected filter type."""
        # Hide all comparison frames
        for frame in [
            self.compare_ma_frame,
            self.compare_bw_frame,
            self.compare_median_frame,
            self.compare_hampel_frame,
            self.compare_zscore_frame,
            self.compare_savgol_frame,
        ]:
            frame.grid_remove()

        # Show relevant frame
        if filter_type == "Moving Average":
            self.compare_ma_frame.grid(row=14, column=0, sticky="ew", padx=10, pady=5)
        elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.compare_bw_frame.grid(row=14, column=0, sticky="ew", padx=10, pady=5)
        elif filter_type == "Median Filter":
            self.compare_median_frame.grid(
                row=14,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )
        elif filter_type == "Hampel Filter":
            self.compare_hampel_frame.grid(
                row=14,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )
        elif filter_type == "Z-Score Filter":
            self.compare_zscore_frame.grid(
                row=14,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )
        elif filter_type == "Savitzky-Golay":
            self.compare_savgol_frame.grid(
                row=14,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )

    def _filter_signals(self, event: tk.Event | None = None) -> None:
        """Filter signals based on search text - optimized for large signal counts."""
        # Check if we're using the new efficient display
        if hasattr(self, "signal_search_entry"):
            search_text = self.signal_search_entry.get().lower()
            print(f"DEBUG: Filtering signals with search text: '{search_text}'")

            # Clear the scrollable frame
            for widget in self.signals_scrollable_frame.winfo_children():
                widget.destroy()

            # Clear signal vars for filtered signals
            self.signal_vars.clear()

            if not search_text:
                # Show first 200 signals when search is cleared
                self._display_signals_batch(self.all_signals[:SIGNAL_BATCH_SIZE], 0)
                self.signals_displayed = min(SIGNAL_BATCH_SIZE, len(self.all_signals))

                # Update load more button
                if hasattr(self, "load_more_button") and len(self.all_signals) > SIGNAL_BATCH_SIZE:
                    remaining = len(self.all_signals) - SIGNAL_BATCH_SIZE
                    self.load_more_button.configure(
                        text=f"Load More Signals ({remaining} remaining)",
                    )
                    self.load_more_button.configure(state="normal")
            else:
                # Filter signals based on search text
                filtered_signals = [
                    signal
                    for signal in self.all_signals
                    if search_text in signal.lower()
                ]
                print(f"DEBUG: Found {len(filtered_signals)} matching signals")

                # Display filtered signals WITHOUT auto-selecting them
                self._display_signals_batch(filtered_signals, 0, auto_select=False)
                self.signals_displayed = len(filtered_signals)

                # Update load more button for filtered results
                if hasattr(self, "load_more_button"):
                    if len(filtered_signals) > LARGE_SIGNAL_THRESHOLD:
                        remaining = len(filtered_signals) - LARGE_SIGNAL_THRESHOLD
                        self.load_more_button.configure(
                            text=f"Load More Filtered Signals ({remaining} remaining)",
                        )
                        self.load_more_button.configure(state="normal")
                    else:
                        self.load_more_button.configure(
                            text=f"All {len(filtered_signals)} Filtered Signals Shown",
                        )
                        self.load_more_button.configure(state="disabled")

            print(
                f"DEBUG: Filtering completed, now showing {self.signals_displayed} signals",
            )
        else:
            # Fallback to original method for small signal counts
            search_text = self.search_entry.get().lower()
            for signal, data in self.signal_vars.items():
                if search_text in signal.lower():
                    data["widget"].grid()
                else:
                    data["widget"].grid_remove()

    def _clear_search(self) -> None:
        """Clear search and show all signals - optimized for large signal counts."""
        if hasattr(self, "signal_search_entry"):
            self.signal_search_entry.delete(0, tk.END)
            self._filter_signals()  # This will handle the clearing properly
        else:
            # Fallback to original method for small signal counts
            self.search_entry.delete(0, tk.END)
            for signal, data in self.signal_vars.items():
                data["widget"].grid()

    def _filter_integrator_signals(self, event: tk.Event | None = None) -> None:
        """Filter integration signals based on search text."""
        search_text = self.integrator_search_entry.get().lower()
        for signal, data in self.integrator_signal_vars.items():
            if search_text in signal.lower():
                data["widget"].pack(anchor="w", padx=5, pady=2)
            else:
                data["widget"].pack_forget()

    def _clear_integrator_search(self) -> None:
        """Clear integration search and show all signals."""
        self.integrator_search_entry.delete(0, tk.END)
        for signal, data in self.integrator_signal_vars.items():
            data["widget"].pack(anchor="w", padx=5, pady=2)

    def _integrator_select_all(self) -> None:
        """Select all integration signals."""
        for signal, data in self.integrator_signal_vars.items():
            data["var"].set(True)

    def _integrator_deselect_all(self) -> None:
        """Deselect all integration signals."""
        for signal, data in self.integrator_signal_vars.items():
            data["var"].set(False)

    def _filter_deriv_signals(self, event: tk.Event | None = None) -> None:
        """Filter differentiation signals based on search text."""
        search_text = self.deriv_search_entry.get().lower()
        for signal, data in self.deriv_signal_vars.items():
            if search_text in signal.lower():
                data["widget"].pack(anchor="w", padx=5, pady=2)
            else:
                data["widget"].pack_forget()

    def _clear_deriv_search(self) -> None:
        """Clear differentiation search and show all signals."""
        self.deriv_search_entry.delete(0, tk.END)
        for signal, data in self.deriv_signal_vars.items():
            data["widget"].pack(anchor="w", padx=5, pady=2)

    def _deriv_select_all(self) -> None:
        """Select all differentiation signals."""
        for signal, data in self.deriv_signal_vars.items():
            data["var"].set(True)

    def _deriv_deselect_all(self) -> None:
        """Deselect all differentiation signals."""
        for signal, data in self.deriv_signal_vars.items():
            data["var"].set(False)

    def _filter_plot_signals(self, event: tk.Event | None = None) -> None:
        """Filter plot signals based on search text and processing signal limit."""
        search_text = self.plot_search_entry.get().lower()

        # Get selected processing signals if limit is enabled
        limited_signals = set()
        if (
            hasattr(self, "limit_plot_signals_var")
            and self.limit_plot_signals_var.get()
        ):
            limited_signals = {
                s for s, data in self.signal_vars.items() if data["var"].get()
            }

        for signal, data in self.plot_signal_vars.items():
            # Check if signal matches search text
            matches_search = search_text in signal.lower()

            # Check if signal is allowed (not limited or in limited set)
            allowed_signal = (
                not hasattr(self, "limit_plot_signals_var")
                or not self.limit_plot_signals_var.get()
                or signal in limited_signals
            )

            if matches_search and allowed_signal:
                data["checkbox"].pack(anchor="w", padx=5, pady=2)
            else:
                data["checkbox"].pack_forget()

    def _on_limit_plot_signals_changed(self) -> None:
        """Handle changes to the limit plotting signals checkbox."""
        # Re-apply the current search filter to update the display
        self._filter_plot_signals()

    def _populate_plotting_signals_from_available(self) -> None:
        """Populate plotting signals with available signals from processing tab."""
        if not hasattr(self, "plot_signal_frame") or not hasattr(self, "signal_vars"):
            return

        # Clear existing plotting signals
        for widget in self.plot_signal_frame.winfo_children():
            widget.destroy()
        self.plot_signal_vars = {}

        # Get all available signals from processing tab
        available_signals = list(self.signal_vars.keys())

        # Create checkboxes for each signal
        for signal in available_signals:
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                self.plot_signal_frame,
                text=signal,
                variable=var,
                command=self._on_plot_signal_checkbox_changed,
            )
            cb.pack(anchor="w", padx=5, pady=2)
            self.plot_signal_vars[signal] = {"var": var, "checkbox": cb}

        # Re-bind mouse wheel to the frame
        self._bind_mousewheel_to_frame(self.plot_signal_frame)

        # Apply current filter
        self._filter_plot_signals()

    def _plot_clear_search(self) -> None:
        """Clear plot search and show all signals."""
        self.plot_search_entry.delete(0, tk.END)
        for signal, data in self.plot_signal_vars.items():
            data["checkbox"].pack(anchor="w", padx=5, pady=2)

    def _plot_select_all(self) -> None:
        """Select all plot signals."""
        for signal, data in self.plot_signal_vars.items():
            data["var"].set(True)

    def _plot_select_none(self) -> None:
        """Deselect all plot signals."""
        for signal, data in self.plot_signal_vars.items():
            data["var"].set(False)

    def _show_selected_signals(self) -> None:
        """Show only selected signals in plot."""
        selected_signals = [
            s for s, data in self.plot_signal_vars.items() if data["var"].get()
        ]
        if selected_signals:
            self.update_plot(selected_signals=selected_signals)
        else:
            messagebox.showwarning(
                "No Signals Selected",
                "Please select at least one signal to plot.",
            )

    def _filter_reference_signals(self, event: tk.Event | None = None) -> None:
        """Filter reference signals for custom variables."""
        search_text = self.custom_var_search_entry.get().lower()
        for signal, widget in self.reference_signal_widgets.items():
            if search_text in signal.lower():
                widget.pack(anchor="w", padx=5, pady=2)
            else:
                widget.pack_forget()

    def _clear_reference_search(self) -> None:
        """Clear reference search and show all signals."""
        self.custom_var_search_entry.delete(0, tk.END)
        for signal, widget in self.reference_signal_widgets.items():
            widget.pack(anchor="w", padx=5, pady=2)

    def _add_custom_variable(self) -> None:
        """Add a custom variable to the list."""
        name = self.custom_var_name_entry.get().strip()
        formula = self.custom_var_formula_entry.get().strip()

        if not name or not formula:
            messagebox.showerror("Error", "Please enter both name and formula.")
            return

        # Check if name already exists
        if any(var["name"] == name for var in self.custom_vars_list):
            messagebox.showerror("Error", f"Variable '{name}' already exists.")
            return

        self.custom_vars_list.append({"name": name, "formula": formula})
        self._update_custom_vars_display()

        # Clear entries
        self.custom_var_name_entry.delete(0, tk.END)
        self.custom_var_formula_entry.delete(0, tk.END)

    def populate_custom_var_sub_tab(self, tab: ctk.CTkFrame) -> None:
        """Fixed custom variables sub-tab with missing listbox."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(9, weight=1)

        ctk.CTkLabel(
            tab,
            text="Custom Variables (Formula Engine)",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkLabel(
            tab,
            text="Create new columns using exact signal names in [brackets].",
            justify="left",
        ).grid(row=1, column=0, padx=10, pady=(0, 5), sticky="w")

        ctk.CTkLabel(tab, text="New Variable Name:").grid(
            row=2,
            column=0,
            padx=10,
            pady=(5, 0),
            sticky="w",
        )
        self.custom_var_name_entry = ctk.CTkEntry(
            tab,
            placeholder_text="e.g., Power_Ratio",
        )
        self.custom_var_name_entry.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(tab, text="Formula:").grid(
            row=4,
            column=0,
            padx=10,
            pady=(5, 0),
            sticky="w",
        )
        self.custom_var_formula_entry = ctk.CTkEntry(
            tab,
            placeholder_text="e.g., ( [SignalA] + [SignalB] ) / 2",
        )
        self.custom_var_formula_entry.grid(
            row=5,
            column=0,
            padx=10,
            pady=5,
            sticky="ew",
        )

        ctk.CTkButton(
            tab,
            text="Add Custom Variable",
            command=self._add_custom_variable,
        ).grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        # Save/Load custom variables
        save_load_frame = ctk.CTkFrame(tab)
        save_load_frame.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        save_load_frame.grid_columnconfigure(0, weight=1)
        save_load_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            save_load_frame,
            text="Save/Load Custom Variables",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        ctk.CTkButton(
            save_load_frame,
            text="Save Variables",
            command=self._save_custom_variables,
        ).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            save_load_frame,
            text="Load Variables",
            command=self._load_custom_variables,
        ).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # FIXED: Add missing custom variables listbox
        custom_vars_list_frame = ctk.CTkFrame(tab)
        custom_vars_list_frame.grid(row=8, column=0, padx=10, pady=5, sticky="ew")
        custom_vars_list_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            custom_vars_list_frame,
            text="Current Custom Variables:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.custom_vars_listbox = ctk.CTkTextbox(custom_vars_list_frame, height=100)
        self.custom_vars_listbox.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkButton(
            custom_vars_list_frame,
            text="Clear All Variables",
            command=self._clear_custom_variables,
        ).grid(row=2, column=0, padx=10, pady=5)

        # Searchable reference list
        reference_frame = ctk.CTkFrame(tab)
        reference_frame.grid(row=9, column=0, padx=10, pady=5, sticky="nsew")
        reference_frame.grid_columnconfigure(0, weight=1)
        reference_frame.grid_rowconfigure(1, weight=1)

        search_bar_frame = ctk.CTkFrame(reference_frame, fg_color="transparent")
        search_bar_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        search_bar_frame.grid_columnconfigure(0, weight=1)

        self.custom_var_search_entry = ctk.CTkEntry(
            search_bar_frame,
            placeholder_text="Search available signals...",
        )
        self.custom_var_search_entry.grid(row=0, column=0, sticky="ew")
        self.custom_var_search_entry.bind(
            "<KeyRelease>",
            self._filter_reference_signals,
        )

        self.custom_var_clear_button = ctk.CTkButton(
            search_bar_frame,
            text="X",
            width=28,
            command=self._clear_reference_search,
        )
        self.custom_var_clear_button.grid(row=0, column=1, padx=(5, 0))

        self.signal_reference_frame = ctk.CTkScrollableFrame(
            reference_frame,
            label_text="Available Signals Reference",
        )
        self.signal_reference_frame.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")

    def _update_custom_vars_display(self) -> None:
        """Update the custom variables display."""
        self.custom_vars_listbox.configure(state="normal")
        self.custom_vars_listbox.delete("1.0", tk.END)

        for var in self.custom_vars_list:
            self.custom_vars_listbox.insert(
                tk.END,
                f"{var['name']}: {var['formula']}\n",
            )

        self.custom_vars_listbox.configure(state="disabled")

    def _clear_custom_variables(self) -> None:
        """Clear all custom variables."""
        self.custom_vars_list.clear()
        self._update_custom_vars_display()

    def _save_custom_variables(self) -> None:
        """Save current custom variables to a JSON file."""
        if not self.custom_vars_list:
            messagebox.showwarning("Warning", "No custom variables to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Custom Variables",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(self.custom_vars_list, f, indent=2)
                messagebox.showinfo("Success", f"Custom variables saved to {file_path}")
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to save custom variables: {e!s}",
                )

    def _load_custom_variables(self) -> None:
        """Load custom variables from a JSON file."""
        file_path = filedialog.askopenfilename(
            title="Load Custom Variables",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path) as f:
                    loaded_vars = json.load(f)

                # Validate the loaded data
                if not isinstance(loaded_vars, list):
                    messagebox.showerror(
                        "Error",
                        "Invalid file format. Expected a list of variables.",
                    )
                    return

                # Check if variables have required fields
                for var in loaded_vars:
                    if (
                        not isinstance(var, dict)
                        or "name" not in var
                        or "formula" not in var
                    ):
                        messagebox.showerror(
                            "Error",
                            "Invalid variable format in file.",
                        )
                        return

                # Ask if user wants to append or replace
                if self.custom_vars_list:
                    response = messagebox.askyesnocancel(
                        "Load Variables",
                        f"Found {len(loaded_vars)} variables in file.\n\n"
                        "• Yes: Add to existing variables\n"
                        "• No: Replace all existing variables\n"
                        "• Cancel: Cancel operation",
                    )

                    if response is None:  # Cancel
                        return
                    if response:  # Yes - append
                        self.custom_vars_list.extend(loaded_vars)
                    else:  # No - replace
                        self.custom_vars_list = loaded_vars.copy()
                else:
                    self.custom_vars_list = loaded_vars.copy()

                self._update_custom_vars_display()
                messagebox.showinfo(
                    "Success",
                    f"Loaded {len(loaded_vars)} custom variables from {file_path}",
                )

            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to load custom variables: {e!s}",
                )

    def _apply_integration(
        self,
        df: pd.DataFrame,
        time_col: str,
        signals_to_integrate: list[str],
        method: str = "Trapezoidal",
    ) -> pd.DataFrame:
        """Apply integration to selected signals.
        
        Args:
            df: Input DataFrame
            time_col: Column name for time values
            signals_to_integrate: List of signal column names to integrate
            method: Integration method ("Trapezoidal", "Rectangular", or "Simpson")
            
        Returns:
            DataFrame with integrated signals added as new columns
        """
        if not signals_to_integrate:
            return df

        try:
            # Make a copy to avoid modifying original
            df = df.copy()

            # Convert time to numeric for integration
            if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(
                df[time_col],
            ):
                # Convert datetime to seconds since start
                time_numeric = (df[time_col] - df[time_col].min()).dt.total_seconds()
            else:
                # Assume it's already numeric
                time_numeric = pd.to_numeric(df[time_col], errors="coerce")

            dt = time_numeric.diff().fillna(0)

            for signal in signals_to_integrate:
                if signal in df.columns and signal != time_col:
                    signal_data = pd.to_numeric(df[signal], errors="coerce")

                    if method == "Trapezoidal":
                        # Trapezoidal rule
                        cumulative = np.zeros(len(signal_data))
                        for i in range(1, len(signal_data)):
                            if not np.isnan(signal_data.iloc[i]) and not np.isnan(
                                signal_data.iloc[i - 1],
                            ):
                                cumulative[i] = (
                                    cumulative[i - 1]
                                    + 0.5
                                    * (signal_data.iloc[i] + signal_data.iloc[i - 1])
                                    * dt.iloc[i]
                                )
                            else:
                                cumulative[i] = cumulative[i - 1]
                    elif method == "Rectangular":
                        # Rectangular rule (left endpoint)
                        cumulative = np.cumsum(signal_data.fillna(0).values * dt.values)
                    else:  # Simpson's rule
                        # Simplified implementation
                        cumulative = np.zeros(len(signal_data))
                        for i in range(1, len(signal_data)):
                            if not np.isnan(signal_data.iloc[i]) and not np.isnan(
                                signal_data.iloc[i - 1],
                            ):
                                cumulative[i] = (
                                    cumulative[i - 1]
                                    + 0.5
                                    * (signal_data.iloc[i] + signal_data.iloc[i - 1])
                                    * dt.iloc[i]
                                )
                            else:
                                cumulative[i] = cumulative[i - 1]

                    df[f"cumulative_{signal}"] = cumulative

        except Exception as e:
            print(f"Error in integration: {e}")

        return df

    def _apply_differentiation(
        self,
        df: pd.DataFrame,
        time_col: str,
        signals_to_differentiate: list[str],
        method: str = "Spline (Acausal)",
    ) -> pd.DataFrame:
        """Apply differentiation to selected signals with support for up to 5th order.
        
        Args:
            df: Input DataFrame
            time_col: Column name for time values
            signals_to_differentiate: List of signal column names to differentiate
            method: Differentiation method
            
        Returns:
            DataFrame with differentiated signals added as new columns
        """
        if not signals_to_differentiate:
            return df

        # Get selected derivative orders
        selected_orders = [
            order for order, var in self.derivative_vars.items() if var.get()
        ]
        if not selected_orders:
            return df

        # Convert time to numeric for differentiation
        time_numeric = pd.to_numeric(df[time_col], errors="coerce")
        dt = time_numeric.diff().fillna(0)

        for signal in signals_to_differentiate:
            if signal in df.columns and signal != time_col:
                signal_data = pd.to_numeric(df[signal], errors="coerce")

                for order in selected_orders:
                    if method == "Spline (Acausal)":
                        # Spline-based differentiation (acausal)
                        try:
                            # Remove NaN values for spline fitting
                            valid_mask = ~(
                                np.isnan(signal_data) | np.isnan(time_numeric)
                            )
                            if np.sum(valid_mask) > order + 1:
                                x_valid = time_numeric[valid_mask]
                                y_valid = signal_data[valid_mask]

                                # Fit spline
                                spline = UnivariateSpline(
                                    x_valid,
                                    y_valid,
                                    s=0,
                                    k=min(5, len(y_valid) - 1),
                                )

                                # Calculate derivatives
                                if order == 1:
                                    derivative = spline.derivative()(time_numeric)
                                elif order == 2:
                                    derivative = spline.derivative().derivative()(
                                        time_numeric,
                                    )
                                elif order == 3:
                                    derivative = (
                                        spline.derivative()
                                        .derivative()
                                        .derivative()(time_numeric)
                                    )
                                elif order == 4:
                                    derivative = (
                                        spline.derivative()
                                        .derivative()
                                        .derivative()
                                        .derivative()(time_numeric)
                                    )
                                elif order == 5:
                                    derivative = (
                                        spline.derivative()
                                        .derivative()
                                        .derivative()
                                        .derivative()
                                        .derivative()(time_numeric)
                                    )
                                else:
                                    continue

                                # Handle NaN values
                                derivative[~valid_mask] = np.nan
                                df[f"{signal}_d{order}"] = derivative
                            else:
                                df[f"{signal}_d{order}"] = np.nan
                        except Exception as e:
                            print(
                                f"Error in spline differentiation for {signal}, order {order}: {e}",
                            )
                            df[f"{signal}_d{order}"] = np.nan

                    elif method == "Rolling Polynomial (Causal)":
                        # Rolling polynomial differentiation (causal)
                        try:
                            # Use the helper function for causal differentiation
                            window_size = 11  # Default window size
                            poly_order = min(
                                5,
                                window_size - 1,
                            )  # Ensure polynomial order < window size

                            if len(signal_data) > window_size:
                                derivative = _poly_derivative(
                                    signal_data,
                                    window_size,
                                    poly_order,
                                    order,
                                    dt.mean(),
                                )
                                df[f"{signal}_d{order}"] = derivative
                            else:
                                df[f"{signal}_d{order}"] = np.nan
                        except Exception as e:
                            print(
                                f"Error in polynomial differentiation for {signal}, "
                                f"order {order}: {e}",
                            )
                            df[f"{signal}_d{order}"] = np.nan

        return df

    def select_files(self) -> None:
        """Select input CSV files."""
        print("DEBUG: select_files() called")
        file_paths = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        print(f"DEBUG: File dialog returned: {file_paths}")

        if file_paths:
            self.input_file_paths = list(file_paths)
            print(f"DEBUG: Set input_file_paths to: {self.input_file_paths}")

            # Set default output directory to the folder of the first selected file
            if self.input_file_paths:
                first_file_dir = os.path.dirname(self.input_file_paths[0])
                self.output_directory = first_file_dir
                print(f"DEBUG: Set output directory to: {self.output_directory}")
                # Update the output label to reflect the new default directory
                if hasattr(self, "output_label"):
                    self.output_label.configure(text=f"Output: {self.output_directory}")
                    print("DEBUG: Updated output label")

            print("DEBUG: Calling update_file_list()")
            self.update_file_list()
            # Auto-load signals for Processing tab immediately to populate lists
            try:
                print("DEBUG: Auto-loading signals after file selection...")
                # Schedule shortly so UI can render the updated file list first
                self.after(UI_UPDATE_DELAY_MS, self.load_signals_from_files)
            except Exception as e:
                print(f"DEBUG: Auto-load scheduling failed: {e}")
        else:
            print("DEBUG: No files selected (user cancelled)")

    def select_output_folder(self) -> None:
        """Select output directory for processed files."""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_directory = folder_path
            self.output_label.configure(text=f"Output: {self.output_directory}")

    def update_file_list(self) -> None:
        """Update the file list display."""
        print("DEBUG: update_file_list() called")
        print(
            f"DEBUG: input_file_paths = {getattr(self, 'input_file_paths', 'NOT SET')}",
        )
        print(
            f"DEBUG: input_file_paths type = {type(getattr(self, 'input_file_paths', None))}",
        )
        print(
            f"DEBUG: input_file_paths length = {len(getattr(self, 'input_file_paths', []))}",
        )

        # Clear existing widgets
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        print("DEBUG: Cleared existing widgets")

        if not self.input_file_paths:
            print("DEBUG: No input file paths, showing default message")
            label = ctk.CTkLabel(
                self.file_list_frame,
                text="Files you select will be listed here.",
            )
            label.pack(padx=5, pady=5)
            print("DEBUG: Default label created and packed")
            return

        total_files = len(self.input_file_paths)
        print(f"DEBUG: Creating display for {total_files} files")
        print(f"DEBUG: total_files > {LARGE_SIGNAL_THRESHOLD}? {total_files > LARGE_SIGNAL_THRESHOLD}")

        # For large numbers of files, use a more efficient display
        if total_files > LARGE_SIGNAL_THRESHOLD:  # Lowered threshold for better performance
            print(f"DEBUG: Using smart summary display for {total_files} files")
            # Create a summary display for large file lists
            summary_frame = ctk.CTkFrame(self.file_list_frame)
            summary_frame.pack(fill="x", padx=5, pady=5)

            # Show file count and first few files
            summary_label = ctk.CTkLabel(
                summary_frame,
                text=f"📁 {total_files} files selected",
                font=ctk.CTkFont(size=14, weight="bold"),
            )
            summary_label.pack(padx=5, pady=5)

            # Show first 5 files as examples
            preview_frame = ctk.CTkFrame(summary_frame)
            preview_frame.pack(fill="x", padx=5, pady=5)

            preview_label = ctk.CTkLabel(
                preview_frame,
                text="📋 Sample files:",
                font=ctk.CTkFont(size=12),
            )
            preview_label.pack(anchor="w", padx=5, pady=2)

            for i in range(min(5, total_files)):
                filename = os.path.basename(self.input_file_paths[i])
                file_label = ctk.CTkLabel(
                    preview_frame,
                    text=f"  • {filename}",
                    font=ctk.CTkFont(size=11),
                )
                file_label.pack(anchor="w", padx=10, pady=1)

            if total_files > 5:
                more_label = ctk.CTkLabel(
                    preview_frame,
                    text=f"  ... and {total_files - 5} more files",
                    font=ctk.CTkFont(size=11, slant="italic"),
                )
                more_label.pack(anchor="w", padx=10, pady=1)

            # Add a button to show all files (optional)
            show_all_button = ctk.CTkButton(
                summary_frame,
                text="Show All Files",
                command=self._show_all_files_dialog,
                width=120,
            )
            show_all_button.pack(pady=5)

            # Add a button to clear all files
            clear_all_button = ctk.CTkButton(
                summary_frame,
                text="Clear All Files",
                command=self._clear_all_files,
                width=120,
                fg_color="red",
                hover_color="darkred",
            )
            clear_all_button.pack(pady=5)

        else:
            # For smaller file lists, use the original detailed display
            print(f"DEBUG: Using detailed display for {total_files} files")
            for i, file_path in enumerate(self.input_file_paths):
                file_frame = ctk.CTkFrame(self.file_list_frame)
                file_frame.pack(fill="x", padx=5, pady=2)

                filename = os.path.basename(file_path)
                label = ctk.CTkLabel(
                    file_frame,
                    text=f"{i+1}. {filename}",
                    font=ctk.CTkFont(size=11),
                )
                label.pack(side="left", padx=5, pady=2)

                button = ctk.CTkButton(
                    file_frame,
                    text="X",
                    width=25,
                    command=lambda f=file_path: self.remove_file(f),
                )
                button.pack(side="right", padx=5, pady=2)

        print("DEBUG: update_file_list() completed")

        # Force GUI update
        self.file_list_frame.update_idletasks()
        print("DEBUG: Forced file_list_frame update_idletasks()")

    def _show_all_files_dialog(self) -> None:
        """Show all files in a separate dialog window."""
        if not self.input_file_paths:
            return

        # Create a new window
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"All Files ({len(self.input_file_paths)} files)")
        dialog.geometry("600x400")
        dialog.resizable(True, True)

        # Create a scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(dialog)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Add all files to the scrollable frame
        for i, file_path in enumerate(self.input_file_paths):
            file_frame = ctk.CTkFrame(scroll_frame)
            file_frame.pack(fill="x", padx=5, pady=2)

            filename = os.path.basename(file_path)
            label = ctk.CTkLabel(
                file_frame,
                text=f"{i+1:4d}. {filename}",
                font=ctk.CTkFont(size=11),
            )
            label.pack(side="left", padx=5, pady=2)

            button = ctk.CTkButton(
                file_frame,
                text="X",
                width=25,
                command=lambda f=file_path: self.remove_file(f),
            )
            button.pack(side="right", padx=5, pady=2)

        # Add close button
        close_button = ctk.CTkButton(dialog, text="Close", command=dialog.destroy)
        close_button.pack(pady=10)

    def _clear_all_files(self) -> None:
        """Clear all selected files."""
        if self.input_file_paths:
            file_count = len(self.input_file_paths)
            self.input_file_paths.clear()
            self.update_file_list()

            # Clear signal lists immediately
            if hasattr(self, "signal_vars"):
                self.signal_vars.clear()

            # Clear signal list display
            if hasattr(self, "signal_list_frame"):
                for widget in self.signal_list_frame.winfo_children():
                    widget.destroy()

                # Show default message
                label = ctk.CTkLabel(
                    self.signal_list_frame,
                    text="No signals available. Select files to load signals.",
                )
                label.pack(padx=5, pady=5)

            # Update status
            if hasattr(self, "status_label"):
                self.status_label.configure(
                    text=f"Cleared {file_count} files. Ready to select new files.",
                )

            print(f"DEBUG: Cleared {file_count} files")
        else:
            print("DEBUG: No files to clear")

    def _cancel_signal_loading(self, progress_window: ctk.CTkToplevel) -> None:
        """Cancel the signal loading process."""
        print("DEBUG: Signal loading cancelled by user")
        self.signal_loading_cancelled = True

        # Clear files if loading was cancelled
        if hasattr(self, "input_file_paths"):
            self.input_file_paths.clear()
            self.update_file_list()

        # Clear signal lists
        if hasattr(self, "signal_vars"):
            self.signal_vars.clear()

        # Clear signal list display
        if hasattr(self, "signal_list_frame"):
            for widget in self.signal_list_frame.winfo_children():
                widget.destroy()

            # Show default message
            label = ctk.CTkLabel(
                self.signal_list_frame,
                text="Signal loading cancelled. Select files to try again.",
            )
            label.pack(padx=5, pady=5)

        # Update status
        if hasattr(self, "status_label"):
            self.status_label.configure(
                text="Signal loading cancelled. Ready to select new files.",
            )

        # Close progress window
        try:
            progress_window.destroy()
        except Exception as e:
            # Log progress window destruction errors for debugging
            print(f"Warning: Failed to destroy progress window: {e}")

        # Clear progress window reference
        if hasattr(self, "current_progress_window"):
            delattr(self, "current_progress_window")

    def _on_bulk_mode_change(self) -> None:
        """Handle bulk processing mode toggle."""
        # First file only option is now in Signal List Management section
        # No need to show/hide it based on bulk mode state

        if hasattr(self, "input_file_paths") and self.input_file_paths:
            # Reload signals with new mode
            self.load_signals_from_files()

    def _manual_load_signals(self) -> None:
        """Manually load signals from files."""
        if not hasattr(self, "input_file_paths") or not self.input_file_paths:
            messagebox.showwarning(
                "No Files",
                "Please select files first before loading signals.",
            )
            return

        print("DEBUG: Manual signal loading triggered")
        self.load_signals_from_files()

    def _load_signals_from_first_file(self) -> None:
        """Load signals from the first file only."""
        if not hasattr(self, "input_file_paths") or not self.input_file_paths:
            messagebox.showwarning(
                "No Files",
                "Please select files first before loading signals.",
            )
            return

        print("DEBUG: Loading signals from first file only")

        # Set the first file only flag
        self.first_file_only_var.set(True)

        # Load signals using the existing function
        self.load_signals_from_files()

    def _create_signal_list(self) -> None:
        """Create a signal list from text file or manual input."""
        print("DEBUG: _create_signal_list() called")

        # Ask user if they want to load from file or enter manually
        choice = messagebox.askyesno(
            "Create Signal List",
            "Do you want to load signals from a text file?\n\n"
            "Yes = Load from text file (one signal per line)\n"
            "No = Enter signals manually",
        )

        if choice:
            # Load from text file
            file_path = filedialog.askopenfilename(
                title="Select Signal List File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )

            if file_path:
                try:
                    with open(file_path) as f:
                        signals = [line.strip() for line in f if line.strip()]

                    if signals:
                        print(f"DEBUG: Loaded {len(signals)} signals from file")
                        self.update_signal_list(signals)
                        self.signal_list_status_label.configure(
                            text=f"Created signal list from file: {len(signals)} signals",
                            text_color="green",
                        )
                    else:
                        messagebox.showwarning(
                            "Empty File",
                            "The selected file contains no signals.",
                        )
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load signal file:\n{e}")
        else:
            # Manual input
            self._show_manual_signal_input()

    def _show_manual_signal_input(self) -> None:
        """Show dialog for manual signal input."""
        # Create a dialog window
        dialog = ctk.CTkToplevel(self)
        dialog.title("Enter Signal Names")
        dialog.geometry("500x400")
        dialog.resizable(True, True)
        dialog.transient(self)
        dialog.grab_set()

        # Instructions
        instruction_label = ctk.CTkLabel(
            dialog,
            text="Enter signal names (one per line):",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        instruction_label.pack(pady=10)

        # Text area for signal input
        text_area = ctk.CTkTextbox(dialog, height=200)
        text_area.pack(fill="both", expand=True, padx=20, pady=10)

        # Buttons
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill="x", padx=20, pady=10)

        def create_signal_list():
            # Get text from text area
            text = text_area.get("1.0", "end-1c")
            signals = [line.strip() for line in text.split("\n") if line.strip()]

            if signals:
                print(f"DEBUG: Created signal list with {len(signals)} signals")
                self.update_signal_list(signals)
                self.signal_list_status_label.configure(
                    text=f"Created signal list manually: {len(signals)} signals",
                    text_color="green",
                )
                dialog.destroy()
            else:
                messagebox.showwarning(
                    "No Signals",
                    "Please enter at least one signal name.",
                )

        def cancel():
            dialog.destroy()

        ctk.CTkButton(
            button_frame,
            text="Create Signal List",
            command=create_signal_list,
        ).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=cancel).pack(
            side="right",
            padx=5,
        )

    def _show_input_file_help(self) -> None:
        """Show comprehensive help for Input File Selection section."""
        help_text = """Input File Selection - Complete Guide

This section helps you select and configure your input files for processing.

📁 FILE SELECTION:

• Select Input Files
  - Opens file dialog to select CSV files for processing
  - Supports multiple file selection
  - Automatically sets output directory to first file's location
  - Files are displayed in a summary view (for large selections)

• Clear All Files
  - Removes all selected files from the list
  - Clears the signal list display
  - Resets the file selection state

• Select Output Folder
  - Choose where processed files will be saved
  - Defaults to the folder of the first selected file
  - Can be changed at any time before processing

⚙️ BULK PROCESSING MODE:

When enabled (default):
• Reads headers from only the first 3 files
• Assumes all files have the same column structure
• Much faster for large datasets (10,000+ files)
• Ideal when all files are from the same source/system

When disabled:
• Reads headers from all files (up to 100 for very large datasets)
• More thorough but slower
• Use when files might have different column structures

This mode only affects signal detection, not data processing.

🔄 TYPICAL WORKFLOW:

1. Select Input Files → Choose your CSV files
2. Configure Bulk Mode → Enable/disable based on your dataset
3. Select Output Folder → Choose where to save results
4. Load Signals → Use Signal List Management to load available signals
5. Process Data → Apply filters, export, etc.

💡 TIPS:

• Use bulk mode for large, uniform datasets (same column structure)
• Disable bulk mode if files might have different structures
• The output folder can be changed anytime before processing
• Large file selections (>50 files) show a summary view for performance
        """

        # Create help dialog
        help_dialog = ctk.CTkToplevel(self)
        help_dialog.title("Input File Selection Help")
        help_dialog.geometry("600x500")
        help_dialog.resizable(True, True)

        # Make dialog modal
        help_dialog.transient(self)
        help_dialog.grab_set()

        # Create scrollable text widget
        text_widget = ctk.CTkTextbox(help_dialog, wrap="word")
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # Insert help text
        text_widget.insert("1.0", help_text)
        text_widget.configure(state="disabled")  # Make read-only

        # Add close button
        close_button = ctk.CTkButton(
            help_dialog,
            text="Close",
            command=help_dialog.destroy,
        )
        close_button.pack(pady=10)

    def _show_signal_list_help(self) -> None:
        """Show comprehensive help for Signal List Management section."""
        help_text = """Signal List Management - Complete Guide

This section helps you manage which signals (columns) to process from your files.

📋 ESSENTIAL BUTTONS:

• Load from Files
  - Reads headers from all selected files to find available signals
  - Populates the "Available Signals to Process" list
  - REQUIRED: Use this after selecting files to see available signals
  - This is now manual since we removed automatic signal loading

• Load from First File
  - Reads headers from only the first file (bulk processing mode)
  - Assumes all files have the same column structure
  - Much faster for large datasets (10,000+ files)
  - Use when you know all files have identical structure

📁 SIGNAL LIST MANAGEMENT:

• Save Signal List
  - Saves your currently selected signals to a file
  - Useful for reusing the same signal selection later
  - Creates a .json file with your signal preferences

• Load Signal List
  - Loads a previously saved signal list
  - Restores your signal selection from a saved file
  - Useful for consistent processing across different datasets

• Create Signal List
  - Creates a signal list from a text file or manual input
  - Option to load from text file (one signal per line)
  - Option to manually enter signal names
  - Useful when you know exactly which signals you want

• Apply Signals
  - Takes a loaded signal list and applies it to current files
  - Selects signals present in both saved list and current files
  - Deselects signals not in the saved list
  - Shows which saved signals are missing from current files

🔄 TYPICAL WORKFLOW:

1. Select Files → Click "Select Input Files"
2. Load Signals → Click "Load from Files" (or "Load from First File" for bulk mode)
3. Select Signals → Choose which signals to process
4. Save List → Optionally save your selection for future use
5. Process Data → Apply filters, export, etc.

💡 TIPS:

• Use "Load from Files" for thorough signal detection
• Use "Load from First File" for speed with large, uniform datasets
• Save signal lists for consistent processing across multiple datasets
• The status label below shows current signal list status
        """

        # Create help dialog
        help_dialog = ctk.CTkToplevel(self)
        help_dialog.title("Signal List Management Help")
        help_dialog.geometry("600x500")
        help_dialog.resizable(True, True)

        # Make dialog modal
        help_dialog.transient(self)
        help_dialog.grab_set()

        # Create scrollable text widget
        text_widget = ctk.CTkTextbox(help_dialog, wrap="word")
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # Insert help text
        text_widget.insert("1.0", help_text)
        text_widget.configure(state="disabled")  # Make read-only

        # Add close button
        close_button = ctk.CTkButton(
            help_dialog,
            text="Close",
            command=help_dialog.destroy,
        )
        close_button.pack(pady=10)

    def remove_file(self, file_path: str) -> None:
        """Remove a file from the list."""
        if file_path in self.input_file_paths:
            self.input_file_paths.remove(file_path)
            self.update_file_list()
            self.load_signals_from_files()

    def load_signals_from_files(self) -> None:
        """Load signals from all selected files (optimized for large file counts)."""
        print("DEBUG: load_signals_from_files() called")

        if not self.input_file_paths:
            print("DEBUG: No input file paths, returning early")
            return

        total_files = len(self.input_file_paths)

        # Create progress window for large file counts
        progress_window: ctk.CTkToplevel | None = None
        status_label: ctk.CTkLabel | None = None
        progress_bar: ctk.CTkProgressBar | None = None

        if total_files > 100:
            progress_window = ctk.CTkToplevel(self)
            progress_window.title("Loading Signals")
            progress_window.geometry("400x200")
            progress_window.resizable(False, False)

            # Make dialog modal
            progress_window.transient(self)
            progress_window.grab_set()

            progress_label = ctk.CTkLabel(
                progress_window,
                text="Loading signals from files...",
            )
            progress_label.pack(pady=20)

            # Add progress bar
            progress_bar = ctk.CTkProgressBar(progress_window)
            progress_bar.pack(pady=10, padx=20, fill="x")
            progress_bar.set(0)

            status_label = ctk.CTkLabel(progress_window, text="Starting...")
            status_label.pack(pady=10)

            # Add cancel button
            cancel_button = ctk.CTkButton(
                progress_window,
                text="Cancel",
                command=lambda: self._cancel_signal_loading(progress_window),
                fg_color="red",
                hover_color="darkred",
            )
            cancel_button.pack(pady=10)

            # Store reference for cleanup
            self.current_progress_window = progress_window

        # Check if bulk processing mode is enabled
        bulk_mode = getattr(self, "bulk_mode_var", None) and self.bulk_mode_var.get()
        print(
            f"DEBUG: bulk_mode_var exists: {getattr(self, 'bulk_mode_var', None) is not None}",
        )
        bulk_mode_value = (
            getattr(self, "bulk_mode_var", None).get()
            if getattr(self, "bulk_mode_var", None)
            else "N/A"
        )
        print(f"DEBUG: bulk_mode_var value: {bulk_mode_value}")
        print(f"DEBUG: bulk_mode result: {bulk_mode}")

        # Check for cancellation
        if hasattr(self, "signal_loading_cancelled") and self.signal_loading_cancelled:
            self.signal_loading_cancelled = False
            if progress_window:
                try:
                    progress_window.destroy()
                except Exception as e:
                    # Log progress window destruction errors for debugging
                    print(f"Warning: Failed to destroy progress window: {e}")
            return

        try:
            if bulk_mode and total_files > 1:
                # Check if first file only option is enabled
                first_file_only = (
                    getattr(self, "first_file_only_var", None)
                    and self.first_file_only_var.get()
                )

                if first_file_only:
                    # First file only mode: most conservative approach
                    print(
                        "DEBUG: Using bulk processing mode - reading headers from first file only",
                    )

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text="Bulk mode: Reading headers from first file only...",
                        )
                        progress_window.update()
                    else:
                        self.update_status(
                            "Bulk mode: Reading headers from first file only...",
                            show_progress=True,
                            progress_value=0.1,
                            progress_text="Reading file headers...",
                        )

                    # Read headers from first file only
                    sample_files = self.input_file_paths[:1]
                    all_signals = set()
                else:
                    # Standard bulk mode: read headers from first few files
                    print(
                        "DEBUG: Using bulk processing mode - "
                        "reading headers from sample files only",
                    )

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text="Bulk mode: Reading headers from sample files...",
                        )
                        progress_window.update()
                    else:
                        self.update_status(
                            "Bulk mode: Reading headers from sample files...",
                            show_progress=True,
                            progress_value=0.1,
                            progress_text="Reading file headers...",
                        )

                    # Read headers from first 3 files only
                    sample_files = self.input_file_paths[:3]
                    all_signals = set()

                for i, file_path in enumerate(sample_files):
                    # Check for cancellation
                    if (
                        hasattr(self, "signal_loading_cancelled")
                        and self.signal_loading_cancelled
                    ):
                        print(
                            "DEBUG: Signal loading cancelled during bulk mode processing",
                        )
                        return

                    try:
                        if total_files > 100:
                            status_label.configure(
                                text=f"Reading sample file {i+1}/3: {os.path.basename(file_path)}",
                            )
                            if progress_bar:
                                progress = (i + 1) / len(sample_files)
                                progress_bar.set(progress)
                            progress_window.update()
                        elif hasattr(self, "status_label"):
                            self.status_label.configure(
                                text=f"Reading sample file {i+1}/3: {os.path.basename(file_path)}",
                            )
                            self.update()

                        df = pd.read_csv(file_path, nrows=1)
                        signals = df.columns.tolist()
                        all_signals.update(signals)

                    except Exception as e:
                        print(f"Error reading sample file {file_path}: {e}")

                if first_file_only:
                    print(
                        f"DEBUG: Bulk mode (first file only) - "
                        f"signals from 1 file: {len(all_signals)} unique signals",
                    )

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text=f"Bulk mode: Using {len(all_signals)} signals from first file only "
                            f"(assumed same for all {total_files} files)",
                        )
                        progress_window.update()
                    elif hasattr(self, "status_label"):
                        self.status_label.configure(
                            text=f"Bulk mode: Using {len(all_signals)} signals from first file only "
                            f"(assumed same for all {total_files} files)",
                        )
                        self.update()
                else:
                    print(
                        f"DEBUG: Bulk mode - signals from {len(sample_files)} sample files: "
                        f"{len(all_signals)} unique signals",
                    )

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text=f"Bulk mode: Using {len(all_signals)} signals from sample files "
                            f"(assumed same for all {total_files} files)",
                        )
                        progress_window.update()
                    elif hasattr(self, "status_label"):
                        self.status_label.configure(
                            text=f"Bulk mode: Using {len(all_signals)} signals from sample files "
                            f"(assumed same for all {total_files} files)",
                        )
                        self.update()

            else:
                # Normal mode: read headers from all files (but limit for very large counts)
                print("DEBUG: Using normal mode - reading headers from all files")

                # For very large file counts, limit to first 100 files to prevent stalling
                files_to_read = (
                    min(total_files, 100) if total_files > 100 else total_files
                )

                # Update status
                if total_files > 100:
                    status_label.configure(
                        text=f"Reading headers from first {files_to_read} files "
                        f"(of {total_files})...",
                    )
                    progress_window.update()
                elif hasattr(self, "status_label"):
                    self.status_label.configure(
                        text=f"Reading headers from {files_to_read} files...",
                    )
                    self.update()

                all_signals = set()

                # For large numbers of files, use batch processing
                batch_size = LARGE_BATCH_SIZE if files_to_read > LARGE_SIGNAL_THRESHOLD else SMALL_BATCH_SIZE

                for i in range(0, files_to_read, batch_size):
                    # Check for cancellation
                    if (
                        hasattr(self, "signal_loading_cancelled")
                        and self.signal_loading_cancelled
                    ):
                        print("DEBUG: Signal loading cancelled during batch processing")
                        return

                    batch_end = min(i + batch_size, files_to_read)
                    batch_files = self.input_file_paths[i:batch_end]

                    # Update status for batch
                    if total_files > 100:
                        try:
                            status_label.configure(
                                text=f"Reading files {i+1}-{batch_end}/"
                                f"{files_to_read}...",
                            )
                            if progress_bar:
                                progress = (i + batch_size) / files_to_read
                                progress_bar.set(min(progress, 1.0))
                            progress_window.update()
                        except Exception as e:
                            print(f"Status update error (ignoring): {e}")
                            # Continue processing even if status update fails
                    elif hasattr(self, "status_label"):
                        self.status_label.configure(
                            text=f"Reading files {i+1}-{batch_end}/"
                            f"{total_files}...",
                        )
                        self.update()

                    for file_path in batch_files:
                        try:
                            # Just read header for efficiency
                            df = pd.read_csv(file_path, nrows=1)
                            signals = df.columns.tolist()
                            all_signals.update(signals)

                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

                    # Force UI update after each batch
                    if total_files > 100:
                        progress_window.update()
                    else:
                        self.update()

                print(
                    f"DEBUG: Normal mode - all signals collected: "
                    f"{len(all_signals)} unique signals",
                )

                # Update status
                if total_files > 100:
                    try:
                        if files_to_read < total_files:
                            status_label.configure(
                                text=f"Found {len(all_signals)} unique signals in first {files_to_read} "
                                f"files (of {total_files})",
                            )
                        else:
                            status_label.configure(
                                text=f"Found {len(all_signals)} unique signals in {total_files} files",
                            )
                        progress_window.update()
                    except Exception as e:
                        print(f"Status update error (ignoring): {e}")
                elif hasattr(self, "status_label"):
                    if "files_to_read" in locals() and files_to_read < total_files:
                        self.status_label.configure(
                            text=f"Found {len(all_signals)} unique signals in first {files_to_read} "
                            f"files (of {total_files})",
                        )
                    else:
                        self.status_label.configure(
                            text=f"Found {len(all_signals)} unique signals in {total_files} files",
                        )
                    self.update()

            # Update signal list
            if total_files > 100:
                try:
                    status_label.configure(text="Updating signal list...")
                    progress_window.update()
                except Exception as e:
                    print(f"Signal list update error (ignoring): {e}")

            self.update_signal_list(sorted(all_signals))

            # Close progress window
            if total_files > 100:
                try:
                    progress_window.destroy()
                    print("DEBUG: Progress window closed")
                except Exception as e:
                    print(f"Progress window close error (ignoring): {e}")

                # Clear progress window reference
                if hasattr(self, "current_progress_window"):
                    delattr(self, "current_progress_window")

        except Exception as e:
            print(f"Error in load_signals_from_files: {e}")
            traceback.print_exc()
            if total_files > 100 and progress_window:
                try:
                    progress_window.destroy()
                except Exception:
                    # Silently ignore progress window destruction errors
                    pass

                # Clear progress window reference
                if hasattr(self, "current_progress_window"):
                    delattr(self, "current_progress_window")

            messagebox.showerror("Error", f"Error loading signals: {e!s}")

        # Update plot file menu with smart defaults
        file_names = ["Select a file..."] + [
            os.path.basename(f) for f in self.input_file_paths
        ]
        if hasattr(self, "plot_file_menu"):
            self.plot_file_menu.configure(values=file_names)

            # Auto-select the first file if there's only one - immediate execution like baseline
            if len(self.input_file_paths) == 1:
                single_file = os.path.basename(self.input_file_paths[0])
                self.plot_file_menu.set(single_file)
                # Direct call - no scheduling delays
                self._auto_select_single_file(single_file)
            else:
                # For multiple files, user can select
                self.plot_file_menu.set("Select a file...")
                if hasattr(self, "status_label"):
                    self.status_label.configure(
                        text=f"Ready - {len(self.input_file_paths)} files loaded. "
                        f"Go to Plotting tab to visualize.",
                    )

        print("DEBUG: load_signals_from_files() completed")

        # Populate plotting signals with available signals
        self._populate_plotting_signals_from_available()

    def _auto_select_single_file(self, filename: str) -> None:
        """Auto-select single file - simplified."""
        try:
            if hasattr(self, "plot_file_menu"):
                current_selection = self.plot_file_menu.get()
                if current_selection == filename:  # Only proceed if still selected
                    self.on_plot_file_select(filename)
        except Exception as e:
            print(f"Error in auto-select: {e}")

    def _ensure_data_loaded(self, filename: str) -> bool:
        """Ensure data is loaded for the given filename."""
        if filename not in self.processed_files:
            # Try to load the file
            full_path = None
            for file_path in self.input_file_paths:
                if os.path.basename(file_path) == filename:
                    full_path = file_path
                    break

            if full_path and os.path.exists(full_path):
                try:
                    df = pd.read_csv(full_path, low_memory=False)
                    # Simple time column conversion
                    for col in df.columns:
                        if any(
                            time_word in col.lower()
                            for time_word in ["time", "timestamp", "date"]
                        ):
                            try:
                                df[col] = pd.to_datetime(df[col])
                                break  # Only convert first time column found
                            except Exception as e:
                                # Log datetime conversion errors for debugging
                                print(f"Warning: Failed to convert column {col} to datetime: {e}")
                    return True
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    return False
        return True

    def update_signal_list(self, signals: list[str]) -> None:
        """Update the signal list with checkboxes - optimized for large signal counts."""
        print(f"DEBUG: update_signal_list called with {len(signals)} signals")

        # Store signals for later use
        self.all_signals = signals

        # Check if signal_list_frame exists
        if not hasattr(self, "signal_list_frame"):
            print("DEBUG: ERROR - signal_list_frame does not exist!")
            return

        print("DEBUG: signal_list_frame exists, clearing widgets")

        # Clear existing widgets
        for widget in self.signal_list_frame.winfo_children():
            widget.destroy()

        self.signal_vars.clear()

        # For large numbers of signals, use a more efficient approach
        if len(signals) > SIGNAL_BATCH_SIZE:
            print(
                f"DEBUG: Large signal count ({len(signals)}), using efficient display",
            )

            # Create a summary frame
            summary_frame = ctk.CTkFrame(self.signal_list_frame)
            summary_frame.pack(fill="x", padx=5, pady=5)

            # Show signal count
            summary_label = ctk.CTkLabel(
                summary_frame,
                text=f"📊 {len(signals)} signals available",
                font=ctk.CTkFont(size=14, weight="bold"),
            )
            summary_label.pack(padx=5, pady=5)

            # Create search frame
            search_frame = ctk.CTkFrame(summary_frame)
            search_frame.pack(fill="x", padx=5, pady=5)

            search_label = ctk.CTkLabel(
                search_frame,
                text="Search signals:",
                font=ctk.CTkFont(size=12),
            )
            search_label.pack(anchor="w", padx=5, pady=2)

            self.signal_search_entry = ctk.CTkEntry(
                search_frame,
                placeholder_text="Type to search signals...",
            )
            self.signal_search_entry.pack(fill="x", padx=5, pady=2)
            self.signal_search_entry.bind("<KeyRelease>", self._filter_signals)

            # Create control buttons
            button_frame = ctk.CTkFrame(summary_frame)
            button_frame.pack(fill="x", padx=5, pady=5)

            ctk.CTkButton(
                button_frame,
                text="Select All",
                command=self.select_all,
            ).pack(side="left", padx=2, pady=2)
            ctk.CTkButton(
                button_frame,
                text="Deselect All",
                command=self.deselect_all,
            ).pack(side="left", padx=2, pady=2)
            ctk.CTkButton(
                button_frame,
                text="Show Selected",
                command=self._show_selected_signals,
            ).pack(side="left", padx=2, pady=2)
            ctk.CTkButton(
                button_frame,
                text="Clear Search",
                command=self._clear_search,
            ).pack(side="left", padx=2, pady=2)

            # Create scrollable frame for signals
            self.signals_scrollable_frame = ctk.CTkScrollableFrame(
                self.signal_list_frame,
                height=300,
            )
            self.signals_scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)

            # Initially show first 200 signals
            self._display_signals_batch(signals[:SIGNAL_BATCH_SIZE], start_index=0)

            # Add "Load More" button if there are more signals
            if len(signals) > SIGNAL_BATCH_SIZE:
                load_more_frame = ctk.CTkFrame(self.signal_list_frame)
                load_more_frame.pack(fill="x", padx=5, pady=5)

                # Show warning about truncated signals
                warning_label = ctk.CTkLabel(
                    load_more_frame,
                    text=f"⚠️ WARNING: Only showing first {SIGNAL_BATCH_SIZE} of {len(signals)} signals",
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color="orange",
                )
                warning_label.pack(pady=2)

                self.load_more_button = ctk.CTkButton(
                    load_more_frame,
                    text=f"Load More Signals ({len(signals) - SIGNAL_BATCH_SIZE} remaining)",
                    command=lambda: self._load_more_signals(signals, SIGNAL_BATCH_SIZE),
                )
                self.load_more_button.pack(pady=5)

                self.signals_displayed = SIGNAL_BATCH_SIZE
                self.all_signals_for_display = signals
            else:
                self.signals_displayed = len(signals)
                self.all_signals_for_display = signals

        else:
            # For smaller signal counts, use the original approach
            print(f"DEBUG: Small signal count ({len(signals)}), using standard display")

            # Create checkboxes directly
            for signal in signals:
                var = tk.BooleanVar(value=True)
                cb = ctk.CTkCheckBox(
                    self.signal_list_frame,
                    text=signal,
                    variable=var,
                    command=self._on_signal_checkbox_changed,
                )
                cb.grid(sticky="w", padx=5, pady=2)
                self.signal_vars[signal] = {"var": var, "widget": cb}

        # Update sort column menu
        sort_values = ["No Sorting"] + signals
        self.sort_col_menu.configure(values=sort_values)

        # Initialize plot signal variables (will be populated when file is selected in plotting tab)
        self.plot_signal_vars = {}

        # Update other signal lists - simplified
        self._update_plots_signals(signals)
        self._update_integration_signals(signals)
        self._update_differentiation_signals(signals)
        self._update_reference_signals(signals)

        print("DEBUG: update_signal_list completed")

    def _schedule_plot_update(self) -> None:
        """Debounce and schedule plot update shortly after a checkbox change."""
        try:
            # Cancel pending job if any
            if (
                hasattr(self, "_plot_update_job_id")
                and self._plot_update_job_id is not None
            ):
                try:
                    self.after_cancel(self._plot_update_job_id)
                except Exception as e:
                    # Log after_cancel errors for debugging
                    print(f"Warning: Failed to cancel plot update job: {e}")
            # Schedule a near-future update to coalesce rapid toggles
            self._plot_update_job_id = self.after(
                200,
                getattr(self, "update_plot", lambda: None),
            )
        except Exception as e:
            print(f"DEBUG: _schedule_plot_update error: {e}")

    def _on_signal_checkbox_changed(self) -> None:
        """Handle signal checkbox toggles by auto-updating plot."""
        self._schedule_plot_update()
        # Update integration and differentiation signals based on selected processing signals
        self._update_processing_dependent_signals()

    def _update_processing_dependent_signals(self) -> None:
        """Update integration and differentiation signals based on selected processing signals."""
        # Get currently selected signals for processing
        selected_signals = [
            s for s, data in self.signal_vars.items() if data["var"].get()
        ]

        # Filter out time-related columns (common time column names)
        time_columns = {
            "local_time",
            "utc_time",
            "time",
            "timestamp",
            "date",
            "datetime",
        }
        non_time_signals = [
            s for s in selected_signals if s.lower() not in time_columns
        ]

        # Update integration signals
        if hasattr(self, "integrator_signals_frame"):
            self._update_integration_signals(non_time_signals)

        # Update differentiation signals
        if hasattr(self, "deriv_signals_frame"):
            self._update_differentiation_signals(non_time_signals)

    def _on_plot_signal_checkbox_changed(self) -> None:
        """Auto-update plot when Plotting tab signal checkboxes change (scoped)."""
        self._schedule_plot_update()

    def _display_signals_batch(self, signals_batch: list[str], start_index: int = 0, auto_select: bool = True) -> None:
        """Display a batch of signals in the scrollable frame."""
        print(
            f"DEBUG: Displaying batch of {len(signals_batch)} signals starting at index "
            f"{start_index}, auto_select={auto_select}",
        )

        for i, signal in enumerate(signals_batch):
            var = tk.BooleanVar(value=auto_select)
            cb = ctk.CTkCheckBox(
                self.signals_scrollable_frame,
                text=signal,
                variable=var,
                command=self._on_signal_checkbox_changed,
            )
            cb.pack(anchor="w", padx=5, pady=1)
            self.signal_vars[signal] = {"var": var, "widget": cb}

        print("DEBUG: Batch display completed")

    def _load_more_signals(self, all_signals: list[str], current_count: int) -> None:
        """Load more signals when the Load More button is clicked."""
        print(
            f"DEBUG: Loading more signals, currently showing {current_count} of {len(all_signals)}",
        )

        # Calculate how many more to load (use 200 as batch size)
        remaining = len(all_signals) - current_count
        batch_size = min(SIGNAL_BATCH_SIZE, remaining)
        end_index = current_count + batch_size

        # Display the next batch
        self._display_signals_batch(all_signals[current_count:end_index], current_count)

        # Update the load more button
        remaining_after = len(all_signals) - end_index
        if remaining_after > 0:
            self.load_more_button.configure(
                text=f"Load More Signals ({remaining_after} remaining)",
            )
        else:
            self.load_more_button.configure(text="All Signals Loaded")
            self.load_more_button.configure(state="disabled")

        self.signals_displayed = end_index
        print(f"DEBUG: Now showing {self.signals_displayed} signals")

    def _update_integration_signals(self, signals: list[str]) -> None:
        """Update integration signals - simplified."""
        # Clear existing widgets
        for widget in self.integrator_signals_frame.winfo_children():
            widget.destroy()
        self.integrator_signal_vars.clear()

        for signal in signals:
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                self.integrator_signals_frame,
                text=signal,
                variable=var,
            )
            cb.grid(sticky="w", padx=5, pady=2)
            self.integrator_signal_vars[signal] = {"var": var, "widget": cb}

    def _update_differentiation_signals(self, signals: list[str]) -> None:
        """Update differentiation signals - simplified."""
        # Clear existing widgets
        for widget in self.deriv_signals_frame.winfo_children():
            widget.destroy()
        self.deriv_signal_vars.clear()

        for signal in signals:
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self.deriv_signals_frame, text=signal, variable=var)
            cb.grid(sticky="w", padx=5, pady=2)
            self.deriv_signal_vars[signal] = {"var": var, "widget": cb}

    def _update_reference_signals(self, signals: list[str]) -> None:
        """Update reference signals - simplified."""
        # Clear existing widgets
        for widget in self.signal_reference_frame.winfo_children():
            widget.destroy()
        self.reference_signal_widgets.clear()

        for signal in signals:
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self.signal_reference_frame, text=signal, variable=var)
            cb.grid(sticky="w", padx=5, pady=2)
            self.reference_signal_widgets[signal] = {"var": var, "widget": cb}

    def select_all(self) -> None:
        """Select all signals - optimized for large signal counts."""
        if hasattr(self, "all_signals"):
            # For large signal counts, select all signals (including those not yet displayed)
            for signal in self.all_signals:
                if signal in self.signal_vars:
                    self.signal_vars[signal]["var"].set(True)
            print(f"DEBUG: Selected all {len(self.all_signals)} signals")
        else:
            # Fallback for small signal counts
            for signal, data in self.signal_vars.items():
                data["var"].set(True)

    def deselect_all(self) -> None:
        """Deselect all signals - optimized for large signal counts."""
        if hasattr(self, "all_signals"):
            # For large signal counts, deselect all signals (including those not yet displayed)
            for signal in self.all_signals:
                if signal in self.signal_vars:
                    self.signal_vars[signal]["var"].set(False)
            print(f"DEBUG: Deselected all {len(self.all_signals)} signals")
        else:
            # Fallback for small signal counts
            for signal, data in self.signal_vars.items():
                data["var"].set(False)

    def process_files(self) -> None:
        """Process all selected files with current settings."""
        print("\n=== STARTING PROCESS_FILES DEBUG ===")
        if not self.input_file_paths:
            print("ERROR: No input file paths selected")
            messagebox.showerror("Error", "Please select input files first.")
            return

        print(f"Input files: {len(self.input_file_paths)} files")
        for i, path in enumerate(self.input_file_paths):
            print(f"  {i+1}: {path}")

        selected_signals = [
            s for s, data in self.signal_vars.items() if data["var"].get()
        ]
        print(f"Selected signals: {len(selected_signals)} signals")
        for signal in selected_signals:
            print(f"  - {signal}")

        if not selected_signals:
            print("ERROR: No signals selected")
            messagebox.showerror(
                "Error",
                "Please select at least one signal to process.",
            )
            return

        # Get processing settings
        settings = {
            "selected_signals": selected_signals,
            "filter_type": self.filter_type_var.get(),
            "resample_enabled": self.resample_var.get(),
            "resample_rule": self._get_resample_rule(),
            "ma_window": (
                int(self.ma_value_entry.get()) if self.ma_value_entry.get() else 10
            ),
            "bw_order": (
                int(self.bw_order_entry.get()) if self.bw_order_entry.get() else 3
            ),
            "bw_cutoff": (
                float(self.bw_cutoff_entry.get()) if self.bw_cutoff_entry.get() else 0.1
            ),
            "median_kernel": (
                int(self.median_kernel_entry.get())
                if self.median_kernel_entry.get()
                else 5
            ),
            "savgol_window": (
                int(self.savgol_window_entry.get())
                if self.savgol_window_entry.get()
                else 11
            ),
            "savgol_polyorder": (
                int(self.savgol_polyorder_entry.get())
                if self.savgol_polyorder_entry.get()
                else 2
            ),
        }

        print("\nProcessing settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

        # Check output directory
        if not self.output_directory:
            print("WARNING: No output directory set, using default")
        else:
            print(f"Output directory: {self.output_directory}")

        # Update status
        self.status_label.configure(text="Processing files...")
        self.update()

        # Process files sequentially (simpler than parallel processing for debugging)
        processed_files = []
        error_count = 0

        print(f"\nStarting processing of {len(self.input_file_paths)} files...")

        for i, file_path in enumerate(self.input_file_paths):
            print(
                f"\n--- Processing file {i+1}/{len(self.input_file_paths)}: "
                f"{os.path.basename(file_path)} ---",
            )
            try:
                self.status_label.configure(
                    text=f"Processing file {i+1}/{len(self.input_file_paths)}: "
                    f"{os.path.basename(file_path)}",
                )
                self.update()

                # Check if file exists
                if not os.path.exists(file_path):
                    print(f"ERROR: File not found: {file_path}")
                    error_count += 1
                    continue

                print(f"File exists, size: {os.path.getsize(file_path)} bytes")

                # Process the file
                processed_df = self._process_single_file(file_path, settings)

                if processed_df is not None and not processed_df.empty:
                    print(f"SUCCESS: File processed. Shape: {processed_df.shape}")
                    print(f"Columns: {list(processed_df.columns)}")
                    processed_files.append((file_path, processed_df))
                    # Store processed data for plotting
                    filename = os.path.basename(file_path)
                    self.processed_files[filename] = processed_df.copy()
                    print(f"Stored in processed_files cache as: {filename}")
                else:
                    print("ERROR: File processing returned None or empty DataFrame")
                    error_count += 1

            except Exception as e:
                print(f"EXCEPTION processing {file_path}: {e}")
                import traceback

                traceback.print_exc()
                error_count += 1

        print("\nProcessing complete. Results:")
        print(f"  Processed files: {len(processed_files)}")
        print(f"  Errors: {error_count}")

        if not processed_files:
            print("ERROR: No files were successfully processed")
            messagebox.showerror("Error", "No files were successfully processed.")
            self.status_label.configure(text="Processing failed - no files processed")
            return

        # Check if we should combine multiple files into a single dataset
        if len(processed_files) > 1:
            # Ask user if they want to combine files for time series analysis
            combine_response = messagebox.askyesno(
                "Combine Files",
                f"You have {len(processed_files)} processed files.\n\n"
                "Would you like to combine them into a single dataset for time series analysis?\n\n"
                "This is useful for multi-day data or continuous time series.\n"
                "Files will be combined chronologically based on their timestamps.",
            )

            if combine_response:
                print("\nUser chose to combine files into single dataset")
                processed_files = self._combine_multiple_files(processed_files)
                # Update processed_files cache with combined dataset
                if processed_files:
                    combined_file_path, combined_df = processed_files[0]
                    self.processed_files[combined_file_path] = combined_df.copy()

        # Export processed files
        print("\nStarting export process...")
        print(f"Export type: {self.export_type_var.get()}")
        try:
            self._export_processed_files(processed_files)
            print("Export completed successfully")

            # Update status
            success_count = len(processed_files)
            total_count = len(self.input_file_paths)

            if error_count > 0:
                self.status_label.configure(
                    text=f"Processing complete: {success_count}/{total_count} "
                    f"files processed successfully",
                )
                messagebox.showwarning(
                    "Processing Complete",
                    f"Processed {success_count} out of {total_count} files.\n"
                    f"{error_count} files had errors.",
                )
            else:
                self.status_label.configure(
                    text=f"All {success_count} files processed successfully!",
                )
                messagebox.showinfo(
                    "Success",
                    f"All {success_count} files processed and exported successfully!",
                )

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting files: {e!s}")
            self.status_label.configure(text="Export failed")

    def _process_single_file(self, file_path: str, settings: dict[str, Any]) -> pd.DataFrame | None:
        """Process a single file with all advanced features."""
        print(f"\n_process_single_file called for: {os.path.basename(file_path)}")
        try:
            print("  Loading CSV file...")
            df = pd.read_csv(file_path, low_memory=False)
            print(f"  Loaded DataFrame shape: {df.shape}")
            print(f"  Original columns: {list(df.columns)}")

            # Determine which signals to keep for this specific file
            signals_in_this_file = [
                s for s in settings["selected_signals"] if s in df.columns
            ]
            time_col = df.columns[0]
            print(f"  Time column: {time_col}")
            print(f"  Signals found in file: {len(signals_in_this_file)}")

            if time_col not in signals_in_this_file:
                signals_in_this_file.insert(0, time_col)
                print("  Added time column to signals")

            print(f"  Final signals to process: {signals_in_this_file}")

            processed_df = df[signals_in_this_file].copy()
            print(f"  Copied DataFrame shape: {processed_df.shape}")

            # Data type conversion
            print("  Converting data types...")
            processed_df[time_col] = pd.to_datetime(
                processed_df[time_col],
                errors="coerce",
            )
            before_drop = len(processed_df)
            processed_df.dropna(subset=[time_col], inplace=True)
            after_drop = len(processed_df)
            print(f"  Dropped {before_drop - after_drop} rows with invalid time")

            for col in processed_df.columns:
                if col != time_col:
                    before_numeric = processed_df[col].notna().sum()
                    processed_df[col] = pd.to_numeric(
                        processed_df[col],
                        errors="coerce",
                    )
                    after_numeric = processed_df[col].notna().sum()
                    print(
                        f"  Column {col}: {before_numeric} -> {after_numeric} valid values",
                    )

            if processed_df.empty:
                print("  ERROR: DataFrame is empty after data type conversion")
                return None

            # Apply time trimming if specified
            trim_date = self.trim_date_entry.get().strip()
            trim_start = self.trim_start_entry.get().strip()
            trim_end = self.trim_end_entry.get().strip()

            if trim_date or trim_start or trim_end:
                print(
                    f"  Applying time trimming: date={trim_date}, start={trim_start}, "
                    f"end={trim_end}",
                )
                try:
                    # Get the date from the data if not specified
                    if not trim_date:
                        trim_date = processed_df[time_col].iloc[0].strftime("%Y-%m-%d")
                        print(f"  Using date from data: {trim_date}")

                    # Create full datetime strings
                    start_time_str = trim_start or DEFAULT_START_TIME
                    end_time_str = trim_end or DEFAULT_END_TIME
                    start_full_str = f"{trim_date} {start_time_str}"
                    end_full_str = f"{trim_date} {end_time_str}"
                    print(f"  Time range: {start_full_str} to {end_full_str}")

                    before_trim = len(processed_df)
                    # Filter the data by time range
                    processed_df = (
                        processed_df.set_index(time_col)
                        .loc[start_full_str:end_full_str]
                        .reset_index()
                    )
                    after_trim = len(processed_df)
                    print(f"  Trimming: {before_trim} -> {after_trim} rows")

                    if processed_df.empty:
                        print("  ERROR: Time trimming resulted in empty dataset")
                        return None

                except Exception as e:
                    print(f"  ERROR in time trimming: {e}")

            print("  Setting time column as index...")
            processed_df.set_index(time_col, inplace=True)
            print(f"  DataFrame shape after indexing: {processed_df.shape}")

            # Apply Filtering
            filter_type = settings.get("filter_type")
            print(f"  Filter type: {filter_type}")
            if filter_type and filter_type != "None":
                print("  Applying filtering...")
                numeric_cols = processed_df.select_dtypes(
                    include=np.number,
                ).columns.tolist()
                print(f"  Numeric columns for filtering: {numeric_cols}")
                for col in numeric_cols:
                    signal_data = processed_df[col].dropna()
                    if len(signal_data) < MIN_SIGNAL_DATA_POINTS:
                        continue

                    # Apply filtering based on type
                    if filter_type == "Moving Average":
                        window_size = settings.get("ma_window", DEFAULT_MA_WINDOW)
                        processed_df[col] = signal_data.rolling(
                            window=window_size,
                            min_periods=MIN_PERIODS_DEFAULT,
                        ).mean()
                    elif filter_type in [
                        "Butterworth Low-pass",
                        "Butterworth High-pass",
                    ]:
                        order = settings.get("bw_order", DEFAULT_BW_ORDER)
                        cutoff = settings.get("bw_cutoff", DEFAULT_BW_CUTOFF)
                        sr = (
                            1.0
                            / pd.to_numeric(
                                signal_data.index.to_series().diff().dt.total_seconds(),
                            ).mean()
                        )
                        if pd.notna(sr) and len(signal_data) > order * MIN_BUTTERWORTH_DATA_MULTIPLIER:
                            btype = (
                                "low"
                                if filter_type == "Butterworth Low-pass"
                                else "high"
                            )
                            b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sr)
                            processed_df[col] = pd.Series(
                                filtfilt(b, a, signal_data),
                                index=signal_data.index,
                            )
                    elif filter_type == "Median Filter":
                        kernel = settings.get("median_kernel", DEFAULT_MEDIAN_KERNEL)
                        if kernel % 2 == 0:
                            kernel += 1
                        if len(signal_data) > kernel:
                            processed_df[col] = pd.Series(
                                medfilt(signal_data, kernel_size=kernel),
                                index=signal_data.index,
                            )
                    elif filter_type == "Hampel Filter":
                        window = settings.get("hampel_window", DEFAULT_HAMPEL_WINDOW)
                        threshold = settings.get("hampel_threshold", DEFAULT_HAMPEL_THRESHOLD)

                        try:
                            signal_data = processed_df[col].ffill().bfill()

                            # Apply Hampel filter
                            median_filtered = pd.Series(
                                medfilt(signal_data, kernel_size=window),
                                index=signal_data.index,
                            )
                            mad = signal_data.rolling(window=window, center=True).apply(
                                lambda x: np.median(np.abs(x - np.median(x))),
                            )
                            threshold_value = (
                                threshold * NORMAL_DISTRIBUTION_CONSTANT * mad
                            )  # 1.4826 is the constant for normal distribution

                            # Replace outliers with median using proper indexing
                            outliers = (
                                np.abs(signal_data - median_filtered) > threshold_value
                            )
                            processed_df.loc[outliers, col] = median_filtered[outliers]
                        except Exception as e:
                            print(f"Error applying Hampel filter: {e}")
                            # Fallback to simple median filter
                            processed_df[col] = pd.Series(
                                medfilt(signal_data, kernel_size=window),
                                index=signal_data.index,
                            )
                    elif filter_type == "Z-Score Filter":
                        threshold = settings.get("zscore_threshold", DEFAULT_ZSCORE_THRESHOLD)
                        method = settings.get("zscore_method", DEFAULT_ZSCORE_METHOD)

                        mean_val = signal_data.mean()
                        std_val = signal_data.std()
                        z_scores = np.abs((signal_data - mean_val) / std_val)

                        if method == "Remove Outliers":
                            processed_df.loc[z_scores > threshold, col] = np.nan
                        elif method == "Clip Outliers":
                            upper_bound = mean_val + threshold * std_val
                            lower_bound = mean_val - threshold * std_val
                            processed_df[col] = processed_df[col].clip(
                                lower=lower_bound,
                                upper=upper_bound,
                            )
                        elif method == "Replace with Median":
                            median_val = signal_data.median()
                            processed_df.loc[z_scores > threshold, col] = median_val
                    elif filter_type == "Savitzky-Golay":
                        window = settings.get("savgol_window", DEFAULT_SAVGOL_WINDOW)
                        polyorder = settings.get("savgol_polyorder", DEFAULT_SAVGOL_POLYORDER)
                        if window % 2 == 0:
                            window += 1
                        if polyorder >= window:
                            polyorder = window - 1
                        if len(signal_data) > window:
                            if _savgol_filter is None:
                                raise RuntimeError(
                                    "scipy.signal.savgol_filter unavailable. "
                                    "Install SciPy or skip smoothing.",
                                )
                            processed_df[col] = pd.Series(
                                _savgol_filter(signal_data, window, polyorder),
                                index=signal_data.index,
                            )

            # Apply Resampling
            if settings.get("resample_enabled"):
                resample_rule = settings.get("resample_rule")
                print(f"  Applying resampling with rule: {resample_rule}")
                if resample_rule:
                    before_resample = len(processed_df)
                    processed_df = (
                        processed_df.resample(resample_rule).mean().dropna(how="all")
                    )
                    after_resample = len(processed_df)
                    print(f"  Resampling: {before_resample} -> {after_resample} rows")
            else:
                print("  Resampling disabled")

            print("  Resetting index...")
            processed_df.reset_index(inplace=True)
            print(f"  DataFrame shape after reset: {processed_df.shape}")

            print("  Applying custom variables...")
            processed_df = self._apply_custom_variables(processed_df, time_col)
            print(f"  DataFrame shape after custom vars: {processed_df.shape}")

            # Apply integration if signals are selected
            signals_to_integrate = [
                s
                for s, data in self.integrator_signal_vars.items()
                if data["var"].get()
            ]
            if signals_to_integrate:
                print(f"  Applying integration to: {signals_to_integrate}")
                integration_method = self.integrator_method_var.get()
                processed_df = self._apply_integration(
                    processed_df,
                    time_col,
                    signals_to_integrate,
                    integration_method,
                )
                print(f"  DataFrame shape after integration: {processed_df.shape}")

            # Apply differentiation if signals are selected
            signals_to_differentiate = [
                s for s, data in self.deriv_signal_vars.items() if data["var"].get()
            ]
            if signals_to_differentiate:
                print(f"  Applying differentiation to: {signals_to_differentiate}")
                differentiation_method = self.deriv_method_var.get()
                processed_df = self._apply_differentiation(
                    processed_df,
                    time_col,
                    signals_to_differentiate,
                    differentiation_method,
                )
                print(f"  DataFrame shape after differentiation: {processed_df.shape}")

            if processed_df.empty:
                print("  ERROR: Final DataFrame is empty")
                return None

            print(
                f"  SUCCESS: Returning processed DataFrame with shape {processed_df.shape}",
            )
            return processed_df

        except Exception as e:
            print(f"  EXCEPTION in _process_single_file: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _apply_custom_variables(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Apply custom variables to the dataframe."""
        if not self.custom_vars_list:
            return df

        # Make a copy to avoid modifying original
        df = df.copy()

        for var in self.custom_vars_list:
            try:
                formula = var["formula"]
                name = var["name"]

                # Create a safe evaluation environment
                safe_dict = {}

                # Add all numeric columns to the safe dictionary
                for col in df.columns:
                    if col != time_col and pd.api.types.is_numeric_dtype(df[col]):
                        # Replace signal name in formula
                        if f"[{col}]" in formula:
                            safe_dict[col] = df[col]
                            formula = formula.replace(f"[{col}]", col)

                # Add safe math functions
                import math

                safe_dict.update(
                    {
                        "abs": abs,
                        "min": min,
                        "max": max,
                        "sum": sum,
                        "len": len,
                        "round": round,
                        "sqrt": math.sqrt,
                        "log": math.log,
                        "log10": math.log10,
                        "exp": math.exp,
                        "pow": pow,
                        "sin": math.sin,
                        "cos": math.cos,
                        "tan": math.tan,
                        "pi": math.pi,
                        "e": math.e,
                    },
                )

                # Evaluate the formula safely
                result = eval(formula, {"__builtins__": {}}, safe_dict)
                df[name] = result

            except Exception as e:
                print(f"Error applying custom variable '{var['name']}': {e}")
                df[var["name"]] = np.nan
                messagebox.showwarning(
                    "Custom Variable Error",
                    f"Error in formula for '{var['name']}':\n{e!s}",
                )

        return df

    def _get_resample_rule(self) -> str:
        """Get the resample rule from UI inputs."""
        if not self.resample_var.get():
            return None

        value = self.resample_value_entry.get()
        unit = self.resample_unit_menu.get()

        if not value:
            return None

        try:
            value = float(value)
            if unit == "ms":
                return f"{value}ms"
            if unit == "s":
                return f"{value}s"  # Fixed: use 's' instead of deprecated 'S'
            if unit == "min":
                return f"{value}T"
            if unit == "hr":
                return f"{value}H"
        except ValueError:
            return None

        return None

    def _export_processed_files(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export processed files based on selected format."""
        export_type = self.export_type_var.get()

        if export_type == "CSV (Separate Files)":
            self._export_csv_separate(processed_files)
        elif export_type == "CSV (Compiled)":
            self._export_csv_compiled(processed_files)
        elif export_type == "Excel (Multi-sheet)":
            self._export_excel_multisheet(processed_files)
        elif export_type == "Excel (Separate Files)":
            self._export_excel_separate(processed_files)
        elif export_type == "MAT (Separate Files)":
            self._export_mat_separate(processed_files)
        elif export_type == "MAT (Compiled)":
            self._export_mat_compiled(processed_files)
        elif export_type == "Parquet (Single File)":
            self._export_parquet_single(processed_files)
        elif export_type == "Parquet (Separate Files)":
            self._export_parquet_separate(processed_files)
        elif export_type == "HDF5 (Single File)":
            self._export_hdf5_single(processed_files)
        elif export_type == "HDF5 (Separate Files)":
            self._export_hdf5_separate(processed_files)
        elif export_type == "Feather (Single File)":
            self._export_feather_single(processed_files)
        elif export_type == "Feather (Separate Files)":
            self._export_feather_separate(processed_files)
        elif export_type == "Pickle (Single File)":
            self._export_pickle_single(processed_files)
        elif export_type == "Pickle (Separate Files)":
            self._export_pickle_separate(processed_files)

    def _export_csv_separate(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export each file as a separate CSV."""
        print(f"_export_csv_separate called with {len(processed_files)} files")
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                self.output_directory,
                f"{base_name}_processed.csv",
            )
            print(f"Exporting to: {output_path}")

            final_path = self._check_file_overwrite(output_path)
            if final_path is None:
                print(f"Export cancelled for {base_name}")
                continue

            print(f"Final export path: {final_path}")
            df = self._apply_sorting(df)
            df.to_csv(final_path, index=False)
            exported_count += 1
            print(f"Successfully exported: {final_path}")

        print(f"Export summary: {exported_count} files exported")
        if exported_count > 0:
            print(f"Showing success message for {exported_count} files")
            messagebox.showinfo(
                "Export Success",
                f"Exported {exported_count} files to {self.output_directory}",
            )
        else:
            print("Showing cancelled message")
            messagebox.showinfo("Export Cancelled", "No files were exported.")

    def _export_csv_compiled(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export all files as a single compiled CSV."""
        if not processed_files:
            return
        compiled_df = pd.concat(
            [
                df.assign(Source_File=os.path.splitext(os.path.basename(fp))[0])
                for fp, df in processed_files
            ],
            ignore_index=True,
        )
        compiled_df = self._apply_sorting(compiled_df)

        output_path = os.path.join(self.output_directory, "compiled_processed_data.csv")
        final_path = self._check_file_overwrite(output_path)
        if final_path:
            compiled_df.to_csv(final_path, index=False)
            messagebox.showinfo("Success", f"Exported compiled data to {final_path}")

    def _export_excel_multisheet(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export all files to a single Excel file with multiple sheets."""
        output_path = os.path.join(self.output_directory, "processed_data.xlsx")
        final_path = self._check_file_overwrite(output_path)
        if not final_path:
            return

        with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
            for file_path, df in processed_files:
                sheet_name = os.path.splitext(os.path.basename(file_path))[0][:EXCEL_SHEET_NAME_MAX_LENGTH]
                df = self._apply_sorting(df)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        messagebox.showinfo("Success", f"Exported to Excel file: {final_path}")

    def _export_excel_separate(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export each file as a separate Excel file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                self.output_directory,
                f"{base_name}_processed.xlsx",
            )

            final_path = self._check_file_overwrite(output_path)
            if final_path is None:
                continue

            df = self._apply_sorting(df)
            df.to_excel(final_path, index=False)
            exported_count += 1

        if exported_count > 0:
            messagebox.showinfo(
                "Success",
                f"Exported {exported_count} Excel files to {self.output_directory}",
            )
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_mat_separate(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export each file as a separate MAT file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                self.output_directory,
                f"{base_name}_processed.mat",
            )

            final_path = self._check_file_overwrite(output_path)
            if final_path is None:
                continue

            df = self._apply_sorting(df)
            mat_data = {col: df[col].values for col in df.columns}
            savemat(final_path, mat_data)
            exported_count += 1

        if exported_count > 0:
            messagebox.showinfo(
                "Success",
                f"Exported {exported_count} MAT files to {self.output_directory}",
            )
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_mat_compiled(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export all files as a single compiled MAT file."""
        if not processed_files:
            return
        compiled_df = pd.concat(
            [
                df.assign(Source_File=os.path.splitext(os.path.basename(fp))[0])
                for fp, df in processed_files
            ],
            ignore_index=True,
        )
        compiled_df = self._apply_sorting(compiled_df)

        output_path = os.path.join(self.output_directory, "compiled_processed_data.mat")
        final_path = self._check_file_overwrite(output_path)
        if final_path:
            mat_data = {col: compiled_df[col].values for col in compiled_df.columns}
            savemat(final_path, mat_data)
            messagebox.showinfo(
                "Success",
                f"Exported compiled MAT file to {final_path}",
            )

    def _export_parquet_single(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export all files as a single Parquet file (optimized for large datasets)."""
        if not processed_files:
            return

        # Check bulk mode for column validation
        bulk_mode = getattr(self, "bulk_mode_var", None) and self.bulk_mode_var.get()
        expected_columns = None
        column_mismatches = []

        # Create progress window
        progress_window = ctk.CTkToplevel(self)
        progress_window.title("Converting to Parquet")
        progress_window.geometry("400x200")
        progress_window.resizable(False, False)

        progress_label = ctk.CTkLabel(
            progress_window,
            text="Converting files to Parquet...",
        )
        progress_label.pack(pady=20)

        progress_bar = ctk.CTkProgressBar(progress_window)
        progress_bar.pack(pady=10, padx=20, fill="x")
        progress_bar.set(0)

        status_label = ctk.CTkLabel(progress_window, text="Starting conversion...")
        status_label.pack(pady=10)

        try:
            # Process files in batches
            batch_size = 100
            all_dataframes = []
            total_files = len(processed_files)

            for i in range(0, total_files, batch_size):
                batch_end = min(i + batch_size, total_files)
                batch_files = processed_files[i:batch_end]

                # Update progress
                progress = (i + batch_size) / total_files
                progress_bar.set(min(progress, 1.0))
                status_label.configure(
                    text=f"Processing files {i+1}-{batch_end}/{total_files}",
                )
                progress_window.update()

                batch_dfs = []
                for file_path, df in batch_files:
                    try:
                        # Column validation in bulk mode
                        if bulk_mode and expected_columns is None:
                            expected_columns = df.columns.tolist()
                        elif bulk_mode and expected_columns is not None:
                            current_columns = df.columns.tolist()
                            if current_columns != expected_columns:
                                column_mismatches.append(
                                    {
                                        "file": os.path.basename(file_path),
                                        "expected": expected_columns,
                                        "found": current_columns,
                                    },
                                )

                        # Add source file information
                        df["source_file"] = os.path.basename(file_path)
                        batch_dfs.append(df)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

                if batch_dfs:
                    # Concatenate batch dataframes
                    batch_combined = pd.concat(batch_dfs, ignore_index=True)
                    all_dataframes.append(batch_combined)

                # Clear batch dataframes to free memory
                del batch_dfs
                if "batch_combined" in locals():
                    del batch_combined

            # Final concatenation
            status_label.configure(text="Combining all data...")
            progress_window.update()

            if all_dataframes:
                final_df = pd.concat(all_dataframes, ignore_index=True)
                final_df = self._apply_sorting(final_df)

                # Save to Parquet
                status_label.configure(text="Saving to Parquet file...")
                progress_window.update()

                output_path = os.path.join(
                    self.output_directory,
                    "compiled_processed_data.parquet",
                )
                final_path = self._check_file_overwrite(output_path)
                if final_path:
                    final_df.to_parquet(
                        final_path,
                        engine="pyarrow",
                        compression="snappy",
                    )

                    # Show success message with column mismatch info if any
                    progress_window.destroy()

                    success_message = f"Successfully exported {total_files} files to:\n{final_path}\n\n"
                    success_message += f"Total rows: {len(final_df):,}\n"
                    success_message += f"Total columns: {len(final_df.columns)}"

                    if bulk_mode and column_mismatches:
                        success_message += f"\n\n⚠️ Column mismatches found in {len(column_mismatches)} files:"
                        for mismatch in column_mismatches[
                            :5
                        ]:  # Show first 5 mismatches
                            success_message += (
                                f"\n• {mismatch['file']}: expected "
                                f"{len(mismatch['expected'])} columns, found {len(mismatch['found'])}"
                            )
                        if len(column_mismatches) > 5:
                            success_message += (
                                f"\n• ... and {len(column_mismatches) - 5} more files"
                            )

                    messagebox.showinfo("Success", success_message)
                else:
                    progress_window.destroy()
                    messagebox.showinfo("Cancelled", "Export was cancelled.")
            else:
                progress_window.destroy()
                messagebox.showerror("Error", "No valid data found to export.")

        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Error", f"Error converting files: {e!s}")
            print(f"Conversion error: {e}")
            traceback.print_exc()

    def _export_parquet_separate(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export each file as a separate Parquet file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                self.output_directory,
                f"{base_name}_processed.parquet",
            )

            final_path = self._check_file_overwrite(output_path)
            if final_path is None:
                continue

            df = self._apply_sorting(df)
            df.to_parquet(final_path, engine="pyarrow", compression="snappy")
            exported_count += 1

        if exported_count > 0:
            messagebox.showinfo(
                "Success",
                f"Exported {exported_count} Parquet files to {self.output_directory}",
            )
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_hdf5_single(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export all files as a single HDF5 file."""
        if not processed_files:
            return
        compiled_df = pd.concat(
            [
                df.assign(Source_File=os.path.splitext(os.path.basename(fp))[0])
                for fp, df in processed_files
            ],
            ignore_index=True,
        )
        compiled_df = self._apply_sorting(compiled_df)

        output_path = os.path.join(self.output_directory, "compiled_processed_data.h5")
        final_path = self._check_file_overwrite(output_path)
        if final_path:
            compiled_df.to_hdf(
                final_path,
                key="data",
                mode="w",
                complevel=9,
                complib="blosc",
            )
            messagebox.showinfo(
                "Success",
                f"Exported compiled HDF5 file to {final_path}",
            )

    def _export_hdf5_separate(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export each file as a separate HDF5 file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                self.output_directory,
                f"{base_name}_processed.h5",
            )

            final_path = self._check_file_overwrite(output_path)
            if final_path is None:
                continue

            df = self._apply_sorting(df)
            df.to_hdf(final_path, key="data", mode="w", complevel=9, complib="blosc")
            exported_count += 1

        if exported_count > 0:
            messagebox.showinfo(
                "Success",
                f"Exported {exported_count} HDF5 files to {self.output_directory}",
            )
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_feather_single(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export all files as a single Feather file."""
        if not processed_files:
            return
        compiled_df = pd.concat(
            [
                df.assign(Source_File=os.path.splitext(os.path.basename(fp))[0])
                for fp, df in processed_files
            ],
            ignore_index=True,
        )
        compiled_df = self._apply_sorting(compiled_df)

        output_path = os.path.join(
            self.output_directory,
            "compiled_processed_data.feather",
        )
        final_path = self._check_file_overwrite(output_path)
        if final_path:
            compiled_df.to_feather(final_path, compression="lz4")
            messagebox.showinfo(
                "Success",
                f"Exported compiled Feather file to {final_path}",
            )

    def _export_feather_separate(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export each file as a separate Feather file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                self.output_directory,
                f"{base_name}_processed.feather",
            )

            final_path = self._check_file_overwrite(output_path)
            if final_path is None:
                continue

            df = self._apply_sorting(df)
            df.to_feather(output_path, compression="lz4")
            exported_count += 1

        if exported_count > 0:
            messagebox.showinfo(
                "Success",
                f"Exported {exported_count} Feather files to {self.output_directory}",
            )
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_pickle_single(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export all files as a single Pickle file."""
        if not processed_files:
            return
        compiled_df = pd.concat(
            [
                df.assign(Source_File=os.path.splitext(os.path.basename(fp))[0])
                for fp, df in processed_files
            ],
            ignore_index=True,
        )
        compiled_df = self._apply_sorting(compiled_df)

        output_path = os.path.join(self.output_directory, "compiled_processed_data.pkl")
        final_path = self._check_file_overwrite(output_path)
        if final_path:
            compiled_df.to_pickle(final_path, compression="gzip")
            messagebox.showinfo(
                "Success",
                f"Exported compiled Pickle file to {final_path}",
            )

    def _export_pickle_separate(self, processed_files: dict[str, pd.DataFrame]) -> None:
        """Export each file as a separate Pickle file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                self.output_directory,
                f"{base_name}_processed.pkl",
            )

            final_path = self._check_file_overwrite(output_path)
            if final_path is None:
                continue

            df = self._apply_sorting(df)
            df.to_pickle(final_path, compression="gzip")
            exported_count += 1

        if exported_count > 0:
            messagebox.showinfo(
                "Success",
                f"Exported {exported_count} Pickle files to {self.output_directory}",
            )
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _combine_multiple_files(self, processed_files: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple processed files into a single dataset for time series data."""
        if not processed_files or len(processed_files) <= 1:
            return processed_files

        print(f"\n=== COMBINING {len(processed_files)} FILES INTO SINGLE DATASET ===")

        # Sort files by time to ensure proper chronological order
        sorted_files = []
        for file_path, df in processed_files:
            try:
                # Get the first timestamp from each file
                time_col = df.columns[0]  # Assuming first column is time
                first_time = pd.to_datetime(df[time_col].iloc[0])
                sorted_files.append((file_path, df, first_time))
            except Exception as e:
                print(f"Warning: Could not parse time for {file_path}: {e}")
                # If time parsing fails, use file modification time
                file_time = pd.to_datetime(os.path.getmtime(file_path), unit="s")
                sorted_files.append((file_path, df, file_time))

        # Sort by time
        sorted_files.sort(key=lambda x: x[2])

        # Combine all dataframes
        combined_dfs = []
        for file_path, df, _ in sorted_files:
            # Add source file identifier
            source_name = os.path.splitext(os.path.basename(file_path))[0]
            df_with_source = df.copy()
            df_with_source["Source_File"] = source_name
            combined_dfs.append(df_with_source)

        # Concatenate all dataframes
        combined_df = pd.concat(combined_dfs, ignore_index=True)

        # Sort by time column
        time_col = combined_df.columns[0]
        combined_df = combined_df.sort_values(time_col)

        print(f"Combined dataset shape: {combined_df.shape}")
        print(
            f"Time range: {combined_df[time_col].min()} to {combined_df[time_col].max()}",
        )
        print(f"Files included: {[os.path.basename(fp) for fp, _, _ in sorted_files]}")

        # Return the combined dataset as a single "file"
        combined_file_path = "combined_dataset"
        return [(combined_file_path, combined_df)]

    def _apply_sorting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sorting to the dataframe."""
        sort_col = self.sort_col_menu.get()
        sort_order = self.sort_order_var.get()

        if sort_col and sort_col != "No Sorting" and sort_col in df.columns:
            ascending = sort_order == "Ascending"
            df = df.sort_values(by=sort_col, ascending=ascending)

        return df

    def create_plotting_tab(self, tab: ctk.CTkFrame) -> None:
        """Create the plotting and analysis tab with all advanced features."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Top control bar
        plot_control_frame = ctk.CTkFrame(tab)
        plot_control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        plot_control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(plot_control_frame, text="File to Plot:").grid(
            row=0,
            column=0,
            padx=(10, 5),
            pady=10,
        )
        self.plot_file_menu = ctk.CTkOptionMenu(
            plot_control_frame,
            values=["Select a file..."],
            command=self.on_plot_file_select,
        )
        self.plot_file_menu.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        ctk.CTkLabel(plot_control_frame, text="X-Axis:").grid(
            row=0,
            column=2,
            padx=(10, 5),
            pady=10,
        )
        self.plot_xaxis_menu = ctk.CTkOptionMenu(
            plot_control_frame,
            values=["default time"],
            command=lambda e: self.update_plot(),
        )
        self.plot_xaxis_menu.grid(row=0, column=3, padx=5, pady=10, sticky="ew")

        # Load Plot Configuration dropdown
        ctk.CTkLabel(plot_control_frame, text="Load Config:").grid(
            row=0,
            column=4,
            padx=(10, 5),
            pady=10,
        )
        self.load_plot_config_menu = ctk.CTkOptionMenu(
            plot_control_frame,
            values=["No saved plots"],
            command=self._on_load_plot_config_select,
        )
        self.load_plot_config_menu.grid(row=0, column=5, padx=5, pady=10, sticky="ew")

        # Save Plot Configuration button
        ctk.CTkButton(
            plot_control_frame,
            text="Save Plot Config",
            height=35,
            command=self._save_current_plot_config,
        ).grid(row=0, column=6, padx=10, pady=10)

        # Modify Plot Configuration button
        ctk.CTkButton(
            plot_control_frame,
            text="Modify Plot Config",
            height=35,
            command=self._modify_plot_config,
        ).grid(row=0, column=7, padx=10, pady=10)

        # Manual Plot Update button for debugging
        ctk.CTkButton(
            plot_control_frame,
            text="🔄 Update Plot",
            height=35,
            command=lambda: self.update_plot(),
        ).grid(row=0, column=8, padx=5, pady=10)

        # Main content frame for splitter
        plot_main_frame = ctk.CTkFrame(tab)
        plot_main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        plot_main_frame.grid_rowconfigure(0, weight=1)
        plot_main_frame.grid_columnconfigure(0, weight=1)

        def create_plot_left_content(left_panel: ctk.CTkFrame) -> None:
            """Create the left panel content for plotting with all advanced features"""
            left_panel.grid_rowconfigure(0, weight=1)
            left_panel.grid_columnconfigure(0, weight=1)

            # The scrollable area for controls (removed title)
            plot_left_panel = ctk.CTkScrollableFrame(left_panel)
            plot_left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            plot_left_panel.grid_columnconfigure(0, weight=1)

            # Plot signal selection
            plot_signal_select_frame = ctk.CTkFrame(plot_left_panel)
            plot_signal_select_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
            plot_signal_select_frame.grid_columnconfigure(0, weight=1)

            self.plot_search_entry = ctk.CTkEntry(
                plot_signal_select_frame,
                placeholder_text="Search plot signals...",
            )
            self.plot_search_entry.grid(
                row=0,
                column=0,
                columnspan=4,
                sticky="ew",
                padx=5,
                pady=5,
            )
            self.plot_search_entry.bind("<KeyRelease>", self._filter_plot_signals)

            # Checkbox to limit plotting signals to only those selected for processing
            self.limit_plot_signals_var = tk.BooleanVar(value=False)
            limit_plot_checkbox = ctk.CTkCheckBox(
                plot_signal_select_frame,
                text="Limit to processing signals only",
                variable=self.limit_plot_signals_var,
                command=self._on_limit_plot_signals_changed,
            )
            limit_plot_checkbox.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="w",
                padx=5,
                pady=2,
            )

            ctk.CTkButton(
                plot_signal_select_frame,
                text="All",
                command=self._plot_select_all,
            ).grid(row=2, column=0, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(
                plot_signal_select_frame,
                text="None",
                command=self._plot_select_none,
            ).grid(row=2, column=1, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(
                plot_signal_select_frame,
                text="Show Selected",
                command=self._show_selected_signals,
            ).grid(row=2, column=2, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(
                plot_signal_select_frame,
                text="X",
                width=28,
                command=self._plot_clear_search,
            ).grid(row=2, column=3, sticky="w", padx=2, pady=5)

            self.plot_signal_frame = ctk.CTkScrollableFrame(
                plot_left_panel,
                label_text="Signals to Plot",
                height=150,
            )
            self.plot_signal_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

            # Bind mouse wheel to the signals frame for proper scrolling
            self._bind_mousewheel_to_frame(self.plot_signal_frame)

            # Filter preview - MOVED ABOVE PLOT APPEARANCE
            plot_filter_frame = ctk.CTkFrame(plot_left_panel)
            plot_filter_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
            plot_filter_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                plot_filter_frame,
                text="Filter Preview",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            self.plot_filter_type = ctk.StringVar(value="None")
            self.plot_filter_menu = ctk.CTkOptionMenu(
                plot_filter_frame,
                variable=self.plot_filter_type,
                values=self.filter_names,
                command=self._update_plot_filter_ui,
            )
            self.plot_filter_menu.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

            # Filter parameter frames
            time_units = ["ms", "s", "min", "hr"]
            (self.plot_ma_frame, self.plot_ma_value_entry, self.plot_ma_unit_menu) = (
                self._create_ma_param_frame(plot_filter_frame, time_units)
            )
            (
                self.plot_bw_frame,
                self.plot_bw_order_entry,
                self.plot_bw_cutoff_entry,
            ) = self._create_bw_param_frame(plot_filter_frame)
            (self.plot_median_frame, self.plot_median_kernel_entry) = (
                self._create_median_param_frame(plot_filter_frame)
            )
            (
                self.plot_hampel_frame,
                self.plot_hampel_window_entry,
                self.plot_hampel_threshold_entry,
            ) = self._create_hampel_param_frame(plot_filter_frame)
            (
                self.plot_zscore_frame,
                self.plot_zscore_threshold_entry,
                self.plot_zscore_method_menu,
            ) = self._create_zscore_param_frame(plot_filter_frame)
            (
                self.plot_savgol_frame,
                self.plot_savgol_window_entry,
                self.plot_savgol_polyorder_entry,
            ) = self._create_savgol_param_frame(plot_filter_frame)
            self._update_plot_filter_ui("None")

            # Show both raw and filtered signals option (moved below parameter frames)
            self.show_both_signals_var = tk.BooleanVar(value=False)
            show_both_checkbox = ctk.CTkCheckBox(
                plot_filter_frame,
                text="Show both raw and filtered signals",
                variable=self.show_both_signals_var,
                command=self._on_plot_setting_change,
            )
            show_both_checkbox.grid(row=10, column=0, sticky="w", padx=10, pady=5)

            # Multiple filter comparison option
            self.compare_filters_var = tk.BooleanVar(value=False)
            compare_filters_checkbox = ctk.CTkCheckBox(
                plot_filter_frame,
                text="Compare multiple filters",
                variable=self.compare_filters_var,
                command=self._on_plot_setting_change,
            )
            compare_filters_checkbox.grid(row=11, column=0, sticky="w", padx=10, pady=5)

            # Second filter for comparison
            ctk.CTkLabel(plot_filter_frame, text="Compare with filter:").grid(
                row=12,
                column=0,
                sticky="w",
                padx=10,
                pady=(10, 0),
            )
            self.compare_filter_type = ctk.StringVar(value="None")
            self.compare_filter_menu = ctk.CTkOptionMenu(
                plot_filter_frame,
                variable=self.compare_filter_type,
                values=self.filter_names,
                command=self._update_compare_filter_ui,
            )
            self.compare_filter_menu.grid(
                row=13,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )

            # Second filter parameter frames (initially hidden)
            (
                self.compare_ma_frame,
                self.compare_ma_value_entry,
                self.compare_ma_unit_menu,
            ) = self._create_ma_param_frame(plot_filter_frame, time_units)
            (
                self.compare_bw_frame,
                self.compare_bw_order_entry,
                self.compare_bw_cutoff_entry,
            ) = self._create_bw_param_frame(plot_filter_frame)
            (self.compare_median_frame, self.compare_median_kernel_entry) = (
                self._create_median_param_frame(plot_filter_frame)
            )
            (
                self.compare_hampel_frame,
                self.compare_hampel_window_entry,
                self.compare_hampel_threshold_entry,
            ) = self._create_hampel_param_frame(plot_filter_frame)
            (
                self.compare_zscore_frame,
                self.compare_zscore_threshold_entry,
                self.compare_zscore_method_menu,
            ) = self._create_zscore_param_frame(plot_filter_frame)
            (
                self.compare_savgol_frame,
                self.compare_savgol_window_entry,
                self.compare_savgol_polyorder_entry,
            ) = self._create_savgol_param_frame(plot_filter_frame)

            # Auto-zoom controls
            auto_zoom_frame = ctk.CTkFrame(plot_filter_frame)
            auto_zoom_frame.grid(row=20, column=0, sticky="ew", padx=10, pady=5)
            auto_zoom_frame.grid_columnconfigure(0, weight=1)
            auto_zoom_frame.grid_columnconfigure(1, weight=1)

            self.auto_zoom_var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                auto_zoom_frame,
                text="Auto-zoom on changes",
                variable=self.auto_zoom_var,
            ).grid(row=0, column=0, sticky="w", padx=5, pady=2)
            ctk.CTkButton(
                auto_zoom_frame,
                text="Fit to Data",
                command=self._auto_fit_plot,
                width=100,
            ).grid(row=0, column=1, sticky="e", padx=5, pady=2)

            ctk.CTkButton(
                plot_filter_frame,
                text="Preview Filter/s",
                command=self.update_plot,
            ).grid(row=21, column=0, sticky="ew", padx=10, pady=5)
            ctk.CTkButton(
                plot_filter_frame,
                text="Copy Settings to Processing Tab",
                command=self._copy_plot_settings_to_processing,
            ).grid(row=22, column=0, sticky="ew", padx=10, pady=5)

            # Plot appearance controls
            appearance_frame = ctk.CTkFrame(plot_left_panel)
            appearance_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
            appearance_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                appearance_frame,
                text="Plot Appearance",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkLabel(appearance_frame, text="Chart Type:").grid(
                row=1,
                column=0,
                sticky="w",
                padx=10,
            )
            self.plot_type_var = ctk.StringVar(value="Line Only")
            plot_type_menu = ctk.CTkOptionMenu(
                appearance_frame,
                variable=self.plot_type_var,
                values=["Line with Markers", "Line Only", "Markers Only (Scatter)"],
                command=self._on_plot_setting_change,
            )
            plot_type_menu.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

            self.plot_title_entry = ctk.CTkEntry(
                appearance_frame,
                placeholder_text="Plot Title",
            )
            self.plot_title_entry.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
            self.plot_title_entry.bind("<Return>", self._on_plot_setting_change)
            self.plot_title_entry.bind("<FocusOut>", self._on_plot_setting_change)
            # Force placeholder to show immediately
            self.plot_title_entry.delete(0, "end")
            self.plot_title_entry.insert(0, "")

            self.plot_xlabel_entry = ctk.CTkEntry(
                appearance_frame,
                placeholder_text="X-Axis Label",
            )
            self.plot_xlabel_entry.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
            self.plot_xlabel_entry.bind("<Return>", self._on_plot_setting_change)
            self.plot_xlabel_entry.bind("<FocusOut>", self._on_plot_setting_change)
            # Force placeholder to show immediately
            self.plot_xlabel_entry.delete(0, "end")
            self.plot_xlabel_entry.insert(0, "")

            self.plot_ylabel_entry = ctk.CTkEntry(
                appearance_frame,
                placeholder_text="Y-Axis Label",
            )
            self.plot_ylabel_entry.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
            self.plot_ylabel_entry.bind("<Return>", self._on_plot_setting_change)
            self.plot_ylabel_entry.bind("<FocusOut>", self._on_plot_setting_change)
            # Force placeholder to show immediately
            self.plot_ylabel_entry.delete(0, "end")
            self.plot_ylabel_entry.insert(0, "")

            # Color scheme controls
            ctk.CTkLabel(appearance_frame, text="Color Scheme:").grid(
                row=6,
                column=0,
                sticky="w",
                padx=10,
                pady=(10, 0),
            )
            self.color_scheme_var = ctk.StringVar(value="Auto (Matplotlib)")
            color_schemes = [
                "Auto (Matplotlib)",
                "Viridis",
                "Plasma",
                "Cool",
                "Warm",
                "Rainbow",
                "Custom Colors",
            ]
            color_scheme_menu = ctk.CTkOptionMenu(
                appearance_frame,
                variable=self.color_scheme_var,
                values=color_schemes,
                command=self._on_color_scheme_change,
            )
            color_scheme_menu.grid(row=6, column=1, sticky="ew", padx=10, pady=(10, 0))

            # Custom Colors Frame (initially hidden)
            self.custom_colors_frame = ctk.CTkFrame(appearance_frame)
            self.custom_colors_frame.grid(
                row=7,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=10,
                pady=5,
            )
            self.custom_colors_frame.grid_remove()  # Initially hidden
            self.custom_colors_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                self.custom_colors_frame,
                text="Custom Colors:",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

            # Scrollable frame for color buttons
            self.colors_scroll_frame = ctk.CTkScrollableFrame(
                self.custom_colors_frame,
                height=80,
            )
            self.colors_scroll_frame.grid(
                row=1,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=10,
                pady=5,
            )

            # Buttons for color management
            colors_buttons_frame = ctk.CTkFrame(
                self.custom_colors_frame,
                fg_color="transparent",
            )
            colors_buttons_frame.grid(
                row=2,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=10,
                pady=5,
            )
            colors_buttons_frame.grid_columnconfigure(0, weight=1)
            colors_buttons_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkButton(
                colors_buttons_frame,
                text="Add Color",
                command=self._add_custom_color,
            ).grid(row=0, column=0, padx=5, sticky="ew")
            ctk.CTkButton(
                colors_buttons_frame,
                text="Reset to Default",
                command=self._reset_custom_colors,
            ).grid(row=0, column=1, padx=5, sticky="ew")

            # Initialize custom colors display
            self._update_custom_colors_display()

            # Line width control
            ctk.CTkLabel(appearance_frame, text="Line Width:").grid(
                row=8,
                column=0,
                sticky="w",
                padx=10,
                pady=(5, 0),
            )
            self.line_width_var = ctk.StringVar(value="1.0")
            line_widths = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]
            line_width_menu = ctk.CTkOptionMenu(
                appearance_frame,
                variable=self.line_width_var,
                values=line_widths,
                command=self._on_plot_setting_change,
            )
            line_width_menu.grid(row=8, column=1, sticky="ew", padx=10, pady=(5, 0))

            # Legend placement control
            ctk.CTkLabel(appearance_frame, text="Legend Position:").grid(
                row=9,
                column=0,
                sticky="w",
                padx=10,
                pady=(5, 0),
            )
            self.legend_position_var = ctk.StringVar(value="best")
            legend_positions = [
                "best",
                "upper right",
                "upper left",
                "lower left",
                "lower right",
                "right",
                "center left",
                "center right",
                "lower center",
                "upper center",
                "center",
                "outside right",
            ]
            legend_position_menu = ctk.CTkOptionMenu(
                appearance_frame,
                variable=self.legend_position_var,
                values=legend_positions,
                command=self._on_plot_setting_change,
            )
            legend_position_menu.grid(
                row=9,
                column=1,
                sticky="ew",
                padx=10,
                pady=(5, 0),
            )

            # Custom Legend Labels control
            legend_header_frame = ctk.CTkFrame(appearance_frame, fg_color="transparent")
            legend_header_frame.grid(
                row=10,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=10,
                pady=(10, 0),
            )
            legend_header_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                legend_header_frame,
                text="Custom Legend Labels:",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkButton(
                legend_header_frame,
                text="?",
                width=25,
                height=25,
                command=self._show_legend_guide,
            ).grid(row=0, column=1, sticky="e", padx=(5, 0))

            ctk.CTkLabel(
                appearance_frame,
                text="For subscripts use: $H_2O$, $CO_2$, $v_{max}$ (LaTeX syntax)",
                font=ctk.CTkFont(size=10),
            ).grid(row=11, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 5))

            # Scrollable frame for legend customization
            self.legend_frame = ctk.CTkScrollableFrame(appearance_frame, height=120)
            self.legend_frame.grid(
                row=12,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=10,
                pady=5,
            )

            ctk.CTkButton(
                appearance_frame,
                text="Refresh Legend Entries",
                command=self._refresh_legend_entries,
            ).grid(row=13, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

            # Custom legend entries dictionary - only initialize if not already exists
            if not hasattr(self, "custom_legend_entries"):
                self.custom_legend_entries = {}

            # Trendline controls
            trend_frame = ctk.CTkFrame(plot_left_panel)
            trend_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
            trend_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                trend_frame,
                text="Trendline",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=5)

            ctk.CTkLabel(trend_frame, text="Signal:").grid(
                row=1,
                column=0,
                sticky="w",
                padx=10,
                pady=(5, 0),
            )
            self.trendline_signal_var = ctk.StringVar(value="Select signal...")
            self.trendline_signal_menu = ctk.CTkOptionMenu(
                trend_frame,
                variable=self.trendline_signal_var,
                values=["Select signal..."],
                command=self._on_plot_setting_change,
            )
            self.trendline_signal_menu.grid(
                row=2,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )

            ctk.CTkLabel(trend_frame, text="Type:").grid(
                row=3,
                column=0,
                sticky="w",
                padx=10,
                pady=(5, 0),
            )
            self.trendline_type_var = ctk.StringVar(value="None")
            trendline_type_menu = ctk.CTkOptionMenu(
                trend_frame,
                variable=self.trendline_type_var,
                values=["None", "Linear", "Exponential", "Power", "Polynomial"],
                command=self._on_plot_setting_change,
            )
            trendline_type_menu.grid(row=4, column=0, sticky="ew", padx=10, pady=5)

            self.poly_order_entry = ctk.CTkEntry(
                trend_frame,
                placeholder_text="Polynomial Order (2-6)",
            )
            self.poly_order_entry.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
            self.poly_order_entry.bind("<Return>", self._on_plot_setting_change)
            self.poly_order_entry.bind("<FocusOut>", self._on_plot_setting_change)

            # Trendline time window controls
            ctk.CTkLabel(trend_frame, text="Time Window:").grid(
                row=6,
                column=0,
                sticky="w",
                padx=10,
                pady=(5, 0),
            )

            # Time window selection method
            self.trendline_window_mode = ctk.StringVar(value="Full Range")
            trendline_window_menu = ctk.CTkOptionMenu(
                trend_frame,
                variable=self.trendline_window_mode,
                values=["Full Range", "Manual Entry", "Visual Selection"],
                command=self._on_trendline_window_mode_change,
            )
            trendline_window_menu.grid(row=7, column=0, sticky="ew", padx=10, pady=5)

            # Manual time window frame (initially hidden)
            self.trendline_manual_frame = ctk.CTkFrame(trend_frame)
            self.trendline_manual_frame.grid(
                row=8,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )
            self.trendline_manual_frame.grid_remove()  # Hide initially
            self.trendline_manual_frame.grid_columnconfigure(0, weight=1)
            self.trendline_manual_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(self.trendline_manual_frame, text="Start:").grid(
                row=0,
                column=0,
                sticky="w",
                padx=5,
                pady=2,
            )
            self.trendline_start_entry = ctk.CTkEntry(
                self.trendline_manual_frame,
                placeholder_text="Start time",
            )
            self.trendline_start_entry.grid(
                row=0,
                column=1,
                sticky="ew",
                padx=5,
                pady=2,
            )
            self.trendline_start_entry.bind("<Return>", self._on_plot_setting_change)

            ctk.CTkLabel(self.trendline_manual_frame, text="End:").grid(
                row=1,
                column=0,
                sticky="w",
                padx=5,
                pady=2,
            )
            self.trendline_end_entry = ctk.CTkEntry(
                self.trendline_manual_frame,
                placeholder_text="End time",
            )
            self.trendline_end_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
            self.trendline_end_entry.bind("<Return>", self._on_plot_setting_change)

            # Visual selection controls
            self.trendline_visual_frame = ctk.CTkFrame(trend_frame)
            self.trendline_visual_frame.grid(
                row=9,
                column=0,
                sticky="ew",
                padx=10,
                pady=5,
            )
            self.trendline_visual_frame.grid_remove()  # Hide initially
            self.trendline_visual_frame.grid_columnconfigure(0, weight=1)

            self.trendline_select_button = ctk.CTkButton(
                self.trendline_visual_frame,
                text="Select Time Window on Plot",
                command=self._start_trendline_selection,
            )
            self.trendline_select_button.grid(
                row=0,
                column=0,
                sticky="ew",
                padx=5,
                pady=5,
            )

            self.trendline_selected_range = ctk.CTkLabel(
                self.trendline_visual_frame,
                text="No range selected",
            )
            self.trendline_selected_range.grid(
                row=1,
                column=0,
                sticky="ew",
                padx=5,
                pady=2,
            )

            self.trendline_textbox = ctk.CTkTextbox(trend_frame, height=70)
            self.trendline_textbox.grid(row=10, column=0, sticky="ew", padx=10, pady=5)

            # Time range controls
            time_range_frame = ctk.CTkFrame(plot_left_panel)
            time_range_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
            time_range_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                time_range_frame,
                text="Plot Time Range",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkLabel(time_range_frame, text="Start Time (HH:MM:SS):").grid(
                row=1,
                column=0,
                sticky="w",
                padx=10,
            )
            self.plotting_start_time_entry = ctk.CTkEntry(
                time_range_frame,
                placeholder_text="e.g., 09:30:00",
            )
            self.plotting_start_time_entry.grid(
                row=2,
                column=0,
                sticky="ew",
                padx=10,
                pady=2,
            )
            ctk.CTkLabel(time_range_frame, text="End Time (HH:MM:SS):").grid(
                row=3,
                column=0,
                sticky="w",
                padx=10,
            )
            self.plotting_end_time_entry = ctk.CTkEntry(
                time_range_frame,
                placeholder_text="e.g., 17:00:00",
            )
            self.plotting_end_time_entry.grid(
                row=4,
                column=0,
                sticky="ew",
                padx=10,
                pady=2,
            )
            ctk.CTkButton(
                time_range_frame,
                text="Apply Time Range to Plot",
                command=self._apply_plot_time_range,
            ).grid(row=5, column=0, sticky="ew", padx=10, pady=5)
            ctk.CTkButton(
                time_range_frame,
                text="Reset Plot Range",
                command=self._reset_plot_range,
            ).grid(row=6, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkButton(
                time_range_frame,
                text="Save Current View",
                command=self._save_current_plot_view,
            ).grid(row=7, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkButton(
                time_range_frame,
                text="Copy Current View to Processing",
                command=self._copy_current_view_to_processing,
            ).grid(row=8, column=0, sticky="ew", padx=10, pady=2)

            # Export controls
            export_chart_frame = ctk.CTkFrame(plot_left_panel)
            export_chart_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
            export_chart_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                export_chart_frame,
                text="Export Chart",
                font=ctk.CTkFont(weight="bold"),
            ).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkButton(
                export_chart_frame,
                text="Save as PNG/PDF",
                command=self._export_chart_image,
            ).grid(row=1, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkButton(
                export_chart_frame,
                text="Export to Excel with Chart",
                command=self._export_chart_excel,
            ).grid(row=2, column=0, sticky="ew", padx=10, pady=2)

        def create_plot_right_content(right_panel: ctk.CTkFrame) -> None:
            """Create the right panel content for plotting"""
            right_panel.grid_rowconfigure(1, weight=1)
            right_panel.grid_columnconfigure(0, weight=1)

            # The plot canvas
            plot_canvas_frame = ctk.CTkFrame(right_panel)
            plot_canvas_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            plot_canvas_frame.grid_rowconfigure(1, weight=1)
            plot_canvas_frame.grid_columnconfigure(0, weight=1)

            self.plot_fig = Figure(figsize=(5, 4), dpi=100)
            self.plot_ax = self.plot_fig.add_subplot(111)
            self.plot_fig.tight_layout()

            self.plot_canvas = FigureCanvasTkAgg(
                self.plot_fig,
                master=plot_canvas_frame,
            )
            self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

            # Initialize plot with welcome message
            self.plot_ax.text(
                0.5,
                0.5,
                "Plotting ready - select a file",
                ha="center",
                va="center",
                transform=self.plot_ax.transAxes,
            )
            self.plot_ax.set_title("Select a file to begin plotting")
            self.plot_canvas.draw()

            toolbar = NavigationToolbar2Tk(
                self.plot_canvas,
                plot_canvas_frame,
                pack_toolbar=False,
            )
            toolbar.grid(row=0, column=0, sticky="ew")

            # Store toolbar reference for custom functionality
            self.plot_toolbar = toolbar

            # Initialize zoom state storage
            self.saved_zoom_state = None

            # Add custom zoom controls
            zoom_frame = ctk.CTkFrame(plot_canvas_frame)
            zoom_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
            zoom_frame.grid_columnconfigure(0, weight=1)
            zoom_frame.grid_columnconfigure(1, weight=1)
            zoom_frame.grid_columnconfigure(2, weight=1)
            zoom_frame.grid_columnconfigure(3, weight=1)

            save_zoom_btn = ctk.CTkButton(
                zoom_frame,
                text="Save Zoom State",
                command=self._save_zoom_state,
            )
            save_zoom_btn.grid(row=0, column=0, padx=2, pady=2, sticky="ew")

            restore_zoom_btn = ctk.CTkButton(
                zoom_frame,
                text="Restore Zoom",
                command=self._restore_zoom_state,
            )
            restore_zoom_btn.grid(row=0, column=1, padx=2, pady=2, sticky="ew")

            zoom_in_btn = ctk.CTkButton(
                zoom_frame,
                text="Zoom In 25%",
                command=self._zoom_in_25,
            )
            zoom_in_btn.grid(row=0, column=2, padx=2, pady=2, sticky="ew")

            zoom_out_btn = ctk.CTkButton(
                zoom_frame,
                text="Zoom Out 25%",
                command=self._zoom_out_25,
            )
            zoom_out_btn.grid(row=0, column=3, padx=2, pady=2, sticky="ew")

        # Create splitter for plotting tab
        splitter_frame = self._create_splitter(
            plot_main_frame,
            create_plot_left_content,
            create_plot_right_content,
            "plotting_left_width",
            400,
        )
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def create_plots_list_tab(self, tab: ctk.CTkFrame) -> None:
        """Create the plots list tab."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(
            header_frame,
            text="Plots List Manager",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left", padx=10, pady=10)

        # Main content
        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Create splitter
        splitter_frame = self._create_splitter(
            main_frame,
            self._create_plots_list_left,
            self._create_plots_list_right,
            "plots_list_left_width",
            300,
        )
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def _create_plots_list_left(self, left_panel: ctk.CTkFrame) -> None:
        """Create left panel for plots list."""
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        # Plot configuration frame
        config_frame = ctk.CTkFrame(left_panel)
        config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        config_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            config_frame,
            text="Plot Configuration",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(config_frame, text="Plot Name:").grid(
            row=1,
            column=0,
            padx=10,
            pady=2,
            sticky="w",
        )
        self.plot_name_entry = ctk.CTkEntry(
            config_frame,
            placeholder_text="e.g., Temperature Analysis",
        )
        self.plot_name_entry.grid(row=2, column=0, padx=10, pady=2, sticky="ew")

        ctk.CTkLabel(config_frame, text="Description:").grid(
            row=3,
            column=0,
            padx=10,
            pady=2,
            sticky="w",
        )
        self.plot_desc_entry = ctk.CTkEntry(
            config_frame,
            placeholder_text="Brief description of this plot",
        )
        self.plot_desc_entry.grid(row=4, column=0, padx=10, pady=2, sticky="ew")

        # Signal selection
        ctk.CTkLabel(config_frame, text="Signals to Include:").grid(
            row=5,
            column=0,
            padx=10,
            pady=(10, 2),
            sticky="w",
        )
        self.plots_signals_frame = ctk.CTkScrollableFrame(config_frame, height=100)
        self.plots_signals_frame.grid(row=6, column=0, padx=10, pady=2, sticky="ew")

        # Bind mouse wheel to the plots signals frame
        self._bind_mousewheel_to_frame(self.plots_signals_frame)

        # Time range
        ctk.CTkLabel(config_frame, text="Time Range (HH:MM:SS):").grid(
            row=7,
            column=0,
            padx=10,
            pady=(10, 2),
            sticky="w",
        )

        time_range_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        time_range_frame.grid(row=8, column=0, padx=10, pady=2, sticky="ew")
        time_range_frame.grid_columnconfigure(0, weight=1)
        time_range_frame.grid_columnconfigure(1, weight=1)

        self.plots_list_start_time_entry = ctk.CTkEntry(
            time_range_frame,
            placeholder_text="Start time",
        )
        self.plots_list_start_time_entry.grid(
            row=0,
            column=0,
            padx=(0, 5),
            pady=2,
            sticky="ew",
        )

        self.plots_list_end_time_entry = ctk.CTkEntry(
            time_range_frame,
            placeholder_text="End time",
        )
        self.plots_list_end_time_entry.grid(
            row=0,
            column=1,
            padx=(5, 0),
            pady=2,
            sticky="ew",
        )

        # Buttons
        button_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        button_frame.grid(row=9, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkButton(
            button_frame,
            text="Add to List",
            command=self._add_plot_to_list,
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            button_frame,
            text="Update Selected",
            command=self._update_selected_plot,
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            button_frame,
            text="Clear Form",
            command=self._clear_plot_form,
        ).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Plots list
        list_frame = ctk.CTkFrame(left_panel)
        list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            list_frame,
            text="Saved Plots",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.plots_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            font=("Arial", 10),
        )
        self.plots_listbox.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.plots_listbox.bind("<<ListboxSelect>>", self._on_plot_select)

        # List buttons
        list_button_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        list_button_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkButton(
            list_button_frame,
            text="Load Selected",
            command=self._load_selected_plot,
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            list_button_frame,
            text="Delete Selected",
            command=self._delete_selected_plot,
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            list_button_frame,
            text="Clear All",
            command=self._clear_all_plots,
        ).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

    def _create_plots_list_right(self, right_panel: ctk.CTkFrame) -> None:
        """Create right panel for plots list."""
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        # Plot preview frame
        preview_frame = ctk.CTkFrame(right_panel)
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            preview_frame,
            text="Plot Preview",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Preview canvas
        self.preview_fig = Figure(figsize=(6, 4), dpi=100)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_fig.tight_layout()

        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="nsew",
        )

        # Preview buttons
        preview_button_frame = ctk.CTkFrame(preview_frame, fg_color="transparent")
        preview_button_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkButton(
            preview_button_frame,
            text="Generate Preview",
            command=self._generate_plot_preview,
        ).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            preview_button_frame,
            text="Export All Plots",
            command=self._export_all_plots,
        ).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def create_dat_import_tab(self, tab: ctk.CTkFrame) -> None:
        """Create the DAT file import tab."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(
            header_frame,
            text="DAT File Import",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left", padx=10, pady=10)

        # Main content
        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Create splitter
        splitter_frame = self._create_splitter(
            main_frame,
            self._create_dat_import_left,
            self._create_dat_import_right,
            "dat_import_left_width",
            300,
        )
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def _create_dat_import_left(self, left_panel: ctk.CTkFrame) -> None:
        """Create left panel for DAT import."""
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        # File selection frame
        file_frame = ctk.CTkFrame(left_panel)
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        file_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            file_frame,
            text="File Selection",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        ctk.CTkButton(
            file_frame,
            text="Select Tag File (.dbf)",
            command=self._select_tag_file,
        ).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            file_frame,
            text="Select Data File (.dat)",
            command=self._select_data_file,
        ).grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.tag_file_label = ctk.CTkLabel(
            file_frame,
            text="No tag file selected",
            font=ctk.CTkFont(size=11),
        )
        self.tag_file_label.grid(row=3, column=0, padx=10, pady=2, sticky="w")

        self.data_file_label = ctk.CTkLabel(
            file_frame,
            text="No data file selected",
            font=ctk.CTkFont(size=11),
        )
        self.data_file_label.grid(row=4, column=0, padx=10, pady=2, sticky="w")

        # Import settings frame
        settings_frame = ctk.CTkFrame(left_panel)
        settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            settings_frame,
            text="Import Settings",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(settings_frame, text="Tag Delimiter:").grid(
            row=1,
            column=0,
            padx=10,
            pady=2,
            sticky="w",
        )
        ctk.CTkOptionMenu(
            settings_frame,
            variable=self.tag_delimiter_var,
            values=["newline", "comma", "semicolon", "tab"],
        ).grid(row=2, column=0, padx=10, pady=2, sticky="ew")

        # Tag selection frame
        tag_frame = ctk.CTkFrame(settings_frame)
        tag_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        tag_frame.grid_columnconfigure(0, weight=1)
        tag_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            tag_frame,
            text="Select Tags to Import:",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.tags_listbox = tk.Listbox(
            tag_frame,
            selectmode=tk.MULTIPLE,
            font=("Arial", 10),
        )
        self.tags_listbox.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Import button
        ctk.CTkButton(
            settings_frame,
            text="Import Selected Tags",
            command=self._import_selected_tags,
        ).grid(row=4, column=0, padx=10, pady=10, sticky="ew")

    def _create_dat_import_right(self, right_panel: ctk.CTkFrame) -> None:
        """Create right panel for DAT import."""
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        # Preview frame
        preview_frame = ctk.CTkFrame(right_panel)
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            preview_frame,
            text="Import Preview",
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.import_preview_text = ctk.CTkTextbox(preview_frame, height=200)
        self.import_preview_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

    def _load_layout_config(self) -> dict[str, Any]:
        """Load layout configuration from file."""
        try:
            if os.path.exists(self.layout_config_file):
                with open(self.layout_config_file) as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading layout config: {e}")
        return {}

    def _save_layout_config(self) -> None:
        """Save layout configuration to file."""
        try:
            # Get current window dimensions
            self.layout_data["window_width"] = self.winfo_width()
            self.layout_data["window_height"] = self.winfo_height()

            # Save splitter positions
            for splitter_key, splitter in self.splitters.items():
                if hasattr(splitter, "winfo_width"):
                    self.layout_data[splitter_key] = splitter.winfo_width()

            with open(self.layout_config_file, "w") as f:
                json.dump(self.layout_data, f, indent=2)
        except Exception as e:
            print(f"Error saving layout config: {e}")

    def _create_splitter(
        self,
        parent: ctk.CTkFrame,
        left_creator: callable,
        right_creator: callable,
        splitter_key: str,
        default_left_width: int,
    ) -> ctk.CTkFrame:
        """Create a splitter with left and right panels."""
        splitter_frame = ctk.CTkFrame(parent)
        # Make the right panel expandable rather than the splitter handle
        splitter_frame.grid_columnconfigure(2, weight=1)
        splitter_frame.grid_rowconfigure(0, weight=1)

        # Get saved width or use default
        left_width = self.layout_data.get(splitter_key, default_left_width)

        # Create left panel
        left_panel = ctk.CTkFrame(splitter_frame, width=left_width)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(5, 0))
        left_panel.grid_propagate(False)
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(0, weight=1)
        left_creator(left_panel)

        # Create splitter handle with better visual feedback
        splitter_handle = ctk.CTkFrame(splitter_frame, width=8, fg_color="#666666")
        splitter_handle.grid(row=0, column=1, sticky="ns", padx=1)

        # Bind events for dragging
        splitter_handle.bind(
            "<Enter>",
            lambda e, h=splitter_handle: self._on_splitter_enter(e, h),
        )
        splitter_handle.bind(
            "<Leave>",
            lambda e, h=splitter_handle: self._on_splitter_leave(e, h),
        )
        splitter_handle.bind(
            "<Button-1>",
            lambda e, h=splitter_handle: self._start_splitter_drag(
                e,
                h,
                left_panel,
                splitter_key,
            ),
        )
        splitter_handle.bind(
            "<B1-Motion>",
            lambda e, h=splitter_handle: self._drag_splitter(
                e,
                h,
                left_panel,
                splitter_key,
            ),
        )
        splitter_handle.bind("<ButtonRelease-1>", lambda e: self._end_splitter_drag())

        # Create right panel
        right_panel = ctk.CTkFrame(splitter_frame)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(0, 5))
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)
        right_creator(right_panel)

        # Store splitter reference
        self.splitters[splitter_key] = left_panel

        return splitter_frame

    def _on_splitter_enter(self, event: tk.Event, handle: ctk.CTkFrame) -> None:
        """Handle mouse enter on splitter handle."""
        handle.configure(fg_color="#888888")
        handle.configure(cursor="sb_h_double_arrow")

    def _on_splitter_leave(self, event: tk.Event, handle: ctk.CTkFrame) -> None:
        """Handle mouse leave on splitter handle."""
        if not hasattr(self, "dragging_splitter") or not self.dragging_splitter:
            handle.configure(fg_color="#666666")

    def _start_splitter_drag(self, event: tk.Event, handle: ctk.CTkFrame, left_panel: ctk.CTkFrame, splitter_key: str) -> None:
        """Start dragging the splitter."""
        self.dragging_splitter = True
        self.drag_splitter_key = splitter_key
        self.drag_left_panel = left_panel
        self.drag_start_x = event.x_root
        self.drag_start_width = left_panel.winfo_width()
        handle.configure(fg_color="#AAAAAA")

    def _drag_splitter(self, event: tk.Event, handle: ctk.CTkFrame, left_panel: ctk.CTkFrame, splitter_key: str) -> None:
        """Drag the splitter."""
        if hasattr(self, "dragging_splitter") and self.dragging_splitter:
            delta_x = event.x_root - self.drag_start_x
            new_width = max(
                150,
                min(800, self.drag_start_width + delta_x),
            )  # Min 150, Max 800
            left_panel.configure(width=new_width)

    def _end_splitter_drag(self) -> None:
        """End dragging the splitter."""
        if hasattr(self, "dragging_splitter") and self.dragging_splitter:
            # Save the current position
            if hasattr(self, "drag_splitter_key") and hasattr(self, "drag_left_panel"):
                self.layout_data[self.drag_splitter_key] = (
                    self.drag_left_panel.winfo_width()
                )
                # Auto-save layout
                self._save_layout_config()

        self.dragging_splitter = False
        # Reset handle color
        for splitter_key, splitter in self.splitters.items():
            if hasattr(splitter, "master") and hasattr(
                splitter.master,
                "winfo_children",
            ):
                for child in splitter.master.winfo_children():
                    if isinstance(child, ctk.CTkFrame) and child.winfo_width() == 8:
                        child.configure(fg_color="#666666")

    def _on_closing(self) -> None:
        """Handle application closing."""
        self._save_layout_config()
        self.quit()

    def _on_window_configure(self, event: tk.Event) -> None:
        """Handle window resize events to save layout."""
        # Only save if this is the main window being resized
        if event.widget == self:
            # Debounce the saving to avoid too frequent saves
            if hasattr(self, "_resize_timer"):
                self.after_cancel(self._resize_timer)
            self._resize_timer = self.after(LAYOUT_SAVE_DELAY_MS, self._save_layout_config)

    def create_status_bar(self) -> None:
        """Create the status bar with progress tracking."""
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(status_frame, text="Ready", anchor="w")
        self.status_label.grid(row=0, column=0, padx=5, pady=2, sticky="ew")

        # Progress bar (initially hidden)
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()  # Hidden by default

        # Progress label
        self.progress_label = ctk.CTkLabel(status_frame, text="", anchor="w")
        self.progress_label.grid(row=2, column=0, padx=5, pady=2, sticky="ew")
        self.progress_label.grid_remove()  # Hidden by default

    def update_status(
        self,
        message: str,
        show_progress: bool = False,
        progress_value: int = 0,
        progress_text: str = "",
    ) -> None:
        """Update status bar with optional progress tracking."""
        self.status_label.configure(text=message)

        if show_progress:
            self.progress_bar.grid()
            self.progress_label.grid()
            self.progress_bar.set(progress_value)
            if progress_text:
                self.progress_label.configure(text=progress_text)
        else:
            self.progress_bar.grid_remove()
            self.progress_label.grid_remove()

        # Force update
        self.update()

    # Placeholder methods for functionality that would be implemented
    def on_plot_file_select(self, value: str) -> None:
        """Handle plot file selection - simplified for better performance."""
        if value == "Select a file...":
            return

        try:
            # Direct execution like baseline - no async scheduling
            df = self.get_data_for_plotting(value)
            if df is not None and not df.empty:
                # Update x-axis options - use actual columns, not "default time"
                x_axis_options = list(df.columns)
                self.plot_xaxis_menu.configure(values=x_axis_options)

                # Set the first column as default x-axis (usually time)
                if x_axis_options:
                    self.plot_xaxis_menu.set(x_axis_options[0])

                # Update signal checkboxes - direct creation like baseline
                self.plot_signal_vars = {}
                for widget in self.plot_signal_frame.winfo_children():
                    widget.destroy()

                for signal in df.columns:
                    var = tk.BooleanVar(value=False)
                    cb = ctk.CTkCheckBox(
                        self.plot_signal_frame,
                        text=signal,
                        variable=var,
                        command=self._on_plot_signal_checkbox_changed,
                    )
                    cb.pack(anchor="w", padx=5, pady=2)
                    self.plot_signal_vars[signal] = {"var": var, "checkbox": cb}

                # Re-bind mouse wheel to all new checkboxes
                self._bind_mousewheel_to_frame(self.plot_signal_frame)

                # Update trendline signal options
                signal_options = ["Select signal..."] + [
                    col for col in df.columns if col != x_axis_options[0]
                ]  # Exclude time column
                self.trendline_signal_menu.configure(values=signal_options)
                self.trendline_signal_var.set("Select signal...")

                # Reset signal tracking when file changes
                if hasattr(self, "last_plotted_signals"):
                    self.last_plotted_signals = set()

                # Update plot immediately - no delays
                self.update_plot()

        except Exception as e:
            print(f"ERROR in on_plot_file_select: {e}")
            import traceback

            traceback.print_exc()
            messagebox.showerror(
                "Error",
                f"Failed to select file for plotting:\n{e!s}",
            )
            if hasattr(self, "status_label"):
                self.status_label.configure(text="Error selecting file for plotting")
                self.status_label.configure(text="Ready")

    def update_plot(self, selected_signals: list[str] | None = None) -> None:
        """Update the plot with fixed error handling and canvas management."""
        # Check if plot canvas is initialized
        if not hasattr(self, "plot_canvas") or not hasattr(self, "plot_ax"):
            print("Warning: Plot canvas not initialized")
            return

        selected_file = self.plot_file_menu.get()
        x_axis_col = self.plot_xaxis_menu.get()

        if selected_file == "Select a file..." or not x_axis_col:
            return

        # Clear the plot first
        self.plot_ax.clear()

        # Get data with specific error handling
        df = None
        try:
            df = self.get_data_for_plotting(selected_file)
        except Exception as e:
            print(f"Error loading data: {e}")
            self.plot_ax.text(
                0.5,
                0.5,
                f"Error loading data:\n{e!s}",
                ha="center",
                va="center",
                wrap=True,
            )
            self.plot_canvas.draw()
            self.status_label.configure(text="Error loading data - check console")
            return

        if df is None or df.empty:
            self.plot_ax.text(
                0.5,
                0.5,
                "No data available for plotting",
                ha="center",
                va="center",
            )
            self.plot_canvas.draw()
            return

        # Validate x_axis_col
        if x_axis_col not in df.columns:
            if len(df.columns) > 0:
                x_axis_col = df.columns[0]
                self.plot_xaxis_menu.set(x_axis_col)
            else:
                self.plot_ax.text(
                    0.5,
                    0.5,
                    "No valid columns found for plotting.",
                    ha="center",
                    va="center",
                )
                self.plot_canvas.draw()
                return

        # Get signals to plot
        signals_to_plot = [
            s for s, data in self.plot_signal_vars.items() if data["var"].get()
        ]

        if not signals_to_plot:
            self.plot_ax.text(
                0.5,
                0.5,
                "Select one or more signals to plot",
                ha="center",
                va="center",
            )
            self.plot_canvas.draw()
            return

        # Now do the actual plotting with more granular error handling
        try:
            # Check if we should show both raw and filtered signals
            show_both = self.show_both_signals_var.get()
            plot_filter = self.plot_filter_type.get()
            compare_filters = self.compare_filters_var.get()

            # Chart customization
            plot_style = self.plot_type_var.get()
            style_args = {"linestyle": "-", "marker": ""}
            if plot_style == "Line with Markers":
                style_args = {"linestyle": "-", "marker": ".", "markersize": 4}
            elif plot_style == "Markers Only (Scatter)":
                style_args = {"linestyle": "None", "marker": "."}

            # Apply filter if needed
            plot_df = df.copy()
            if plot_filter != "None" and not show_both and not compare_filters:
                try:
                    plot_df = self._apply_plot_filter(
                        plot_df,
                        signals_to_plot,
                        x_axis_col,
                    )
                except Exception as e:
                    print(f"Warning: Filter failed - {e}")
                    # Continue with unfiltered data

            # Plot signals
            for i, signal in enumerate(signals_to_plot):
                if signal not in plot_df.columns:
                    print(f"Warning: Signal {signal} not found in data")
                    continue

                signal_data = plot_df[[x_axis_col, signal]].dropna()
                if len(signal_data) == 0:
                    print(f"Warning: No valid data for signal {signal}")
                    continue

                try:
                    # Convert data to numeric if possible
                    try:
                        signal_data[signal] = pd.to_numeric(
                            signal_data[signal],
                            errors="coerce",
                        )
                    except Exception:
                        print("Warning: Could not convert signal to numeric")
                        continue

                    # Skip if all values are NaN after conversion
                    if signal_data[signal].isna().all():
                        print(f"Warning: Signal {signal} has no valid numeric data")
                        continue

                    # Get color
                    color_scheme = self.color_scheme_var.get()
                    if color_scheme == "Default":
                        color = plt.cm.tab10(i % 10)
                    elif color_scheme == "Colorblind-friendly":
                        colors = [
                            "#0173B2",
                            "#DE8F05",
                            "#029E73",
                            "#CC78BC",
                            "#CA9161",
                            "#FBAFE4",
                            "#949494",
                            "#ECE133",
                            "#56B4E9",
                        ]
                        color = colors[i % len(colors)]
                    else:
                        color = self.custom_colors[i % len(self.custom_colors)]

                    # Plot with custom legend if available
                    label = self.custom_legend_entries.get(signal, signal)
                    line_width = self.line_width_var.get()

                    self.plot_ax.plot(
                        signal_data[x_axis_col],
                        signal_data[signal],
                        label=label,
                        color=color,
                        linewidth=line_width,
                        **style_args,
                    )

                    # Show both raw and filtered if requested
                    if show_both and plot_filter != "None":
                        raw_data = pd.to_numeric(df[signal], errors="coerce")
                        if not raw_data.isna().all():
                            raw_label = f"{label} (raw)"
                            self.plot_ax.plot(
                                df[x_axis_col],
                                raw_data,
                                label=raw_label,
                                color=color,
                                alpha=0.3,
                                linewidth=line_width * 0.7,
                            )

                    # Compare multiple filters if requested
                    if compare_filters and plot_filter != "None":
                        compare_filter = self.compare_filter_type.get()

                        # Plot raw data first (ensure numeric)
                        raw_data = pd.to_numeric(df[signal], errors="coerce")
                        if not raw_data.isna().all():
                            raw_label = f"{label} (raw)"
                            self.plot_ax.plot(
                                df[x_axis_col],
                                raw_data,
                                label=raw_label,
                                color=color,
                                alpha=0.5,
                                linewidth=line_width * ZOOM_IN_FACTOR,
                                linestyle="--",
                            )

                        # Plot main filter
                        try:
                            filtered_df = self._apply_plot_filter(
                                df.copy(),
                                [signal],
                                x_axis_col,
                                plot_filter,
                                False,
                            )
                            filtered_label = f"{label} ({plot_filter})"
                            self.plot_ax.plot(
                                filtered_df[x_axis_col],
                                filtered_df[signal],
                                label=filtered_label,
                                color=color,
                                linewidth=line_width,
                            )
                        except Exception as e:
                            print(f"Warning: Main filter failed - {e}")

                        # Plot comparison filter if different from main filter
                        if compare_filter != "None" and compare_filter != plot_filter:
                            try:
                                compare_df = self._apply_plot_filter(
                                    df.copy(),
                                    [signal],
                                    x_axis_col,
                                    compare_filter,
                                    True,
                                )
                                compare_label = f"{label} ({compare_filter})"
                                self.plot_ax.plot(
                                    compare_df[x_axis_col],
                                    compare_df[signal],
                                    label=compare_label,
                                    color=color,
                                    linewidth=line_width,
                                    linestyle=":",
                                )
                            except Exception as e:
                                print(f"Warning: Comparison filter failed - {e}")

                except Exception as e:
                    print(f"Error plotting signal {signal}: {e}")
                    continue

            # Add trendline if configured
            try:
                trendline_signal = self.trendline_signal_var.get()
                trendline_type = self.trendline_type_var.get()

                if trendline_signal != "None" and trendline_signal in signals_to_plot:
                    self._add_trendline(
                        plot_df,
                        x_axis_col,
                        trendline_signal,
                        trendline_type,
                    )
            except Exception as e:
                print(f"Warning: Trendline failed - {e}")

            # Configure plot appearance
            title = self.plot_title_entry.get() or f"Signals from {selected_file}"
            xlabel = self.plot_xlabel_entry.get() or x_axis_col
            ylabel = self.plot_ylabel_entry.get() or "Value"

            self.plot_ax.set_title(title, fontsize=14)
            self.plot_ax.set_xlabel(xlabel)
            self.plot_ax.set_ylabel(ylabel)

            # Legend
            legend_position = self.legend_position_var.get()
            if legend_position == "outside right":
                self.plot_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                self.plot_ax.legend(loc=legend_position)

            self.plot_ax.grid(True, linestyle="--", alpha=0.6)

            # Format datetime axis if applicable
            if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
                self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                self.plot_ax.tick_params(axis="x", rotation=0)

            # Smart zoom handling
            zoom_state = getattr(self, "saved_zoom_state", None)

            # Detect if new signals were added
            has_new_signals = self._detect_new_signals(signals_to_plot)

            # Determine zoom behavior
            if has_new_signals:
                should_auto_zoom = self._should_auto_zoom("new_signal")
                zoom_reason = "new_signal"
            else:
                should_auto_zoom = self._should_auto_zoom("filter_change")
                zoom_reason = "filter_change"

            if should_auto_zoom:
                # Auto-fit to show all data
                try:
                    self.plot_ax.autoscale_view()
                    print(f"Auto-zoom applied ({zoom_reason}): fitting to all data")
                except Exception as e:
                    print(f"Warning: Could not auto-fit plot - {e}")
            elif zoom_state:
                # Restore previous zoom state
                try:
                    self._apply_zoom_state(zoom_state)
                    print(f"Zoom state restored ({zoom_reason}): preserving user view")
                except Exception as e:
                    print(f"Warning: Could not apply zoom state - {e}")

            # Force canvas update
            self.plot_canvas.draw_idle()
            self.status_label.configure(text="Plot updated successfully")

        except Exception as e:
            print(f"Error in plotting: {e}")
            import traceback

            traceback.print_exc()

            # Show error on plot
            self.plot_ax.clear()
            self.plot_ax.text(
                0.5,
                0.5,
                f"Error creating plot:\n{e!s}",
                ha="center",
                va="center",
                wrap=True,
            )
            self.plot_canvas.draw()
            self.status_label.configure(text="Plot error - check console for details")

    def _ensure_plot_canvas_ready(self) -> None:
        """Ensure plot canvas is properly initialized."""
        if not hasattr(self, "plot_canvas") or self.plot_canvas is None:
            print("ERROR: Plot canvas not initialized!")
            return False

        if not hasattr(self, "plot_ax") or self.plot_ax is None:
            print("ERROR: Plot axes not initialized!")
            return False

        # Force a draw to ensure canvas is ready
        try:
            self.plot_canvas.draw()
            return True
        except Exception as e:
            print(f"ERROR: Canvas draw failed - {e}")
            return False

    def enable_plot_debugging(self) -> None:
        """Enable verbose debugging for plot operations."""
        self.plot_debug = True

    def debug_print(self, message: str) -> None:
        """Print debug message if debugging is enabled."""
        if hasattr(self, "plot_debug") and self.plot_debug:
            print(f"[PLOT DEBUG] {message}")

    def _apply_plot_filter(
        self,
        df: pd.DataFrame,
        signal_cols: list[str],
        x_axis_col: str,
        filter_type: str | None = None,
        is_comparison: bool = False,
    ) -> pd.DataFrame:
        """Apply filter to plot data with support for comparison filters."""
        if filter_type is None:
            filter_type = self.plot_filter_type.get()

        if filter_type == "None":
            return df

        # Get filter parameters based on whether this is a comparison filter
        if is_comparison:
            # Use comparison filter parameters
            if filter_type == "Moving Average":
                window = int(self.compare_ma_value_entry.get())
                unit = self.compare_ma_unit_menu.get()
                # Convert to samples based on unit
                if unit == "ms":
                    window = int(window * self.sample_rate / MILLISECONDS_PER_SECOND)
                elif unit == "s":
                    window = int(window * self.sample_rate)
                elif unit == "min":
                    window = int(window * self.sample_rate * SECONDS_PER_MINUTE)
                elif unit == "hr":
                    window = int(window * self.sample_rate * SECONDS_PER_HOUR)

                for col in signal_cols:
                    if col in df.columns:
                        df[col] = df[col].rolling(window=window, center=True).mean()

            elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
                order = int(self.compare_bw_order_entry.get())
                cutoff = float(self.compare_bw_cutoff_entry.get())

                for col in signal_cols:
                    if col in df.columns:
                        if filter_type == "Butterworth Low-pass":
                            df[col] = self._apply_butterworth_lowpass(
                                df[col],
                                cutoff,
                                order,
                            )
                        else:
                            df[col] = self._apply_butterworth_highpass(
                                df[col],
                                cutoff,
                                order,
                            )

            elif filter_type == "Median Filter":
                kernel_size = int(self.compare_median_kernel_entry.get())
                for col in signal_cols:
                    if col in df.columns:
                        df[col] = (
                            df[col].rolling(window=kernel_size, center=True).median()
                        )

            elif filter_type == "Hampel Filter":
                window = int(self.compare_hampel_window_entry.get())
                threshold = float(self.compare_hampel_threshold_entry.get())
                for col in signal_cols:
                    if col in df.columns:
                        df[col] = self._apply_hampel_filter(df[col], window, threshold)

            elif filter_type == "Z-Score Filter":
                threshold = float(self.compare_zscore_threshold_entry.get())
                method = self.compare_zscore_method_menu.get()
                for col in signal_cols:
                    if col in df.columns:
                        df[col] = self._apply_zscore_filter(df[col], threshold, method)

            elif filter_type == "Savitzky-Golay":
                window = int(self.compare_savgol_window_entry.get())
                polyorder = int(self.compare_savgol_polyorder_entry.get())
                for col in signal_cols:
                    if col in df.columns:
                        if _savgol_filter is None:
                            raise RuntimeError(
                                "scipy.signal.savgol_filter unavailable. Install SciPy or skip smoothing.",
                            )
                        df[col] = _savgol_filter(df[col], window, polyorder)
        elif filter_type == "Moving Average":
            window = int(self.plot_ma_value_entry.get())
            unit = self.plot_ma_unit_menu.get()
            # Convert to samples based on unit
            if unit == "ms":
                window = int(window * self.sample_rate / MILLISECONDS_PER_SECOND)
            elif unit == "s":
                window = int(window * self.sample_rate)
            elif unit == "min":
                window = int(window * self.sample_rate * SECONDS_PER_MINUTE)
            elif unit == "hr":
                window = int(window * self.sample_rate * SECONDS_PER_HOUR)

            for col in signal_cols:
                if col in df.columns:
                    df[col] = df[col].rolling(window=window, center=True).mean()

        elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
            order = int(self.plot_bw_order_entry.get())
            cutoff = float(self.plot_bw_cutoff_entry.get())

            for col in signal_cols:
                if col in df.columns:
                    if filter_type == "Butterworth Low-pass":
                        df[col] = self._apply_butterworth_lowpass(
                            df[col],
                            cutoff,
                            order,
                        )
                    else:
                        df[col] = self._apply_butterworth_highpass(
                            df[col],
                            cutoff,
                            order,
                        )

        elif filter_type == "Median Filter":
            kernel_size = int(self.plot_median_kernel_entry.get())
            for col in signal_cols:
                if col in df.columns:
                    df[col] = df[col].rolling(window=kernel_size, center=True).median()

        elif filter_type == "Hampel Filter":
            window = int(self.plot_hampel_window_entry.get())
            threshold = float(self.plot_hampel_threshold_entry.get())
            for col in signal_cols:
                if col in df.columns:
                    df[col] = self._apply_hampel_filter(df[col], window, threshold)

        elif filter_type == "Z-Score Filter":
            threshold = float(self.plot_zscore_threshold_entry.get())
            method = self.plot_zscore_method_menu.get()
            for col in signal_cols:
                if col in df.columns:
                    df[col] = self._apply_zscore_filter(df[col], threshold, method)

        elif filter_type == "Savitzky-Golay":
            window = int(self.plot_savgol_window_entry.get())
            polyorder = int(self.plot_savgol_polyorder_entry.get())
            for col in signal_cols:
                if col in df.columns:
                    if _savgol_filter is None:
                        raise RuntimeError(
                            "scipy.signal.savgol_filter unavailable. Install SciPy or skip smoothing.",
                        )
                    df[col] = _savgol_filter(df[col], window, polyorder)

        return df
        """Apply filter preview to the plot data."""
        filter_type = self.plot_filter_type.get()

        if filter_type == "None":
            return df

        filtered_df = df.copy()

        for signal in signal_cols:
            if signal not in df.columns:
                continue

            if filter_type == "Moving Average":
                window = float(self.plot_ma_value_entry.get() or "10")
                unit = self.plot_ma_unit_menu.get()
                if unit == "ms":
                    window = window / MILLISECONDS_PER_SECOND
                elif unit == "min":
                    window = window * SECONDS_PER_MINUTE
                elif unit == "hr":
                    window = window * SECONDS_PER_HOUR

                # Convert window to number of samples
                if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
                    time_diff = df[x_axis_col].diff().dt.total_seconds().median()
                    if time_diff > 0:
                        window_samples = int(window / time_diff)
                        filtered_df[signal] = (
                            df[signal]
                            .rolling(window=max(1, window_samples), center=True)
                            .mean()
                        )
                else:
                    filtered_df[signal] = (
                        df[signal].rolling(window=int(window), center=True).mean()
                    )

            elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
                order = int(self.plot_bw_order_entry.get() or "2")
                cutoff = float(self.plot_bw_cutoff_entry.get() or "0.1")

                try:
                    from scipy.signal import butter, filtfilt

                    # Calculate sampling frequency from time data
                    if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
                        time_diff = df[x_axis_col].diff().dt.total_seconds()
                        fs = 1.0 / time_diff.median() if time_diff.median() > 0 else 1.0
                    else:
                        # Assume uniform sampling
                        fs = 1.0

                    # Normalize cutoff frequency
                    nyquist = fs / DEFAULT_BW_NYQUIST
                    normalized_cutoff = cutoff / nyquist

                    # Design filter
                    b, a = butter(
                        order,
                        normalized_cutoff,
                        btype=(
                            "low" if filter_type == "Butterworth Low-pass" else "high"
                        ),
                    )

                    # Apply filter
                    signal_data = (
                        df[signal].fillna(method="ffill").fillna(method="bfill")
                    )
                    filtered_df[signal] = filtfilt(b, a, signal_data)

                except ImportError:
                    # Fallback to simple smoothing if scipy not available
                    filtered_df[signal] = (
                        df[signal].rolling(window=order * MIN_BUTTERWORTH_DATA_MULTIPLIER + 1, center=True).mean()
                    )
                except Exception as e:
                    print(f"Error applying Butterworth filter: {e}")
                    # Fallback to simple smoothing
                    filtered_df[signal] = (
                        df[signal].rolling(window=order * MIN_BUTTERWORTH_DATA_MULTIPLIER + 1, center=True).mean()
                    )

            elif filter_type == "Median Filter":
                kernel = int(self.plot_median_kernel_entry.get() or "5")
                filtered_df[signal] = (
                    df[signal].rolling(window=kernel, center=True).median()
                )

            elif filter_type == "Hampel Filter":
                window = int(self.plot_hampel_window_entry.get() or "7")
                threshold = float(self.plot_hampel_threshold_entry.get() or "3.0")

                try:
                    from scipy.signal import medfilt

                    signal_data = df[signal].ffill().bfill()

                    # Apply Hampel filter
                    median_filtered = pd.Series(
                        medfilt(signal_data, kernel_size=window),
                        index=signal_data.index,
                    )
                    mad = signal_data.rolling(window=window, center=True).apply(
                        lambda x: np.median(np.abs(x - np.median(x))),
                    )
                                            threshold_value = (
                            threshold * NORMAL_DISTRIBUTION_CONSTANT * mad
                        )  # 1.4826 is the constant for normal distribution

                    # Replace outliers with median using proper indexing
                    outliers = np.abs(signal_data - median_filtered) > threshold_value
                    filtered_df = (
                        filtered_df.copy()
                    )  # Ensure we have a copy to avoid warnings
                    filtered_df.loc[outliers, signal] = median_filtered.loc[outliers]

                except ImportError:
                    # Fallback to simple median filter
                    filtered_df[signal] = (
                        df[signal].rolling(window=window, center=True).median()
                    )
                except Exception as e:
                    print(f"Error applying Hampel filter: {e}")
                    # Fallback to simple median filter
                    filtered_df[signal] = (
                        df[signal].rolling(window=window, center=True).median()
                    )

            elif filter_type == "Z-Score Filter":
                threshold = float(self.plot_zscore_threshold_entry.get() or "3.0")
                method = self.plot_zscore_method_menu.get()

                signal_data = df[signal].fillna(method="ffill").fillna(method="bfill")
                mean_val = signal_data.mean()
                std_val = signal_data.std()
                z_scores = np.abs((signal_data - mean_val) / std_val)

                if method == "Remove Outliers":
                    # Replace outliers with NaN
                    filtered_df[signal] = signal_data.copy()
                    filtered_df[signal].loc[z_scores > threshold] = np.nan
                elif method == "Clip Outliers":
                    # Clip outliers to threshold
                    filtered_df[signal] = signal_data.copy()
                    upper_bound = mean_val + threshold * std_val
                    lower_bound = mean_val - threshold * std_val
                    filtered_df[signal] = filtered_df[signal].clip(
                        lower=lower_bound,
                        upper=upper_bound,
                    )
                elif method == "Replace with Median":
                    # Replace outliers with median
                    median_val = signal_data.median()
                    filtered_df[signal] = signal_data.copy()
                    filtered_df[signal].loc[z_scores > threshold] = median_val

            elif filter_type == "Savitzky-Golay":
                window = int(self.plot_savgol_window_entry.get() or "11")
                polyorder = int(self.plot_savgol_polyorder_entry.get() or "3")

                try:
                    if _savgol_filter is None:
                        raise RuntimeError(
                            "scipy.signal.savgol_filter unavailable. Install SciPy or skip smoothing.",
                        )

                    signal_data = (
                        df[signal].fillna(method="ffill").fillna(method="bfill")
                    )
                    filtered_df[signal] = _savgol_filter(signal_data, window, polyorder)
                except ImportError:
                    # Fallback to simple smoothing if scipy not available
                    filtered_df[signal] = (
                        df[signal].rolling(window=window, center=True).mean()
                    )
                except Exception as e:
                    print(f"Error applying Savitzky-Golay filter: {e}")
                    # Fallback to simple smoothing
                    filtered_df[signal] = (
                        df[signal].rolling(window=window, center=True).mean()
                    )

        return filtered_df

    def _add_trendline(self, df: pd.DataFrame, signal: str, x_axis_col: str) -> None:
        """Add trendline to the plot."""
        trend_type = self.trendline_type_var.get()

        if trend_type == "None":
            return

        plot_df = df[[x_axis_col, signal]].dropna()
        if len(plot_df) < 2:
            return

        # Apply time window filtering based on selected mode
        window_mode = self.trendline_window_mode.get()

        if window_mode == "Manual Entry":
            start_str = self.trendline_start_entry.get().strip()
            end_str = self.trendline_end_entry.get().strip()

            if start_str or end_str:
                try:
                    if pd.api.types.is_datetime64_any_dtype(plot_df[x_axis_col]):
                        # Convert to datetime
                        if start_str:
                            start_time = pd.to_datetime(start_str)
                            plot_df = plot_df[plot_df[x_axis_col] >= start_time]
                        if end_str:
                            end_time = pd.to_datetime(end_str)
                            plot_df = plot_df[plot_df[x_axis_col] <= end_time]
                    else:
                        # Numeric data
                        if start_str:
                            start_val = float(start_str)
                            plot_df = plot_df[plot_df[x_axis_col] >= start_val]
                        if end_str:
                            end_val = float(end_str)
                            plot_df = plot_df[plot_df[x_axis_col] <= end_val]
                except (ValueError, TypeError):
                    messagebox.showwarning(
                        "Warning",
                        "Invalid time window format. Using full range.",
                    )

        elif window_mode == "Visual Selection":
            if hasattr(self, "trendline_selection_start") and hasattr(
                self,
                "trendline_selection_end",
            ):
                if (
                    self.trendline_selection_start is not None
                    and self.trendline_selection_end is not None
                ):
                    try:
                        if pd.api.types.is_datetime64_any_dtype(plot_df[x_axis_col]):
                            # Convert numeric selection back to datetime
                            x_min = plot_df[x_axis_col].min()
                            start_time = x_min + pd.Timedelta(
                                seconds=self.trendline_selection_start,
                            )
                            end_time = x_min + pd.Timedelta(
                                seconds=self.trendline_selection_end,
                            )
                            plot_df = plot_df[
                                (plot_df[x_axis_col] >= start_time)
                                & (plot_df[x_axis_col] <= end_time)
                            ]
                        else:
                            # Numeric data
                            plot_df = plot_df[
                                (plot_df[x_axis_col] >= self.trendline_selection_start)
                                & (plot_df[x_axis_col] <= self.trendline_selection_end)
                            ]
                    except Exception as e:
                        print(f"Error applying visual selection: {e}")

        # Check if we still have enough data after filtering
        if len(plot_df) < 2:
            messagebox.showwarning(
                "Warning",
                "Not enough data points in selected time window for trendline.",
            )
            return

        x_data = plot_df[x_axis_col].values
        y_data = plot_df[signal].values

        # Convert datetime to numeric for fitting
        if pd.api.types.is_datetime64_any_dtype(plot_df[x_axis_col]):
            x_numeric = (
                (plot_df[x_axis_col] - plot_df[x_axis_col].min())
                .dt.total_seconds()
                .values
            )
        else:
            x_numeric = x_data.astype(float)

        try:
            if trend_type == "Linear":
                coeffs = np.polyfit(x_numeric, y_data, 1)
                trend = np.poly1d(coeffs)
                trendline = trend(x_numeric)
                equation = f"y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}"

            elif trend_type == "Exponential":
                # Log-linear fit for exponential
                y_positive = y_data[y_data > 0]
                x_positive = x_numeric[y_data > 0]
                if len(y_positive) > 1:
                    log_y = np.log(y_positive)
                    coeffs = np.polyfit(x_positive, log_y, 1)
                    a = np.exp(coeffs[1])
                    b = coeffs[0]
                    trendline = a * np.exp(b * x_numeric)
                    equation = f"y = {a:.4f} * e^({b:.4f}x)"
                else:
                    messagebox.showwarning(
                        "Warning",
                        "Not enough positive values for exponential trendline.",
                    )
                    return

            elif trend_type == "Power":
                # Log-log fit for power law
                mask = (y_data > 0) & (x_numeric > 0)
                y_positive = y_data[mask]
                x_positive = x_numeric[mask]
                if len(y_positive) > 1:
                    log_x = np.log(x_positive)
                    log_y = np.log(y_positive)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    a = np.exp(coeffs[1])
                    b = coeffs[0]
                    trendline = a * (x_numeric**b)
                    equation = f"y = {a:.4f} * x^({b:.4f})"
                else:
                    messagebox.showwarning(
                        "Warning",
                        "Not enough positive values for power trendline.",
                    )
                    return

            elif trend_type == "Polynomial":
                order = int(self.poly_order_entry.get() or "2")
                order = max(2, min(6, order))  # Limit to 2-6
                coeffs = np.polyfit(x_numeric, y_data, order)
                trend = np.poly1d(coeffs)
                trendline = trend(x_numeric)
                # Build equation string
                terms = []
                for i in range(order + 1):
                    power = order - i
                    coeff = coeffs[i]
                    if power == 0:
                        terms.append(f"{coeff:.4f}")
                    elif power == 1:
                        terms.append(f"{coeff:.4f}x")
                    else:
                        terms.append(f"{coeff:.4f}x^{power}")
                equation = f"Polynomial (order {order}): " + " + ".join(terms)

            # Plot trendline - use the original x_data for plotting
            self.plot_ax.plot(
                plot_df[x_axis_col],
                trendline,
                "--",
                color="red",
                linewidth=2,
                label=f"{signal} Trendline ({trend_type})",
                alpha=DEFAULT_ALPHA,
            )

            # Force redraw the legend
            handles, labels = self.plot_ax.get_legend_handles_labels()
            legend_position = self.legend_position_var.get()
            if legend_position == "outside right":
                self.plot_ax.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
            else:
                self.plot_ax.legend(handles, labels, loc=legend_position)

            # Update trendline textbox
            self.trendline_textbox.delete("1.0", tk.END)
            self.trendline_textbox.insert("1.0", equation)

            # Redraw the canvas
            self.plot_canvas.draw()

        except Exception as e:
            messagebox.showerror("Trendline Error", f"Error adding trendline: {e!s}")
            print(f"Error adding trendline: {e}")
            import traceback

            traceback.print_exc()

    def get_data_for_plotting(self, filename: str) -> pd.DataFrame | None:
        """Get data for plotting from the specified file - simplified baseline approach."""
        try:
            # First check if it's in processed files
            if filename in self.processed_files:
                return self.processed_files[filename]

            # Find the full path of the file
            full_path = None
            for file_path in self.input_file_paths:
                if os.path.basename(file_path) == filename:
                    full_path = file_path
                    break

            if full_path and os.path.exists(full_path):
                df = pd.read_csv(full_path)
                # Try to identify time column
                time_col = None
                for col in df.columns:
                    if any(
                        time_word in col.lower()
                        for time_word in ["time", "timestamp", "date"]
                    ):
                        time_col = col
                        break

                if time_col and pd.api.types.is_object_dtype(df[time_col]):
                    try:
                        df[time_col] = pd.to_datetime(df[time_col])
                    except Exception as e:
                        # Log datetime conversion errors for debugging
                        print(f"Warning: Failed to convert time column to datetime: {e}")

                return df
        except Exception as e:
            print(f"Error loading data for plotting: {e}")
            return None

    def _debug_plot_state(self) -> None:
        """Debug helper to print current plotting state."""
        print("\n=== PLOT DEBUG STATE ===")
        print(f"plot_file_menu: {getattr(self, 'plot_file_menu', None)}")
        if hasattr(self, "plot_file_menu"):
            print(f"  selected file: {self.plot_file_menu.get()}")

        print(f"plot_xaxis_menu: {getattr(self, 'plot_xaxis_menu', None)}")
        if hasattr(self, "plot_xaxis_menu"):
            print(f"  selected x-axis: {self.plot_xaxis_menu.get()}")

        print(f"plot_signal_vars: {getattr(self, 'plot_signal_vars', None)}")
        if hasattr(self, "plot_signal_vars"):
            print(f"  number of signals: {len(self.plot_signal_vars)}")
            selected = [
                s for s, data in self.plot_signal_vars.items() if data["var"].get()
            ]
            print(f"  selected signals: {selected}")

        print(f"plot_canvas: {getattr(self, 'plot_canvas', None)}")
        print(f"plot_ax: {getattr(self, 'plot_ax', None)}")
        print(
            f"processed_files: {len(getattr(self, 'processed_files', {})) if hasattr(self, 'processed_files') else 'None'}",
        )
        print(
            f"loaded_data_cache: {len(getattr(self, 'loaded_data_cache', {})) if hasattr(self, 'loaded_data_cache') else 'None'}",
        )
        print("========================\n")

    def _force_signal_selection(self) -> None:
        """Force select at least one signal for debugging."""
        if hasattr(self, "plot_signal_vars") and self.plot_signal_vars:
            # Check if any signals are selected
            selected = [
                s for s, data in self.plot_signal_vars.items() if data["var"].get()
            ]
            if not selected:
                # Auto-select first non-time signal
                for signal, data in self.plot_signal_vars.items():
                    if not any(
                        word in signal.lower() for word in ["time", "date", "timestamp"]
                    ):
                        data["var"].set(True)
                        print(f"DEBUG: Force-selected signal: {signal}")
                        break

    def _show_setup_help(self) -> None:
        """Show setup help."""
        messagebox.showinfo(
            "Setup Help",
            "This tab allows you to configure file processing settings.",
        )

    def _show_plot_help(self) -> None:
        """Show plotting help."""
        messagebox.showinfo(
            "Plotting Help",
            "This tab allows you to visualize and analyze your data.",
        )

    def _show_plots_list_help(self) -> None:
        """Show plots list help."""
        messagebox.showinfo(
            "Plots List Help",
            "This tab allows you to save and manage plot configurations.",
        )

    def _show_dat_import_help(self) -> None:
        """Show DAT import help."""
        messagebox.showinfo(
            "DAT Import Help",
            "This tab allows you to import DAT files with DBF tag files.",
        )

    def _show_legend_guide(self) -> None:
        """Show comprehensive legend formatting guide."""
        guide_window = ctk.CTkToplevel(self)
        guide_window.title("Custom Legend Guide - LaTeX Formatting")
        guide_window.geometry("600x700")
        guide_window.transient(self)
        guide_window.grab_set()

        # Make window resizable
        guide_window.grid_columnconfigure(0, weight=1)
        guide_window.grid_rowconfigure(0, weight=1)

        # Create scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(guide_window)
        scrollable_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        scrollable_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ctk.CTkLabel(
            scrollable_frame,
            text="Custom Legend Formatting Guide",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title_label.grid(row=0, column=0, pady=(0, 20), sticky="w")

        guide_text = """
BASIC SUBSCRIPTS:
• $H_2O$ → H₂O
• $CO_2$ → CO₂
• $CH_4$ → CH₄
• $O_2$ → O₂

MULTI-CHARACTER SUBSCRIPTS:
• $v_{max}$ → vₘₐₓ
• $P_{total}$ → Pₜₒₜₐₗ
• $T_{ambient}$ → Tₐₘᵦᵢₑₙₜ
• $Flow_{air}$ → Flowₐᵢᵣ

SUPERSCRIPTS:
• $x^2$ → x²
• $m^3$ → m³
• $T^{-1}$ → T⁻¹
• $10^{-6}$ → 10⁻⁶

COMBINED SUB & SUPERSCRIPTS:
• $H_2O^+$ → H₂O⁺
• $CO_2^{-}$ → CO₂⁻
• $x_1^2$ → x₁²

GREEK LETTERS:
• $\\alpha$ → α (alpha)
• $\\beta$ → β (beta)
• $\\gamma$ → γ (gamma)
• $\\delta$ → δ (delta)
• $\\theta$ → θ (theta)
• $\\lambda$ → λ (lambda)
• $\\mu$ → μ (mu)
• $\\pi$ → π (pi)
• $\\sigma$ → σ (sigma)
• $\\omega$ → ω (omega)
• $\\Delta$ → Δ (Delta - capital)
• $\\Omega$ → Ω (Omega - capital)

ENGINEERING EXAMPLES:
• $\\dot{m}_{air}$ → ṁₐᵢᵣ (mass flow rate)
• $T_{in}$ → Tᵢₙ (inlet temperature)
• $P_1$ → P₁ (pressure point 1)
• $[CO_2]$ → [CO₂] (concentration)
• $\\eta_{thermal}$ → ηₜₕₑᵣₘₐₗ (thermal efficiency)
• $\\Delta P$ → ΔP (pressure difference)
• $f_{Hz}$ → fₕᵨ (frequency in Hz)

FRACTIONS & MATH:
• $\\frac{m}{s}$ → m/s (as fraction)
• $m/s^2$ → m/s² (acceleration)
• $kg \\cdot m^2$ → kg·m² 
• $\\pm$ → ± (plus-minus)

TIPS:
• Always wrap LaTeX in dollar signs: $...$
• Use curly braces {} for multi-character sub/superscripts
• Backslash \\ before Greek letters
• Use \\dot{} for dot notation (derivatives)
• Use \\frac{numerator}{denominator} for fractions

COMMON MISTAKES TO AVOID:
• Don't forget the $ symbols
• Use {max} not just max for subscripts
• Remember \\ before Greek letters
• Close all braces properly
        """

        # Create text widget for the guide
        text_widget = ctk.CTkTextbox(
            scrollable_frame,
            width=550,
            height=500,
            wrap="word",
        )
        text_widget.grid(row=1, column=0, pady=10, sticky="ew")
        text_widget.insert("1.0", guide_text)
        text_widget.configure(state="disabled")

        # Close button
        close_button = ctk.CTkButton(
            guide_window,
            text="Close",
            command=guide_window.destroy,
        )
        close_button.grid(row=1, column=0, pady=10)

        # Center the window
        guide_window.update_idletasks()
        x = (guide_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (guide_window.winfo_screenheight() // 2) - (700 // 2)
        guide_window.geometry(f"600x700+{x}+{y}")

    def save_settings(self) -> None:
        """Save current settings to a configuration file."""
        try:
            # Collect all current settings
            settings = {
                "filter_settings": {
                    "filter_type": (
                        self.filter_type_var.get()
                        if hasattr(self, "filter_type_var")
                        else "None"
                    ),
                    "ma_window": (
                        self.ma_value_entry.get()
                        if hasattr(self, "ma_value_entry")
                        else "10"
                    ),
                    "ma_unit": (
                        self.ma_unit_menu.get()
                        if hasattr(self, "ma_unit_menu")
                        else "s"
                    ),
                    "bw_order": (
                        self.bw_order_entry.get()
                        if hasattr(self, "bw_order_entry")
                        else "3"
                    ),
                    "bw_cutoff": (
                        self.bw_cutoff_entry.get()
                        if hasattr(self, "bw_cutoff_entry")
                        else "0.1"
                    ),
                    "median_kernel": (
                        self.median_kernel_entry.get()
                        if hasattr(self, "median_kernel_entry")
                        else "5"
                    ),
                    "hampel_window": (
                        self.hampel_window_entry.get()
                        if hasattr(self, "hampel_window_entry")
                        else "7"
                    ),
                    "hampel_threshold": (
                        self.hampel_threshold_entry.get()
                        if hasattr(self, "hampel_threshold_entry")
                        else "3.0"
                    ),
                    "zscore_threshold": (
                        self.zscore_threshold_entry.get()
                        if hasattr(self, "zscore_threshold_entry")
                        else "3.0"
                    ),
                    "zscore_method": (
                        self.zscore_method_menu.get()
                        if hasattr(self, "zscore_method_menu")
                        else "Remove Outliers"
                    ),
                    "savgol_window": (
                        self.savgol_window_entry.get()
                        if hasattr(self, "savgol_window_entry")
                        else "11"
                    ),
                    "savgol_polyorder": (
                        self.savgol_polyorder_entry.get()
                        if hasattr(self, "savgol_polyorder_entry")
                        else "2"
                    ),
                },
                "resample_settings": {
                    "enabled": (
                        self.resample_var.get()
                        if hasattr(self, "resample_var")
                        else False
                    ),
                    "value": (
                        self.resample_value_entry.get()
                        if hasattr(self, "resample_value_entry")
                        else "10"
                    ),
                    "unit": (
                        self.resample_unit_menu.get()
                        if hasattr(self, "resample_unit_menu")
                        else "s"
                    ),
                },
                "trim_settings": {
                    "date": (
                        self.trim_date_entry.get()
                        if hasattr(self, "trim_date_entry")
                        else ""
                    ),
                    "start_time": (
                        self.trim_start_entry.get()
                        if hasattr(self, "trim_start_entry")
                        else ""
                    ),
                    "end_time": (
                        self.trim_end_entry.get()
                        if hasattr(self, "trim_end_entry")
                        else ""
                    ),
                },
                "integration_settings": {
                    "method": (
                        self.integrator_method_var.get()
                        if hasattr(self, "integrator_method_var")
                        else "Trapezoidal"
                    ),
                },
                "differentiation_settings": {
                    "method": (
                        self.deriv_method_var.get()
                        if hasattr(self, "deriv_method_var")
                        else "Spline (Acausal)"
                    ),
                    "orders": (
                        {str(i): var.get() for i, var in self.derivative_vars.items()}
                        if hasattr(self, "derivative_vars")
                        else {}
                    ),
                },
                "export_settings": {
                    "type": (
                        self.export_type_var.get()
                        if hasattr(self, "export_type_var")
                        else "CSV (Separate Files)"
                    ),
                    "sort_column": (
                        self.sort_col_menu.get()
                        if hasattr(self, "sort_col_menu")
                        else "No Sorting"
                    ),
                    "sort_order": (
                        self.sort_order_var.get()
                        if hasattr(self, "sort_order_var")
                        else "Ascending"
                    ),
                },
                "dataset_naming": {
                    "mode": (
                        self.dataset_naming_var.get()
                        if hasattr(self, "dataset_naming_var")
                        else "auto"
                    ),
                    "custom_name": (
                        self.custom_dataset_entry.get()
                        if hasattr(self, "custom_dataset_entry")
                        else ""
                    ),
                },
                "custom_variables": (
                    self.custom_vars_list if hasattr(self, "custom_vars_list") else []
                ),
                "output_directory": self.output_directory,
                "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                title="Save Configuration Settings",
                defaultextension=".json",
                filetypes=[("JSON Configuration", "*.json"), ("All files", "*.*")],
                initialfile="csv_processor_config.json",
            )

            if file_path:
                with open(file_path, "w") as f:
                    json.dump(settings, f, indent=2)
                messagebox.showinfo("Success", f"Settings saved to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{e!s}")

    def load_settings(self) -> None:
        """Load settings from a configuration file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Configuration Settings",
                filetypes=[("JSON Configuration", "*.json"), ("All files", "*.*")],
            )

            if not file_path:
                return

            with open(file_path) as f:
                settings = json.load(f)

            # Apply filter settings
            if "filter_settings" in settings:
                fs = settings["filter_settings"]
                if hasattr(self, "filter_type_var"):
                    self.filter_type_var.set(fs.get("filter_type", "None"))
                    self._update_filter_ui(fs.get("filter_type", "None"))
                if hasattr(self, "ma_value_entry"):
                    self.ma_value_entry.delete(0, tk.END)
                    self.ma_value_entry.insert(0, fs.get("ma_window", "10"))
                if hasattr(self, "ma_unit_menu"):
                    self.ma_unit_menu.set(fs.get("ma_unit", "s"))
                if hasattr(self, "bw_order_entry"):
                    self.bw_order_entry.delete(0, tk.END)
                    self.bw_order_entry.insert(0, fs.get("bw_order", "3"))
                if hasattr(self, "bw_cutoff_entry"):
                    self.bw_cutoff_entry.delete(0, tk.END)
                    self.bw_cutoff_entry.insert(0, fs.get("bw_cutoff", "0.1"))
                if hasattr(self, "median_kernel_entry"):
                    self.median_kernel_entry.delete(0, tk.END)
                    self.median_kernel_entry.insert(0, fs.get("median_kernel", "5"))
                if hasattr(self, "hampel_window_entry"):
                    self.hampel_window_entry.delete(0, tk.END)
                    self.hampel_window_entry.insert(0, fs.get("hampel_window", "7"))
                if hasattr(self, "hampel_threshold_entry"):
                    self.hampel_threshold_entry.delete(0, tk.END)
                    self.hampel_threshold_entry.insert(
                        0,
                        fs.get("hampel_threshold", "3.0"),
                    )
                if hasattr(self, "zscore_threshold_entry"):
                    self.zscore_threshold_entry.delete(0, tk.END)
                    self.zscore_threshold_entry.insert(
                        0,
                        fs.get("zscore_threshold", "3.0"),
                    )
                if hasattr(self, "zscore_method_menu"):
                    self.zscore_method_menu.set(
                        fs.get("zscore_method", "Remove Outliers"),
                    )
                if hasattr(self, "savgol_window_entry"):
                    self.savgol_window_entry.delete(0, tk.END)
                    self.savgol_window_entry.insert(0, fs.get("savgol_window", "11"))
                if hasattr(self, "savgol_polyorder_entry"):
                    self.savgol_polyorder_entry.delete(0, tk.END)
                    self.savgol_polyorder_entry.insert(
                        0,
                        fs.get("savgol_polyorder", "2"),
                    )

            # Apply resample settings
            if "resample_settings" in settings:
                rs = settings["resample_settings"]
                if hasattr(self, "resample_var"):
                    self.resample_var.set(rs.get("enabled", False))
                if hasattr(self, "resample_value_entry"):
                    self.resample_value_entry.delete(0, tk.END)
                    self.resample_value_entry.insert(0, rs.get("value", "10"))
                if hasattr(self, "resample_unit_menu"):
                    self.resample_unit_menu.set(rs.get("unit", "s"))

            # Apply trim settings
            if "trim_settings" in settings:
                ts = settings["trim_settings"]
                if hasattr(self, "trim_date_entry"):
                    self.trim_date_entry.delete(0, tk.END)
                    self.trim_date_entry.insert(0, ts.get("date", ""))
                if hasattr(self, "trim_start_entry"):
                    self.trim_start_entry.delete(0, tk.END)
                    self.trim_start_entry.insert(0, ts.get("start_time", ""))
                if hasattr(self, "trim_end_entry"):
                    self.trim_end_entry.delete(0, tk.END)
                    self.trim_end_entry.insert(0, ts.get("end_time", ""))

            # Apply integration settings
            if "integration_settings" in settings:
                its = settings["integration_settings"]
                if hasattr(self, "integrator_method_var"):
                    self.integrator_method_var.set(its.get("method", "Trapezoidal"))

            # Apply differentiation settings
            if "differentiation_settings" in settings:
                ds = settings["differentiation_settings"]
                if hasattr(self, "deriv_method_var"):
                    self.deriv_method_var.set(ds.get("method", "Spline (Acausal)"))
                if hasattr(self, "derivative_vars") and "orders" in ds:
                    for order_str, value in ds["orders"].items():
                        order = int(order_str)
                        if order in self.derivative_vars:
                            self.derivative_vars[order].set(value)

            # Apply export settings
            if "export_settings" in settings:
                es = settings["export_settings"]
                if hasattr(self, "export_type_var"):
                    self.export_type_var.set(es.get("type", "CSV (Separate Files)"))
                if hasattr(self, "sort_col_menu"):
                    self.sort_col_menu.set(es.get("sort_column", "No Sorting"))
                if hasattr(self, "sort_order_var"):
                    self.sort_order_var.set(es.get("sort_order", "Ascending"))

            # Apply dataset naming settings
            if "dataset_naming" in settings:
                dns = settings["dataset_naming"]
                if hasattr(self, "dataset_naming_var"):
                    self.dataset_naming_var.set(dns.get("mode", "auto"))
                    self._on_dataset_naming_change()
                if hasattr(self, "custom_dataset_entry"):
                    self.custom_dataset_entry.delete(0, tk.END)
                    self.custom_dataset_entry.insert(0, dns.get("custom_name", ""))

            # Apply custom variables
            if "custom_variables" in settings and hasattr(self, "custom_vars_list"):
                self.custom_vars_list = settings["custom_variables"]
                if hasattr(self, "_update_custom_vars_display"):
                    self._update_custom_vars_display()

            # Apply output directory
            if "output_directory" in settings:
                self.output_directory = settings["output_directory"]
                if hasattr(self, "output_label"):
                    self.output_label.configure(text=f"Output: {self.output_directory}")

            saved_at = settings.get("saved_at", "Unknown time")
            messagebox.showinfo(
                "Success",
                f"Settings loaded successfully!\n\nConfiguration saved at: {saved_at}",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings:\n{e!s}")

    def manage_configurations(self) -> None:
        """Open a window to manage saved configuration files."""
        try:
            # Create a new window for configuration management
            config_window = ctk.CTkToplevel(self)
            config_window.title("Manage Saved Configurations")
            config_window.geometry("600x400")
            config_window.resizable(True, True)

            # Make it modal
            config_window.transient(self)
            config_window.grab_set()

            # Create main frame
            main_frame = ctk.CTkFrame(config_window)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)

            # Title
            ctk.CTkLabel(
                main_frame,
                text="Saved Configuration Files",
                font=ctk.CTkFont(weight="bold", size=16),
            ).pack(pady=(0, 10))

            # Create a frame for the list and buttons
            content_frame = ctk.CTkFrame(main_frame)
            content_frame.pack(fill="both", expand=True, padx=5, pady=5)

            # Create listbox with scrollbar
            list_frame = ctk.CTkFrame(content_frame)
            list_frame.pack(fill="both", expand=True, padx=5, pady=5)

            # Listbox for configurations
            self.config_listbox = tk.Listbox(
                list_frame,
                selectmode=tk.SINGLE,
                font=("Arial", 10),
            )
            config_scrollbar = tk.Scrollbar(
                list_frame,
                orient="vertical",
                command=self.config_listbox.yview,
            )
            self.config_listbox.configure(yscrollcommand=config_scrollbar.set)

            self.config_listbox.pack(
                side="left",
                fill="both",
                expand=True,
                padx=(5, 0),
                pady=5,
            )
            config_scrollbar.pack(side="right", fill="y", pady=5)

            # Button frame
            button_frame = ctk.CTkFrame(content_frame)
            button_frame.pack(fill="x", padx=5, pady=5)

            # Buttons
            ctk.CTkButton(
                button_frame,
                text="Refresh List",
                command=self._refresh_config_list,
            ).pack(side="left", padx=5, pady=5)
            ctk.CTkButton(
                button_frame,
                text="Load Selected",
                command=self._load_selected_config,
            ).pack(side="left", padx=5, pady=5)
            ctk.CTkButton(
                button_frame,
                text="Delete Selected",
                command=self._delete_selected_config,
            ).pack(side="left", padx=5, pady=5)
            ctk.CTkButton(
                button_frame,
                text="Open File Location",
                command=self._open_config_location,
            ).pack(side="left", padx=5, pady=5)
            ctk.CTkButton(
                button_frame,
                text="Close",
                command=config_window.destroy,
            ).pack(side="right", padx=5, pady=5)

            # Status label
            self.config_status_label = ctk.CTkLabel(
                main_frame,
                text="",
                font=ctk.CTkFont(size=11),
            )
            self.config_status_label.pack(pady=5)

            # Store the window reference
            self.config_management_window = config_window

            # Load initial list
            self._refresh_config_list()

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open configuration manager:\n{e!s}",
            )

    def _refresh_config_list(self) -> None:
        """Refresh the list of saved configuration files."""
        try:
            self.config_listbox.delete(0, tk.END)
            config_files = []

            # Get the current directory and look for .json files
            current_dir = os.getcwd()
            for file in os.listdir(current_dir):
                if file.endswith(".json"):
                    file_path = os.path.join(current_dir, file)
                    try:
                        # Try to read the file to see if it's a valid configuration
                        with open(file_path) as f:
                            data = json.load(f)
                            # Check if it has the expected structure (processing configs have 'saved_at', plotting configs have 'plot_name')
                            if isinstance(data, dict) and (
                                "saved_at" in data or "plot_name" in data
                            ):
                                if "saved_at" in data:
                                    config_files.append(
                                        (
                                            file,
                                            file_path,
                                            data.get("saved_at", "Unknown"),
                                            "Processing Config",
                                        ),
                                    )
                                elif "plot_name" in data:
                                    config_files.append(
                                        (
                                            file,
                                            file_path,
                                            data.get("created_date", "Unknown"),
                                            "Plot Config",
                                        ),
                                    )
                    except Exception:
                        # Skip files that can't be read as JSON or don't have the right structure
                        continue

            # Sort by creation date (newest first)
            config_files.sort(key=lambda x: x[2], reverse=True)

            # Add to listbox
            for filename, filepath, saved_at, config_type in config_files:
                display_text = f"{filename} ({config_type} - {saved_at})"
                self.config_listbox.insert(tk.END, display_text)
                # Store the filepath as item data
                self.config_listbox.itemconfig(
                    tk.END,
                    {
                        "bg": (
                            "lightgray"
                            if self.config_listbox.size() % 2 == 0
                            else "white"
                        ),
                    },
                )

            self.config_status_label.configure(
                text=f"Found {len(config_files)} configuration file(s)",
            )

        except Exception as e:
            self.config_status_label.configure(text=f"Error refreshing list: {e!s}")

    def _load_selected_config(self) -> None:
        """Load the selected configuration file."""
        try:
            selection = self.config_listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "Warning",
                    "Please select a configuration file to load.",
                )
                return

            # Get the selected item
            selected_index = selection[0]
            selected_text = self.config_listbox.get(selected_index)

            # Extract filename from the display text
            filename = selected_text.split(" (")[0]
            filepath = os.path.join(os.getcwd(), filename)

            # Load the configuration
            with open(filepath) as f:
                settings = json.load(f)

            # Check if it's a processing config or plot config
            if "saved_at" in settings:
                # Processing configuration
                self._apply_loaded_settings(settings)
            elif "plot_name" in settings:
                # Plot configuration - apply to plotting tab
                self._apply_plot_config(settings)
            else:
                messagebox.showerror("Error", "Unknown configuration file format.")
                return

            self.config_status_label.configure(text=f"Loaded configuration: {filename}")
            messagebox.showinfo(
                "Success",
                f"Configuration loaded successfully:\n{filename}",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e!s}")

    def _delete_selected_config(self) -> None:
        """Delete the selected configuration file."""
        try:
            selection = self.config_listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "Warning",
                    "Please select a configuration file to delete.",
                )
                return

            # Get the selected item
            selected_index = selection[0]
            selected_text = self.config_listbox.get(selected_index)

            # Extract filename from the display text
            filename = selected_text.split(" (")[0]
            filepath = os.path.join(os.getcwd(), filename)

            # Confirm deletion
            result = messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete this configuration file?\n\n{filename}\n\nThis action cannot be undone.",
            )
            if result:
                os.remove(filepath)
                self.config_status_label.configure(
                    text=f"Deleted configuration: {filename}",
                )
                self._refresh_config_list()
                messagebox.showinfo(
                    "Success",
                    f"Configuration deleted successfully:\n{filename}",
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete configuration:\n{e!s}")

    def _open_config_location(self) -> None:
        """Open the folder containing configuration files."""
        try:
            current_dir = os.getcwd()
            if os.name == "nt":  # Windows
                os.startfile(current_dir)
            elif os.name == "posix":  # macOS and Linux
                import subprocess

                subprocess.run(["open", current_dir], check=False)  # macOS
            else:
                import subprocess

                subprocess.run(["xdg-open", current_dir], check=False)  # Linux
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder:\n{e!s}")

    def _apply_loaded_settings(self, settings: dict[str, Any]) -> None:
        """Apply loaded settings to the UI (extracted from load_settings)."""
        try:
            # Apply filter settings
            if "filter_settings" in settings:
                fs = settings["filter_settings"]
                if hasattr(self, "filter_type_var"):
                    self.filter_type_var.set(fs.get("filter_type", "None"))
                    self._update_filter_ui(fs.get("filter_type", "None"))
                if hasattr(self, "ma_value_entry"):
                    self.ma_value_entry.delete(0, tk.END)
                    self.ma_value_entry.insert(0, fs.get("ma_window", "10"))
                if hasattr(self, "ma_unit_menu"):
                    self.ma_unit_menu.set(fs.get("ma_unit", "s"))
                if hasattr(self, "bw_order_entry"):
                    self.bw_order_entry.delete(0, tk.END)
                    self.bw_order_entry.insert(0, fs.get("bw_order", "3"))
                if hasattr(self, "bw_cutoff_entry"):
                    self.bw_cutoff_entry.delete(0, tk.END)
                    self.bw_cutoff_entry.insert(0, fs.get("bw_cutoff", "0.1"))
                if hasattr(self, "median_kernel_entry"):
                    self.median_kernel_entry.delete(0, tk.END)
                    self.median_kernel_entry.insert(0, fs.get("median_kernel", "5"))
                if hasattr(self, "hampel_window_entry"):
                    self.hampel_window_entry.delete(0, tk.END)
                    self.hampel_window_entry.insert(0, fs.get("hampel_window", "7"))
                if hasattr(self, "hampel_threshold_entry"):
                    self.hampel_threshold_entry.delete(0, tk.END)
                    self.hampel_threshold_entry.insert(
                        0,
                        fs.get("hampel_threshold", "3.0"),
                    )
                if hasattr(self, "zscore_threshold_entry"):
                    self.zscore_threshold_entry.delete(0, tk.END)
                    self.zscore_threshold_entry.insert(
                        0,
                        fs.get("zscore_threshold", "3.0"),
                    )
                if hasattr(self, "zscore_method_menu"):
                    self.zscore_method_menu.set(
                        fs.get("zscore_method", "Remove Outliers"),
                    )
                if hasattr(self, "savgol_window_entry"):
                    self.savgol_window_entry.delete(0, tk.END)
                    self.savgol_window_entry.insert(0, fs.get("savgol_window", "11"))
                if hasattr(self, "savgol_polyorder_entry"):
                    self.savgol_polyorder_entry.delete(0, tk.END)
                    self.savgol_polyorder_entry.insert(
                        0,
                        fs.get("savgol_polyorder", "2"),
                    )

            # Apply resample settings
            if "resample_settings" in settings:
                rs = settings["resample_settings"]
                if hasattr(self, "resample_var"):
                    self.resample_var.set(rs.get("enabled", False))
                if hasattr(self, "resample_value_entry"):
                    self.resample_value_entry.delete(0, tk.END)
                    self.resample_value_entry.insert(0, rs.get("value", "10"))
                if hasattr(self, "resample_unit_menu"):
                    self.resample_unit_menu.set(rs.get("unit", "s"))

            # Apply trim settings
            if "trim_settings" in settings:
                ts = settings["trim_settings"]
                if hasattr(self, "trim_date_entry"):
                    self.trim_date_entry.delete(0, tk.END)
                    self.trim_date_entry.insert(0, ts.get("date", ""))
                if hasattr(self, "trim_start_entry"):
                    self.trim_start_entry.delete(0, tk.END)
                    self.trim_start_entry.insert(0, ts.get("start_time", ""))
                if hasattr(self, "trim_end_entry"):
                    self.trim_end_entry.delete(0, tk.END)
                    self.trim_end_entry.insert(0, ts.get("end_time", ""))

            # Apply integration settings
            if "integration_settings" in settings:
                is_settings = settings["integration_settings"]
                if hasattr(self, "integrator_method_var"):
                    self.integrator_method_var.set(
                        is_settings.get("method", "Trapezoidal"),
                    )

            # Apply differentiation settings
            if "differentiation_settings" in settings:
                ds = settings["differentiation_settings"]
                if hasattr(self, "deriv_method_var"):
                    self.deriv_method_var.set(ds.get("method", "Spline (Acausal)"))
                if hasattr(self, "derivative_vars") and "orders" in ds:
                    for order, enabled in ds["orders"].items():
                        if order in self.derivative_vars:
                            self.derivative_vars[order].set(enabled)

            # Apply export settings
            if "export_settings" in settings:
                es = settings["export_settings"]
                if hasattr(self, "export_type_var"):
                    self.export_type_var.set(es.get("type", "CSV (Separate Files)"))
                if hasattr(self, "sort_col_menu"):
                    self.sort_col_menu.set(es.get("sort_column", "No Sorting"))
                if hasattr(self, "sort_order_var"):
                    self.sort_order_var.set(es.get("sort_order", "Ascending"))

            # Apply dataset naming settings
            if "dataset_naming" in settings:
                dn = settings["dataset_naming"]
                if hasattr(self, "dataset_naming_var"):
                    self.dataset_naming_var.set(dn.get("mode", "auto"))
                if hasattr(self, "custom_dataset_entry"):
                    self.custom_dataset_entry.delete(0, tk.END)
                    self.custom_dataset_entry.insert(0, dn.get("custom_name", ""))

            # Apply custom variables
            if "custom_variables" in settings:
                self.custom_vars_list = settings["custom_variables"]
                self._update_custom_vars_display()

            # Apply output directory
            if "output_directory" in settings:
                self.output_directory = settings["output_directory"]
                if hasattr(self, "output_label"):
                    self.output_label.configure(text=f"Output: {self.output_directory}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings:\n{e!s}")

    def save_signal_list(self) -> None:
        """Save the currently selected signals as a signal list."""
        if not self.signal_vars:
            messagebox.showwarning(
                "Warning",
                "No signals available to save. Please load a file first.",
            )
            return

        # Get currently selected signals
        selected_signals = [
            signal for signal, data in self.signal_vars.items() if data["var"].get()
        ]

        if not selected_signals:
            messagebox.showwarning(
                "Warning",
                "No signals are currently selected. Please select signals to save.",
            )
            return

        # Ask user for a name for this signal list
        signal_list_name = tk.simpledialog.askstring(
            "Save Signal List",
            "Enter a name for this signal list:",
            initialvalue="My Signal List",
        )

        if not signal_list_name:
            return  # User cancelled

        # Create the signal list data
        signal_list_data = {
            "name": signal_list_name,
            "signals": selected_signals,
            "created_date": pd.Timestamp.now().isoformat(),
        }

        # Save to file
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Signal List",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"{signal_list_name}.json",
            )

            if file_path:
                with open(file_path, "w") as f:
                    json.dump(signal_list_data, f, indent=2)

                # No popup message - just update status bar for better user experience
                self.status_label.configure(
                    text=f"Signal list saved: {signal_list_name} ({len(selected_signals)} signals)",
                )
                print(
                    f"DEBUG: Signal list '{signal_list_name}' saved successfully with {len(selected_signals)} signals",
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save signal list:\n{e}")

    def load_signal_list(self) -> None:
        """Load a saved signal list from file."""
        print("DEBUG: load_signal_list() called")
        try:
            print("DEBUG: Opening file dialog")
            file_path = filedialog.askopenfilename(
                title="Load Signal List",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            print(f"DEBUG: File dialog returned: {file_path}")

            if not file_path:
                print("DEBUG: No file selected, returning")
                return  # User cancelled

            print(f"DEBUG: Loading file: {file_path}")
            with open(file_path) as f:
                signal_list_data = json.load(f)
            print(f"DEBUG: Successfully loaded JSON data: {signal_list_data}")
            print(f"DEBUG: Successfully loaded JSON data: {signal_list_data}")

            # Validate the loaded data
            print("DEBUG: Validating loaded data")
            if (
                not isinstance(signal_list_data, dict)
                or "signals" not in signal_list_data
            ):
                print("DEBUG: Invalid signal list file format")
                messagebox.showerror("Error", "Invalid signal list file format.")
                return

            # Store the loaded signal list
            print("DEBUG: Storing loaded signal list")
            self.saved_signal_list = signal_list_data.get("signals", [])
            self.saved_signal_list_name = signal_list_data.get("name", "Unknown")
            print(f"DEBUG: Saved signal list: {len(self.saved_signal_list)} signals")

            # Update status
            print("DEBUG: Updating status label")
            self.signal_list_status_label.configure(
                text=f"Loaded: {self.saved_signal_list_name} ({len(self.saved_signal_list)} signals)",
                text_color="green",
            )

            # Create a signal list from the loaded signals even if no files are loaded
            print("DEBUG: Creating signal list from loaded signals")
            self.update_signal_list(self.saved_signal_list)

            # Automatically apply the loaded signals if we have signals available
            print(f"DEBUG: Checking if signal_vars exist: {bool(self.signal_vars)}")
            if self.signal_vars:
                print("DEBUG: Applying loaded signals internally")
                self._apply_loaded_signals_internal()

            print("DEBUG: Signal list loaded successfully without popup")
            # No popup message - just update status bar for better user experience
            self.status_label.configure(
                text=f"Signal list loaded: {self.saved_signal_list_name} ({len(self.saved_signal_list)} signals)",
            )
            print("DEBUG: load_signal_list() completed successfully")

        except Exception as e:
            print(f"DEBUG: Exception in load_signal_list: {e}")
            import traceback

            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load signal list:\n{e}")

    def _apply_loaded_signals_internal(self) -> None:
        """Internal method to apply loaded signals without showing message boxes."""
        print("DEBUG: _apply_loaded_signals_internal() called")
        if not self.saved_signal_list or not self.signal_vars:
            print(
                f"DEBUG: Early return - saved_signal_list: {bool(self.saved_signal_list)}, signal_vars: {bool(self.signal_vars)}",
            )
            return

        # Get current available signals
        available_signals = list(self.signal_vars.keys())
        print(f"DEBUG: Available signals: {len(available_signals)}")

        # Find which saved signals are present
        present_signals = []
        missing_signals = []

        print("DEBUG: Checking saved signals against available signals")
        for saved_signal in self.saved_signal_list:
            if saved_signal in available_signals:
                present_signals.append(saved_signal)
            else:
                missing_signals.append(saved_signal)

        print(
            f"DEBUG: Present signals: {len(present_signals)}, Missing signals: {len(missing_signals)}",
        )

        # Apply the saved signals (select present ones, deselect others)
        print("DEBUG: Applying signal selections")
        for signal, data in self.signal_vars.items():
            if signal in present_signals:
                data["var"].set(True)
            else:
                data["var"].set(False)

        # Update status
        print("DEBUG: Updating status label")
        self.signal_list_status_label.configure(
            text=f"Applied: {self.saved_signal_list_name} ({len(present_signals)}/{len(self.saved_signal_list)} signals)",
            text_color="blue",
        )
        print("DEBUG: _apply_loaded_signals_internal() completed")

    def apply_saved_signals(self) -> None:
        """Apply the saved signal list to the current file's signals.

        This function takes a previously saved signal list and applies it to the currently loaded files.
        It will:
        1. Select all signals that are present in both the saved list and current files
        2. Deselect all signals that are not in the saved list
        3. Show you which signals from the saved list are missing from current files
        """
        if not self.saved_signal_list:
            messagebox.showwarning(
                "Warning",
                "No saved signal list loaded. Please load a signal list first.",
            )
            return

        if not self.signal_vars:
            messagebox.showwarning(
                "Warning",
                "No signals available. Please load a file first.",
            )
            return

        # Get current available signals
        available_signals = list(self.signal_vars.keys())

        # Find which saved signals are present and which are missing
        present_signals = []
        missing_signals = []

        for saved_signal in self.saved_signal_list:
            if saved_signal in available_signals:
                present_signals.append(saved_signal)
            else:
                missing_signals.append(saved_signal)

        # Apply the saved signals (select present ones, deselect others)
        for signal, data in self.signal_vars.items():
            if signal in present_signals:
                data["var"].set(True)
            else:
                data["var"].set(False)

        # Show results to user
        if missing_signals:
            missing_text = "\n".join([f"• {signal}" for signal in missing_signals])
            messagebox.showinfo(
                "Signals Applied",
                f"Applied {len(present_signals)} signals from '{self.saved_signal_list_name}'.\n\n"
                f"Missing signals ({len(missing_signals)}):\n{missing_text}",
            )
        else:
            messagebox.showinfo(
                "Signals Applied",
                f"Successfully applied all {len(present_signals)} signals from '{self.saved_signal_list_name}'.",
            )

        # Update status
        self.signal_list_status_label.configure(
            text=f"Applied: {self.saved_signal_list_name} ({len(present_signals)}/{len(self.saved_signal_list)} signals)",
            text_color="blue",
        )

        self.status_label.configure(
            text=f"Applied {len(present_signals)} signals from saved list",
        )

    def _copy_plot_settings_to_processing(self) -> None:
        """Copies filter settings from the plot tab to the main processing tab."""
        plot_filter = self.plot_filter_type.get()
        self.filter_type_var.set(plot_filter)
        self._update_filter_ui(plot_filter)

        # Copy filter parameters
        if plot_filter == "Moving Average":
            if hasattr(self, "plot_ma_value_entry") and hasattr(
                self,
                "plot_ma_unit_menu",
            ):
                self.ma_value_entry.delete(0, tk.END)
                self.ma_value_entry.insert(0, self.plot_ma_value_entry.get())
                self.ma_unit_menu.set(self.plot_ma_unit_menu.get())
        elif plot_filter == "Butterworth":
            if hasattr(self, "plot_bw_order_entry") and hasattr(
                self,
                "plot_bw_cutoff_entry",
            ):
                self.bw_order_entry.delete(0, tk.END)
                self.bw_order_entry.insert(0, self.plot_bw_order_entry.get())
                self.bw_cutoff_entry.delete(0, tk.END)
                self.bw_cutoff_entry.insert(0, self.plot_bw_cutoff_entry.get())
        elif plot_filter == "Median Filter":
            if hasattr(self, "plot_median_kernel_entry"):
                self.median_kernel_entry.delete(0, tk.END)
                self.median_kernel_entry.insert(0, self.plot_median_kernel_entry.get())
        elif plot_filter == "Hampel Filter":
            if hasattr(self, "plot_hampel_window_entry") and hasattr(
                self,
                "plot_hampel_threshold_entry",
            ):
                self.hampel_window_entry.delete(0, tk.END)
                self.hampel_window_entry.insert(0, self.plot_hampel_window_entry.get())
                self.hampel_threshold_entry.delete(0, tk.END)
                self.hampel_threshold_entry.insert(
                    0,
                    self.plot_hampel_threshold_entry.get(),
                )
        elif plot_filter == "Z-Score Filter":
            if hasattr(self, "plot_zscore_threshold_entry") and hasattr(
                self,
                "plot_zscore_method_menu",
            ):
                self.zscore_threshold_entry.delete(0, tk.END)
                self.zscore_threshold_entry.insert(
                    0,
                    self.plot_zscore_threshold_entry.get(),
                )
                self.zscore_method_menu.set(self.plot_zscore_method_menu.get())
        elif plot_filter == "Savitzky-Golay":
            if hasattr(self, "plot_savgol_window_entry") and hasattr(
                self,
                "plot_savgol_polyorder_entry",
            ):
                self.savgol_window_entry.delete(0, tk.END)
                self.savgol_window_entry.insert(0, self.plot_savgol_window_entry.get())
                self.savgol_polyorder_entry.delete(0, tk.END)
                self.savgol_polyorder_entry.insert(
                    0,
                    self.plot_savgol_polyorder_entry.get(),
                )

        messagebox.showinfo(
            "Settings Copied",
            "Filter settings from the plot tab have been applied to the main processing configuration.",
        )

    def _export_chart_image(self) -> None:
        """Export the current chart as an image file."""
        if not hasattr(self, "plot_fig") or not self.plot_fig.get_axes():
            messagebox.showwarning(
                "Warning",
                "No plot to export. Please create a plot first.",
            )
            return

        try:
            file_types = [
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg"),
            ]

            save_path = filedialog.asksaveasfilename(
                title="Export Chart As Image",
                filetypes=file_types,
                defaultextension=".png",
            )

            if save_path:
                # Check for overwrite and get final path
                final_path = self._check_file_overwrite(save_path)
                if final_path is None:  # User cancelled
                    return

                self.plot_fig.savefig(
                    final_path,
                    dpi=DEFAULT_DPI,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )
                messagebox.showinfo("Success", f"Chart exported to:\n{final_path}")
                self.status_label.configure(
                    text=f"Chart exported: {os.path.basename(final_path)}",
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart:\n{e}")

    def _export_chart_excel(self) -> None:
        """Export the current plot data and chart to Excel."""
        selected_file = self.plot_file_menu.get()

        if selected_file == "Select a file...":
            messagebox.showwarning("Warning", "Please select a file to plot first.")
            return

        try:
            save_path = filedialog.asksaveasfilename(
                title="Export Chart Data to Excel",
                filetypes=[("Excel files", "*.xlsx")],
                defaultextension=".xlsx",
            )

            if save_path:
                # Check for overwrite and get final path
                final_path = self._check_file_overwrite(save_path)
                if final_path is None:  # User cancelled
                    return

                df = self.get_data_for_plotting(selected_file)
                if df is not None and not df.empty:
                    signals_to_plot = [
                        s
                        for s, data in self.plot_signal_vars.items()
                        if data["var"].get()
                    ]

                    if signals_to_plot:
                        # Filter data to only include plotted signals
                        export_df = df[signals_to_plot].copy()

                        # Add time column if it exists
                        time_col = None
                        for col in df.columns:
                            if pd.api.types.is_datetime64_any_dtype(df[col]):
                                time_col = col
                                break

                        if time_col:
                            export_df.insert(0, time_col, df[time_col])

                        # Export to Excel
                        with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
                            export_df.to_excel(
                                writer,
                                sheet_name="Chart Data",
                                index=False,
                            )

                            # Add chart information
                            info_df = pd.DataFrame(
                                {
                                    "Property": [
                                        "File",
                                        "Title",
                                        "X-Axis",
                                        "Y-Axis",
                                        "Signals Plotted",
                                    ],
                                    "Value": [
                                        selected_file,
                                        self.plot_title_entry.get() or "No title",
                                        self.plot_xlabel_entry.get() or "No label",
                                        self.plot_ylabel_entry.get() or "No label",
                                        ", ".join(signals_to_plot),
                                    ],
                                },
                            )
                            info_df.to_excel(
                                writer,
                                sheet_name="Chart Info",
                                index=False,
                            )

                        messagebox.showinfo(
                            "Success",
                            f"Chart data exported to:\n{final_path}",
                        )
                        self.status_label.configure(
                            text=f"Chart data exported: {os.path.basename(final_path)}",
                        )
                    else:
                        messagebox.showwarning(
                            "Warning",
                            "No signals selected for plotting.",
                        )
                else:
                    messagebox.showerror("Error", "Could not load data for export.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart data:\n{e}")

    def _add_plot_to_list(self) -> None:
        """Add plot to the plots list."""
        plot_name = self.plot_name_entry.get().strip()
        plot_desc = self.plot_desc_entry.get().strip()

        if not plot_name:
            messagebox.showerror("Error", "Please enter a plot name.")
            return

        # Get selected signals from plots signals frame
        selected_signals = []
        if hasattr(self, "plots_signal_vars"):
            selected_signals = [
                signal for signal, var in self.plots_signal_vars.items() if var.get()
            ]

        plot_config = {
            "name": plot_name,
            "description": plot_desc
            or f"Plot configuration created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "signals": selected_signals,
            "start_time": self.plots_list_start_time_entry.get(),
            "end_time": self.plots_list_end_time_entry.get(),
            "created_date": pd.Timestamp.now().isoformat(),
        }

        self.plots_list.append(plot_config)
        self._update_plots_listbox()
        self._save_plots_to_file()
        self._clear_plot_form()

        messagebox.showinfo("Success", f"Plot '{plot_name}' added to list!")

    def _update_selected_plot(self) -> None:
        """Update selected plot in the list."""
        selection = self.plots_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a plot to update.")
            return

        plot_name = self.plot_name_entry.get().strip()
        if not plot_name:
            messagebox.showerror("Error", "Please enter a plot name.")
            return

        idx = selection[0]
        selected_signals = []
        if hasattr(self, "plots_signal_vars"):
            selected_signals = [
                signal for signal, var in self.plots_signal_vars.items() if var.get()
            ]

        self.plots_list[idx].update(
            {
                "name": plot_name,
                "description": self.plot_desc_entry.get().strip(),
                "signals": selected_signals,
                "start_time": self.plots_list_start_time_entry.get(),
                "end_time": self.plots_list_end_time_entry.get(),
                "modified_date": pd.Timestamp.now().isoformat(),
            },
        )

        self._update_plots_listbox()
        self._save_plots_to_file()
        messagebox.showinfo("Success", "Plot configuration updated!")

    def _clear_plot_form(self) -> None:
        """Clear the plot form."""
        self.plot_name_entry.delete(0, tk.END)
        self.plot_desc_entry.delete(0, tk.END)
        self.plots_list_start_time_entry.delete(0, tk.END)
        self.plots_list_end_time_entry.delete(0, tk.END)

        # Clear signal selections
        if hasattr(self, "plots_signal_vars"):
            for var in self.plots_signal_vars.values():
                var.set(False)

    def _on_plot_select(self, event: tk.Event) -> None:
        """Handle plot selection in listbox."""
        selection = self.plots_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        plot_config = self.plots_list[idx]

        # Populate form with selected plot data
        self.plot_name_entry.delete(0, tk.END)
        self.plot_name_entry.insert(0, plot_config.get("name", ""))

        self.plot_desc_entry.delete(0, tk.END)
        self.plot_desc_entry.insert(0, plot_config.get("description", ""))

        self.plots_list_start_time_entry.delete(0, tk.END)
        self.plots_list_start_time_entry.insert(0, plot_config.get("start_time", ""))

        self.plots_list_end_time_entry.delete(0, tk.END)
        self.plots_list_end_time_entry.insert(0, plot_config.get("end_time", ""))

        # Update signal selections
        if hasattr(self, "plots_signal_vars"):
            saved_signals = plot_config.get("signals", [])
            for signal, var in self.plots_signal_vars.items():
                var.set(signal in saved_signals)

    def _load_selected_plot(self) -> None:
        """Load selected plot configuration."""
        selection = self.plots_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a plot to load.")
            return

        idx = selection[0]
        plot_config = self.plots_list[idx]

        print(f"DEBUG: Loading plot config: {plot_config.get('name', 'Unknown')}")
        print(f"DEBUG: File in config: '{plot_config.get('file', '')}'")

        # Apply the plot configuration using the same method as the main plotting tab
        self._apply_plot_config(plot_config)

        # Switch to plotting tab to show the loaded configuration
        if hasattr(self, "tabview"):
            self.tabview.set("Plotting")

        messagebox.showinfo(
            "Success",
            f"Plot configuration '{plot_config['name']}' loaded and applied to Plotting tab!",
        )

    def _delete_selected_plot(self) -> None:
        """Delete selected plot from list."""
        selection = self.plots_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a plot to delete.")
            return

        idx = selection[0]
        plot_name = self.plots_list[idx]["name"]

        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete plot '{plot_name}'?",
        ):
            del self.plots_list[idx]
            self._update_plots_listbox()
            self._save_plots_to_file()
            self._clear_plot_form()
            messagebox.showinfo("Success", f"Plot '{plot_name}' deleted.")

    def _clear_all_plots(self) -> None:
        """Clear all plots from list."""
        if self.plots_list and messagebox.askyesno(
            "Confirm Clear",
            "Are you sure you want to clear all plots?",
        ):
            self.plots_list.clear()
            self._update_plots_listbox()
            self._save_plots_to_file()
            self._clear_plot_form()
            messagebox.showinfo("Success", "All plots cleared.")

    def _update_plots_listbox(self) -> None:
        """Update the plots listbox with current plots."""
        self.plots_listbox.delete(0, tk.END)
        for plot in self.plots_list:
            display_text = f"{plot['name']} ({len(plot.get('signals', []))} signals)"
            self.plots_listbox.insert(tk.END, display_text)

    def _save_plots_to_file(self) -> None:
        """Save plots list to file."""
        try:
            plots_file = os.path.join(
                os.path.expanduser("~"),
                ".csv_processor_plots.json",
            )
            with open(plots_file, "w") as f:
                json.dump(self.plots_list, f, indent=2)
        except Exception as e:
            print(f"Error saving plots to file: {e}")

    def _load_plots_from_file(self) -> None:
        """Load plots list from file."""
        try:
            plots_file = os.path.join(
                os.path.expanduser("~"),
                ".csv_processor_plots.json",
            )
            if os.path.exists(plots_file):
                with open(plots_file) as f:
                    self.plots_list = json.load(f)
                self._update_plots_listbox()
                self._update_load_plot_config_menu()
        except Exception as e:
            print(f"Error loading plots from file: {e}")
            self.plots_list = []

    def _select_tag_file(self) -> None:
        """Select tag file for DAT import."""
        filepath = filedialog.askopenfilename(
            title="Select Tag File",
            filetypes=[("DBF files", "*.dbf"), ("All files", "*.*")],
        )
        if filepath:
            self.dat_import_tag_file_path = filepath
            self.tag_file_label.configure(text=os.path.basename(filepath))

    def _select_data_file(self) -> None:
        """Select data file for DAT import."""
        filepath = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
        )
        if filepath:
            self.dat_import_data_file_path = filepath
            self.data_file_label.configure(text=os.path.basename(filepath))

            # Set default output directory to the folder of the selected DAT file
            dat_file_dir = os.path.dirname(filepath)
            self.output_directory = dat_file_dir
            # Update the output label to reflect the new default directory
            if hasattr(self, "output_label"):
                self.output_label.configure(text=f"Output: {self.output_directory}")

    def _import_selected_tags(self) -> None:
        """Import selected tags."""
        if not self.dat_import_data_file_path:
            messagebox.showerror("Error", "Please select a data file first.")
            return

        selected_tags = [tag for tag, var in self.dat_tag_vars.items() if var.get()]
        if not selected_tags:
            messagebox.showerror("Error", "Please select at least one tag to import.")
            return

        try:
            # Load and process the DAT file with selected tags
            df = pd.read_csv(self.dat_import_data_file_path, sep="\t", low_memory=False)

            # Filter to only selected tags
            if "Time" in df.columns:
                selected_columns = ["Time"] + [
                    col for col in selected_tags if col in df.columns
                ]
            else:
                selected_columns = [col for col in selected_tags if col in df.columns]

            filtered_df = df[selected_columns]

            # Save to CSV
            output_path = filedialog.asksaveasfilename(
                title="Save Imported DAT Data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )

            if output_path:
                filtered_df.to_csv(output_path, index=False)
                messagebox.showinfo(
                    "Success",
                    f"Data imported and saved to {output_path}",
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to import data: {e!s}")

    def trim_and_save(self) -> None:
        """Trim data and save."""
        if not self.input_file_paths:
            messagebox.showerror("Error", "Please select input files first.")
            return

        trim_date = self.trim_date_entry.get()
        trim_start = self.trim_start_entry.get()
        trim_end = self.trim_end_entry.get()

        if not any([trim_date, trim_start, trim_end]):
            messagebox.showerror(
                "Error",
                "Please specify at least one time parameter for trimming.",
            )
            return

        for file_path in self.input_file_paths:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                time_col = df.columns[0]

                # Convert time column
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df.dropna(subset=[time_col], inplace=True)

                # Apply time trimming
                if trim_date or trim_start or trim_end:
                    # Get the date from the data if not specified
                    if not trim_date:
                        trim_date = df[time_col].iloc[0].strftime("%Y-%m-%d")

                    # Create full datetime strings
                    start_time_str = trim_start or DEFAULT_START_TIME
                    end_time_str = trim_end or DEFAULT_END_TIME
                    start_full_str = f"{trim_date} {start_time_str}"
                    end_full_str = f"{trim_date} {end_time_str}"

                    # Filter the data by time range
                    df = (
                        df.set_index(time_col)
                        .loc[start_full_str:end_full_str]
                        .reset_index()
                    )

                # Save trimmed file
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(
                    self.output_directory,
                    f"{base_name}_Trimmed.csv",
                )
                df.to_csv(output_path, index=False)

            except Exception as e:
                print(f"Error trimming {file_path}: {e}")

        messagebox.showinfo(
            "Success",
            f"Files trimmed and saved to {self.output_directory}",
        )

    def _apply_plot_time_range(self) -> None:
        """Apply time range to plot."""
        start_time_str = self.plotting_start_time_entry.get()
        end_time_str = self.plotting_end_time_entry.get()

        if not start_time_str and not end_time_str:
            return

        selected_file = self.plot_file_menu.get()
        if selected_file == "Select a file...":
            return

        df = self.get_data_for_plotting(selected_file)
        if df is None or df.empty:
            return

        # Find time column
        time_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = col
                break

        if not time_col:
            messagebox.showerror("Error", "No datetime column found in the data.")
            return

        try:
            # Get the date from the data
            date_str = df[time_col].iloc[0].strftime("%Y-%m-%d")

            # Create full datetime strings
            start_full_str = (
                f"{date_str} {start_time_str}"
                if start_time_str
                else f"{date_str} {DEFAULT_START_TIME}"
            )
            end_full_str = (
                f"{date_str} {end_time_str}" if end_time_str else f"{date_str} {DEFAULT_END_TIME}"
            )

            # Filter the data
            filtered_df = (
                df.set_index(time_col).loc[start_full_str:end_full_str].reset_index()
            )

            if filtered_df.empty:
                messagebox.showwarning(
                    "Warning",
                    "The specified time range resulted in an empty dataset.",
                )
                return

            # Update the plot with filtered data
            self.plot_ax.clear()

            signals_to_plot = [
                s for s, data in self.plot_signal_vars.items() if data["var"].get()
            ]

            if not signals_to_plot:
                self.plot_ax.text(
                    0.5,
                    0.5,
                    "Select one or more signals to plot",
                    ha="center",
                    va="center",
                )
            else:
                # Apply filter preview if selected
                plot_filter = self.plot_filter_type.get()
                if plot_filter != "None":
                    filtered_df = self._apply_plot_filter(
                        filtered_df,
                        signals_to_plot,
                        time_col,
                    )

                # Chart customization
                plot_style = self.plot_type_var.get()
                style_args = {"linestyle": "-", "marker": ""}
                if plot_style == "Line with Markers":
                    style_args = {"linestyle": "-", "marker": ".", "markersize": 4}
                elif plot_style == "Markers Only (Scatter)":
                    style_args = {"linestyle": "None", "marker": ".", "markersize": 5}

                # Apply line width
                line_width = float(self.line_width_var.get())
                style_args["linewidth"] = line_width

                # Get color scheme
                color_scheme = self.color_scheme_var.get()
                if color_scheme == "Auto (Matplotlib)":
                    colors = plt.cm.tab10(np.linspace(0, 1, len(signals_to_plot)))
                elif color_scheme == "Viridis":
                    colors = plt.cm.viridis(np.linspace(0, 1, len(signals_to_plot)))
                elif color_scheme == "Plasma":
                    colors = plt.cm.plasma(np.linspace(0, 1, len(signals_to_plot)))
                elif color_scheme == "Cool":
                    colors = plt.cm.cool(np.linspace(0, 1, len(signals_to_plot)))
                elif color_scheme == "Warm":
                    colors = plt.cm.autumn(np.linspace(0, 1, len(signals_to_plot)))
                elif color_scheme == "Rainbow":
                    colors = plt.cm.rainbow(np.linspace(0, 1, len(signals_to_plot)))
                else:  # Custom Colors - default to tab10
                    colors = plt.cm.Set1(np.linspace(0, 1, len(signals_to_plot)))

                # Plot each selected signal
                for i, signal in enumerate(signals_to_plot):
                    if signal not in filtered_df.columns:
                        continue

                    plot_df = filtered_df[[time_col, signal]].dropna()
                    plot_style = style_args.copy()
                    plot_style["color"] = colors[i]
                    signal_label = self.custom_legend_entries.get(signal, signal)
                    self.plot_ax.plot(
                        plot_df[time_col],
                        plot_df[signal],
                        label=signal_label,
                        **plot_style,
                    )

                # Add trendline if selected
                if self.trendline_type_var.get() != "None":
                    selected_trendline_signal = self.trendline_signal_var.get()
                    if (
                        selected_trendline_signal != "Select signal..."
                        and selected_trendline_signal in filtered_df.columns
                    ):
                        self._add_trendline(
                            filtered_df,
                            selected_trendline_signal,
                            time_col,
                        )

            # Apply custom labels and title
            title = (
                self.plot_title_entry.get()
                or f"Signals from {selected_file} (Time Range: {start_time_str} - {end_time_str})"
            )
            xlabel = self.plot_xlabel_entry.get() or time_col
            ylabel = self.plot_ylabel_entry.get() or "Value"
            self.plot_ax.set_title(title, fontsize=14)
            self.plot_ax.set_xlabel(xlabel)
            self.plot_ax.set_ylabel(ylabel)

            # Apply legend with custom position
            legend_position = self.legend_position_var.get()
            if legend_position == "outside right":
                # For outside right, use bbox_to_anchor to place legend outside the plot area
                self.plot_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                self.plot_ax.legend(loc=legend_position)
            self.plot_ax.grid(True, linestyle="--", alpha=0.6)

            if pd.api.types.is_datetime64_any_dtype(filtered_df[time_col]):
                # Use simpler HH:MM format for cleaner plot appearance
                self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                # Keep labels horizontal for better readability
                self.plot_ax.tick_params(axis="x", rotation=0)

            self.plot_canvas.draw()

        except Exception as e:
            messagebox.showerror(
                "Time Range Error",
                f"Invalid time format. Please use HH:MM:SS.\n{e}",
            )

    def _reset_plot_range(self) -> None:
        """Reset plot range."""
        self.plotting_start_time_entry.delete(0, tk.END)
        self.plotting_end_time_entry.delete(0, tk.END)
        self.update_plot()

    def _copy_trim_to_plot_range(self) -> None:
        """Copy trim times to plot range."""
        start_time = self.trim_start_entry.get()
        end_time = self.trim_end_entry.get()

        if start_time:
            self.plotting_start_time_entry.delete(0, tk.END)
            self.plotting_start_time_entry.insert(0, start_time)

        if end_time:
            self.plotting_end_time_entry.delete(0, tk.END)
            self.plotting_end_time_entry.insert(0, end_time)

        self._apply_plot_time_range()

    def _copy_plot_range_to_trim(self) -> None:
        """Copy current plot x-axis range to time trimming fields."""
        try:
            # Check if plot exists and has data
            if not hasattr(self, "plot_ax") or not self.plot_ax.lines:
                messagebox.showwarning(
                    "Warning",
                    "No plot data available. Please create a plot first.",
                )
                return

            # Get current x-axis limits
            xlim = self.plot_ax.get_xlim()

            # Convert matplotlib date numbers to datetime
            start_datetime = mdates.num2date(xlim[0])
            end_datetime = mdates.num2date(xlim[1])

            # Extract date and time components
            date_str = start_datetime.strftime("%Y-%m-%d")
            start_time_str = start_datetime.strftime("%H:%M:%S")
            end_time_str = end_datetime.strftime("%H:%M:%S")

            # Update the trim fields
            self.trim_date_entry.delete(0, tk.END)
            self.trim_date_entry.insert(0, date_str)

            self.trim_start_entry.delete(0, tk.END)
            self.trim_start_entry.insert(0, start_time_str)

            self.trim_end_entry.delete(0, tk.END)
            self.trim_end_entry.insert(0, end_time_str)

            messagebox.showinfo(
                "Success",
                f"Copied plot range to time trimming:\nDate: {date_str}\nStart: {start_time_str}\nEnd: {end_time_str}",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy plot range: {e!s}")

    def _save_current_plot_view(self) -> None:
        """Save the current plot view state."""
        try:
            if not hasattr(self, "plot_ax"):
                messagebox.showwarning("Warning", "No plot available.")
                return

            # Save current view limits
            self.saved_plot_view = {
                "xlim": self.plot_ax.get_xlim(),
                "ylim": self.plot_ax.get_ylim(),
            }

            messagebox.showinfo(
                "Success",
                "Current plot view saved! Use the Home button on the toolbar to return to this view.",
            )

            # Override the home button functionality
            self._override_home_button()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot view: {e!s}")

    def _copy_current_view_to_processing(self) -> None:
        """Copy current plot view range to processing tab time trimming."""
        try:
            # This is essentially the same as _copy_plot_range_to_trim but with a different message
            if not hasattr(self, "plot_ax") or not self.plot_ax.lines:
                messagebox.showwarning(
                    "Warning",
                    "No plot data available. Please create a plot first.",
                )
                return

            # Get current x-axis limits
            xlim = self.plot_ax.get_xlim()

            # Convert matplotlib date numbers to datetime
            start_datetime = mdates.num2date(xlim[0])
            end_datetime = mdates.num2date(xlim[1])

            # Extract date and time components
            date_str = start_datetime.strftime("%Y-%m-%d")
            start_time_str = start_datetime.strftime("%H:%M:%S")
            end_time_str = end_datetime.strftime("%H:%M:%S")

            # Update the trim fields
            self.trim_date_entry.delete(0, tk.END)
            self.trim_date_entry.insert(0, date_str)

            self.trim_start_entry.delete(0, tk.END)
            self.trim_start_entry.insert(0, start_time_str)

            self.trim_end_entry.delete(0, tk.END)
            self.trim_end_entry.insert(0, end_time_str)

            # Switch to the Processing tab
            self.main_tab_view.set("Processing")

            messagebox.showinfo(
                "Success",
                f"Copied current view to Processing tab time trimming:\nDate: {date_str}\nStart: {start_time_str}\nEnd: {end_time_str}",
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy current view: {e!s}")

    def _override_home_button(self) -> None:
        """Override the matplotlib toolbar home button to use saved view."""
        if hasattr(self, "plot_toolbar") and self.saved_plot_view:
            # Store original home function
            if not hasattr(self, "_original_home"):
                self._original_home = self.plot_toolbar.home

            # Create custom home function
            def custom_home() -> None:
                """Custom home function that restores saved plot view."""
                try:
                    if self.saved_plot_view:
                        self.plot_ax.set_xlim(self.saved_plot_view["xlim"])
                        self.plot_ax.set_ylim(self.saved_plot_view["ylim"])
                        self.plot_canvas.draw()
                    else:
                        # Fall back to original home if no saved view
                        self._original_home()
                except Exception:
                    # Fall back to original home on any error
                    self._original_home()

            # Replace the home function
            self.plot_toolbar.home = custom_home

    def _refresh_legend_entries(self) -> None:
        """Refresh legend entries based on currently selected signals."""
        # Clear existing legend widgets
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        # Get currently selected signals
        selected_signals = []
        if hasattr(self, "plot_signal_vars"):
            selected_signals = [
                signal
                for signal, data in self.plot_signal_vars.items()
                if data["var"].get()
            ]

        if not selected_signals:
            ctk.CTkLabel(
                self.legend_frame,
                text="Select signals to customize legend labels",
            ).pack(padx=5, pady=5)
            ctk.CTkLabel(
                self.legend_frame,
                text="Tip: For subscripts use $H_2O$, $CO_2$, $v_{max}$",
                font=ctk.CTkFont(size=10),
                text_color="gray",
            ).pack(padx=5, pady=2)
            return

        # Initialize legend order if not exists
        if not hasattr(self, "legend_order"):
            self.legend_order = selected_signals.copy()
        else:
            # Update legend order to include new signals
            for signal in selected_signals:
                if signal not in self.legend_order:
                    self.legend_order.append(signal)
            # Remove signals that are no longer selected
            self.legend_order = [s for s in self.legend_order if s in selected_signals]

        # Create entry widgets for each selected signal in legend order
        for i, signal in enumerate(self.legend_order):
            if signal not in selected_signals:
                continue

            signal_frame = ctk.CTkFrame(self.legend_frame)
            signal_frame.pack(fill="x", padx=5, pady=2)

            # Move up button
            if i > 0:
                up_btn = ctk.CTkButton(
                    signal_frame,
                    text="↑",
                    width=25,
                    height=25,
                    command=lambda s=signal: self._move_legend_up(s),
                )
                up_btn.pack(side="left", padx=2, pady=2)
            else:
                # Placeholder for alignment
                placeholder = ctk.CTkLabel(signal_frame, text="", width=25)
                placeholder.pack(side="left", padx=2, pady=2)

            # Move down button
            if i < len(self.legend_order) - 1:
                down_btn = ctk.CTkButton(
                    signal_frame,
                    text="↓",
                    width=25,
                    height=25,
                    command=lambda s=signal: self._move_legend_down(s),
                )
                down_btn.pack(side="left", padx=2, pady=2)
            else:
                # Placeholder for alignment
                placeholder = ctk.CTkLabel(signal_frame, text="", width=25)
                placeholder.pack(side="left", padx=2, pady=2)

            # Signal name label
            ctk.CTkLabel(signal_frame, text=f"{signal}:", width=100).pack(
                side="left",
                padx=5,
                pady=2,
            )

            # Custom legend entry
            current_value = self.custom_legend_entries.get(signal, signal)
            legend_entry = ctk.CTkEntry(
                signal_frame,
                placeholder_text=f"Custom label for {signal}",
            )
            legend_entry.pack(side="right", fill="x", expand=True, padx=5, pady=2)
            legend_entry.insert(0, current_value)
            legend_entry.bind(
                "<Return>",
                lambda e, s=signal: self._on_legend_change(s, e.widget.get()),
            )
            legend_entry.bind(
                "<FocusOut>",
                lambda e, s=signal: self._on_legend_change(s, e.widget.get()),
            )

    def _on_legend_change(self, signal: str, new_label: str) -> None:
        """Handle changes to legend labels."""
        self.custom_legend_entries[signal] = new_label
        # Trigger immediate plot update
        self._on_plot_setting_change()

    def _move_legend_up(self, signal: str) -> None:
        """Move a signal up in the legend order."""
        if hasattr(self, "legend_order") and signal in self.legend_order:
            idx = self.legend_order.index(signal)
            if idx > 0:
                self.legend_order[idx], self.legend_order[idx - 1] = (
                    self.legend_order[idx - 1],
                    self.legend_order[idx],
                )
                self._refresh_legend_entries()
                self._on_plot_setting_change()

    def _move_legend_down(self, signal: str) -> None:
        """Move a signal down in the legend order."""
        if hasattr(self, "legend_order") and signal in self.legend_order:
            idx = self.legend_order.index(signal)
            if idx < len(self.legend_order) - 1:
                self.legend_order[idx], self.legend_order[idx + 1] = (
                    self.legend_order[idx + 1],
                    self.legend_order[idx],
                )
                self._refresh_legend_entries()
                self._on_plot_setting_change()

    def _add_trendline(self) -> None:
        """Add trendline to plot."""
        if not hasattr(self, "plot_ax") or not self.plot_ax:
            messagebox.showerror(
                "Error",
                "No plot available. Please create a plot first.",
            )
            return

        trendline_signal = self.trendline_signal_var.get()
        trendline_type = self.trendline_type_var.get()

        if trendline_signal == "Select signal..." or trendline_type == "None":
            return

        try:
            # Get current plot data
            selected_file = self.plot_file_menu.get()
            if selected_file == "Select a file...":
                messagebox.showerror("Error", "Please select a file for plotting.")
                return

            # Find the matching file path
            file_path = None
            for path in self.input_file_paths:
                if os.path.basename(path) == selected_file:
                    file_path = path
                    break

            if not file_path:
                messagebox.showerror("Error", "Selected file not found.")
                return

            # Load data for trendline
            df = self._load_data_for_plotting(file_path)
            if df is None:
                return

            # Get time and signal data
            time_col = self.plot_xaxis_menu.get()
            if time_col not in df.columns or trendline_signal not in df.columns:
                messagebox.showerror("Error", "Selected columns not found in data.")
                return

            x_data = df[time_col]
            y_data = df[trendline_signal]

            # Remove NaN values
            valid_mask = ~(pd.isna(x_data) | pd.isna(y_data))
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]

            if len(x_clean) < 2:
                messagebox.showerror(
                    "Error",
                    "Not enough valid data points for trendline.",
                )
                return

            # Convert datetime to numeric for fitting if needed
            if pd.api.types.is_datetime64_any_dtype(x_clean):
                x_numeric = pd.to_numeric(x_clean)
            else:
                x_numeric = pd.to_numeric(x_clean, errors="coerce")

            y_numeric = pd.to_numeric(y_clean, errors="coerce")

            # Calculate trendline based on type
            if trendline_type == "Linear":
                coeffs = np.polyfit(x_numeric, y_numeric, 1)
                trendline_y = np.polyval(coeffs, x_numeric)
                label = f"Linear Trend: {trendline_signal}"
            elif trendline_type == "Polynomial (2nd)":
                coeffs = np.polyfit(x_numeric, y_numeric, 2)
                trendline_y = np.polyval(coeffs, x_numeric)
                label = f"Poly (2nd) Trend: {trendline_signal}"
            elif trendline_type == "Polynomial (3rd)":
                coeffs = np.polyfit(x_numeric, y_numeric, 3)
                trendline_y = np.polyval(coeffs, x_numeric)
                label = f"Poly (3rd) Trend: {trendline_signal}"
            else:
                return

            # Plot trendline
            self.plot_ax.plot(
                x_clean,
                trendline_y,
                "--",
                linewidth=2,
                label=label,
                alpha=DEFAULT_ALPHA,
            )
            self.plot_ax.legend()
            self.plot_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add trendline: {e!s}")

    def create_help_tab(self, tab: ctk.CTkFrame) -> None:
        """Create the help tab with comprehensive documentation."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(
            header_frame,
            text="Help & Documentation",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(side="left", padx=10, pady=10)

        # Main content with scrollable help
        help_frame = ctk.CTkScrollableFrame(tab)
        help_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        help_frame.grid_columnconfigure(0, weight=1)

        # Help content
        help_content = """
# Advanced CSV Processor & DAT Importer - Help Guide

## Overview
This application provides comprehensive tools for processing, analyzing, and visualizing time series data from CSV files and DAT files with DBF tag files.

## New Features (Latest Update)

### 🎯 Smart Auto-Zoom System
- **Auto-zoom Control**: Toggle "Auto-zoom on changes" in Filter Preview section
- **Smart Detection**: Automatically detects new signals vs filter changes
- **Manual Control**: "Fit to Data" button for manual auto-zoom
- **Preserve Zoom**: Keeps your view when changing filters (if auto-zoom disabled)

### 🔧 Configuration Management
- **Manage Configurations**: New button in "Configuration Save and Load" section
- **Delete Configurations**: Remove unwanted saved settings
- **Load Configurations**: Direct loading from management window
- **File Location**: Open folder containing configuration files

### ⚡ Performance Improvements
- **Smooth Typing**: Text boxes only update on Enter key (not every keystroke)
- **Fast Signal Selection**: No automatic plot updates when clicking checkboxes
- **Manual Updates**: Use "🔄 Update Plot" button when ready

## Tab Descriptions

### Processing Tab
**Purpose**: Configure file processing settings and batch export data.

**Features**:
- **Setup Sub-tab**:
  - Select input CSV files and output directory
  - Save/load processing configurations
  - Manage saved configurations (new!)
  - Configure export format (CSV, Excel, MAT)
  - Set sorting options

- **Processing Sub-tab**:
  - **Signal Filtering**: Apply various filters (Moving Average, Butterworth, Median, Savitzky-Golay)
  - **Time Resampling**: Resample data to different time intervals
  - **Signal Integration**: Create cumulative columns for flow calculations
  - **Signal Differentiation**: Calculate derivatives up to 5th order

- **Custom Vars Sub-tab**:
  - Create custom variables using mathematical formulas
  - Reference existing signals using [SignalName] syntax
  - Build complex calculations

**Usage**:
1. Select input CSV files
2. Choose output directory
3. Configure processing options
4. Select signals to process
5. Click "Process & Batch Export Files"

### Plotting & Analysis Tab
**Purpose**: Visualize and analyze processed data.

**Features**:
- Interactive plotting with matplotlib
- **Smart Auto-Zoom**: Intelligent zoom behavior (new!)
- Multiple chart types (Line Only default, Scatter, etc.)
- Trendline analysis (Linear, Exponential, Power, Polynomial)
- Export plots as images or Excel files
- Real-time signal filtering and selection

**Usage**:
1. Select file to plot from dropdown
2. Choose signals to display
3. Configure plot appearance
4. Add trendlines if needed
5. Export results

### Plots List Tab
**Purpose**: Save and manage plot configurations for batch processing.

**Features**:
- Save plot configurations with names and descriptions
- Preview plots before saving
- Batch export all saved plots
- Manage plot library

**Usage**:
1. Configure a plot in Plotting & Analysis tab
2. Add plot configuration to list
3. Generate previews
4. Export all plots at once

### DAT File Import Tab
**Purpose**: Import data from DAT files with DBF tag files.

**Features**:
- Import DAT files with associated DBF tag files
- Select specific tags to import
- Preview import data
- Convert to CSV format

**Usage**:
1. Select DBF tag file (.dbf)
2. Select DAT data file (.dat)
3. Choose tags to import
4. Preview and import data

## Advanced Features

### Smart Auto-Zoom System
- **New Signal Detection**: Always auto-zoom when adding new signals
- **Filter Change Control**: Respect user preference for filter changes
- **Manual Override**: "Fit to Data" button for immediate reset
- **Zoom Preservation**: Maintains user's zoom state when appropriate

### Configuration Management
- **View All Configurations**: See all saved settings in one place
- **Delete Unwanted Configs**: Remove old or unused configurations
- **Direct Loading**: Load configurations without file dialog
- **File System Access**: Open folder to manage files directly

### Signal Filtering
- **Moving Average**: Smooth data using rolling average
- **Butterworth**: Low-pass or high-pass filtering
- **Median Filter**: Remove outliers
- **Savitzky-Golay**: Preserve signal shape while smoothing

### Signal Integration
- **Trapezoidal**: Most accurate for most applications
- **Rectangular**: Simple left-endpoint method
- **Simpson**: Higher order accuracy

### Signal Differentiation
- **Spline (Acausal)**: Uses entire dataset for each point
- **Rolling Polynomial (Causal)**: Uses only past data points

### Custom Variables
Use mathematical formulas with signal references:
- Example: `([Temperature] * 1.8) + 32` (Celsius to Fahrenheit)
- Example: `([Flow_Rate] * [Pressure]) / 1000` (Power calculation)

## Tips & Best Practices

1. **File Selection**: Use consistent time formats across files
2. **Signal Selection**: Only select signals you need to reduce processing time
3. **Filtering**: Start with "None" and add filters as needed
4. **Integration**: Use Trapezoidal method for most accurate results
5. **Custom Variables**: Test formulas with simple calculations first
6. **Export**: Use "CSV (Separate Files)" for individual analysis, "CSV (Compiled)" for combined analysis
7. **Auto-Zoom**: Disable for stable filter comparison, enable for exploration
8. **Configuration Management**: Regularly clean up old configurations

## Troubleshooting

**Common Issues**:
- **Time parsing errors**: Ensure consistent datetime format
- **Memory issues**: Process fewer files or signals at once
- **Filter errors**: Check signal length vs. filter parameters
- **Integration errors**: Verify time column is properly formatted
- **Performance issues**: Use manual plot updates instead of live updates

**Performance Tips**:
- Close other applications when processing large files
- Use appropriate filter parameters for your data
- Consider resampling for very large datasets
- Disable auto-zoom when comparing filters
- Use "Fit to Data" button for quick overview

## Keyboard Shortcuts

- **Ctrl+O**: Select files (in file dialogs)
- **Ctrl+S**: Save settings
- **Ctrl+L**: Load settings
- **Enter**: Update plot (in text boxes)
- **F1**: Show this help (when help tab is active)

## Support

For additional support or feature requests, please refer to the application documentation or contact the development team.
        """

        # Create help text widget
        help_text = ctk.CTkTextbox(help_frame, wrap="word", font=ctk.CTkFont(size=12))
        help_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        help_text.insert("1.0", help_content)
        help_text.configure(state="disabled")  # Make read-only

    def _generate_unique_filename(self, base_path: str, extension: str) -> str:
        """Generate a unique filename to prevent overwriting existing files."""
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]

        # Remove any existing suffix like _processed, _1, _2, etc.
        base_name = base_name.removesuffix("_processed")  # Remove '_processed'

        counter = 1
        while True:
            if counter == 1:
                filename = f"{base_name}_processed{extension}"
            else:
                filename = f"{base_name}_processed_{counter}{extension}"

            full_path = os.path.join(directory, filename)
            if not os.path.exists(full_path):
                return full_path
            counter += 1

    def _check_file_overwrite(self, file_path: str) -> str | None:
        """Check if file exists and prompt user for action."""
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            response = messagebox.askyesnocancel(
                "File Already Exists",
                f"The file '{filename}' already exists.\n\n"
                f"Would you like to:\n"
                f"• Yes: Overwrite the existing file\n"
                f"• No: Generate a unique filename\n"
                f"• Cancel: Cancel the operation",
                icon="warning",
            )

            if response is None:  # Cancel
                return None
            if response:  # Yes - overwrite
                return file_path
            # No - generate unique name
            directory = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            extension = os.path.splitext(file_path)[1]
            return self._generate_unique_filename(
                os.path.join(directory, base_name),
                extension,
            )

        return file_path

    def _save_current_plot_config(self) -> None:
        """Save the current plot configuration."""
        # Get current plot settings
        plot_name = simpledialog.askstring(
            "Save Plot Configuration",
            "Enter a name for this plot configuration:",
        )
        if not plot_name:
            return

        # Get currently selected signals for plotting
        selected_signals = []
        if hasattr(self, "plot_signal_vars"):
            selected_signals = [
                signal
                for signal, data in self.plot_signal_vars.items()
                if data["var"].get()
            ]

        # Get current plot settings
        plot_config = {
            "name": plot_name,
            "description": f"Plot configuration saved on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "file": (
                self.plot_file_menu.get() if hasattr(self, "plot_file_menu") else ""
            ),
            "x_axis": (
                self.plot_xaxis_menu.get() if hasattr(self, "plot_xaxis_menu") else ""
            ),
            "signals": selected_signals,
            "filter_type": (
                self.plot_filter_type.get()
                if hasattr(self, "plot_filter_type")
                else "None"
            ),
            "show_both_signals": (
                self.show_both_signals_var.get()
                if hasattr(self, "show_both_signals_var")
                else False
            ),
            "plot_title": (
                self.plot_title_entry.get() if hasattr(self, "plot_title_entry") else ""
            ),
            "plot_xlabel": (
                self.plot_xlabel_entry.get()
                if hasattr(self, "plot_xlabel_entry")
                else ""
            ),
            "plot_ylabel": (
                self.plot_ylabel_entry.get()
                if hasattr(self, "plot_ylabel_entry")
                else ""
            ),
            "start_time": (
                self.plotting_start_time_entry.get()
                if hasattr(self, "plotting_start_time_entry")
                else ""
            ),
            "end_time": (
                self.plotting_end_time_entry.get()
                if hasattr(self, "plotting_end_time_entry")
                else ""
            ),
            "color_scheme": (
                self.color_scheme_var.get()
                if hasattr(self, "color_scheme_var")
                else "Auto (Matplotlib)"
            ),
            "line_width": (
                self.line_width_var.get() if hasattr(self, "line_width_var") else "1.0"
            ),
            "legend_position": (
                self.legend_position_var.get()
                if hasattr(self, "legend_position_var")
                else "best"
            ),
            "plot_type": (
                self.plot_type_var.get()
                if hasattr(self, "plot_type_var")
                else "Line with Markers"
            ),
            "trendline_signal": (
                self.trendline_signal_var.get()
                if hasattr(self, "trendline_signal_var")
                else "Select signal..."
            ),
            "trendline_type": (
                self.trendline_type_var.get()
                if hasattr(self, "trendline_type_var")
                else "None"
            ),
            "custom_legend_entries": dict(
                self.custom_legend_entries,
            ),  # Save custom legend labels
            "custom_colors": list(self.custom_colors),  # Save custom colors
            "created_date": pd.Timestamp.now().isoformat(),
        }

        # Add filter-specific parameters for plot preview
        if plot_config["filter_type"] == "Moving Average":
            plot_config["ma_value"] = (
                self.plot_ma_value_entry.get()
                if hasattr(self, "plot_ma_value_entry")
                else ""
            )
            plot_config["ma_unit"] = (
                self.plot_ma_unit_menu.get()
                if hasattr(self, "plot_ma_unit_menu")
                else ""
            )
        elif plot_config["filter_type"] in [
            "Butterworth Low-pass",
            "Butterworth High-pass",
        ]:
            plot_config["bw_order"] = (
                self.plot_bw_order_entry.get()
                if hasattr(self, "plot_bw_order_entry")
                else ""
            )
            plot_config["bw_cutoff"] = (
                self.plot_bw_cutoff_entry.get()
                if hasattr(self, "plot_bw_cutoff_entry")
                else ""
            )
        elif plot_config["filter_type"] == "Median Filter":
            plot_config["median_kernel"] = (
                self.plot_median_kernel_entry.get()
                if hasattr(self, "plot_median_kernel_entry")
                else ""
            )
        elif plot_config["filter_type"] == "Hampel Filter":
            plot_config["hampel_window"] = (
                self.plot_hampel_window_entry.get()
                if hasattr(self, "plot_hampel_window_entry")
                else ""
            )
            plot_config["hampel_threshold"] = (
                self.plot_hampel_threshold_entry.get()
                if hasattr(self, "plot_hampel_threshold_entry")
                else ""
            )
        elif plot_config["filter_type"] == "Z-Score Filter":
            plot_config["zscore_threshold"] = (
                self.plot_zscore_threshold_entry.get()
                if hasattr(self, "plot_zscore_threshold_entry")
                else ""
            )
            plot_config["zscore_method"] = (
                self.plot_zscore_method_menu.get()
                if hasattr(self, "plot_zscore_method_menu")
                else ""
            )
        elif plot_config["filter_type"] == "Savitzky-Golay":
            plot_config["savgol_window"] = (
                self.plot_savgol_window_entry.get()
                if hasattr(self, "plot_savgol_window_entry")
                else ""
            )
            plot_config["savgol_polyorder"] = (
                self.plot_savgol_polyorder_entry.get()
                if hasattr(self, "plot_savgol_polyorder_entry")
                else ""
            )

        # Add to plots list
        self.plots_list.append(plot_config)
        self._update_plots_listbox()
        self._update_load_plot_config_menu()
        self._save_plots_to_file()

        messagebox.showinfo(
            "Success",
            f"Plot configuration '{plot_name}' saved successfully!",
        )

    def _modify_plot_config(self) -> None:
        """Modify an existing plot configuration."""
        if not hasattr(self, "plots_list") or not self.plots_list:
            messagebox.showwarning(
                "No Configurations",
                "No saved plot configurations found. Please save a configuration first.",
            )
            return

        # Create a dialog to select which configuration to modify
        dialog = ctk.CTkToplevel(self)
        dialog.title("Modify Plot Configuration")
        dialog.geometry("400x300")
        dialog.grab_set()  # Make dialog modal

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (300 // 2)
        dialog.geometry(f"400x300+{x}+{y}")

        # Create listbox for configurations
        ctk.CTkLabel(
            dialog,
            text="Select configuration to modify:",
            font=ctk.CTkFont(weight="bold"),
        ).pack(pady=10)

        listbox_frame = ctk.CTkFrame(dialog)
        listbox_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Create listbox
        listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=10)

        # Populate listbox
        for i, config in enumerate(self.plots_list):
            listbox.insert(
                tk.END,
                f"{config['name']} ({config.get('created_date', 'Unknown date')})",
            )

        # Buttons frame
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill="x", padx=20, pady=10)

        def on_modify():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "No Selection",
                    "Please select a configuration to modify.",
                )
                return

            selected_index = selection[0]
            selected_config = self.plots_list[selected_index]

            # Load the configuration into the current UI
            self._apply_plot_config(selected_config)

            # Update the configuration with current settings
            self._update_plot_config(selected_index)

            dialog.destroy()
            messagebox.showinfo(
                "Success",
                f"Configuration '{selected_config['name']}' has been updated with current settings!",
            )

        def on_delete():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "No Selection",
                    "Please select a configuration to delete.",
                )
                return

            selected_index = selection[0]
            selected_config = self.plots_list[selected_index]

            # Ask for confirmation
            result = messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete the configuration '{selected_config['name']}'?\n\nThis action cannot be undone.",
            )
            if result:
                # Remove the configuration from the list
                deleted_config = self.plots_list.pop(selected_index)

                # Update the listbox
                listbox.delete(selection[0])

                # Update the plots listbox in the main UI if it exists
                if hasattr(self, "plots_listbox"):
                    self._update_plots_listbox()

                messagebox.showinfo(
                    "Success",
                    f"Configuration '{deleted_config['name']}' has been deleted!",
                )

        def on_cancel():
            dialog.destroy()

        ctk.CTkButton(button_frame, text="Modify Selected", command=on_modify).pack(
            side="left",
            padx=5,
        )
        ctk.CTkButton(
            button_frame,
            text="Delete Selected",
            command=on_delete,
            fg_color="red",
            hover_color="darkred",
        ).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=on_cancel).pack(
            side="right",
            padx=5,
        )

    def _update_plot_config(self, config_index: int) -> None:
        """Update an existing plot configuration with current settings."""
        if not hasattr(self, "plots_list") or config_index >= len(self.plots_list):
            return

        # Get currently selected signals for plotting
        selected_signals = []
        if hasattr(self, "plot_signal_vars"):
            selected_signals = [
                signal
                for signal, data in self.plot_signal_vars.items()
                if data["var"].get()
            ]

        # Update the configuration with current settings
        self.plots_list[config_index].update(
            {
                "file": (
                    self.plot_file_menu.get() if hasattr(self, "plot_file_menu") else ""
                ),
                "x_axis": (
                    self.plot_xaxis_menu.get()
                    if hasattr(self, "plot_xaxis_menu")
                    else ""
                ),
                "signals": selected_signals,
                "filter_type": (
                    self.plot_filter_type.get()
                    if hasattr(self, "plot_filter_type")
                    else "None"
                ),
                "show_both_signals": (
                    self.show_both_signals_var.get()
                    if hasattr(self, "show_both_signals_var")
                    else False
                ),
                "compare_filters": (
                    self.compare_filters_var.get()
                    if hasattr(self, "compare_filters_var")
                    else False
                ),
                "plot_title": (
                    self.plot_title_entry.get()
                    if hasattr(self, "plot_title_entry")
                    else ""
                ),
                "plot_xlabel": (
                    self.plot_xlabel_entry.get()
                    if hasattr(self, "plot_xlabel_entry")
                    else ""
                ),
                "plot_ylabel": (
                    self.plot_ylabel_entry.get()
                    if hasattr(self, "plot_ylabel_entry")
                    else ""
                ),
                "start_time": (
                    self.plotting_start_time_entry.get()
                    if hasattr(self, "plotting_start_time_entry")
                    else ""
                ),
                "end_time": (
                    self.plotting_end_time_entry.get()
                    if hasattr(self, "plotting_end_time_entry")
                    else ""
                ),
                "color_scheme": (
                    self.color_scheme_var.get()
                    if hasattr(self, "color_scheme_var")
                    else "Auto (Matplotlib)"
                ),
                "line_width": (
                    self.line_width_var.get()
                    if hasattr(self, "line_width_var")
                    else "1.0"
                ),
                "legend_position": (
                    self.legend_position_var.get()
                    if hasattr(self, "legend_position_var")
                    else "best"
                ),
                "plot_type": (
                    self.plot_type_var.get()
                    if hasattr(self, "plot_type_var")
                    else "Line with Markers"
                ),
                "trendline_signal": (
                    self.trendline_signal_var.get()
                    if hasattr(self, "trendline_signal_var")
                    else "Select signal..."
                ),
                "trendline_type": (
                    self.trendline_type_var.get()
                    if hasattr(self, "trendline_type_var")
                    else "None"
                ),
                "custom_legend_entries": dict(self.custom_legend_entries),
                "custom_colors": list(self.custom_colors),
                "modified_date": pd.Timestamp.now().isoformat(),
            },
        )

        # Add filter-specific parameters
        if self.plots_list[config_index]["filter_type"] == "Moving Average":
            self.plots_list[config_index]["ma_value"] = (
                self.plot_ma_value_entry.get()
                if hasattr(self, "plot_ma_value_entry")
                else ""
            )
            self.plots_list[config_index]["ma_unit"] = (
                self.plot_ma_unit_menu.get()
                if hasattr(self, "plot_ma_unit_menu")
                else ""
            )
        elif self.plots_list[config_index]["filter_type"] in [
            "Butterworth Low-pass",
            "Butterworth High-pass",
        ]:
            self.plots_list[config_index]["bw_order"] = (
                self.plot_bw_order_entry.get()
                if hasattr(self, "plot_bw_order_entry")
                else ""
            )
            self.plots_list[config_index]["bw_cutoff"] = (
                self.plot_bw_cutoff_entry.get()
                if hasattr(self, "plot_bw_cutoff_entry")
                else ""
            )
        elif self.plots_list[config_index]["filter_type"] == "Median Filter":
            self.plots_list[config_index]["median_kernel"] = (
                self.plot_median_kernel_entry.get()
                if hasattr(self, "plot_median_kernel_entry")
                else ""
            )
        elif self.plots_list[config_index]["filter_type"] == "Hampel Filter":
            self.plots_list[config_index]["hampel_window"] = (
                self.plot_hampel_window_entry.get()
                if hasattr(self, "plot_hampel_window_entry")
                else ""
            )
            self.plots_list[config_index]["hampel_threshold"] = (
                self.plot_hampel_threshold_entry.get()
                if hasattr(self, "plot_hampel_threshold_entry")
                else ""
            )
        elif self.plots_list[config_index]["filter_type"] == "Z-Score Filter":
            self.plots_list[config_index]["zscore_threshold"] = (
                self.plot_zscore_threshold_entry.get()
                if hasattr(self, "plot_zscore_threshold_entry")
                else ""
            )
            self.plots_list[config_index]["zscore_method"] = (
                self.plot_zscore_method_menu.get()
                if hasattr(self, "plot_zscore_method_menu")
                else ""
            )
        elif self.plots_list[config_index]["filter_type"] == "Savitzky-Golay":
            self.plots_list[config_index]["savgol_window"] = (
                self.plot_savgol_window_entry.get()
                if hasattr(self, "plot_savgol_window_entry")
                else ""
            )
            self.plots_list[config_index]["savgol_polyorder"] = (
                self.plot_savgol_polyorder_entry.get()
                if hasattr(self, "plot_savgol_polyorder_entry")
                else ""
            )

        # Save the updated configuration
        self._save_plots_to_file()

    def _on_load_plot_config_select(self, selected_plot_name: str) -> None:
        """Handle selection from the load plot config dropdown."""
        if selected_plot_name == "No saved plots":
            return

        # Find the plot config by name
        plot_config = None
        for config in self.plots_list:
            if config["name"] == selected_plot_name:
                plot_config = config
                break

        if not plot_config:
            messagebox.showerror(
                "Error",
                f"Plot configuration '{selected_plot_name}' not found.",
            )
            return

        # Apply the plot configuration
        self._apply_plot_config(plot_config)
        messagebox.showinfo(
            "Success",
            f"Plot configuration '{selected_plot_name}' loaded!",
        )

    def _apply_plot_config(self, plot_config: dict[str, Any]) -> None:
        """Apply a plot configuration to the current plotting tab."""
        # Apply file selection first
        if (
            "file" in plot_config
            and plot_config["file"]
            and hasattr(self, "plot_file_menu")
        ):
            self.plot_file_menu.set(plot_config["file"])
            # Trigger file selection to populate signals
            self.on_plot_file_select(plot_config["file"])

            # Give time for signals to load, then apply signal selections
            self.after(100, lambda: self._apply_plot_config_signals(plot_config))
        else:
            # If no file, just apply what we can
            self._apply_plot_config_signals(plot_config)

    def _apply_plot_config_signals(self, plot_config: dict[str, Any]) -> None:
        """Apply signal selections and other settings after file is loaded."""
        # Apply x-axis selection
        if (
            "x_axis" in plot_config
            and plot_config["x_axis"]
            and hasattr(self, "plot_xaxis_menu")
        ):
            self.plot_xaxis_menu.set(plot_config["x_axis"])

        # Apply signal selections - now that signals should be loaded
        if hasattr(self, "plot_signal_vars") and "signals" in plot_config:
            saved_signals = plot_config["signals"]
            for signal, data in self.plot_signal_vars.items():
                data["var"].set(signal in saved_signals)

        # Apply filter settings
        if "filter_type" in plot_config and hasattr(self, "plot_filter_type"):
            self.plot_filter_type.set(plot_config["filter_type"])
            self._update_plot_filter_ui(plot_config["filter_type"])

        # Apply filter parameters - enhanced with all filter types
        if plot_config.get("filter_type") == "Moving Average":
            if "ma_value" in plot_config and hasattr(self, "plot_ma_value_entry"):
                self.plot_ma_value_entry.delete(0, tk.END)
                self.plot_ma_value_entry.insert(0, plot_config["ma_value"])
            if "ma_unit" in plot_config and hasattr(self, "plot_ma_unit_menu"):
                self.plot_ma_unit_menu.set(plot_config["ma_unit"])
        elif plot_config.get("filter_type") in [
            "Butterworth Low-pass",
            "Butterworth High-pass",
        ]:
            if "bw_order" in plot_config and hasattr(self, "plot_bw_order_entry"):
                self.plot_bw_order_entry.delete(0, tk.END)
                self.plot_bw_order_entry.insert(0, plot_config["bw_order"])
            if "bw_cutoff" in plot_config and hasattr(self, "plot_bw_cutoff_entry"):
                self.plot_bw_cutoff_entry.delete(0, tk.END)
                self.plot_bw_cutoff_entry.insert(0, plot_config["bw_cutoff"])
        elif plot_config.get("filter_type") == "Median Filter":
            if "median_kernel" in plot_config and hasattr(
                self,
                "plot_median_kernel_entry",
            ):
                self.plot_median_kernel_entry.delete(0, tk.END)
                self.plot_median_kernel_entry.insert(0, plot_config["median_kernel"])
        elif plot_config.get("filter_type") == "Hampel Filter":
            if "hampel_window" in plot_config and hasattr(
                self,
                "plot_hampel_window_entry",
            ):
                self.plot_hampel_window_entry.delete(0, tk.END)
                self.plot_hampel_window_entry.insert(0, plot_config["hampel_window"])
            if "hampel_threshold" in plot_config and hasattr(
                self,
                "plot_hampel_threshold_entry",
            ):
                self.plot_hampel_threshold_entry.delete(0, tk.END)
                self.plot_hampel_threshold_entry.insert(
                    0,
                    plot_config["hampel_threshold"],
                )
        elif plot_config.get("filter_type") == "Z-Score Filter":
            if "zscore_threshold" in plot_config and hasattr(
                self,
                "plot_zscore_threshold_entry",
            ):
                self.plot_zscore_threshold_entry.delete(0, tk.END)
                self.plot_zscore_threshold_entry.insert(
                    0,
                    plot_config["zscore_threshold"],
                )
            if "zscore_method" in plot_config and hasattr(
                self,
                "plot_zscore_method_menu",
            ):
                self.plot_zscore_method_menu.set(plot_config["zscore_method"])
        elif plot_config.get("filter_type") == "Savitzky-Golay":
            if "savgol_window" in plot_config and hasattr(
                self,
                "plot_savgol_window_entry",
            ):
                self.plot_savgol_window_entry.delete(0, tk.END)
                self.plot_savgol_window_entry.insert(0, plot_config["savgol_window"])
            if "savgol_polyorder" in plot_config and hasattr(
                self,
                "plot_savgol_polyorder_entry",
            ):
                self.plot_savgol_polyorder_entry.delete(0, tk.END)
                self.plot_savgol_polyorder_entry.insert(
                    0,
                    plot_config["savgol_polyorder"],
                )

        # Apply custom legend entries
        if "custom_legend_entries" in plot_config:
            self.custom_legend_entries = plot_config["custom_legend_entries"]

        # Apply custom colors
        if "custom_colors" in plot_config:
            self.custom_colors = plot_config["custom_colors"]

        # Apply other plot settings
        if "show_both_signals" in plot_config and hasattr(
            self,
            "show_both_signals_var",
        ):
            self.show_both_signals_var.set(plot_config["show_both_signals"])

        if "plot_title" in plot_config and hasattr(self, "plot_title_entry"):
            self.plot_title_entry.delete(0, tk.END)
            self.plot_title_entry.insert(0, plot_config["plot_title"])

        if "plot_xlabel" in plot_config and hasattr(self, "plot_xlabel_entry"):
            self.plot_xlabel_entry.delete(0, tk.END)
            self.plot_xlabel_entry.insert(0, plot_config["plot_xlabel"])

        if "plot_ylabel" in plot_config and hasattr(self, "plot_ylabel_entry"):
            self.plot_ylabel_entry.delete(0, tk.END)
            self.plot_ylabel_entry.insert(0, plot_config["plot_ylabel"])

        if "start_time" in plot_config and hasattr(self, "plotting_start_time_entry"):
            self.plotting_start_time_entry.delete(0, tk.END)
            self.plotting_start_time_entry.insert(0, plot_config["start_time"])

        if "end_time" in plot_config and hasattr(self, "plotting_end_time_entry"):
            self.plotting_end_time_entry.delete(0, tk.END)
            self.plotting_end_time_entry.insert(0, plot_config["end_time"])

        if "color_scheme" in plot_config and hasattr(self, "color_scheme_var"):
            self.color_scheme_var.set(plot_config["color_scheme"])

        if "line_width" in plot_config and hasattr(self, "line_width_var"):
            self.line_width_var.set(plot_config["line_width"])

        if "legend_position" in plot_config and hasattr(self, "legend_position_var"):
            self.legend_position_var.set(plot_config["legend_position"])

        if "plot_type" in plot_config and hasattr(self, "plot_type_var"):
            self.plot_type_var.set(plot_config["plot_type"])

        if "trendline_signal" in plot_config and hasattr(self, "trendline_signal_var"):
            self.trendline_signal_var.set(plot_config["trendline_signal"])

        if "trendline_type" in plot_config and hasattr(self, "trendline_type_var"):
            self.trendline_type_var.set(plot_config["trendline_type"])

        # Finally, update the plot
        self.update_plot()
        # Update the plot
        self.update_plot()

    def _update_load_plot_config_menu(self) -> None:
        """Update the load plot config dropdown menu."""
        if not hasattr(self, "load_plot_config_menu"):
            return

        if self.plots_list:
            plot_names = [config["name"] for config in self.plots_list]
            self.load_plot_config_menu.configure(values=plot_names)
            self.load_plot_config_menu.set("Select a plot config...")
        else:
            self.load_plot_config_menu.configure(values=["No saved plots"])
            self.load_plot_config_menu.set("No saved plots")

    def _update_plots_signals(self, signals: list[str]) -> None:
        """Update signals available in plots list tab."""
        if not hasattr(self, "plots_signals_frame"):
            return

        # Clear existing widgets
        for widget in self.plots_signals_frame.winfo_children():
            widget.destroy()

        # Initialize plots signal vars if not exists
        if not hasattr(self, "plots_signal_vars"):
            self.plots_signal_vars = {}

        self.plots_signal_vars.clear()

        # Add checkboxes for each signal
        for signal in signals:
            if signal != signals[0]:  # Skip time column
                var = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(
                    self.plots_signals_frame,
                    text=signal,
                    variable=var,
                )
                cb.grid(sticky="w", padx=5, pady=2)
                self.plots_signal_vars[signal] = var

        # Re-bind mouse wheel to all new checkboxes
        self._bind_mousewheel_to_frame(self.plots_signals_frame)

    def _generate_plot_preview(self) -> None:
        """Generate plot preview."""
        selection = self.plots_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a plot to preview.")
            return

        try:
            # Clear previous plot
            self.preview_ax.clear()

            idx = selection[0]
            plot_config = self.plots_list[idx]

            # Get the actual data and plot it exactly like the main plotting tab
            signals = plot_config.get("signals", [])
            file_name = plot_config.get("file", "")

            print(
                f"DEBUG: Preview plot config - File: '{file_name}', Signals: {signals}",
            )

            if not signals:
                self.preview_ax.text(
                    0.5,
                    0.5,
                    "No signals selected in this configuration",
                    transform=self.preview_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return

            if not file_name or file_name == "Select a file...":
                # Show available files for debugging
                available_files = []
                if hasattr(self, "plot_file_menu") and hasattr(
                    self.plot_file_menu,
                    "_values",
                ):
                    available_files = [
                        f
                        for f in self.plot_file_menu._values
                        if f != "Select a file..."
                    ]

                debug_text = f"No data file specified in plot configuration\n\nSaved file: '{file_name}'"
                if available_files:
                    debug_text += "\n\nAvailable files:\n" + "\n".join(
                        available_files[:3],
                    )
                    if len(available_files) > 3:
                        debug_text += f"\n... and {len(available_files)-3} more"
                else:
                    debug_text += "\n\nNo files currently loaded.\nPlease load files on Setup tab first."

                self.preview_ax.text(
                    0.5,
                    0.5,
                    debug_text,
                    transform=self.preview_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return

            # Get the actual data using the same method as main plotting
            df = self.get_data_for_plotting(file_name)

            if df is None or df.empty:
                # Show available files for debugging
                available_files = []
                if hasattr(self, "processed_files") and self.processed_files:
                    available_files.extend(
                        [os.path.basename(fp) for fp in self.processed_files.keys()],
                    )
                if hasattr(self, "input_file_paths") and self.input_file_paths:
                    available_files.extend(
                        [os.path.basename(fp) for fp in self.input_file_paths],
                    )

                if available_files:
                    debug_text = (
                        f"Data file '{file_name}' not found\n\nAvailable files:\n"
                        + "\n".join(set(available_files)[:5])
                    )
                    if len(set(available_files)) > 5:
                        debug_text += f"\n... and {len(set(available_files))-5} more"
                else:
                    debug_text = "No data files loaded\n\nPlease:\n1. Select CSV files on Setup tab\n2. Process files or plot directly"

                self.preview_ax.text(
                    0.5,
                    0.5,
                    debug_text,
                    transform=self.preview_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return

            # Get time column and available signals
            time_col = df.columns[0]
            # Try to find a better time column if the first column doesn't look like time
            for col in df.columns:
                if any(
                    time_word in col.lower()
                    for time_word in ["time", "timestamp", "date"]
                ):
                    time_col = col
                    break

            available_signals = [s for s in signals if s in df.columns]

            if not available_signals:
                self.preview_ax.text(
                    0.5,
                    0.5,
                    "None of the selected signals\nare available in the data",
                    transform=self.preview_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return

            # Apply time range if specified
            plot_df = df.copy()
            start_time = plot_config.get("start_time", "")
            end_time = plot_config.get("end_time", "")

            if start_time or end_time:
                if pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                    if start_time:
                        try:
                            start_datetime = pd.to_datetime(
                                f"{plot_df[time_col].dt.date.iloc[0]} {start_time}",
                            )
                            plot_df = plot_df[plot_df[time_col] >= start_datetime]
                        except Exception as e:
                            # Log time range filtering errors for debugging
                            print(f"Warning: Failed to apply start time filter: {e}")
                    if end_time:
                        try:
                            end_datetime = pd.to_datetime(
                                f"{plot_df[time_col].dt.date.iloc[0]} {end_time}",
                            )
                            plot_df = plot_df[plot_df[time_col] <= end_datetime]
                        except Exception as e:
                            # Log time range filtering errors for debugging
                            print(f"Warning: Failed to apply end time filter: {e}")

            # Plot all available signals
            colors = plt.cm.tab10(np.linspace(0, 1, len(available_signals)))
            for i, signal in enumerate(available_signals):
                signal_data = plot_df[[time_col, signal]].dropna()
                if len(signal_data) > 0:
                    self.preview_ax.plot(
                        signal_data[time_col],
                        signal_data[signal],
                        label=signal,
                        linewidth=1,
                        color=colors[i],
                    )

            # Apply plot configuration
            title = (
                plot_config.get("plot_title", "") or f"Preview: {plot_config['name']}"
            )
            xlabel = plot_config.get("plot_xlabel", "") or time_col
            ylabel = plot_config.get("plot_ylabel", "") or "Value"

            self.preview_ax.set_title(title, fontsize=14)
            self.preview_ax.set_xlabel(xlabel)
            self.preview_ax.set_ylabel(ylabel)

            # Use legend position from plot config if available, otherwise default to 'best'
            legend_position = plot_config.get("legend_position", "best")
            if legend_position == "outside right":
                self.preview_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                self.preview_ax.legend(loc=legend_position)

            self.preview_ax.grid(True, linestyle="--", alpha=0.6)

            # Format x-axis for time data
            if pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                import matplotlib.dates as mdates

                self.preview_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                self.preview_ax.tick_params(axis="x", rotation=0)

            self.preview_canvas.draw()

        except Exception as e:
            self.preview_ax.clear()
            self.preview_ax.text(
                0.5,
                0.5,
                f"Error generating preview:\n{e!s}",
                transform=self.preview_ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            self.preview_ax.set_title("Preview Error")
            self.preview_canvas.draw()

    def _export_all_plots(self) -> None:
        """Export all plots."""
        if not self.plots_list:
            messagebox.showwarning("Warning", "No plots to export.")
            return

        # Ask user for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return

        try:
            exported_count = 0
            for plot_config in self.plots_list:
                # Create a simple text file with plot configuration
                filename = f"{plot_config['name'].replace(' ', '_')}_config.txt"
                filepath = os.path.join(export_dir, filename)

                with open(filepath, "w") as f:
                    f.write(f"Plot Configuration: {plot_config['name']}\n")
                    f.write(f"Description: {plot_config.get('description', 'N/A')}\n")
                    f.write(f"Created: {plot_config.get('created_date', 'N/A')}\n")
                    f.write(f"Signals: {', '.join(plot_config.get('signals', []))}\n")
                    f.write(f"Start Time: {plot_config.get('start_time', 'N/A')}\n")
                    f.write(f"End Time: {plot_config.get('end_time', 'N/A')}\n")

                    if "filter_type" in plot_config:
                        f.write(f"Filter: {plot_config['filter_type']}\n")

                    f.write("\nFull Configuration:\n")
                    f.writelines(
                        f"  {key}: {value}\n" for key, value in plot_config.items()
                    )

                exported_count += 1

            messagebox.showinfo(
                "Export Complete",
                f"Exported {exported_count} plot configurations to {export_dir}",
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting plots: {e}")

    def _on_plot_setting_change(self, *args: Any) -> None:
        """Automatically update plot when appearance settings change."""
        # Only update if we have data and signals selected
        if hasattr(self, "plot_signal_vars"):
            selected_count = sum(
                1 for data in self.plot_signal_vars.values() if data["var"].get()
            )

            if selected_count > 0:
                # Use after_idle to prevent too many rapid updates
                if hasattr(self, "_update_pending"):
                    self.after_cancel(self._update_pending)
                self._update_pending = self.after_idle(self.update_plot)

    def _on_color_scheme_change(self, scheme: str) -> None:
        """Handle color scheme change and show/hide custom colors interface."""
        if scheme == "Custom Colors":
            self.custom_colors_frame.grid()
        else:
            self.custom_colors_frame.grid_remove()

        # Trigger plot update
        self._on_plot_setting_change()

    def _update_custom_colors_display(self) -> None:
        """Update the display of custom colors with color preview buttons."""
        # Clear existing widgets
        for widget in self.colors_scroll_frame.winfo_children():
            widget.destroy()

        for i, color in enumerate(self.custom_colors):
            color_frame = ctk.CTkFrame(self.colors_scroll_frame)
            color_frame.pack(fill="x", padx=5, pady=2)

            # Color preview button
            color_button = ctk.CTkButton(
                color_frame,
                text=f"Color {i+1}",
                width=80,
                height=30,
                fg_color=color,
                hover_color=color,
                command=lambda idx=i: self._edit_custom_color(idx),
            )
            color_button.pack(side="left", padx=5, pady=5)

            # Color hex code label
            color_label = ctk.CTkLabel(
                color_frame,
                text=color,
                font=ctk.CTkFont(size=10),
            )
            color_label.pack(side="left", padx=5, pady=5)

            # Remove button
            remove_button = ctk.CTkButton(
                color_frame,
                text="✕",
                width=30,
                height=30,
                command=lambda idx=i: self._remove_custom_color(idx),
            )
            remove_button.pack(side="right", padx=5, pady=5)

    def _add_custom_color(self) -> None:
        """Add a new custom color using color picker."""
        color = colorchooser.askcolor(title="Choose Color")[1]  # Get hex value
        if color:
            self.custom_colors.append(color)
            self._update_custom_colors_display()
            if self.color_scheme_var.get() == "Custom Colors":
                self._on_plot_setting_change()

    def _edit_custom_color(self, index: int) -> None:
        """Edit an existing custom color."""
        if 0 <= index < len(self.custom_colors):
            current_color = self.custom_colors[index]
            color = colorchooser.askcolor(
                color=current_color,
                title=f"Edit Color {index+1}",
            )[1]
            if color:
                self.custom_colors[index] = color
                self._update_custom_colors_display()
                if self.color_scheme_var.get() == "Custom Colors":
                    self._on_plot_setting_change()

    def _remove_custom_color(self, index: int) -> None:
        """Remove a custom color."""
        if (
            0 <= index < len(self.custom_colors) and len(self.custom_colors) > 1
        ):  # Keep at least one color
            self.custom_colors.pop(index)
            self._update_custom_colors_display()
            if self.color_scheme_var.get() == "Custom Colors":
                self._on_plot_setting_change()

    def _reset_custom_colors(self) -> None:
        """Reset custom colors to default set."""
        self.custom_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        self._update_custom_colors_display()
        if self.color_scheme_var.get() == "Custom Colors":
            self._on_plot_setting_change()

    def _bind_mousewheel_to_frame(self, frame: ctk.CTkFrame) -> None:
        """Bind mouse wheel events to a frame for proper scrolling."""

        def on_mousewheel(event):
            # Scroll the frame's canvas
            try:
                frame._parent_canvas.yview_scroll(
                    int(-1 * (event.delta / 120)),
                    "units",
                )
            except Exception:
                # Fallback for different systems
                frame._parent_canvas.yview_scroll(int(-1 * event.delta), "units")

        # Bind mousewheel to the frame and all its children
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)
            widget.bind(
                "<Button-4>",
                lambda e: frame._parent_canvas.yview_scroll(-1, "units"),
            )  # Linux
            widget.bind(
                "<Button-5>",
                lambda e: frame._parent_canvas.yview_scroll(1, "units"),
            )  # Linux

            for child in widget.winfo_children():
                bind_mousewheel(child)

        bind_mousewheel(frame)

    def _on_trendline_window_mode_change(self, mode: str) -> None:
        """Handle trendline window mode change."""
        if mode == "Manual Entry":
            self.trendline_manual_frame.grid()
            self.trendline_visual_frame.grid_remove()
        elif mode == "Visual Selection":
            self.trendline_manual_frame.grid_remove()
            self.trendline_visual_frame.grid()
        else:  # Full Range
            self.trendline_manual_frame.grid_remove()
            self.trendline_visual_frame.grid_remove()

        self._on_plot_setting_change()

    def _start_trendline_selection(self) -> None:
        """Start visual selection of trendline window."""
        if not hasattr(self, "plot_canvas") or not self.plot_canvas:
            messagebox.showwarning("Warning", "Please generate a plot first.")
            return

        # Enable selection mode
        self.trendline_selection_active = True
        self.trendline_selection_start = None
        self.trendline_selection_end = None

        # Connect mouse events
        self.plot_canvas.mpl_connect(
            "button_press_event",
            self._on_trendline_selection_start,
        )
        self.plot_canvas.mpl_connect(
            "button_release_event",
            self._on_trendline_selection_end,
        )

        # Update button text
        self.trendline_select_button.configure(
            text="Click and drag on plot to select range",
        )
        self.trendline_selected_range.configure(text="Selection active...")

    def _on_trendline_selection_start(self, event: Any) -> None:
        """Handle start of trendline selection."""
        if (
            hasattr(self, "trendline_selection_active")
            and self.trendline_selection_active
            and event.inaxes
        ):
            self.trendline_selection_start = event.xdata

    def _on_trendline_selection_end(self, event: Any) -> None:
        """Handle end of trendline selection."""
        if (
            hasattr(self, "trendline_selection_active")
            and self.trendline_selection_active
            and event.inaxes
        ):
            if self.trendline_selection_start is not None:
                self.trendline_selection_end = event.xdata

                # Ensure start < end
                if self.trendline_selection_start > self.trendline_selection_end:
                    self.trendline_selection_start, self.trendline_selection_end = (
                        self.trendline_selection_end,
                        self.trendline_selection_start,
                    )

                # Update display
                start_str = f"{self.trendline_selection_start:.2f}"
                end_str = f"{self.trendline_selection_end:.2f}"
                self.trendline_selected_range.configure(
                    text=f"Range: {start_str} to {end_str}",
                )

                # Disable selection mode
                self.trendline_selection_active = False
                self.trendline_select_button.configure(
                    text="Select Time Window on Plot",
                )

                # Update plot
                self._on_plot_setting_change()

    def _on_dataset_naming_change(self) -> None:
        """Handle changes to dataset naming mode."""
        if self.dataset_naming_var.get() == "custom":
            self.custom_dataset_entry.configure(state="normal")
            self.custom_dataset_entry.bind(
                "<KeyRelease>",
                self._check_custom_name_overwrite,
            )
        else:
            self.custom_dataset_entry.configure(state="disabled")
            self.overwrite_warning_label.configure(text="")

    def _check_custom_name_overwrite(self, event: Any = None) -> None:
        """Check if custom dataset name will cause file overwrite."""
        if not hasattr(self, "custom_dataset_entry") or not hasattr(
            self,
            "output_directory",
        ):
            return

        custom_name = self.custom_dataset_entry.get().strip()
        if not custom_name:
            self.overwrite_warning_label.configure(text="")
            return

        # Check for existing files with the custom name
        output_dir = self.output_directory
        if os.path.exists(output_dir):
            # Check for various file extensions that might be created
            extensions = [".csv", ".xlsx", ".mat"]
            existing_files = []

            for ext in extensions:
                potential_file = os.path.join(output_dir, f"{custom_name}{ext}")
                if os.path.exists(potential_file):
                    existing_files.append(f"{custom_name}{ext}")

            if existing_files:
                warning_text = f"⚠️ Warning: Will overwrite existing files: {', '.join(existing_files)}"
                self.overwrite_warning_label.configure(
                    text=warning_text,
                    text_color="orange",
                )
            else:
                self.overwrite_warning_label.configure(
                    text="✓ No file conflicts found",
                    text_color="green",
                )
        else:
            self.overwrite_warning_label.configure(text="")

    def _save_zoom_state(self) -> None:
        """Save current zoom/pan state of the plot."""
        if hasattr(self, "plot_ax"):
            self.saved_zoom_state = {
                "xlim": self.plot_ax.get_xlim(),
                "ylim": self.plot_ax.get_ylim(),
            }
            messagebox.showinfo("Zoom State", "Current zoom state saved!")

    def _restore_zoom_state(self) -> None:
        """Restore previously saved zoom/pan state."""
        if hasattr(self, "saved_zoom_state") and self.saved_zoom_state:
            if hasattr(self, "plot_ax"):
                self.plot_ax.set_xlim(self.saved_zoom_state["xlim"])
                self.plot_ax.set_ylim(self.saved_zoom_state["ylim"])
                self.plot_canvas.draw()
                messagebox.showinfo("Zoom State", "Zoom state restored!")
        else:
            messagebox.showwarning("Warning", "No saved zoom state found.")

    def _zoom_out_25(self) -> None:
        """Zoom out by 25% while maintaining center."""
        if hasattr(self, "plot_ax"):
            xlim = self.plot_ax.get_xlim()
            ylim = self.plot_ax.get_ylim()

            # Calculate current center and range
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]

            # Expand range by 25%
            new_x_range = x_range * ZOOM_OUT_FACTOR
            new_y_range = y_range * ZOOM_OUT_FACTOR

            # Set new limits
            self.plot_ax.set_xlim(
                x_center - new_x_range / 2,
                x_center + new_x_range / 2,
            )
            self.plot_ax.set_ylim(
                y_center - new_y_range / 2,
                y_center + new_y_range / 2,
            )
            self.plot_canvas.draw()

    def _zoom_in_25(self) -> None:
        """Zoom in by 25% while maintaining center."""
        if hasattr(self, "plot_ax"):
            xlim = self.plot_ax.get_xlim()
            ylim = self.plot_ax.get_ylim()

            # Calculate current center and range
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]

            # Shrink range by 25%
            new_x_range = x_range * 0.75
            new_y_range = y_range * 0.75

            # Set new limits
            self.plot_ax.set_xlim(
                x_center - new_x_range / 2,
                x_center + new_x_range / 2,
            )
            self.plot_ax.set_ylim(
                y_center - new_y_range / 2,
                y_center + new_y_range / 2,
            )
            self.plot_canvas.draw()

    def _preserve_zoom_during_update(self) -> None:
        """Store zoom state before plot update and restore after."""
        zoom_state = None
        if hasattr(self, "plot_ax"):
            zoom_state = {
                "xlim": self.plot_ax.get_xlim(),
                "ylim": self.plot_ax.get_ylim(),
            }
        return zoom_state

    def _auto_fit_plot(self) -> None:
        """Auto-fit the plot to show all data."""
        if hasattr(self, "plot_ax"):
            try:
                self.plot_ax.autoscale_view()
                self.plot_canvas.draw()
                self.status_label.configure(text="Plot auto-fitted to data")
            except Exception as e:
                print(f"Error auto-fitting plot: {e}")

    def _should_auto_zoom(self, reason: str = "filter_change") -> bool:
        """Determine if auto-zoom should be applied based on the reason."""
        if not hasattr(self, "auto_zoom_var"):
            return True  # Default to auto-zoom if control doesn't exist

        # Always auto-zoom when adding new signals
        if reason == "new_signal":
            return True

        # Use user preference for other changes
        return self.auto_zoom_var.get()

    def _detect_new_signals(self, current_signals: list[str]) -> bool:
        """Detect if new signals have been added since last plot update."""
        if not hasattr(self, "last_plotted_signals"):
            self.last_plotted_signals = set()
            return True  # First time plotting, treat as new signals

        current_set = set(current_signals)
        new_signals = current_set - self.last_plotted_signals

        # Update the last plotted signals
        self.last_plotted_signals = current_set

        return len(new_signals) > 0

    def _apply_zoom_state(self, zoom_state: dict[str, Any]) -> None:
        """Apply stored zoom state after plot update."""
        if zoom_state and hasattr(self, "plot_ax"):
            try:
                self.plot_ax.set_xlim(zoom_state["xlim"])
                self.plot_ax.set_ylim(zoom_state["ylim"])
            except Exception as e:
                print(f"Error restoring zoom state: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Starting Advanced CSV Processor - Complete Version...")
    app = CSVProcessorApp()
    app.mainloop()

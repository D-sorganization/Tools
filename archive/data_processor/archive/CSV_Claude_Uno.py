# =============================================================================
# Enhanced Plant Data Processor & Analyzer with ML Format Support
#
# Description:
# A comprehensive GUI application for processing, analyzing, and converting
# plant data from CSV/DAT files with support for machine learning formats
# including Parquet, Feather, and HDF5.
#
# Dependencies for Python 3.8+:
# pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow
# pip install pyarrow fastparquet tables h5py simpledbf
#
# =============================================================================

import configparser
import io
import json
import os
import re
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from scipy.signal import butter, filtfilt, medfilt, savgol_filter
from scipy.stats import linregress
from simpledbf import Dbf5

# ML Format imports
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import pyarrow.feather as feather

    FEATHER_AVAILABLE = True
except ImportError:
    FEATHER_AVAILABLE = False

try:
    import h5py
    import tables

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


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
        if filter_type != "None":
            numeric_cols = processed_df.select_dtypes(
                include=np.number
            ).columns.tolist()
            for col in numeric_cols:
                signal_data = processed_df[col].dropna()
                if len(signal_data) < 2:
                    continue
                # Apply the filtering logic here
                pass

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
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
        return None


class EnhancedPlantDataProcessor(ctk.CTk):
    """Enhanced data processor with ML format support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Enhanced Plant Data Processor & ML Format Converter")
        self.geometry("1400x950")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- App State Variables ---
        self.input_file_paths = []
        self.loaded_data_cache = {}
        self.output_directory = os.path.expanduser("~/Documents")
        self.signal_vars = {}
        self.plot_signal_vars = {}
        self.filter_names = [
            "None",
            "Moving Average",
            "Median Filter",
            "Butterworth Low-pass",
            "Butterworth High-pass",
            "Savitzky-Golay",
        ]
        self.custom_vars_list = []
        self.reference_signal_widgets = {}
        self.dat_import_tag_file_path = None
        self.dat_import_data_file_path = None
        self.dat_tag_vars = {}

        # Enhanced export formats
        self.export_formats = [
            "CSV (Separate Files)",
            "CSV (Compiled)",
            "Excel (Multi-sheet)",
            "Excel (Separate Files)",
            "MAT (Separate Files)",
            "MAT (Compiled)",
        ]

        # Add ML formats if available
        if PARQUET_AVAILABLE:
            self.export_formats.extend(
                ["Parquet (Separate Files)", "Parquet (Compiled)"]
            )
        if FEATHER_AVAILABLE:
            self.export_formats.extend(
                ["Feather (Separate Files)", "Feather (Compiled)"]
            )
        if HDF5_AVAILABLE:
            self.export_formats.extend(["HDF5 (Separate Files)", "HDF5 (Compiled)"])

        # --- Create Main UI ---
        self.main_tab_view = ctk.CTkTabview(self)
        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.main_tab_view.add("Setup & Process")
        self.main_tab_view.add("Format Converter")
        self.main_tab_view.add("Time Range Extractor")
        self.main_tab_view.add("Plotting & Analysis")
        self.main_tab_view.add("DAT File Import")

        self.create_setup_and_process_tab(self.main_tab_view.tab("Setup & Process"))
        self.create_format_converter_tab(self.main_tab_view.tab("Format Converter"))
        self.create_time_range_extractor_tab(
            self.main_tab_view.tab("Time Range Extractor")
        )
        self.create_plotting_tab(self.main_tab_view.tab("Plotting & Analysis"))
        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))

        self.create_status_bar()
        self.status_label.configure(
            text="Ready. Enhanced processor with ML format support loaded."
        )

    def create_setup_and_process_tab(self, parent_tab):
        """Creates the main setup and processing tab."""
        parent_tab.grid_columnconfigure(1, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(parent_tab, width=380)
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_panel.grid_rowconfigure(1, weight=1)

        # Header with Help Button
        header_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=15, pady=10, sticky="ew")
        ctk.CTkLabel(
            header_frame, text="Control Panel", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left")
        ctk.CTkButton(
            header_frame, text="Help", width=70, command=self._show_setup_help
        ).pack(side="right")

        processing_tab_view = ctk.CTkTabview(left_panel)
        processing_tab_view.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        processing_tab_view.add("Setup")
        processing_tab_view.add("Processing")
        processing_tab_view.add("Custom Vars")

        self.populate_setup_sub_tab(processing_tab_view.tab("Setup"))
        self.populate_processing_sub_tab(processing_tab_view.tab("Processing"))
        self.populate_custom_var_sub_tab(processing_tab_view.tab("Custom Vars"))

        self.process_button = ctk.CTkButton(
            left_panel,
            text="Process & Batch Export Files",
            height=40,
            command=self.process_files,
        )
        self.process_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # Right panel for file list and signal selection
        right_panel = ctk.CTkFrame(parent_tab)
        right_panel.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        right_panel.grid_rowconfigure(2, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self.file_list_frame = ctk.CTkScrollableFrame(
            right_panel, label_text="Selected Input Files", height=120
        )
        self.file_list_frame.grid(row=0, column=0, padx=10, pady=(0, 10), sticky="new")
        self.initial_file_label = ctk.CTkLabel(
            self.file_list_frame, text="Files you select will be listed here."
        )
        self.initial_file_label.pack(padx=5, pady=5)

        signal_control_frame = ctk.CTkFrame(right_panel)
        signal_control_frame.grid(row=1, column=0, padx=10, pady=0, sticky="ew")
        signal_control_frame.grid_columnconfigure(0, weight=1)

        self.search_entry = ctk.CTkEntry(
            signal_control_frame, placeholder_text="Search for signals..."
        )
        self.search_entry.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.search_entry.bind("<KeyRelease>", self._filter_signals)
        self.clear_search_button = ctk.CTkButton(
            signal_control_frame, text="X", width=28, command=self._clear_search
        )
        self.clear_search_button.grid(row=0, column=1, padx=5)
        ctk.CTkButton(
            signal_control_frame, text="Select All", width=100, command=self.select_all
        ).grid(row=0, column=2, padx=5)
        ctk.CTkButton(
            signal_control_frame,
            text="Deselect All",
            width=100,
            command=self.deselect_all,
        ).grid(row=0, column=3)

        self.signal_list_frame = ctk.CTkScrollableFrame(
            right_panel, label_text="Available Signals to Process"
        )
        self.signal_list_frame.grid(
            row=2, column=0, padx=10, pady=(5, 10), sticky="nsew"
        )
        self.signal_list_frame.grid_columnconfigure(0, weight=1)

    def create_format_converter_tab(self, parent_tab):
        """Creates the format converter tab for converting between different data formats."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(1, weight=1)

        # Top panel for controls
        control_frame = ctk.CTkFrame(parent_tab)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            control_frame,
            text="Format Converter",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # Input format selection
        ctk.CTkLabel(control_frame, text="Input Format:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        self.input_format_var = ctk.StringVar(value="CSV")
        input_formats = ["CSV", "Excel", "Parquet", "Feather", "HDF5", "MAT"]
        self.input_format_menu = ctk.CTkOptionMenu(
            control_frame, variable=self.input_format_var, values=input_formats
        )
        self.input_format_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Output format selection
        ctk.CTkLabel(control_frame, text="Output Format:").grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )
        self.output_format_var = ctk.StringVar(value="Parquet")
        output_formats = ["CSV", "Excel", "Parquet", "Feather", "HDF5", "JSON"]
        self.output_format_menu = ctk.CTkOptionMenu(
            control_frame, variable=self.output_format_var, values=output_formats
        )
        self.output_format_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Buttons
        button_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_frame.grid(row=1, column=2, rowspan=2, padx=10, pady=5, sticky="ns")

        ctk.CTkButton(
            button_frame, text="Select Files", command=self.select_conversion_files
        ).pack(pady=2)
        ctk.CTkButton(button_frame, text="Convert", command=self.convert_formats).pack(
            pady=2
        )
        ctk.CTkButton(
            button_frame, text="Help", command=self._show_converter_help
        ).pack(pady=2)

        # Main panel for file list and options
        main_frame = ctk.CTkFrame(parent_tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # File list
        self.conversion_file_list = ctk.CTkScrollableFrame(
            main_frame, label_text="Files to Convert"
        )
        self.conversion_file_list.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Conversion options
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(
            options_frame, text="Conversion Options", font=ctk.CTkFont(weight="bold")
        ).pack(padx=10, pady=10)

        # Compression options
        compression_frame = ctk.CTkFrame(options_frame)
        compression_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(compression_frame, text="Compression:").pack(
            anchor="w", padx=10, pady=5
        )
        self.compression_var = ctk.StringVar(value="snappy")
        ctk.CTkOptionMenu(
            compression_frame,
            variable=self.compression_var,
            values=["none", "snappy", "gzip", "brotli"],
        ).pack(fill="x", padx=10, pady=5)

        # Batch processing options
        batch_frame = ctk.CTkFrame(options_frame)
        batch_frame.pack(fill="x", padx=10, pady=5)
        self.combine_files_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            batch_frame,
            text="Combine all files into single output",
            variable=self.combine_files_var,
        ).pack(anchor="w", padx=10, pady=5)

        # Memory optimization
        memory_frame = ctk.CTkFrame(options_frame)
        memory_frame.pack(fill="x", padx=10, pady=5)
        self.chunk_processing_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            memory_frame,
            text="Use chunk processing for large files",
            variable=self.chunk_processing_var,
        ).pack(anchor="w", padx=10, pady=5)

        self.conversion_files = []

    def create_time_range_extractor_tab(self, parent_tab):
        """Creates the time range extractor tab for extracting specific time periods."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(1, weight=1)

        # Top control panel
        control_frame = ctk.CTkFrame(parent_tab)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            control_frame,
            text="Time Range Extractor",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        ctk.CTkButton(
            control_frame, text="Select Files", command=self.select_extraction_files
        ).grid(row=1, column=0, padx=10, pady=5)
        ctk.CTkButton(
            control_frame, text="Extract Ranges", command=self.extract_time_ranges
        ).grid(row=1, column=1, padx=10, pady=5)
        ctk.CTkButton(
            control_frame, text="Help", width=70, command=self._show_extractor_help
        ).grid(row=1, column=2, padx=10, pady=5)

        # Main content
        main_frame = ctk.CTkFrame(parent_tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # File list
        self.extraction_file_list = ctk.CTkScrollableFrame(
            main_frame, label_text="Files for Time Range Extraction"
        )
        self.extraction_file_list.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Time range configuration
        range_config_frame = ctk.CTkFrame(main_frame)
        range_config_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(
            range_config_frame,
            text="Time Range Configuration",
            font=ctk.CTkFont(weight="bold"),
        ).pack(padx=10, pady=10)

        # Date range
        date_frame = ctk.CTkFrame(range_config_frame)
        date_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(date_frame, text="Start Date (YYYY-MM-DD):").pack(
            anchor="w", padx=10, pady=2
        )
        self.start_date_entry = ctk.CTkEntry(date_frame, placeholder_text="2024-01-01")
        self.start_date_entry.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(date_frame, text="End Date (YYYY-MM-DD):").pack(
            anchor="w", padx=10, pady=2
        )
        self.end_date_entry = ctk.CTkEntry(date_frame, placeholder_text="2024-12-31")
        self.end_date_entry.pack(fill="x", padx=10, pady=2)

        # Time range
        time_frame = ctk.CTkFrame(range_config_frame)
        time_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(time_frame, text="Start Time (HH:MM:SS):").pack(
            anchor="w", padx=10, pady=2
        )
        self.start_time_entry = ctk.CTkEntry(time_frame, placeholder_text="00:00:00")
        self.start_time_entry.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(time_frame, text="End Time (HH:MM:SS):").pack(
            anchor="w", padx=10, pady=2
        )
        self.end_time_entry = ctk.CTkEntry(time_frame, placeholder_text="23:59:59")
        self.end_time_entry.pack(fill="x", padx=10, pady=2)

        # Output options
        output_frame = ctk.CTkFrame(range_config_frame)
        output_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(output_frame, text="Output Format:").pack(
            anchor="w", padx=10, pady=2
        )
        self.extraction_format_var = ctk.StringVar(value="CSV")
        ctk.CTkOptionMenu(
            output_frame,
            variable=self.extraction_format_var,
            values=["CSV", "Parquet", "Feather", "HDF5", "Excel"],
        ).pack(fill="x", padx=10, pady=2)

        # Resampling options
        resample_frame = ctk.CTkFrame(range_config_frame)
        resample_frame.pack(fill="x", padx=10, pady=5)
        self.resample_extraction_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            resample_frame, text="Resample data", variable=self.resample_extraction_var
        ).pack(anchor="w", padx=10, pady=2)
        self.resample_frequency_entry = ctk.CTkEntry(
            resample_frame, placeholder_text="10S"
        )
        self.resample_frequency_entry.pack(fill="x", padx=10, pady=2)

        self.extraction_files = []

    def create_plotting_tab(self, parent_tab):
        """Creates the plotting and analysis tab."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(1, weight=1)

        # Top control bar
        plot_control_frame = ctk.CTkFrame(parent_tab)
        plot_control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        plot_control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(plot_control_frame, text="File to Plot:").grid(
            row=0, column=0, padx=(10, 5), pady=10
        )
        self.plot_file_menu = ctk.CTkOptionMenu(
            plot_control_frame,
            values=["Select a file..."],
            command=self.on_plot_file_select,
        )
        self.plot_file_menu.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        ctk.CTkLabel(plot_control_frame, text="X-Axis:").grid(
            row=0, column=2, padx=(10, 5), pady=10
        )
        self.plot_xaxis_menu = ctk.CTkOptionMenu(
            plot_control_frame,
            values=["default time"],
            command=lambda e: self.update_plot(),
        )
        self.plot_xaxis_menu.grid(row=0, column=3, padx=5, pady=10, sticky="ew")

        ctk.CTkButton(
            plot_control_frame, text="Help", width=70, command=self._show_plot_help
        ).grid(row=0, column=4, padx=(10, 5), pady=10)

        # Main content frame
        plot_main_frame = ctk.CTkFrame(parent_tab)
        plot_main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        plot_main_frame.grid_rowconfigure(0, weight=1)
        plot_main_frame.grid_columnconfigure(1, weight=1)

        # Left panel for controls
        plot_left_panel = ctk.CTkScrollableFrame(
            plot_main_frame, label_text="Plotting Controls", width=350
        )
        plot_left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")

        # Signal selection
        signal_frame = ctk.CTkFrame(plot_left_panel)
        signal_frame.pack(fill="x", padx=5, pady=5)
        self.plot_signal_frame = ctk.CTkScrollableFrame(
            signal_frame, label_text="Signals to Plot", height=200
        )
        self.plot_signal_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Plot canvas
        plot_canvas_frame = ctk.CTkFrame(plot_main_frame)
        plot_canvas_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        plot_canvas_frame.grid_rowconfigure(1, weight=1)
        plot_canvas_frame.grid_columnconfigure(0, weight=1)

        self.plot_fig = Figure(figsize=(8, 6), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_fig.tight_layout()

        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_canvas_frame)
        self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(
            self.plot_canvas, plot_canvas_frame, pack_toolbar=False
        )
        toolbar.grid(row=0, column=0, sticky="ew")

    def create_dat_import_tab(self, parent_tab):
        """Creates the DAT file import tab with enhanced timestamp generation."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)

        # Scrollable main frame
        main_frame = ctk.CTkScrollableFrame(parent_tab)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(
            header_frame,
            text="DAT File Import & Processing",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left")
        ctk.CTkButton(
            header_frame, text="Help", width=70, command=self._show_dat_help
        ).pack(side="right")

        # File selection
        file_frame = ctk.CTkFrame(main_frame)
        file_frame.pack(fill="x", padx=10, pady=10)
        file_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            file_frame, text="Step 1: Select Tag File", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkButton(
            file_frame, text="Select Tag File", command=self._select_tag_file
        ).grid(row=1, column=0, padx=10, pady=5)
        self.tag_file_label = ctk.CTkLabel(
            file_frame, text="No file selected", anchor="w"
        )
        self.tag_file_label.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(file_frame, text="Preview", command=self._preview_tag_file).grid(
            row=1, column=2, padx=10, pady=5
        )

        ctk.CTkLabel(
            file_frame, text="Step 2: Select Data File", font=ctk.CTkFont(weight="bold")
        ).grid(row=2, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkButton(
            file_frame, text="Select Data File", command=self._select_dat_file
        ).grid(row=3, column=0, padx=10, pady=5)
        self.dat_file_label = ctk.CTkLabel(
            file_frame, text="No file selected", anchor="w"
        )
        self.dat_file_label.grid(
            row=3, column=1, columnspan=2, padx=10, pady=5, sticky="ew"
        )

        # Timestamp configuration
        timestamp_frame = ctk.CTkFrame(main_frame)
        timestamp_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            timestamp_frame,
            text="Step 3: Configure Timestamps",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=5)

        # Sample rate
        sample_frame = ctk.CTkFrame(timestamp_frame, fg_color="transparent")
        sample_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(sample_frame, text="Sample Interval:").pack(side="left")
        self.sample_interval_entry = ctk.CTkEntry(sample_frame, width=100)
        self.sample_interval_entry.pack(side="left", padx=5)
        self.sample_interval_entry.insert(0, "10")
        self.sample_unit_menu = ctk.CTkOptionMenu(
            sample_frame, values=["seconds", "minutes", "hours"]
        )
        self.sample_unit_menu.pack(side="left")
        self.sample_unit_menu.set("seconds")

        # Manual timestamp entry
        manual_frame = ctk.CTkFrame(timestamp_frame, fg_color="transparent")
        manual_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(manual_frame, text="Start Time (if not in filename):").pack(
            side="left"
        )
        self.manual_start_time = ctk.CTkEntry(
            manual_frame, placeholder_text="YYYY-MM-DD HH:MM:SS"
        )
        self.manual_start_time.pack(side="left", fill="x", expand=True, padx=5)

        # Tag selection
        self.dat_tags_frame = ctk.CTkScrollableFrame(
            main_frame, label_text="Step 4: Select Tags", height=200
        )
        self.dat_tags_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Processing options
        processing_frame = ctk.CTkFrame(main_frame)
        processing_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            processing_frame,
            text="Step 5: Processing Options",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=5)

        options_inner = ctk.CTkFrame(processing_frame, fg_color="transparent")
        options_inner.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(options_inner, text="Data Reduction Factor:").pack(side="left")
        self.dat_reduction_entry = ctk.CTkEntry(
            options_inner, width=100, placeholder_text="1"
        )
        self.dat_reduction_entry.pack(side="left", padx=5)

        ctk.CTkLabel(options_inner, text="Output Format:").pack(
            side="left", padx=(20, 5)
        )
        self.dat_output_format = ctk.StringVar(value="CSV")
        ctk.CTkOptionMenu(
            options_inner,
            variable=self.dat_output_format,
            values=["CSV", "Parquet", "Feather", "HDF5"],
        ).pack(side="left")

        # Convert button
        self.convert_dat_button = ctk.CTkButton(
            main_frame,
            text="Step 6: Convert and Load",
            height=40,
            command=self._run_dat_conversion,
        )
        self.convert_dat_button.pack(fill="x", padx=10, pady=20)

    def create_status_bar(self):
        """Creates the status bar at the bottom."""
        status_frame = ctk.CTkFrame(self, height=40)
        status_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="sew")
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_propagate(False)

        self.status_label = ctk.CTkLabel(status_frame, text="", anchor="w")
        self.status_label.grid(row=0, column=0, padx=10, sticky="ew")

        self.progressbar = ctk.CTkProgressBar(
            status_frame, orientation="horizontal", width=200
        )
        self.progressbar.set(0)
        self.progressbar.grid(row=0, column=1, padx=10, sticky="e")

    def populate_setup_sub_tab(self, tab):
        """Populates the setup sub-tab."""
        tab.grid_columnconfigure(0, weight=1)

        # File selection
        file_frame = ctk.CTkFrame(tab)
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        file_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            file_frame, text="File Selection", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkButton(
            file_frame, text="Select Input Files", command=self.select_files
        ).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            file_frame, text="Select Output Folder", command=self.select_output_folder
        ).grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.output_label = ctk.CTkLabel(
            file_frame,
            text=f"Output: {self.output_directory}",
            wraplength=300,
            justify="left",
        )
        self.output_label.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")

        # Export options with enhanced formats
        export_frame = ctk.CTkFrame(tab)
        export_frame.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        export_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            export_frame, text="Export Options", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        ctk.CTkLabel(export_frame, text="Format:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        self.export_type_var = ctk.StringVar(value="CSV (Separate Files)")
        self.export_type_menu = ctk.CTkOptionMenu(
            export_frame, variable=self.export_type_var, values=self.export_formats
        )
        self.export_type_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

    def populate_processing_sub_tab(self, tab):
        """Populates the processing sub-tab."""
        tab.grid_columnconfigure(0, weight=1)

        # Filter options
        filter_frame = ctk.CTkFrame(tab)
        filter_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        filter_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            filter_frame, text="Signal Filtering", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(filter_frame, text="Filter Type:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        self.filter_type_var = ctk.StringVar(value="None")
        ctk.CTkOptionMenu(
            filter_frame, variable=self.filter_type_var, values=self.filter_names
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")

    def populate_custom_var_sub_tab(self, tab):
        """Populates the custom variables sub-tab."""
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tab, text="Custom Variables", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")

        var_frame = ctk.CTkFrame(tab)
        var_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        var_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(var_frame, text="Variable Name:").grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )
        self.custom_var_name_entry = ctk.CTkEntry(var_frame)
        self.custom_var_name_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(var_frame, text="Formula:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        self.custom_var_formula_entry = ctk.CTkEntry(var_frame)
        self.custom_var_formula_entry.grid(
            row=1, column=1, padx=10, pady=5, sticky="ew"
        )

        ctk.CTkButton(
            var_frame, text="Add Variable", command=self._add_custom_variable
        ).grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    # Format conversion methods
    def select_conversion_files(self):
        """Select files for format conversion."""
        input_format = self.input_format_var.get().lower()

        if input_format == "csv":
            filetypes = [("CSV files", "*.csv")]
        elif input_format == "excel":
            filetypes = [("Excel files", "*.xlsx"), ("Excel files", "*.xls")]
        elif input_format == "parquet":
            filetypes = [("Parquet files", "*.parquet")]
        elif input_format == "feather":
            filetypes = [("Feather files", "*.feather")]
        elif input_format == "hdf5":
            filetypes = [("HDF5 files", "*.h5"), ("HDF5 files", "*.hdf5")]
        elif input_format == "mat":
            filetypes = [("MAT files", "*.mat")]
        else:
            filetypes = [("All files", "*.*")]

        files = filedialog.askopenfilenames(
            title="Select Files to Convert", filetypes=filetypes
        )

        if files:
            self.conversion_files = list(files)
            self._update_conversion_file_list()

    def _update_conversion_file_list(self):
        """Update the conversion file list display."""
        for widget in self.conversion_file_list.winfo_children():
            widget.destroy()

        for file_path in self.conversion_files:
            ctk.CTkLabel(
                self.conversion_file_list, text=os.path.basename(file_path)
            ).pack(anchor="w", padx=5, pady=2)

    def convert_formats(self):
        """Convert files between different formats."""
        if not self.conversion_files:
            messagebox.showwarning("Warning", "Please select files to convert.")
            return

        input_format = self.input_format_var.get().lower()
        output_format = self.output_format_var.get().lower()

        if input_format == output_format:
            messagebox.showwarning("Warning", "Input and output formats are the same.")
            return

        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        try:
            self.progressbar.set(0)
            self.status_label.configure(text="Converting files...")

            combine_files = self.combine_files_var.get()
            compression = (
                self.compression_var.get()
                if self.compression_var.get() != "none"
                else None
            )
            use_chunks = self.chunk_processing_var.get()

            if combine_files:
                self._convert_files_combined(
                    self.conversion_files,
                    input_format,
                    output_format,
                    output_dir,
                    compression,
                    use_chunks,
                )
            else:
                self._convert_files_separate(
                    self.conversion_files,
                    input_format,
                    output_format,
                    output_dir,
                    compression,
                    use_chunks,
                )

            messagebox.showinfo(
                "Success",
                f"Successfully converted {len(self.conversion_files)} files to {output_format.upper()}",
            )
            self.status_label.configure(text="Conversion completed.")

        except Exception as e:
            messagebox.showerror("Error", f"Conversion failed: {str(e)}")
            self.status_label.configure(text="Conversion failed.")
        finally:
            self.progressbar.set(0)

    def _convert_files_separate(
        self, files, input_format, output_format, output_dir, compression, use_chunks
    ):
        """Convert files separately."""
        for i, file_path in enumerate(files):
            self.status_label.configure(
                text=f"Converting {os.path.basename(file_path)}..."
            )
            self.progressbar.set((i + 1) / len(files))
            self.update_idletasks()

            # Load data based on input format
            df = self._load_data_by_format(file_path, input_format, use_chunks)

            # Save data based on output format
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(
                output_dir, f"{base_name}.{self._get_file_extension(output_format)}"
            )

            self._save_data_by_format(df, output_path, output_format, compression)

    def _convert_files_combined(
        self, files, input_format, output_format, output_dir, compression, use_chunks
    ):
        """Convert and combine multiple files into one."""
        combined_data = []

        for i, file_path in enumerate(files):
            self.status_label.configure(
                text=f"Loading {os.path.basename(file_path)}..."
            )
            self.progressbar.set(
                (i + 1) / (len(files) * 2)
            )  # Half progress for loading
            self.update_idletasks()

            df = self._load_data_by_format(file_path, input_format, use_chunks)
            df["source_file"] = os.path.basename(file_path)
            combined_data.append(df)

        # Combine all dataframes
        self.status_label.configure(text="Combining data...")
        self.progressbar.set(0.75)
        self.update_idletasks()

        combined_df = pd.concat(combined_data, ignore_index=True)

        # Save combined data
        output_path = os.path.join(
            output_dir, f"combined_data.{self._get_file_extension(output_format)}"
        )
        self._save_data_by_format(combined_df, output_path, output_format, compression)

    def _load_data_by_format(self, file_path, format_type, use_chunks=False):
        """Load data based on format type."""
        if format_type == "csv":
            if use_chunks:
                chunks = pd.read_csv(file_path, chunksize=10000, low_memory=False)
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.read_csv(file_path, low_memory=False)

        elif format_type == "excel":
            return pd.read_excel(file_path)

        elif format_type == "parquet":
            if not PARQUET_AVAILABLE:
                raise ImportError("Parquet support not available. Install pyarrow.")
            return pd.read_parquet(file_path)

        elif format_type == "feather":
            if not FEATHER_AVAILABLE:
                raise ImportError("Feather support not available. Install pyarrow.")
            return pd.read_feather(file_path)

        elif format_type == "hdf5":
            if not HDF5_AVAILABLE:
                raise ImportError(
                    "HDF5 support not available. Install tables and h5py."
                )
            return pd.read_hdf(file_path, key="data")

        elif format_type == "mat":
            from scipy.io import loadmat

            mat_data = loadmat(file_path)
            # Convert MAT data to DataFrame (simplified)
            data_dict = {
                k: v.flatten() for k, v in mat_data.items() if not k.startswith("__")
            }
            return pd.DataFrame(data_dict)

        else:
            raise ValueError(f"Unsupported input format: {format_type}")

    def _save_data_by_format(self, df, file_path, format_type, compression=None):
        """Save data based on format type."""
        if format_type == "csv":
            df.to_csv(file_path, index=False, compression=compression)

        elif format_type == "excel":
            df.to_excel(file_path, index=False)

        elif format_type == "parquet":
            if not PARQUET_AVAILABLE:
                raise ImportError("Parquet support not available. Install pyarrow.")
            df.to_parquet(file_path, compression=compression)

        elif format_type == "feather":
            if not FEATHER_AVAILABLE:
                raise ImportError("Feather support not available. Install pyarrow.")
            df.to_feather(file_path, compression=compression)

        elif format_type == "hdf5":
            if not HDF5_AVAILABLE:
                raise ImportError(
                    "HDF5 support not available. Install tables and h5py."
                )
            df.to_hdf(
                file_path, key="data", mode="w", complib="zlib" if compression else None
            )

        elif format_type == "json":
            df.to_json(file_path, orient="records", compression=compression)

        else:
            raise ValueError(f"Unsupported output format: {format_type}")

    def _get_file_extension(self, format_type):
        """Get appropriate file extension for format."""
        extensions = {
            "csv": "csv",
            "excel": "xlsx",
            "parquet": "parquet",
            "feather": "feather",
            "hdf5": "h5",
            "json": "json",
        }
        return extensions.get(format_type, "txt")

    # Time range extraction methods
    def select_extraction_files(self):
        """Select files for time range extraction."""
        files = filedialog.askopenfilenames(
            title="Select Files for Time Range Extraction",
            filetypes=[
                ("All supported", "*.csv;*.parquet;*.feather;*.h5"),
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("Feather files", "*.feather"),
                ("HDF5 files", "*.h5"),
            ],
        )

        if files:
            self.extraction_files = list(files)
            self._update_extraction_file_list()

    def _update_extraction_file_list(self):
        """Update the extraction file list display."""
        for widget in self.extraction_file_list.winfo_children():
            widget.destroy()

        for file_path in self.extraction_files:
            ctk.CTkLabel(
                self.extraction_file_list, text=os.path.basename(file_path)
            ).pack(anchor="w", padx=5, pady=2)

    def extract_time_ranges(self):
        """Extract specified time ranges from selected files."""
        if not self.extraction_files:
            messagebox.showwarning("Warning", "Please select files for extraction.")
            return

        try:
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            start_time = self.start_time_entry.get() or "00:00:00"
            end_time = self.end_time_entry.get() or "23:59:59"

            if not start_date or not end_date:
                messagebox.showwarning("Warning", "Please specify start and end dates.")
                return

            start_datetime = pd.to_datetime(f"{start_date} {start_time}")
            end_datetime = pd.to_datetime(f"{end_date} {end_time}")

            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Extracted Files"
            )
            if not output_dir:
                return

            output_format = self.extraction_format_var.get().lower()
            resample = self.resample_extraction_var.get()
            resample_freq = self.resample_frequency_entry.get() if resample else None

            self.progressbar.set(0)
            extracted_count = 0

            for i, file_path in enumerate(self.extraction_files):
                self.status_label.configure(
                    text=f"Extracting from {os.path.basename(file_path)}..."
                )
                self.progressbar.set((i + 1) / len(self.extraction_files))
                self.update_idletasks()

                # Determine file format and load
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == ".csv":
                    df = pd.read_csv(file_path, low_memory=False)
                elif file_ext == ".parquet":
                    df = pd.read_parquet(file_path)
                elif file_ext == ".feather":
                    df = pd.read_feather(file_path)
                elif file_ext in [".h5", ".hdf5"]:
                    df = pd.read_hdf(file_path, key="data")
                else:
                    continue

                # Ensure time column is datetime
                time_col = df.columns[0]
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                df = df.dropna(subset=[time_col])

                if df.empty:
                    continue

                # Extract time range
                df_filtered = df[
                    (df[time_col] >= start_datetime) & (df[time_col] <= end_datetime)
                ]

                if df_filtered.empty:
                    continue

                # Resample if requested
                if resample and resample_freq:
                    df_filtered = (
                        df_filtered.set_index(time_col)
                        .resample(resample_freq)
                        .mean()
                        .reset_index()
                    )

                # Save extracted data
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = f"{base_name}_extracted_{start_date}_{end_date}.{self._get_file_extension(output_format)}"
                output_path = os.path.join(output_dir, output_filename)

                self._save_data_by_format(df_filtered, output_path, output_format)
                extracted_count += 1

            messagebox.showinfo(
                "Success",
                f"Successfully extracted time ranges from {extracted_count} files.",
            )
            self.status_label.configure(text="Time range extraction completed.")

        except Exception as e:
            messagebox.showerror("Error", f"Time range extraction failed: {str(e)}")
            self.status_label.configure(text="Extraction failed.")
        finally:
            self.progressbar.set(0)

    # Enhanced DAT processing methods
    def _select_tag_file(self):
        """Select the tag file for DAT import."""
        filepath = filedialog.askopenfilename(
            title="Select Tag File",
            filetypes=[("DBF files", "*.dbf"), ("All files", "*.*")],
        )
        if filepath:
            self.dat_import_tag_file_path = filepath
            self.tag_file_label.configure(text=os.path.basename(filepath))

    def _select_dat_file(self):
        """Select the data file for DAT import."""
        filepath = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
        )
        if filepath:
            self.dat_import_data_file_path = filepath
            self.dat_file_label.configure(text=os.path.basename(filepath))

    def _preview_tag_file(self):
        """Preview the tag file contents."""
        if not self.dat_import_tag_file_path:
            messagebox.showwarning("Warning", "Please select a tag file first.")
            return

        try:
            dbf = Dbf5(self.dat_import_tag_file_path, codec="latin-1")
            df = dbf.to_dataframe()

            if "Tagname" in df.columns:
                tags = df["Tagname"].tolist()
                tags = [
                    tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()
                ]

                self._populate_dat_tag_list(tags)
                messagebox.showinfo("Success", f"Found {len(tags)} tags in the file.")
            else:
                messagebox.showerror(
                    "Error", "No 'Tagname' column found in the DBF file."
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview tag file: {str(e)}")

    def _populate_dat_tag_list(self, tags):
        """Populate the tag selection list."""
        for widget in self.dat_tags_frame.winfo_children():
            widget.destroy()

        self.dat_tag_vars = {}
        for tag in tags:
            var = tk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.dat_tags_frame, text=tag, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            self.dat_tag_vars[tag] = {"var": var, "widget": cb}

    def _run_dat_conversion(self):
        """Convert DAT file to specified format with enhanced timestamp handling."""
        if not self.dat_import_tag_file_path or not self.dat_import_data_file_path:
            messagebox.showwarning("Warning", "Please select both tag and data files.")
            return

        try:
            # Get tags from file
            dbf = Dbf5(self.dat_import_tag_file_path, codec="latin-1")
            df_tags = dbf.to_dataframe()
            all_tags = df_tags["Tagname"].tolist()
            all_tags = [
                tag.strip() for tag in all_tags if isinstance(tag, str) and tag.strip()
            ]

            # Get selected tags
            selected_tags = [
                tag for tag, data in self.dat_tag_vars.items() if data["var"].get()
            ]
            if not selected_tags:
                messagebox.showwarning("Warning", "Please select at least one tag.")
                return

            # Load binary data
            data_blob = np.fromfile(self.dat_import_data_file_path, dtype=np.float32)
            num_tags = len(all_tags)
            num_rows = len(data_blob) // num_tags
            data_reshaped = data_blob[: num_rows * num_tags].reshape(num_rows, num_tags)

            # Create DataFrame
            df = pd.DataFrame(data_reshaped, columns=all_tags)

            # Generate timestamps
            start_time = (
                self._extract_timestamp_from_filename() or self._get_manual_timestamp()
            )
            if not start_time:
                messagebox.showerror(
                    "Error", "Could not determine start time. Please enter manually."
                )
                return

            sample_interval = int(self.sample_interval_entry.get())
            sample_unit = self.sample_unit_menu.get()

            if sample_unit == "minutes":
                freq = f"{sample_interval}min"
            elif sample_unit == "hours":
                freq = f"{sample_interval}H"
            else:  # seconds
                freq = f"{sample_interval}S"

            timestamps = pd.date_range(start=start_time, periods=num_rows, freq=freq)

            # Add timestamp columns
            df.insert(0, "Timestamp", timestamps)
            df.insert(1, "Date", timestamps.date)
            df.insert(2, "Time", timestamps.time)

            # Filter to selected tags
            df = df[["Timestamp", "Date", "Time"] + selected_tags]

            # Apply data reduction if specified
            reduction_factor = self.dat_reduction_entry.get()
            if reduction_factor and int(reduction_factor) > 1:
                factor = int(reduction_factor)
                df = df.iloc[::factor].reset_index(drop=True)

            # Save file
            output_format = self.dat_output_format.get().lower()
            save_path = filedialog.asksaveasfilename(
                title="Save Converted File",
                defaultextension=f".{self._get_file_extension(output_format)}",
                filetypes=[
                    (
                        f"{output_format.upper()} files",
                        f"*.{self._get_file_extension(output_format)}",
                    )
                ],
            )

            if save_path:
                self._save_data_by_format(df, save_path, output_format)

                # Add to loaded files
                if save_path not in self.input_file_paths:
                    self.input_file_paths.append(save_path)
                    self._update_file_list_ui()

                messagebox.showinfo(
                    "Success",
                    f"DAT file converted and saved as {output_format.upper()}",
                )
                self.status_label.configure(
                    text=f"DAT conversion completed: {os.path.basename(save_path)}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"DAT conversion failed: {str(e)}")

    def _extract_timestamp_from_filename(self):
        """Extract timestamp from DAT filename."""
        if not self.dat_import_data_file_path:
            return None

        filename = os.path.basename(self.dat_import_data_file_path)

        # Try different patterns
        patterns = [
            r"(\d{4})\s(\d{2})\s(\d{2})\s(\d{4})",  # YYYY MM DD HHMM
            r"(\d{4})(\d{2})(\d{2})(\d{4})",  # YYYYMMDDHHMM
            r"(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})",  # YYYY-MM-DD_HHMM
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                if len(groups) == 4:
                    year, month, day, hhmm = groups
                    hour = hhmm[:2]
                    minute = hhmm[2:]
                    return pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")
                elif len(groups) == 5:
                    year, month, day, hour, minute = groups
                    return pd.to_datetime(f"{year}-{month}-{day} {hour}:{minute}:00")

        return None

    def _get_manual_timestamp(self):
        """Get manually entered timestamp."""
        manual_time = self.manual_start_time.get().strip()
        if manual_time:
            try:
                return pd.to_datetime(manual_time)
            except:
                return None
        return None

    # Standard methods from original code
    def select_files(self):
        """Select input files."""
        paths = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[
                ("All supported", "*.csv;*.parquet;*.feather;*.h5"),
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("Feather files", "*.feather"),
                ("HDF5 files", "*.h5"),
            ],
        )

        if paths:
            self.input_file_paths = list(paths)
            self._update_file_list_ui()

    def _update_file_list_ui(self):
        """Update the file list UI."""
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()

        if self.input_file_paths:
            self.loaded_data_cache.clear()

            for f_path in self.input_file_paths:
                label = ctk.CTkLabel(
                    self.file_list_frame, text=os.path.basename(f_path)
                )
                label.pack(anchor="w", padx=5)

            self.update_signal_list()

            file_names = [os.path.basename(p) for p in self.input_file_paths]
            self.plot_file_menu.configure(values=file_names)
            if file_names:
                self.plot_file_menu.set(file_names[0])
                self.on_plot_file_select(file_names[0])

            self.status_label.configure(
                text=f"Loaded {len(self.input_file_paths)} files."
            )
        else:
            ctk.CTkLabel(self.file_list_frame, text="No files selected.").pack(
                padx=5, pady=5
            )

    def update_signal_list(self):
        """Update the signal list from loaded files."""
        for widget in self.signal_list_frame.winfo_children():
            widget.destroy()
        self.signal_vars.clear()

        if not self.input_file_paths:
            return

        all_columns = set()
        try:
            for f in self.input_file_paths:
                # Determine file type and read accordingly
                file_ext = os.path.splitext(f)[1].lower()
                if file_ext == ".csv":
                    df = pd.read_csv(f, nrows=0)
                elif file_ext == ".parquet":
                    df = pd.read_parquet(f)
                    df = df.iloc[:0]  # Keep structure, remove data
                elif file_ext == ".feather":
                    df = pd.read_feather(f)
                    df = df.iloc[:0]
                elif file_ext in [".h5", ".hdf5"]:
                    df = pd.read_hdf(f, key="data")
                    df = df.iloc[:0]
                else:
                    continue

                all_columns.update(df.columns)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file headers: {e}")
            return

        sorted_columns = sorted(list(all_columns))

        for signal in sorted_columns:
            var = tk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.signal_list_frame, text=signal, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            self.signal_vars[signal] = {"var": var, "widget": cb}

    def select_output_folder(self):
        """Select output folder."""
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_directory = path
            self.output_label.configure(text=f"Output: {self.output_directory}")

    def select_all(self):
        """Select all signals."""
        for data in self.signal_vars.values():
            data["var"].set(True)

    def deselect_all(self):
        """Deselect all signals."""
        for data in self.signal_vars.values():
            data["var"].set(False)

    def _filter_signals(self, event=None):
        """Filter signals based on search."""
        search_term = self.search_entry.get().lower()
        for signal_name, data in self.signal_vars.items():
            widget = data["widget"]
            if search_term in signal_name.lower():
                widget.pack(anchor="w", padx=10, pady=2)
            else:
                widget.pack_forget()

    def _clear_search(self):
        """Clear search filter."""
        self.search_entry.delete(0, "end")
        self._filter_signals()

    def on_plot_file_select(self, filename):
        """Handle plot file selection."""
        if filename == "Select a file...":
            return

        # Clear existing plot signal widgets
        for widget in self.plot_signal_frame.winfo_children():
            widget.destroy()

        self.plot_signal_vars = {}

        # Load file and get columns
        filepath = next(
            (p for p in self.input_file_paths if os.path.basename(p) == filename), None
        )
        if not filepath:
            return

        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            if file_ext == ".csv":
                df = pd.read_csv(filepath, nrows=5)
            elif file_ext == ".parquet":
                df = pd.read_parquet(filepath)
                df = df.head(5)
            elif file_ext == ".feather":
                df = pd.read_feather(filepath)
                df = df.head(5)
            elif file_ext in [".h5", ".hdf5"]:
                df = pd.read_hdf(filepath, key="data")
                df = df.head(5)
            else:
                return

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            for signal in numeric_cols:
                var = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(
                    self.plot_signal_frame,
                    text=signal,
                    variable=var,
                    command=self.update_plot,
                )
                cb.pack(anchor="w", padx=5, pady=2)
                self.plot_signal_vars[signal] = {"var": var, "widget": cb}

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file for plotting: {e}")

    def update_plot(self):
        """Update the plot display."""
        # Basic plot update - would need full implementation
        self.plot_ax.clear()
        self.plot_ax.text(
            0.5,
            0.5,
            "Plotting functionality would be implemented here",
            ha="center",
            va="center",
            transform=self.plot_ax.transAxes,
        )
        self.plot_canvas.draw()

    def process_files(self):
        """Process selected files."""
        if not self.input_file_paths:
            messagebox.showwarning("Warning", "Please select input files.")
            return

        selected_signals = [
            s for s, data in self.signal_vars.items() if data["var"].get()
        ]
        if not selected_signals:
            messagebox.showwarning("Warning", "Please select signals to retain.")
            return

        try:
            self.process_button.configure(state="disabled", text="Processing...")
            self.progressbar.set(0)

            export_type = self.export_type_var.get()

            if "Separate" in export_type:
                self._process_files_separate(selected_signals, export_type)
            else:
                self._process_files_combined(selected_signals, export_type)

            messagebox.showinfo("Success", "File processing completed successfully.")
            self.status_label.configure(text="Processing completed.")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_label.configure(text="Processing failed.")
        finally:
            self.process_button.configure(
                state="normal", text="Process & Batch Export Files"
            )
            self.progressbar.set(0)

    def _process_files_separate(self, selected_signals, export_type):
        """Process files separately."""
        for i, file_path in enumerate(self.input_file_paths):
            self.status_label.configure(
                text=f"Processing {os.path.basename(file_path)}..."
            )
            self.progressbar.set((i + 1) / len(self.input_file_paths))
            self.update_idletasks()

            # Load and process file
            df = self._load_and_process_file(file_path, selected_signals)

            # Save in specified format
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            if "CSV" in export_type:
                output_path = os.path.join(
                    self.output_directory, f"{base_name}_processed.csv"
                )
                df.to_csv(output_path, index=False)
            elif "Parquet" in export_type:
                output_path = os.path.join(
                    self.output_directory, f"{base_name}_processed.parquet"
                )
                df.to_parquet(output_path)
            elif "Feather" in export_type:
                output_path = os.path.join(
                    self.output_directory, f"{base_name}_processed.feather"
                )
                df.to_feather(output_path)
            elif "HDF5" in export_type:
                output_path = os.path.join(
                    self.output_directory, f"{base_name}_processed.h5"
                )
                df.to_hdf(output_path, key="data", mode="w")

    def _process_files_combined(self, selected_signals, export_type):
        """Process and combine files."""
        combined_data = []

        for i, file_path in enumerate(self.input_file_paths):
            self.status_label.configure(
                text=f"Processing {os.path.basename(file_path)}..."
            )
            self.progressbar.set((i + 1) / (len(self.input_file_paths) + 1))
            self.update_idletasks()

            df = self._load_and_process_file(file_path, selected_signals)
            df["source_file"] = os.path.basename(file_path)
            combined_data.append(df)

        # Combine all data
        self.status_label.configure(text="Combining data...")
        self.progressbar.set(1.0)
        self.update_idletasks()

        combined_df = pd.concat(combined_data, ignore_index=True)

        # Save combined data
        if "CSV" in export_type:
            output_path = os.path.join(self.output_directory, "combined_data.csv")
            combined_df.to_csv(output_path, index=False)
        elif "Parquet" in export_type:
            output_path = os.path.join(self.output_directory, "combined_data.parquet")
            combined_df.to_parquet(output_path)
        elif "Feather" in export_type:
            output_path = os.path.join(self.output_directory, "combined_data.feather")
            combined_df.to_feather(output_path)
        elif "HDF5" in export_type:
            output_path = os.path.join(self.output_directory, "combined_data.h5")
            combined_df.to_hdf(output_path, key="data", mode="w")

    def _load_and_process_file(self, file_path, selected_signals):
        """Load and process a single file."""
        # Determine file type and load
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif file_ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_ext == ".feather":
            df = pd.read_feather(file_path)
        elif file_ext in [".h5", ".hdf5"]:
            df = pd.read_hdf(file_path, key="data")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        # Filter columns
        available_signals = [s for s in selected_signals if s in df.columns]
        time_col = df.columns[0]
        if time_col not in available_signals:
            available_signals.insert(0, time_col)

        return df[available_signals].copy()

    def _add_custom_variable(self):
        """Add a custom variable."""
        var_name = self.custom_var_name_entry.get().strip()
        formula = self.custom_var_formula_entry.get().strip()

        if not var_name or not formula:
            messagebox.showwarning(
                "Warning", "Please enter both variable name and formula."
            )
            return

        self.custom_vars_list.append((var_name, formula))
        self.custom_var_name_entry.delete(0, "end")
        self.custom_var_formula_entry.delete(0, "end")

        messagebox.showinfo("Success", f"Custom variable '{var_name}' added.")

    # Help methods
    def _show_setup_help(self):
        """Show setup help."""
        help_text = """
Enhanced Plant Data Processor - Setup & Process Help

This tab allows you to:
1. Select input files (CSV, Parquet, Feather, HDF5)
2. Choose signals to process
3. Configure processing options
4. Export in various formats optimized for machine learning

Enhanced formats supported:
- Parquet: Excellent for large datasets, column-oriented
- Feather: Fast read/write, good for intermediate processing
- HDF5: Hierarchical storage, good for complex data structures

Use the Format Converter tab to convert between different formats.
Use the Time Range Extractor to create smaller datasets from specific time periods.
        """
        self._show_help_window("Setup & Process Help", help_text)

    def _show_converter_help(self):
        """Show converter help."""
        help_text = """
Format Converter Help

Convert between different data formats optimized for various use cases:

CSV: Universal compatibility, human-readable
Excel: Business reporting, multiple sheets
Parquet: Columnar storage, excellent compression, fast analytics
Feather: Fast binary format, good for temporary files
HDF5: Hierarchical data, scientific computing
JSON: Web applications, API integration

Options:
- Compression: Reduce file size (snappy recommended for speed)
- Combine files: Merge multiple files into one
- Chunk processing: Handle large files efficiently

Recommended for ML:
- Parquet: Best for training datasets
- Feather: Good for intermediate processing
- HDF5: Complex hierarchical data
        """
        self._show_help_window("Format Converter Help", help_text)

    def _show_extractor_help(self):
        """Show extractor help."""
        help_text = """
Time Range Extractor Help

Extract specific time periods from your data files:

1. Select files containing timestamped data
2. Specify date and time ranges
3. Choose output format
4. Optionally resample data

Use cases:
- Create training datasets for specific periods
- Extract events or anomalies
- Reduce dataset size for testing
- Focus on specific operational periods

Time formats:
- Dates: YYYY-MM-DD (e.g., 2024-01-01)
- Times: HH:MM:SS (e.g., 09:30:00)
- Resampling: Use pandas frequency strings (10S, 1min, 1H)

Output formats optimized for different purposes:
- CSV: Universal compatibility
- Parquet: Fast loading, compressed
- Feather: Quick processing
- HDF5: Complex data structures
        """
        self._show_help_window("Time Range Extractor Help", help_text)

    def _show_plot_help(self):
        """Show plotting help."""
        help_text = """
Plotting & Analysis Help

Interactive visualization and analysis of your plant data:

Features:
- Multi-signal plotting
- Time series analysis
- Zoom and pan capabilities
- Export plots as images
- Data inspection

This is a simplified plotting interface. For advanced analysis,
consider exporting to Parquet or Feather format and using
specialized tools like Jupyter notebooks with matplotlib,
plotly, or other visualization libraries.
        """
        self._show_help_window("Plotting & Analysis Help", help_text)

    def _show_dat_help(self):
        """Show DAT import help."""
        help_text = """
DAT File Import Help

Convert proprietary binary DAT files to modern formats:

Process:
1. Select tag file (DBF format with signal names)
2. Select binary data file (DAT format)
3. Configure timestamps (auto-detected from filename or manual)
4. Select which tags to include
5. Choose output format and options

Timestamp Detection:
- Automatically parsed from filenames with date/time patterns
- Manual entry if automatic detection fails
- Configurable sample intervals (typically 10 seconds for plant data)

Output Formats:
- CSV: Universal compatibility
- Parquet: Optimized for analysis
- Feather: Fast binary format
- HDF5: Scientific data storage

Data Reduction:
Use reduction factor to downsample large files
(e.g., factor of 6 converts 10-second data to 1-minute data)
        """
        self._show_help_window("DAT Import Help", help_text)

    def _show_help_window(self, title, content):
        """Show a help window."""
        help_window = ctk.CTkToplevel(self)
        help_window.title(title)
        help_window.geometry("700x600")
        help_window.transient(self)
        help_window.grab_set()

        textbox = ctk.CTkTextbox(help_window, wrap="word")
        textbox.pack(expand=True, fill="both", padx=15, pady=15)
        textbox.insert("1.0", content)
        textbox.configure(state="disabled")

        close_button = ctk.CTkButton(
            help_window, text="Close", command=help_window.destroy
        )
        close_button.pack(pady=10)


if __name__ == "__main__":
    # Check for required packages
    missing_packages = []

    if not PARQUET_AVAILABLE:
        missing_packages.append("pyarrow (for Parquet/Feather support)")
    if not HDF5_AVAILABLE:
        missing_packages.append("tables and h5py (for HDF5 support)")

    if missing_packages:
        print("Optional packages not found:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall with: pip install pyarrow tables h5py")
        print("The application will run with limited format support.\n")

    # Set appearance
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    print("Starting Enhanced Plant Data Processor...")

    # Create and run the application
    app = EnhancedPlantDataProcessor()
    app.mainloop()

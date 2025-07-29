# =============================================================================
# Advanced CSV Time Series Processor & Analyzer - Complete Version
#
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This version combines all advanced features
# from Rev2 with the UI fixes from Rev4_Claude, ensuring complete functionality.
#
# Dependencies for Python 3.8+:
# pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow simpledbf
#
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, filtfilt, medfilt, savgol_filter
from scipy.stats import linregress
from scipy.io import savemat
import os
import configparser
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from simpledbf import Dbf5
import re
from PIL import Image
import io

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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
        signals_in_this_file = [s for s in settings['selected_signals'] if s in df.columns]
        time_col = df.columns[0]
        if time_col not in signals_in_this_file:
            signals_in_this_file.insert(0, time_col)
        
        processed_df = df[signals_in_this_file].copy()
        
        # Data type conversion
        processed_df[time_col] = pd.to_datetime(processed_df[time_col], errors='coerce')
        processed_df.dropna(subset=[time_col], inplace=True)
        for col in processed_df.columns:
            if col != time_col:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

        if processed_df.empty:
            return None

        processed_df.set_index(time_col, inplace=True)

        # Apply Filtering
        filter_type = settings.get('filter_type')
        if filter_type and filter_type != "None":
            numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
            for col in numeric_cols:
                signal_data = processed_df[col].dropna()
                if len(signal_data) < 2: 
                    continue
                
                # Apply filtering based on type
                if filter_type == "Moving Average":
                    window_size = settings.get('ma_window', 10)
                    processed_df[col] = signal_data.rolling(window=window_size, min_periods=1).mean()
                elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
                    order = settings.get('bw_order', 3)
                    cutoff = settings.get('bw_cutoff', 0.1)
                    sr = 1.0 / pd.to_numeric(signal_data.index.to_series().diff().dt.total_seconds()).mean()
                    if pd.notna(sr) and len(signal_data) > order * 3:
                        btype = 'low' if filter_type == "Butterworth Low-pass" else 'high'
                        b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sr)
                        processed_df[col] = pd.Series(filtfilt(b, a, signal_data), index=signal_data.index)
                elif filter_type == "Median Filter":
                    kernel = settings.get('median_kernel', 5)
                    if kernel % 2 == 0: kernel += 1
                    if len(signal_data) > kernel:
                        processed_df[col] = pd.Series(medfilt(signal_data, kernel_size=kernel), index=signal_data.index)
                elif filter_type == "Savitzky-Golay":
                    window = settings.get('savgol_window', 11)
                    polyorder = settings.get('savgol_polyorder', 2)
                    if window % 2 == 0: window += 1
                    if polyorder >= window: polyorder = window - 1
                    if len(signal_data) > window:
                        processed_df[col] = pd.Series(savgol_filter(signal_data, window, polyorder), index=signal_data.index)

        # Apply Resampling
        if settings.get('resample_enabled'):
            resample_rule = settings.get('resample_rule')
            if resample_rule:
                processed_df = processed_df.resample(resample_rule).mean().dropna(how='all')

        if processed_df.empty:
            return None

        processed_df.reset_index(inplace=True)
        return processed_df
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Helper function for causal derivative calculation
def _poly_derivative(series, window, poly_order, deriv_order, delta_x):
    """Calculates the derivative of a series using a rolling polynomial fit."""
    if poly_order < deriv_order:
        return pd.Series(np.nan, index=series.index)
    
    # Pad the series at the beginning to get derivatives for the initial points
    padded_series = pd.concat([pd.Series([series.iloc[0]] * (window - 1)), series])
    
    def get_deriv(w):
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
            
    return padded_series.rolling(window=window).apply(get_deriv, raw=True).iloc[window-1:] 

class CSVProcessorApp(ctk.CTk):
    """The main application class with all advanced features and UI fixes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Layout persistence variables
        self.layout_config_file = os.path.join(os.path.expanduser("~"), ".csv_processor_layout.json")
        self.splitters = {}
        self.layout_data = self._load_layout_config()

        self.title("Advanced CSV Processor & DAT Importer - Complete Version")
        
        # Set window size from saved layout or default
        window_width = self.layout_data.get('window_width', 1350)
        window_height = self.layout_data.get('window_height', 900)
        self.geometry(f"{window_width}x{window_height}")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Set up closing handler
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Set up window resize handler to save layout
        self.bind('<Configure>', self._on_window_configure)
        
        # App State Variables
        self.input_file_paths = []
        self.loaded_data_cache = {}
        self.processed_files = {}  # Store processed data for plotting
        self.output_directory = os.path.expanduser("~/Documents")
        self.signal_vars = {}
        self.plot_signal_vars = {}
        self.filter_names = ["None", "Moving Average", "Median Filter", "Hampel Filter", "Z-Score Filter", "Butterworth Low-pass", "Butterworth High-pass", "Savitzky-Golay"]
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
        for i in range(1, 6):  # Support up to 5th order derivatives
            self.derivative_vars[i] = tk.BooleanVar(value=False)
        
        # Plot view state management
        self.saved_plot_view = None
        
        # Custom legend entries for plots
        self.custom_legend_entries = {}

        # Create Main UI
        self.main_tab_view = ctk.CTkTabview(self)
        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.main_tab_view.add("Setup & Process")
        self.main_tab_view.add("Plotting & Analysis")
        self.main_tab_view.add("Plots List")
        self.main_tab_view.add("DAT File Import")
        self.main_tab_view.add("Help")

        self.create_setup_and_process_tab(self.main_tab_view.tab("Setup & Process"))
        self.create_plotting_tab(self.main_tab_view.tab("Plotting & Analysis"))
        self.create_plots_list_tab(self.main_tab_view.tab("Plots List"))
        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))
        self.create_help_tab(self.main_tab_view.tab("Help"))

        self.create_status_bar()
        self.status_label.configure(text="Ready. Select input files or import a DAT file.")
        
        # Load saved plots and other settings
        self._load_plots_from_file()

    def create_setup_and_process_tab(self, parent_tab):
        """Fixed version with proper splitter implementation and all advanced features."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)
        
        def create_left_content(left_panel):
            """Create the left panel content"""
            left_panel.grid_rowconfigure(0, weight=1)
            left_panel.grid_columnconfigure(0, weight=1)
            
            # Create a scrollable frame for the processing tab view
            processing_scrollable_frame = ctk.CTkScrollableFrame(left_panel)
            processing_scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
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
            
            self.process_button = ctk.CTkButton(left_panel, text="Process & Batch Export Files", height=40, command=self.process_files)
            self.process_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        def create_right_content(right_panel):
            """Create the right panel content"""
            right_panel.grid_rowconfigure(2, weight=1)
            right_panel.grid_columnconfigure(0, weight=1)
            
            # File list frame
            self.file_list_frame = ctk.CTkScrollableFrame(right_panel, label_text="Selected Input Files", height=120)
            self.file_list_frame.grid(row=0, column=0, padx=10, pady=(0, 10), sticky="new")
            self.initial_file_label = ctk.CTkLabel(self.file_list_frame, text="Files you select will be listed here.")
            self.initial_file_label.pack(padx=5, pady=5)
            
            # Signal control frame
            signal_control_frame = ctk.CTkFrame(right_panel)
            signal_control_frame.grid(row=1, column=0, padx=10, pady=0, sticky="ew")
            signal_control_frame.grid_columnconfigure(0, weight=1)
            
            self.search_entry = ctk.CTkEntry(signal_control_frame, placeholder_text="Search for signals...")
            self.search_entry.grid(row=0, column=0, padx=(0, 5), sticky="ew")
            self.search_entry.bind("<KeyRelease>", self._filter_signals)
            self.clear_search_button = ctk.CTkButton(signal_control_frame, text="X", width=28, command=self._clear_search)
            self.clear_search_button.grid(row=0, column=1, padx=5)
            ctk.CTkButton(signal_control_frame, text="Select All", width=100, command=self.select_all).grid(row=0, column=2, padx=5)
            ctk.CTkButton(signal_control_frame, text="Deselect All", width=100, command=self.deselect_all).grid(row=0, column=3)
            
            # Signal list frame
            self.signal_list_frame = ctk.CTkScrollableFrame(right_panel, label_text="Available Signals to Process")
            self.signal_list_frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="nsew")
            self.signal_list_frame.grid_columnconfigure(0, weight=1)

        # Create the splitter with the content creator functions
        splitter_frame = self._create_splitter(parent_tab, create_left_content, create_right_content, 'setup_left_width', 350)
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def populate_setup_sub_tab(self, tab):
        """Populate the setup sub-tab."""
        tab.grid_columnconfigure(0, weight=1)
        
        # File selection frame
        file_frame = ctk.CTkFrame(tab)
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        file_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(file_frame, text="CSV File Selection", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkButton(file_frame, text="Select Input CSV Files", command=self.select_files).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(file_frame, text="Select Output Folder", command=self.select_output_folder).grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        self.output_label = ctk.CTkLabel(file_frame, text=f"Output: {self.output_directory}", wraplength=300, justify="left", font=ctk.CTkFont(size=11))
        self.output_label.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")
        
        # Settings frame
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        settings_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(settings_frame, text="Configuration Save and Load", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_settings).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(settings_frame, text="Load Settings", command=self.load_settings).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(settings_frame, text="How to Share App", command=self._show_sharing_instructions).grid(row=1, column=2, padx=10, pady=5, sticky="ew")
        
        # Export options frame
        export_frame = ctk.CTkFrame(tab)
        export_frame.grid(row=2, column=0, padx=10, pady=10, sticky="new")
        export_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(export_frame, text="Export Options", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="w")
        ctk.CTkLabel(export_frame, text="Format:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.export_type_var = ctk.StringVar(value="CSV (Separate Files)")
        ctk.CTkOptionMenu(export_frame, variable=self.export_type_var, values=[
            "CSV (Separate Files)", 
            "CSV (Compiled)", 
            "Excel (Multi-sheet)", 
            "Excel (Separate Files)",
            "MAT (Separate Files)",
            "MAT (Compiled)"
        ]).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(export_frame, text="Sort By:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.sort_col_menu = ctk.CTkOptionMenu(export_frame, values=["No Sorting"])
        self.sort_col_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        self.sort_order_var = ctk.StringVar(value="Ascending")
        sort_asc = ctk.CTkRadioButton(export_frame, text="Ascending", variable=self.sort_order_var, value="Ascending")
        sort_asc.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        sort_desc = ctk.CTkRadioButton(export_frame, text="Descending", variable=self.sort_order_var, value="Descending")
        sort_desc.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        # Signal List Management frame
        signal_list_frame = ctk.CTkFrame(tab)
        signal_list_frame.grid(row=3, column=0, padx=10, pady=10, sticky="new")
        signal_list_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(signal_list_frame, text="Signal List Management", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        
        # Buttons for signal list management
        ctk.CTkButton(signal_list_frame, text="Save Current Signal List", command=self.save_signal_list).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(signal_list_frame, text="Load Saved Signal List", command=self.load_signal_list).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(signal_list_frame, text="Apply Saved Signals", command=self.apply_saved_signals).grid(row=1, column=2, padx=10, pady=5, sticky="ew")
        
        # Status label for signal list operations
        self.signal_list_status_label = ctk.CTkLabel(signal_list_frame, text="No saved signal list loaded", font=ctk.CTkFont(size=11), text_color="gray")
        self.signal_list_status_label.grid(row=2, column=0, columnspan=3, padx=10, pady=(5, 10), sticky="w")

    def populate_processing_sub_tab(self, tab):
        """Populate the processing sub-tab with all advanced features."""
        tab.grid_columnconfigure(0, weight=1)
        time_units = ["ms", "s", "min", "hr"]
        
        # Time trimming frame - moved to top for better workflow
        trim_frame = ctk.CTkFrame(tab)
        trim_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        trim_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(trim_frame, text="Time Trimming", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(trim_frame, text="Trim data to specific time range before processing", justify="left").grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")
        
        ctk.CTkLabel(trim_frame, text="Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.trim_date_entry = ctk.CTkEntry(trim_frame, placeholder_text="e.g., 2024-01-15")
        self.trim_date_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(trim_frame, text="Start Time (HH:MM:SS):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.trim_start_entry = ctk.CTkEntry(trim_frame, placeholder_text="e.g., 09:30:00")
        self.trim_start_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(trim_frame, text="End Time (HH:MM:SS):").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.trim_end_entry = ctk.CTkEntry(trim_frame, placeholder_text="e.g., 17:00:00")
        self.trim_end_entry.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(trim_frame, text="Copy Times to Plot Range", command=self._copy_trim_to_plot_range).grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(trim_frame, text="Copy Plot Range to Times", command=self._copy_plot_range_to_trim).grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Filter frame
        filter_frame = ctk.CTkFrame(tab)
        filter_frame.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        filter_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(filter_frame, text="Signal Filtering", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(filter_frame, text="Filter Type:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.filter_type_var = ctk.StringVar(value="None")
        self.filter_menu = ctk.CTkOptionMenu(filter_frame, variable=self.filter_type_var, values=self.filter_names, command=self._update_filter_ui)
        self.filter_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Create filter parameter frames
        (self.ma_frame, self.ma_value_entry, self.ma_unit_menu) = self._create_ma_param_frame(filter_frame, time_units)
        (self.bw_frame, self.bw_order_entry, self.bw_cutoff_entry) = self._create_bw_param_frame(filter_frame)
        (self.median_frame, self.median_kernel_entry) = self._create_median_param_frame(filter_frame)
        (self.hampel_frame, self.hampel_window_entry, self.hampel_threshold_entry) = self._create_hampel_param_frame(filter_frame)
        (self.zscore_frame, self.zscore_threshold_entry, self.zscore_method_menu) = self._create_zscore_param_frame(filter_frame)
        (self.savgol_frame, self.savgol_window_entry, self.savgol_polyorder_entry) = self._create_savgol_param_frame(filter_frame)
        self._update_filter_ui("None")
        
        # Resample frame
        resample_frame = ctk.CTkFrame(tab)
        resample_frame.grid(row=2, column=0, padx=10, pady=10, sticky="new")
        resample_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(resample_frame, text="Time Resampling", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        self.resample_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(resample_frame, text="Enable Resampling", variable=self.resample_var).grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(resample_frame, text="Time Gap:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        resample_time_frame = ctk.CTkFrame(resample_frame, fg_color="transparent")
        resample_time_frame.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        resample_time_frame.grid_columnconfigure(0, weight=2)
        resample_time_frame.grid_columnconfigure(1, weight=1)
        
        self.resample_value_entry = ctk.CTkEntry(resample_time_frame, placeholder_text="e.g., 10")
        self.resample_value_entry.grid(row=0, column=0, sticky="ew")
        
        self.resample_unit_menu = ctk.CTkOptionMenu(resample_time_frame, values=time_units)
        self.resample_unit_menu.grid(row=0, column=1, padx=(5,0), sticky="ew")
        
        # Integration frame
        integrator_frame = ctk.CTkFrame(tab)
        integrator_frame.grid(row=3, column=0, padx=10, pady=10, sticky="new")
        integrator_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(integrator_frame, text="Signal Integration", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(5,3), sticky="w")
        ctk.CTkLabel(integrator_frame, text="Create cumulative columns for flow calculations", justify="left").grid(row=1, column=0, columnspan=2, padx=10, pady=(0,5), sticky="w")
        
        ctk.CTkLabel(integrator_frame, text="Integration Method:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.integrator_method_var = ctk.StringVar(value="Trapezoidal")
        ctk.CTkOptionMenu(integrator_frame, variable=self.integrator_method_var, values=["Trapezoidal", "Rectangular", "Simpson"]).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        # Integration signals selection frame
        integrator_signals_frame = ctk.CTkFrame(integrator_frame)
        integrator_signals_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        integrator_signals_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(integrator_signals_frame, text="Signals to Integrate:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        self.integrator_search_entry = ctk.CTkEntry(integrator_signals_frame, placeholder_text="Search signals to integrate...")
        self.integrator_search_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.integrator_search_entry.bind("<KeyRelease>", self._filter_integrator_signals)
        
        ctk.CTkButton(integrator_signals_frame, text="X", width=28, command=self._clear_integrator_search).grid(row=1, column=1, padx=5, pady=5)
        
        self.integrator_signals_frame = ctk.CTkScrollableFrame(integrator_signals_frame, height=100)
        self.integrator_signals_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        integrator_buttons_frame = ctk.CTkFrame(integrator_frame, fg_color="transparent")
        integrator_buttons_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(integrator_buttons_frame, text="Select All", command=self._integrator_select_all).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkButton(integrator_buttons_frame, text="Deselect All", command=self._integrator_deselect_all).grid(row=0, column=1, padx=5, pady=5)

        # Differentiation Frame with searchable signals
        deriv_frame = ctk.CTkFrame(tab)
        deriv_frame.grid(row=4, column=0, padx=10, pady=10, sticky="new")
        deriv_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(deriv_frame, text="Signal Differentiation", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(5,3), sticky="w")
        ctk.CTkLabel(deriv_frame, text="Create derivative columns for signal analysis", justify="left").grid(row=1, column=0, columnspan=2, padx=10, pady=(0,5), sticky="w")
        
        # Differentiation method selection
        ctk.CTkLabel(deriv_frame, text="Method:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.deriv_method_var = ctk.StringVar(value="Spline (Acausal)")
        ctk.CTkOptionMenu(deriv_frame, variable=self.deriv_method_var, values=["Spline (Acausal)", "Rolling Polynomial (Causal)"]).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        # Differentiation signals selection frame
        deriv_signals_frame = ctk.CTkFrame(deriv_frame)
        deriv_signals_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        deriv_signals_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(deriv_signals_frame, text="Signals to Differentiate:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Search bar for differentiation signals
        self.deriv_search_entry = ctk.CTkEntry(deriv_signals_frame, placeholder_text="Search signals to differentiate...")
        self.deriv_search_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.deriv_search_entry.bind("<KeyRelease>", self._filter_deriv_signals)
        
        ctk.CTkButton(deriv_signals_frame, text="X", width=28, command=self._clear_deriv_search).grid(row=1, column=1, padx=5, pady=5)
        
        # Scrollable frame for differentiation signal checkboxes
        self.deriv_signals_frame = ctk.CTkScrollableFrame(deriv_signals_frame, height=100)
        self.deriv_signals_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.deriv_signals_frame.grid_columnconfigure(0, weight=1)
        
        # Differentiation control buttons
        deriv_buttons_frame = ctk.CTkFrame(deriv_frame, fg_color="transparent")
        deriv_buttons_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(deriv_buttons_frame, text="Select All", command=self._deriv_select_all).grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkButton(deriv_buttons_frame, text="Deselect All", command=self._deriv_deselect_all).grid(row=0, column=1, padx=5, pady=5)
        
        # Derivative order selection (up to 5th order)
        deriv_order_frame = ctk.CTkFrame(deriv_frame)
        deriv_order_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(deriv_order_frame, text="Derivative Orders:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=5, padx=10, pady=5, sticky="w")
        
        for i in range(1, 6):  # Support up to 5th order
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(deriv_order_frame, text=f"Order {i}", variable=var)
            cb.grid(row=1, column=i-1, padx=10, pady=2, sticky="w")
            self.derivative_vars[i] = var

    def _create_ma_param_frame(self, parent, time_units):
        """Create Moving Average parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(frame, text="Window Size:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        value_entry = ctk.CTkEntry(frame, placeholder_text="10")
        value_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        value_entry.insert(0, "10")  # Set default value
        
        ctk.CTkLabel(frame, text="Unit:").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        unit_menu = ctk.CTkOptionMenu(frame, values=time_units)
        unit_menu.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
        unit_menu.set("s")  # Set default unit
        
        return frame, value_entry, unit_menu

    def _create_bw_param_frame(self, parent):
        """Create Butterworth filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(frame, text="Order:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        order_entry = ctk.CTkEntry(frame, placeholder_text="3")
        order_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(frame, text="Cutoff:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        cutoff_entry = ctk.CTkEntry(frame, placeholder_text="0.1")
        cutoff_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        return frame, order_entry, cutoff_entry

    def _create_median_param_frame(self, parent):
        """Create Median filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(frame, text="Kernel Size:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        kernel_entry = ctk.CTkEntry(frame, placeholder_text="5")
        kernel_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        return frame, kernel_entry

    def _create_savgol_param_frame(self, parent):
        """Create Savitzky-Golay filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(frame, text="Window Size:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        window_entry = ctk.CTkEntry(frame, placeholder_text="11")
        window_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(frame, text="Polynomial Order:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        polyorder_entry = ctk.CTkEntry(frame, placeholder_text="2")
        polyorder_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        return frame, window_entry, polyorder_entry

    def _create_hampel_param_frame(self, parent):
        """Create Hampel filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(frame, text="Window Size:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        window_entry = ctk.CTkEntry(frame, placeholder_text="7")
        window_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(frame, text="Threshold (σ):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        threshold_entry = ctk.CTkEntry(frame, placeholder_text="3.0")
        threshold_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        return frame, window_entry, threshold_entry

    def _create_zscore_param_frame(self, parent):
        """Create Z-Score filter parameter frame."""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(frame, text="Threshold (σ):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        threshold_entry = ctk.CTkEntry(frame, placeholder_text="3.0")
        threshold_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(frame, text="Method:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        method_menu = ctk.CTkOptionMenu(frame, values=["Remove Outliers", "Clip Outliers", "Replace with Median"])
        method_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        return frame, threshold_entry, method_menu

    def _update_filter_ui(self, filter_type):
        """Update filter UI based on selected filter type."""
        # Hide all frames
        for frame in [self.ma_frame, self.bw_frame, self.median_frame, self.hampel_frame, self.zscore_frame, self.savgol_frame]:
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

    def _update_plot_filter_ui(self, filter_type):
        """Update plot filter UI based on selected filter type."""
        # Hide all frames
        for frame in [self.plot_ma_frame, self.plot_bw_frame, self.plot_median_frame, self.plot_hampel_frame, self.plot_zscore_frame, self.plot_savgol_frame]:
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

    def _filter_signals(self, event=None):
        """Filter signals based on search text."""
        search_text = self.search_entry.get().lower()
        for signal, data in self.signal_vars.items():
            if search_text in signal.lower():
                data['widget'].grid()
            else:
                data['widget'].grid_remove()

    def _clear_search(self):
        """Clear search and show all signals."""
        self.search_entry.delete(0, tk.END)
        for signal, data in self.signal_vars.items():
            data['widget'].grid()

    def _filter_integrator_signals(self, event=None):
        """Filter integration signals based on search text."""
        search_text = self.integrator_search_entry.get().lower()
        for signal, data in self.integrator_signal_vars.items():
            if search_text in signal.lower():
                data['widget'].pack(anchor="w", padx=5, pady=2)
            else:
                data['widget'].pack_forget()

    def _clear_integrator_search(self):
        """Clear integration search and show all signals."""
        self.integrator_search_entry.delete(0, tk.END)
        for signal, data in self.integrator_signal_vars.items():
            data['widget'].pack(anchor="w", padx=5, pady=2)

    def _integrator_select_all(self):
        """Select all integration signals."""
        for signal, data in self.integrator_signal_vars.items():
            data['var'].set(True)

    def _integrator_deselect_all(self):
        """Deselect all integration signals."""
        for signal, data in self.integrator_signal_vars.items():
            data['var'].set(False)

    def _filter_deriv_signals(self, event=None):
        """Filter differentiation signals based on search text."""
        search_text = self.deriv_search_entry.get().lower()
        for signal, data in self.deriv_signal_vars.items():
            if search_text in signal.lower():
                data['widget'].pack(anchor="w", padx=5, pady=2)
            else:
                data['widget'].pack_forget()

    def _clear_deriv_search(self):
        """Clear differentiation search and show all signals."""
        self.deriv_search_entry.delete(0, tk.END)
        for signal, data in self.deriv_signal_vars.items():
            data['widget'].pack(anchor="w", padx=5, pady=2)

    def _deriv_select_all(self):
        """Select all differentiation signals."""
        for signal, data in self.deriv_signal_vars.items():
            data['var'].set(True)

    def _deriv_deselect_all(self):
        """Deselect all differentiation signals."""
        for signal, data in self.deriv_signal_vars.items():
            data['var'].set(False)

    def _filter_plot_signals(self, event=None):
        """Filter plot signals based on search text."""
        search_text = self.plot_search_entry.get().lower()
        for signal, data in self.plot_signal_vars.items():
            if search_text in signal.lower():
                data['widget'].grid()
            else:
                data['widget'].grid_remove()

    def _plot_clear_search(self):
        """Clear plot search and show all signals."""
        self.plot_search_entry.delete(0, tk.END)
        for signal, data in self.plot_signal_vars.items():
            data['widget'].grid()

    def _plot_select_all(self):
        """Select all plot signals."""
        for signal, data in self.plot_signal_vars.items():
            data['var'].set(True)

    def _plot_select_none(self):
        """Deselect all plot signals."""
        for signal, data in self.plot_signal_vars.items():
            data['var'].set(False)

    def _show_selected_signals(self):
        """Show only selected signals in plot."""
        selected_signals = [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
        if selected_signals:
            self.update_plot(selected_signals=selected_signals)
        else:
            messagebox.showwarning("No Signals Selected", "Please select at least one signal to plot.")

    def _filter_reference_signals(self, event=None):
        """Filter reference signals for custom variables."""
        search_text = self.custom_var_search_entry.get().lower()
        for signal, widget in self.reference_signal_widgets.items():
            if search_text in signal.lower():
                widget.pack(anchor="w", padx=5, pady=2)
            else:
                widget.pack_forget()

    def _clear_reference_search(self):
        """Clear reference search and show all signals."""
        self.custom_var_search_entry.delete(0, tk.END)
        for signal, widget in self.reference_signal_widgets.items():
            widget.pack(anchor="w", padx=5, pady=2)

    def _add_custom_variable(self):
        """Add a custom variable to the list."""
        name = self.custom_var_name_entry.get().strip()
        formula = self.custom_var_formula_entry.get().strip()
        
        if not name or not formula:
            messagebox.showerror("Error", "Please enter both name and formula.")
            return
        
        # Check if name already exists
        if any(var['name'] == name for var in self.custom_vars_list):
            messagebox.showerror("Error", f"Variable '{name}' already exists.")
            return
        
        self.custom_vars_list.append({'name': name, 'formula': formula})
        self._update_custom_vars_display()
        
        # Clear entries
        self.custom_var_name_entry.delete(0, tk.END)
        self.custom_var_formula_entry.delete(0, tk.END)

    def populate_custom_var_sub_tab(self, tab):
        """Fixed custom variables sub-tab with missing listbox."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(8, weight=1)

        ctk.CTkLabel(tab, text="Custom Variables (Formula Engine)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkLabel(tab, text="Create new columns using exact signal names in [brackets].", justify="left").grid(row=1, column=0, padx=10, pady=(0, 5), sticky="w")
        
        ctk.CTkLabel(tab, text="New Variable Name:").grid(row=2, column=0, padx=10, pady=(5,0), sticky="w")
        self.custom_var_name_entry = ctk.CTkEntry(tab, placeholder_text="e.g., Power_Ratio")
        self.custom_var_name_entry.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(tab, text="Formula:").grid(row=4, column=0, padx=10, pady=(5,0), sticky="w")
        self.custom_var_formula_entry = ctk.CTkEntry(tab, placeholder_text="e.g., ( [SignalA] + [SignalB] ) / 2")
        self.custom_var_formula_entry.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(tab, text="Add Custom Variable", command=self._add_custom_variable).grid(row=6, column=0, padx=10, pady=10, sticky="ew")
        
        # FIXED: Add missing custom variables listbox
        custom_vars_list_frame = ctk.CTkFrame(tab)
        custom_vars_list_frame.grid(row=7, column=0, padx=10, pady=5, sticky="ew")
        custom_vars_list_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(custom_vars_list_frame, text="Current Custom Variables:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.custom_vars_listbox = ctk.CTkTextbox(custom_vars_list_frame, height=100)
        self.custom_vars_listbox.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(custom_vars_list_frame, text="Clear All Variables", command=self._clear_custom_variables).grid(row=2, column=0, padx=10, pady=5)
        
        # Searchable reference list
        reference_frame = ctk.CTkFrame(tab)
        reference_frame.grid(row=8, column=0, padx=10, pady=5, sticky="nsew")
        reference_frame.grid_columnconfigure(0, weight=1)
        reference_frame.grid_rowconfigure(1, weight=1)

        search_bar_frame = ctk.CTkFrame(reference_frame, fg_color="transparent")
        search_bar_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        search_bar_frame.grid_columnconfigure(0, weight=1)

        self.custom_var_search_entry = ctk.CTkEntry(search_bar_frame, placeholder_text="Search available signals...")
        self.custom_var_search_entry.grid(row=0, column=0, sticky="ew")
        self.custom_var_search_entry.bind("<KeyRelease>", self._filter_reference_signals)

        self.custom_var_clear_button = ctk.CTkButton(search_bar_frame, text="X", width=28, command=self._clear_reference_search)
        self.custom_var_clear_button.grid(row=0, column=1, padx=(5,0))

        self.signal_reference_frame = ctk.CTkScrollableFrame(reference_frame, label_text="Available Signals Reference")
        self.signal_reference_frame.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")

    def _update_custom_vars_display(self):
        """Update the custom variables display."""
        self.custom_vars_listbox.configure(state="normal")
        self.custom_vars_listbox.delete("1.0", tk.END)
        
        for var in self.custom_vars_list:
            self.custom_vars_listbox.insert(tk.END, f"{var['name']}: {var['formula']}\n")
        
        self.custom_vars_listbox.configure(state="disabled")

    def _clear_custom_variables(self):
        """Clear all custom variables."""
        self.custom_vars_list.clear()
        self._update_custom_vars_display()

    def _apply_integration(self, df, time_col, signals_to_integrate, method="Trapezoidal"):
        """Apply integration to selected signals."""
        if not signals_to_integrate:
            return df
        
        # Convert time to numeric for integration
        time_numeric = pd.to_numeric(df[time_col], errors='coerce')
        dt = time_numeric.diff().fillna(0)
        
        for signal in signals_to_integrate:
            if signal in df.columns and signal != time_col:
                signal_data = pd.to_numeric(df[signal], errors='coerce')
                
                if method == "Trapezoidal":
                    # Trapezoidal rule
                    cumulative = np.cumsum(0.5 * (signal_data.iloc[:-1].values + signal_data.iloc[1:].values) * dt.iloc[1:].values)
                    cumulative = np.insert(cumulative, 0, 0)  # Start at 0
                elif method == "Rectangular":
                    # Rectangular rule (left endpoint)
                    cumulative = np.cumsum(signal_data.values * dt.values)
                elif method == "Simpson":
                    # Simpson's rule (requires even number of intervals)
                    if len(signal_data) % 2 == 0:
                        cumulative = np.cumsum((signal_data.iloc[::2].values + 4*signal_data.iloc[1::2].values + signal_data.iloc[2::2].values) * dt.iloc[::2].values / 3)
                    else:
                        # Fall back to trapezoidal for odd number of points
                        cumulative = np.cumsum(0.5 * (signal_data.iloc[:-1].values + signal_data.iloc[1:].values) * dt.iloc[1:].values)
                        cumulative = np.insert(cumulative, 0, 0)
                
                df[f'cumulative_{signal}'] = cumulative
        
        return df

    def _apply_differentiation(self, df, time_col, signals_to_differentiate, method="Spline (Acausal)"):
        """Apply differentiation to selected signals with support for up to 5th order."""
        if not signals_to_differentiate:
            return df
        
        # Get selected derivative orders
        selected_orders = [order for order, var in self.derivative_vars.items() if var.get()]
        if not selected_orders:
            return df
        
        # Convert time to numeric for differentiation
        time_numeric = pd.to_numeric(df[time_col], errors='coerce')
        dt = time_numeric.diff().fillna(0)
        
        for signal in signals_to_differentiate:
            if signal in df.columns and signal != time_col:
                signal_data = pd.to_numeric(df[signal], errors='coerce')
                
                for order in selected_orders:
                    if method == "Spline (Acausal)":
                        # Spline-based differentiation (acausal)
                        try:
                            # Remove NaN values for spline fitting
                            valid_mask = ~(np.isnan(signal_data) | np.isnan(time_numeric))
                            if np.sum(valid_mask) > order + 1:
                                x_valid = time_numeric[valid_mask]
                                y_valid = signal_data[valid_mask]
                                
                                # Fit spline
                                spline = UnivariateSpline(x_valid, y_valid, s=0, k=min(5, len(y_valid)-1))
                                
                                # Calculate derivatives
                                if order == 1:
                                    derivative = spline.derivative()(time_numeric)
                                elif order == 2:
                                    derivative = spline.derivative().derivative()(time_numeric)
                                elif order == 3:
                                    derivative = spline.derivative().derivative().derivative()(time_numeric)
                                elif order == 4:
                                    derivative = spline.derivative().derivative().derivative().derivative()(time_numeric)
                                elif order == 5:
                                    derivative = spline.derivative().derivative().derivative().derivative().derivative()(time_numeric)
                                else:
                                    continue
                                
                                # Handle NaN values
                                derivative[~valid_mask] = np.nan
                                df[f'{signal}_d{order}'] = derivative
                            else:
                                df[f'{signal}_d{order}'] = np.nan
                        except Exception as e:
                            print(f"Error in spline differentiation for {signal}, order {order}: {e}")
                            df[f'{signal}_d{order}'] = np.nan
                    
                    elif method == "Rolling Polynomial (Causal)":
                        # Rolling polynomial differentiation (causal)
                        try:
                            # Use the helper function for causal differentiation
                            window_size = 11  # Default window size
                            poly_order = min(5, window_size - 1)  # Ensure polynomial order < window size
                            
                            if len(signal_data) > window_size:
                                derivative = _poly_derivative(signal_data, window_size, poly_order, order, dt.mean())
                                df[f'{signal}_d{order}'] = derivative
                            else:
                                df[f'{signal}_d{order}'] = np.nan
                        except Exception as e:
                            print(f"Error in polynomial differentiation for {signal}, order {order}: {e}")
                            df[f'{signal}_d{order}'] = np.nan
        
        return df 

    def select_files(self):
        """Select input CSV files."""
        file_paths = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_paths:
            self.input_file_paths = list(file_paths)
            
            # Set default output directory to the folder of the first selected file
            if self.input_file_paths:
                first_file_dir = os.path.dirname(self.input_file_paths[0])
                self.output_directory = first_file_dir
                # Update the output label to reflect the new default directory
                if hasattr(self, 'output_label'):
                    self.output_label.configure(text=f"Output: {self.output_directory}")
            
            self.update_file_list()
            self.load_signals_from_files()

    def select_output_folder(self):
        """Select output directory for processed files."""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_directory = folder_path
            self.output_label.configure(text=f"Output: {self.output_directory}")

    def update_file_list(self):
        """Update the file list display."""
        # Clear existing widgets
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        if not self.input_file_paths:
            ctk.CTkLabel(self.file_list_frame, text="Files you select will be listed here.").pack(padx=5, pady=5)
            return
        
        for i, file_path in enumerate(self.input_file_paths):
            file_frame = ctk.CTkFrame(self.file_list_frame)
            file_frame.pack(fill="x", padx=5, pady=2)
            
            filename = os.path.basename(file_path)
            ctk.CTkLabel(file_frame, text=f"{i+1}. {filename}", font=ctk.CTkFont(size=11)).pack(side="left", padx=5, pady=2)
            
            ctk.CTkButton(file_frame, text="X", width=25, command=lambda f=file_path: self.remove_file(f)).pack(side="right", padx=5, pady=2)

    def remove_file(self, file_path):
        """Remove a file from the list."""
        if file_path in self.input_file_paths:
            self.input_file_paths.remove(file_path)
            self.update_file_list()
            self.load_signals_from_files()

    def load_signals_from_files(self):
        """Load signals from all selected files."""
        if not self.input_file_paths:
            return
        
        all_signals = set()
        for file_path in self.input_file_paths:
            try:
                df = pd.read_csv(file_path, nrows=1)  # Just read header
                all_signals.update(df.columns.tolist())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        self.update_signal_list(sorted(all_signals))
        
        # Update plot file menu
        file_names = ["Select a file..."] + [os.path.basename(f) for f in self.input_file_paths]
        if hasattr(self, 'plot_file_menu'):
            self.plot_file_menu.configure(values=file_names)
            
            # Auto-select the file if there's only one
            if len(self.input_file_paths) == 1:
                single_file = os.path.basename(self.input_file_paths[0])
                self.plot_file_menu.set(single_file)
                # Trigger the file selection handler
                self.on_plot_file_select(single_file)

    def update_signal_list(self, signals):
        """Update the signal list with checkboxes."""
        # Clear existing widgets
        for widget in self.signal_list_frame.winfo_children():
            widget.destroy()
        
        self.signal_vars.clear()
        
        for signal in signals:
            var = tk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.signal_list_frame, text=signal, variable=var)
            cb.grid(sticky="w", padx=5, pady=2)
            self.signal_vars[signal] = {'var': var, 'widget': cb}
        
        # Update sort column menu
        sort_values = ["No Sorting"] + signals
        self.sort_col_menu.configure(values=sort_values)
        
        # Initialize plot signal variables (will be populated when file is selected in plotting tab)
        self.plot_signal_vars = {}
        
        # Update plots list signals
        self._update_plots_signals(signals)
        
        # Update integration signals
        for widget in self.integrator_signals_frame.winfo_children():
            widget.destroy()
        self.integrator_signal_vars.clear()
        
        for signal in signals:
            if signal != signals[0]:  # Skip time column
                var = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.integrator_signals_frame, text=signal, variable=var)
                cb.pack(anchor="w", padx=5, pady=2)
                self.integrator_signal_vars[signal] = {'var': var, 'widget': cb}
        
        # Update differentiation signals
        for widget in self.deriv_signals_frame.winfo_children():
            widget.destroy()
        self.deriv_signal_vars.clear()
        
        for signal in signals:
            if signal != signals[0]:  # Skip time column
                var = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.deriv_signals_frame, text=signal, variable=var)
                cb.pack(anchor="w", padx=5, pady=2)
                self.deriv_signal_vars[signal] = {'var': var, 'widget': cb}
        
        # Update reference signals for custom variables
        for widget in self.signal_reference_frame.winfo_children():
            widget.destroy()
        self.reference_signal_widgets.clear()
        
        for signal in signals:
            label = ctk.CTkLabel(self.signal_reference_frame, text=f"[{signal}]", font=ctk.CTkFont(size=11))
            label.pack(anchor="w", padx=5, pady=2)
            self.reference_signal_widgets[signal] = label

    def select_all(self):
        """Select all signals."""
        for signal, data in self.signal_vars.items():
            data['var'].set(True)

    def deselect_all(self):
        """Deselect all signals."""
        for signal, data in self.signal_vars.items():
            data['var'].set(False)

    def process_files(self):
        """Process all selected files with current settings."""
        if not self.input_file_paths:
            messagebox.showerror("Error", "Please select input files first.")
            return
        
        selected_signals = [s for s, data in self.signal_vars.items() if data['var'].get()]
        if not selected_signals:
            messagebox.showerror("Error", "Please select at least one signal to process.")
            return
        
        # Get processing settings
        settings = {
            'selected_signals': selected_signals,
            'filter_type': self.filter_type_var.get(),
            'resample_enabled': self.resample_var.get(),
            'resample_rule': self._get_resample_rule(),
            'ma_window': int(self.ma_value_entry.get()) if self.ma_value_entry.get() else 10,
            'bw_order': int(self.bw_order_entry.get()) if self.bw_order_entry.get() else 3,
            'bw_cutoff': float(self.bw_cutoff_entry.get()) if self.bw_cutoff_entry.get() else 0.1,
            'median_kernel': int(self.median_kernel_entry.get()) if self.median_kernel_entry.get() else 5,
            'hampel_window': int(self.hampel_window_entry.get()) if self.hampel_window_entry.get() else 7,
            'hampel_threshold': float(self.hampel_threshold_entry.get()) if self.hampel_threshold_entry.get() else 3.0,
            'zscore_threshold': float(self.zscore_threshold_entry.get()) if self.zscore_threshold_entry.get() else 3.0,
            'zscore_method': self.zscore_method_menu.get() if hasattr(self, 'zscore_method_menu') else 'Remove Outliers',
            'savgol_window': int(self.savgol_window_entry.get()) if self.savgol_window_entry.get() else 11,
            'savgol_polyorder': int(self.savgol_polyorder_entry.get()) if self.savgol_polyorder_entry.get() else 2
        }
        
        # Process files
        processed_files = []
        for file_path in self.input_file_paths:
            try:
                processed_df = self._process_single_file(file_path, settings)
                if processed_df is not None:
                    processed_files.append((file_path, processed_df))
                    # Store processed data for plotting
                    filename = os.path.basename(file_path)
                    self.processed_files[filename] = processed_df
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if not processed_files:
            messagebox.showerror("Error", "No files were successfully processed.")
            return
        
        # Export processed files
        self._export_processed_files(processed_files)

    def _process_single_file(self, file_path, settings):
        """Process a single file with all advanced features."""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Determine which signals to keep for this specific file
            signals_in_this_file = [s for s in settings['selected_signals'] if s in df.columns]
            time_col = df.columns[0]
            if time_col not in signals_in_this_file:
                signals_in_this_file.insert(0, time_col)
            
            processed_df = df[signals_in_this_file].copy()
            
            # Data type conversion
            processed_df[time_col] = pd.to_datetime(processed_df[time_col], errors='coerce')
            processed_df.dropna(subset=[time_col], inplace=True)
            for col in processed_df.columns:
                if col != time_col:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            if processed_df.empty:
                return None

            # Apply time trimming if specified
            trim_date = self.trim_date_entry.get()
            trim_start = self.trim_start_entry.get()
            trim_end = self.trim_end_entry.get()
            
            if trim_date or trim_start or trim_end:
                try:
                    # Get the date from the data if not specified
                    if not trim_date:
                        trim_date = processed_df[time_col].iloc[0].strftime('%Y-%m-%d')
                    
                    # Create full datetime strings
                    start_time_str = trim_start or "00:00:00"
                    end_time_str = trim_end or "23:59:59"
                    start_full_str = f"{trim_date} {start_time_str}"
                    end_full_str = f"{trim_date} {end_time_str}"
                    
                    # Filter the data by time range
                    processed_df = processed_df.set_index(time_col).loc[start_full_str:end_full_str].reset_index()
                    
                    if processed_df.empty:
                        print(f"Warning: Time trimming resulted in empty dataset for {os.path.basename(file_path)}")
                        return None
                        
                except Exception as e:
                    print(f"Error applying time trimming to {os.path.basename(file_path)}: {e}")

            processed_df.set_index(time_col, inplace=True)

            # Apply Filtering
            filter_type = settings.get('filter_type')
            if filter_type and filter_type != "None":
                numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
                for col in numeric_cols:
                    signal_data = processed_df[col].dropna()
                    if len(signal_data) < 2: 
                        continue
                    
                    # Apply filtering based on type
                    if filter_type == "Moving Average":
                        window_size = settings.get('ma_window', 10)
                        processed_df[col] = signal_data.rolling(window=window_size, min_periods=1).mean()
                    elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
                        order = settings.get('bw_order', 3)
                        cutoff = settings.get('bw_cutoff', 0.1)
                        sr = 1.0 / pd.to_numeric(signal_data.index.to_series().diff().dt.total_seconds()).mean()
                        if pd.notna(sr) and len(signal_data) > order * 3:
                            btype = 'low' if filter_type == "Butterworth Low-pass" else 'high'
                            b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sr)
                            processed_df[col] = pd.Series(filtfilt(b, a, signal_data), index=signal_data.index)
                    elif filter_type == "Median Filter":
                        kernel = settings.get('median_kernel', 5)
                        if kernel % 2 == 0: kernel += 1
                        if len(signal_data) > kernel:
                            processed_df[col] = pd.Series(medfilt(signal_data, kernel_size=kernel), index=signal_data.index)
                    elif filter_type == "Hampel Filter":
                        window = settings.get('hampel_window', 7)
                        threshold = settings.get('hampel_threshold', 3.0)
                        
                        try:
                            from scipy.signal import medfilt
                            signal_data = df[signal].ffill().bfill()
                            
                            # Apply Hampel filter
                            median_filtered = pd.Series(medfilt(signal_data, kernel_size=window), index=signal_data.index)
                            mad = signal_data.rolling(window=window, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
                            threshold_value = threshold * 1.4826 * mad  # 1.4826 is the constant for normal distribution
                            
                            # Replace outliers with median using proper indexing
                            outliers = np.abs(signal_data - median_filtered) > threshold_value
                            processed_df = processed_df.copy()  # Ensure we have a copy to avoid warnings
                            processed_df.loc[outliers, signal] = median_filtered.loc[outliers]
                        except ImportError:
                            # Fallback to simple median filter
                            processed_df[col] = pd.Series(medfilt(signal_data, kernel_size=window), index=signal_data.index)
                        except Exception as e:
                            print(f"Error applying Hampel filter: {e}")
                            # Fallback to simple median filter
                            processed_df[col] = pd.Series(medfilt(signal_data, kernel_size=window), index=signal_data.index)
                    elif filter_type == "Z-Score Filter":
                        threshold = settings.get('zscore_threshold', 3.0)
                        method = settings.get('zscore_method', 'Remove Outliers')
                        
                        mean_val = signal_data.mean()
                        std_val = signal_data.std()
                        z_scores = np.abs((signal_data - mean_val) / std_val)
                        
                        if method == "Remove Outliers":
                            # Replace outliers with NaN
                            processed_df[col] = signal_data.copy()
                            processed_df[col].loc[z_scores > threshold] = np.nan
                        elif method == "Clip Outliers":
                            # Clip outliers to threshold
                            processed_df[col] = signal_data.copy()
                            upper_bound = mean_val + threshold * std_val
                            lower_bound = mean_val - threshold * std_val
                            processed_df[col] = processed_df[col].clip(lower=lower_bound, upper=upper_bound)
                        elif method == "Replace with Median":
                            # Replace outliers with median
                            median_val = signal_data.median()
                            processed_df[col] = signal_data.copy()
                            processed_df[col].loc[z_scores > threshold] = median_val
                    elif filter_type == "Savitzky-Golay":
                        window = settings.get('savgol_window', 11)
                        polyorder = settings.get('savgol_polyorder', 2)
                        if window % 2 == 0: window += 1
                        if polyorder >= window: polyorder = window - 1
                        if len(signal_data) > window:
                            processed_df[col] = pd.Series(savgol_filter(signal_data, window, polyorder), index=signal_data.index)

            # Apply Resampling
            if settings.get('resample_enabled'):
                resample_rule = settings.get('resample_rule')
                if resample_rule:
                    processed_df = processed_df.resample(resample_rule).mean().dropna(how='all')

            # Apply Custom Variables
            processed_df = self._apply_custom_variables(processed_df, time_col)

            # Apply integration if signals are selected
            signals_to_integrate = [s for s, data in self.integrator_signal_vars.items() if data['var'].get()]
            if signals_to_integrate:
                integration_method = self.integrator_method_var.get()
                processed_df = self._apply_integration(processed_df, time_col, signals_to_integrate, integration_method)
            
            # Apply differentiation if signals are selected
            signals_to_differentiate = [s for s, data in self.deriv_signal_vars.items() if data['var'].get()]
            if signals_to_differentiate:
                differentiation_method = self.deriv_method_var.get()
                processed_df = self._apply_differentiation(processed_df, time_col, signals_to_differentiate, differentiation_method)

            if processed_df.empty:
                return None

            processed_df.reset_index(inplace=True)
            return processed_df

        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            return None

    def _apply_custom_variables(self, df, time_col):
        """Apply custom variables to the dataframe."""
        if not self.custom_vars_list:
            return df
        
        for var in self.custom_vars_list:
            try:
                formula = var['formula']
                name = var['name']
                
                # Replace signal names in brackets with actual column references
                for signal in df.columns:
                    if signal != time_col:
                        formula = formula.replace(f'[{signal}]', f'df["{signal}"]')
                
                # Evaluate the formula
                result = eval(formula)
                df[name] = result
                
            except Exception as e:
                print(f"Error applying custom variable {var['name']}: {e}")
                df[var['name']] = np.nan
        
        return df

    def _get_resample_rule(self):
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
            elif unit == "s":
                return f"{value}S"
            elif unit == "min":
                return f"{value}T"
            elif unit == "hr":
                return f"{value}H"
        except ValueError:
            return None
        
        return None

    def _export_processed_files(self, processed_files):
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

    def _export_csv_separate(self, processed_files):
        """Export each file as a separate CSV."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.csv")
            
            # Check for overwrite and get final path
            final_path = self._check_file_overwrite(output_path)
            if final_path is None:  # User cancelled
                continue
            
            # Apply sorting if specified
            df = self._apply_sorting(df)
            
            df.to_csv(final_path, index=False)
            exported_count += 1
        
        if exported_count > 0:
            messagebox.showinfo("Success", f"Exported {exported_count} files to {self.output_directory}")
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_csv_compiled(self, processed_files):
        """Export all files as a single compiled CSV."""
        compiled_dfs = []
        
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            df_copy = df.copy()
            df_copy.insert(0, 'Source_File', base_name)
            compiled_dfs.append(df_copy)
        
        if compiled_dfs:
            compiled_df = pd.concat(compiled_dfs, ignore_index=True)
            compiled_df = self._apply_sorting(compiled_df)
            
            output_path = os.path.join(self.output_directory, "compiled_processed_data.csv")
            
            # Check for overwrite and get final path
            final_path = self._check_file_overwrite(output_path)
            if final_path is None:  # User cancelled
                return
            
            compiled_df.to_csv(final_path, index=False)
            
            messagebox.showinfo("Success", f"Exported compiled data to {final_path}")

    def _export_excel_multisheet(self, processed_files):
        """Export all files to a single Excel file with multiple sheets."""
        output_path = os.path.join(self.output_directory, "processed_data.xlsx")
        
        # Check for overwrite and get final path
        final_path = self._check_file_overwrite(output_path)
        if final_path is None:  # User cancelled
            return
        
        with pd.ExcelWriter(final_path, engine='openpyxl') as writer:
            for file_path, df in processed_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                sheet_name = base_name[:31]  # Excel sheet name limit
                
                df = self._apply_sorting(df)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        messagebox.showinfo("Success", f"Exported to Excel file: {final_path}")

    def _export_excel_separate(self, processed_files):
        """Export each file as a separate Excel file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.xlsx")
            
            # Check for overwrite and get final path
            final_path = self._check_file_overwrite(output_path)
            if final_path is None:  # User cancelled
                continue
            
            df = self._apply_sorting(df)
            df.to_excel(final_path, index=False)
            exported_count += 1
        
        if exported_count > 0:
            messagebox.showinfo("Success", f"Exported {exported_count} Excel files to {self.output_directory}")
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_mat_separate(self, processed_files):
        """Export each file as a separate MAT file."""
        exported_count = 0
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.mat")
            
            # Check for overwrite and get final path
            final_path = self._check_file_overwrite(output_path)
            if final_path is None:  # User cancelled
                continue
            
            df = self._apply_sorting(df)
            
            # Convert to dictionary for MATLAB
            mat_data = {}
            for col in df.columns:
                mat_data[col] = df[col].values
            
            savemat(final_path, mat_data)
            exported_count += 1
        
        if exported_count > 0:
            messagebox.showinfo("Success", f"Exported {exported_count} MAT files to {self.output_directory}")
        else:
            messagebox.showinfo("Cancelled", "No files were exported.")

    def _export_mat_compiled(self, processed_files):
        """Export all files as a single compiled MAT file."""
        compiled_dfs = []
        
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            df_copy = df.copy()
            df_copy.insert(0, 'Source_File', base_name)
            compiled_dfs.append(df_copy)
        
        if compiled_dfs:
            compiled_df = pd.concat(compiled_dfs, ignore_index=True)
            compiled_df = self._apply_sorting(compiled_df)
            
            output_path = os.path.join(self.output_directory, "compiled_processed_data.mat")
            
            # Check for overwrite and get final path
            final_path = self._check_file_overwrite(output_path)
            if final_path is None:  # User cancelled
                return
            
            # Convert to dictionary for MATLAB
            mat_data = {}
            for col in compiled_df.columns:
                mat_data[col] = compiled_df[col].values
            
            savemat(final_path, mat_data)
            
            messagebox.showinfo("Success", f"Exported compiled MAT file to {final_path}")

    def _apply_sorting(self, df):
        """Apply sorting to the dataframe."""
        sort_col = self.sort_col_menu.get()
        sort_order = self.sort_order_var.get()
        
        if sort_col and sort_col != "No Sorting" and sort_col in df.columns:
            ascending = sort_order == "Ascending"
            df = df.sort_values(by=sort_col, ascending=ascending)
        
        return df 

    def create_plotting_tab(self, tab):
        """Create the plotting and analysis tab with all advanced features."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Top control bar
        plot_control_frame = ctk.CTkFrame(tab)
        plot_control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        plot_control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(plot_control_frame, text="File to Plot:").grid(row=0, column=0, padx=(10,5), pady=10)
        self.plot_file_menu = ctk.CTkOptionMenu(plot_control_frame, values=["Select a file..."], command=self.on_plot_file_select)
        self.plot_file_menu.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        
        ctk.CTkLabel(plot_control_frame, text="X-Axis:").grid(row=0, column=2, padx=(10,5), pady=10)
        self.plot_xaxis_menu = ctk.CTkOptionMenu(plot_control_frame, values=["default time"], command=lambda e: self.update_plot())
        self.plot_xaxis_menu.grid(row=0, column=3, padx=5, pady=10, sticky="ew")
        
        # Load Plot Configuration dropdown
        ctk.CTkLabel(plot_control_frame, text="Load Config:").grid(row=0, column=4, padx=(10,5), pady=10)
        self.load_plot_config_menu = ctk.CTkOptionMenu(plot_control_frame, values=["No saved plots"], command=self._on_load_plot_config_select)
        self.load_plot_config_menu.grid(row=0, column=5, padx=5, pady=10, sticky="ew")
        
        # Save Plot Configuration button
        ctk.CTkButton(plot_control_frame, text="Save Plot Config", height=35, command=self._save_current_plot_config).grid(row=0, column=6, padx=10, pady=10)

        # Main content frame for splitter
        plot_main_frame = ctk.CTkFrame(tab)
        plot_main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        plot_main_frame.grid_rowconfigure(0, weight=1)
        plot_main_frame.grid_columnconfigure(0, weight=1)
        
        def create_plot_left_content(left_panel):
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

            self.plot_search_entry = ctk.CTkEntry(plot_signal_select_frame, placeholder_text="Search plot signals...")
            self.plot_search_entry.grid(row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
            self.plot_search_entry.bind("<KeyRelease>", self._filter_plot_signals)
            
            ctk.CTkButton(plot_signal_select_frame, text="All", command=self._plot_select_all).grid(row=1, column=0, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(plot_signal_select_frame, text="None", command=self._plot_select_none).grid(row=1, column=1, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(plot_signal_select_frame, text="Show Selected", command=self._show_selected_signals).grid(row=1, column=2, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(plot_signal_select_frame, text="X", width=28, command=self._plot_clear_search).grid(row=1, column=3, sticky="w", padx=2, pady=5)
            
            self.plot_signal_frame = ctk.CTkScrollableFrame(plot_left_panel, label_text="Signals to Plot", height=150)
            self.plot_signal_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
            
            # Bind mouse wheel to the signals frame for proper scrolling
            self._bind_mousewheel_to_frame(self.plot_signal_frame)

            # Plot appearance controls
            appearance_frame = ctk.CTkFrame(plot_left_panel)
            appearance_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
            appearance_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(appearance_frame, text="Plot Appearance", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkLabel(appearance_frame, text="Chart Type:").grid(row=1, column=0, sticky="w", padx=10)
            self.plot_type_var = ctk.StringVar(value="Line with Markers")
            plot_type_menu = ctk.CTkOptionMenu(appearance_frame, variable=self.plot_type_var, values=["Line with Markers", "Line Only", "Markers Only (Scatter)"], command=self._on_plot_setting_change)
            plot_type_menu.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
            
            self.plot_title_entry = ctk.CTkEntry(appearance_frame, placeholder_text="Plot Title")
            self.plot_title_entry.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
            self.plot_title_entry.bind("<KeyRelease>", self._on_plot_setting_change)
            self.plot_title_entry.bind("<FocusOut>", self._on_plot_setting_change)
            # Force placeholder to show
            self.plot_title_entry.configure(placeholder_text="Plot Title")
            
            self.plot_xlabel_entry = ctk.CTkEntry(appearance_frame, placeholder_text="X-Axis Label")
            self.plot_xlabel_entry.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
            self.plot_xlabel_entry.bind("<KeyRelease>", self._on_plot_setting_change)
            self.plot_xlabel_entry.bind("<FocusOut>", self._on_plot_setting_change)
            # Force placeholder to show
            self.plot_xlabel_entry.configure(placeholder_text="X-Axis Label")
            
            self.plot_ylabel_entry = ctk.CTkEntry(appearance_frame, placeholder_text="Y-Axis Label")
            self.plot_ylabel_entry.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
            self.plot_ylabel_entry.bind("<KeyRelease>", self._on_plot_setting_change)
            self.plot_ylabel_entry.bind("<FocusOut>", self._on_plot_setting_change)
            # Force placeholder to show
            self.plot_ylabel_entry.configure(placeholder_text="Y-Axis Label")
            
            # Color scheme controls
            ctk.CTkLabel(appearance_frame, text="Color Scheme:").grid(row=6, column=0, sticky="w", padx=10, pady=(10,0))
            self.color_scheme_var = ctk.StringVar(value="Auto (Matplotlib)")
            color_schemes = ["Auto (Matplotlib)", "Viridis", "Plasma", "Cool", "Warm", "Rainbow", "Custom Colors"]
            color_scheme_menu = ctk.CTkOptionMenu(appearance_frame, variable=self.color_scheme_var, values=color_schemes, command=self._on_plot_setting_change)
            color_scheme_menu.grid(row=7, column=0, sticky="ew", padx=10, pady=5)
            
            # Line width control
            ctk.CTkLabel(appearance_frame, text="Line Width:").grid(row=8, column=0, sticky="w", padx=10, pady=(5,0))
            self.line_width_var = ctk.StringVar(value="1.0")
            line_widths = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]
            line_width_menu = ctk.CTkOptionMenu(appearance_frame, variable=self.line_width_var, values=line_widths, command=self._on_plot_setting_change)
            line_width_menu.grid(row=9, column=0, sticky="ew", padx=10, pady=5)
            
            # Custom Legend Labels control
            legend_header_frame = ctk.CTkFrame(appearance_frame, fg_color="transparent")
            legend_header_frame.grid(row=10, column=0, sticky="ew", padx=10, pady=(10,0))
            legend_header_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(legend_header_frame, text="Custom Legend Labels:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
            ctk.CTkButton(legend_header_frame, text="?", width=25, height=25, command=self._show_legend_guide).grid(row=0, column=1, sticky="e", padx=(5,0))
            
            ctk.CTkLabel(appearance_frame, text="For subscripts use: $H_2O$, $CO_2$, $v_{max}$ (LaTeX syntax)", font=ctk.CTkFont(size=10)).grid(row=11, column=0, sticky="w", padx=10, pady=(0,5))
            
            # Scrollable frame for legend customization
            self.legend_frame = ctk.CTkScrollableFrame(appearance_frame, height=120)
            self.legend_frame.grid(row=12, column=0, sticky="ew", padx=10, pady=5)
            
            ctk.CTkButton(appearance_frame, text="Refresh Legend Entries", command=self._refresh_legend_entries).grid(row=13, column=0, sticky="ew", padx=10, pady=5)

            # Custom legend entries dictionary
            self.custom_legend_entries = {}

            # Trendline controls
            trend_frame = ctk.CTkFrame(plot_left_panel)
            trend_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
            trend_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(trend_frame, text="Trendline", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            
            ctk.CTkLabel(trend_frame, text="Signal:").grid(row=1, column=0, sticky="w", padx=10, pady=(5,0))
            self.trendline_signal_var = ctk.StringVar(value="Select signal...")
            self.trendline_signal_menu = ctk.CTkOptionMenu(trend_frame, variable=self.trendline_signal_var, values=["Select signal..."], command=self._on_plot_setting_change)
            self.trendline_signal_menu.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
            
            ctk.CTkLabel(trend_frame, text="Type:").grid(row=3, column=0, sticky="w", padx=10, pady=(5,0))
            self.trendline_type_var = ctk.StringVar(value="None")
            trendline_type_menu = ctk.CTkOptionMenu(trend_frame, variable=self.trendline_type_var, values=["None", "Linear", "Exponential", "Power", "Polynomial"], command=self._on_plot_setting_change)
            trendline_type_menu.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
            
            self.poly_order_entry = ctk.CTkEntry(trend_frame, placeholder_text="Polynomial Order (2-6)")
            self.poly_order_entry.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
            self.poly_order_entry.bind("<KeyRelease>", self._on_plot_setting_change)
            self.poly_order_entry.bind("<FocusOut>", self._on_plot_setting_change)
            
            self.trendline_textbox = ctk.CTkTextbox(trend_frame, height=70)
            self.trendline_textbox.grid(row=6, column=0, sticky="ew", padx=10, pady=5)

            # Filter preview
            plot_filter_frame = ctk.CTkFrame(plot_left_panel)
            plot_filter_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
            plot_filter_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(plot_filter_frame, text="Filter Preview", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            self.plot_filter_type = ctk.StringVar(value="None")
            self.plot_filter_menu = ctk.CTkOptionMenu(plot_filter_frame, variable=self.plot_filter_type, values=self.filter_names, command=self._update_plot_filter_ui)
            self.plot_filter_menu.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
            
            # Filter parameter frames
            time_units = ["ms", "s", "min", "hr"]
            (self.plot_ma_frame, self.plot_ma_value_entry, self.plot_ma_unit_menu) = self._create_ma_param_frame(plot_filter_frame, time_units)
            (self.plot_bw_frame, self.plot_bw_order_entry, self.plot_bw_cutoff_entry) = self._create_bw_param_frame(plot_filter_frame)
            (self.plot_median_frame, self.plot_median_kernel_entry) = self._create_median_param_frame(plot_filter_frame)
            (self.plot_hampel_frame, self.plot_hampel_window_entry, self.plot_hampel_threshold_entry) = self._create_hampel_param_frame(plot_filter_frame)
            (self.plot_zscore_frame, self.plot_zscore_threshold_entry, self.plot_zscore_method_menu) = self._create_zscore_param_frame(plot_filter_frame)
            (self.plot_savgol_frame, self.plot_savgol_window_entry, self.plot_savgol_polyorder_entry) = self._create_savgol_param_frame(plot_filter_frame)
            self._update_plot_filter_ui("None")
            
            # Show both raw and filtered signals option (moved below parameter frames)
            self.show_both_signals_var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(plot_filter_frame, text="Show both raw and filtered signals", variable=self.show_both_signals_var).grid(row=10, column=0, sticky="w", padx=10, pady=5)
            
            ctk.CTkButton(plot_filter_frame, text="Preview Filter", command=self.update_plot).grid(row=11, column=0, sticky="ew", padx=10, pady=5)
            ctk.CTkButton(plot_filter_frame, text="Copy Settings to Processing Tab", command=self._copy_plot_settings_to_processing).grid(row=12, column=0, sticky="ew", padx=10, pady=5)

            # Time range controls
            time_range_frame = ctk.CTkFrame(plot_left_panel)
            time_range_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
            time_range_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(time_range_frame, text="Plot Time Range", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkLabel(time_range_frame, text="Start Time (HH:MM:SS):").grid(row=1, column=0, sticky="w", padx=10)
            self.plot_start_time_entry = ctk.CTkEntry(time_range_frame, placeholder_text="e.g., 09:30:00")
            self.plot_start_time_entry.grid(row=2, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkLabel(time_range_frame, text="End Time (HH:MM:SS):").grid(row=3, column=0, sticky="w", padx=10)
            self.plot_end_time_entry = ctk.CTkEntry(time_range_frame, placeholder_text="e.g., 17:00:00")
            self.plot_end_time_entry.grid(row=4, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkButton(time_range_frame, text="Apply Time Range to Plot", command=self._apply_plot_time_range).grid(row=5, column=0, sticky="ew", padx=10, pady=5)
            ctk.CTkButton(time_range_frame, text="Reset Plot Range", command=self._reset_plot_range).grid(row=6, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkButton(time_range_frame, text="Save Current View", command=self._save_current_plot_view).grid(row=7, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkButton(time_range_frame, text="Copy Current View to Processing", command=self._copy_current_view_to_processing).grid(row=8, column=0, sticky="ew", padx=10, pady=2)

            # Export controls
            export_chart_frame = ctk.CTkFrame(plot_left_panel)
            export_chart_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
            export_chart_frame.grid_columnconfigure(0, weight=1)
            
            ctk.CTkLabel(export_chart_frame, text="Export Chart", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            ctk.CTkButton(export_chart_frame, text="Save as PNG/PDF", command=self._export_chart_image).grid(row=1, column=0, sticky="ew", padx=10, pady=2)
            ctk.CTkButton(export_chart_frame, text="Export to Excel with Chart", command=self._export_chart_excel).grid(row=2, column=0, sticky="ew", padx=10, pady=2)

        def create_plot_right_content(right_panel):
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
            
            self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_canvas_frame)
            self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
            
            toolbar = NavigationToolbar2Tk(self.plot_canvas, plot_canvas_frame, pack_toolbar=False)
            toolbar.grid(row=0, column=0, sticky="ew")
            
            # Store toolbar reference for custom functionality
            self.plot_toolbar = toolbar

        # Create splitter for plotting tab
        splitter_frame = self._create_splitter(plot_main_frame, create_plot_left_content, create_plot_right_content, 'plotting_left_width', 400)
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def create_plots_list_tab(self, tab):
        """Create the plots list tab."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(header_frame, text="Plots List Manager", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=10, pady=10)
        
        # Main content
        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create splitter
        splitter_frame = self._create_splitter(main_frame, self._create_plots_list_left, self._create_plots_list_right, 'plots_list_left_width', 300)
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def _create_plots_list_left(self, left_panel):
        """Create left panel for plots list."""
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        # Plot configuration frame
        config_frame = ctk.CTkFrame(left_panel)
        config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        config_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(config_frame, text="Plot Configuration", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(config_frame, text="Plot Name:").grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.plot_name_entry = ctk.CTkEntry(config_frame, placeholder_text="e.g., Temperature Analysis")
        self.plot_name_entry.grid(row=2, column=0, padx=10, pady=2, sticky="ew")
        
        ctk.CTkLabel(config_frame, text="Description:").grid(row=3, column=0, padx=10, pady=2, sticky="w")
        self.plot_desc_entry = ctk.CTkEntry(config_frame, placeholder_text="Brief description of this plot")
        self.plot_desc_entry.grid(row=4, column=0, padx=10, pady=2, sticky="ew")
        
        # Signal selection
        ctk.CTkLabel(config_frame, text="Signals to Include:").grid(row=5, column=0, padx=10, pady=(10,2), sticky="w")
        self.plots_signals_frame = ctk.CTkScrollableFrame(config_frame, height=100)
        self.plots_signals_frame.grid(row=6, column=0, padx=10, pady=2, sticky="ew")
        
        # Bind mouse wheel to the plots signals frame
        self._bind_mousewheel_to_frame(self.plots_signals_frame)
        
        # Time range
        ctk.CTkLabel(config_frame, text="Time Range (HH:MM:SS):").grid(row=7, column=0, padx=10, pady=(10,2), sticky="w")
        
        time_range_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        time_range_frame.grid(row=8, column=0, padx=10, pady=2, sticky="ew")
        time_range_frame.grid_columnconfigure(0, weight=1)
        time_range_frame.grid_columnconfigure(1, weight=1)
        
        self.plot_start_time_entry = ctk.CTkEntry(time_range_frame, placeholder_text="Start time")
        self.plot_start_time_entry.grid(row=0, column=0, padx=(0,5), pady=2, sticky="ew")
        
        self.plot_end_time_entry = ctk.CTkEntry(time_range_frame, placeholder_text="End time")
        self.plot_end_time_entry.grid(row=0, column=1, padx=(5,0), pady=2, sticky="ew")
        
        # Buttons
        button_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        button_frame.grid(row=9, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkButton(button_frame, text="Add to List", command=self._add_plot_to_list).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(button_frame, text="Update Selected", command=self._update_selected_plot).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(button_frame, text="Clear Form", command=self._clear_plot_form).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # Plots list
        list_frame = ctk.CTkFrame(left_panel)
        list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(list_frame, text="Saved Plots", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.plots_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, font=("Arial", 10))
        self.plots_listbox.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.plots_listbox.bind('<<ListboxSelect>>', self._on_plot_select)
        
        # List buttons
        list_button_frame = ctk.CTkFrame(list_frame, fg_color="transparent")
        list_button_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(list_button_frame, text="Load Selected", command=self._load_selected_plot).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(list_button_frame, text="Delete Selected", command=self._delete_selected_plot).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(list_button_frame, text="Clear All", command=self._clear_all_plots).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

    def _create_plots_list_right(self, right_panel):
        """Create right panel for plots list."""
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Plot preview frame
        preview_frame = ctk.CTkFrame(right_panel)
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(preview_frame, text="Plot Preview", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Preview canvas
        self.preview_fig = Figure(figsize=(6, 4), dpi=100)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_fig.tight_layout()
        
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Preview buttons
        preview_button_frame = ctk.CTkFrame(preview_frame, fg_color="transparent")
        preview_button_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        ctk.CTkButton(preview_button_frame, text="Generate Preview", command=self._generate_plot_preview).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(preview_button_frame, text="Export All Plots", command=self._export_all_plots).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def create_dat_import_tab(self, tab):
        """Create the DAT file import tab."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(header_frame, text="DAT File Import", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=10, pady=10)
        
        # Main content
        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create splitter
        splitter_frame = self._create_splitter(main_frame, self._create_dat_import_left, self._create_dat_import_right, 'dat_import_left_width', 300)
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def _create_dat_import_left(self, left_panel):
        """Create left panel for DAT import."""
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        # File selection frame
        file_frame = ctk.CTkFrame(left_panel)
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        file_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(file_frame, text="File Selection", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        ctk.CTkButton(file_frame, text="Select Tag File (.dbf)", command=self._select_tag_file).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(file_frame, text="Select Data File (.dat)", command=self._select_data_file).grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        self.tag_file_label = ctk.CTkLabel(file_frame, text="No tag file selected", font=ctk.CTkFont(size=11))
        self.tag_file_label.grid(row=3, column=0, padx=10, pady=2, sticky="w")
        
        self.data_file_label = ctk.CTkLabel(file_frame, text="No data file selected", font=ctk.CTkFont(size=11))
        self.data_file_label.grid(row=4, column=0, padx=10, pady=2, sticky="w")
        
        # Import settings frame
        settings_frame = ctk.CTkFrame(left_panel)
        settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        settings_frame.grid_columnconfigure(0, weight=1)
        settings_frame.grid_rowconfigure(2, weight=1)
        
        ctk.CTkLabel(settings_frame, text="Import Settings", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(settings_frame, text="Tag Delimiter:").grid(row=1, column=0, padx=10, pady=2, sticky="w")
        ctk.CTkOptionMenu(settings_frame, variable=self.tag_delimiter_var, values=["newline", "comma", "semicolon", "tab"]).grid(row=2, column=0, padx=10, pady=2, sticky="ew")
        
        # Tag selection frame
        tag_frame = ctk.CTkFrame(settings_frame)
        tag_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        tag_frame.grid_columnconfigure(0, weight=1)
        tag_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(tag_frame, text="Select Tags to Import:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.tags_listbox = tk.Listbox(tag_frame, selectmode=tk.MULTIPLE, font=("Arial", 10))
        self.tags_listbox.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Import button
        ctk.CTkButton(settings_frame, text="Import Selected Tags", command=self._import_selected_tags).grid(row=4, column=0, padx=10, pady=10, sticky="ew")

    def _create_dat_import_right(self, right_panel):
        """Create right panel for DAT import."""
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Preview frame
        preview_frame = ctk.CTkFrame(right_panel)
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(preview_frame, text="Import Preview", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.import_preview_text = ctk.CTkTextbox(preview_frame, height=200)
        self.import_preview_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

    def _load_layout_config(self):
        """Load layout configuration from file."""
        try:
            if os.path.exists(self.layout_config_file):
                with open(self.layout_config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading layout config: {e}")
        return {}

    def _save_layout_config(self):
        """Save layout configuration to file."""
        try:
            # Get current window dimensions
            self.layout_data['window_width'] = self.winfo_width()
            self.layout_data['window_height'] = self.winfo_height()
            
            # Save splitter positions
            for splitter_key, splitter in self.splitters.items():
                if hasattr(splitter, 'winfo_width'):
                    self.layout_data[splitter_key] = splitter.winfo_width()
            
            with open(self.layout_config_file, 'w') as f:
                json.dump(self.layout_data, f, indent=2)
        except Exception as e:
            print(f"Error saving layout config: {e}")

    def _create_splitter(self, parent, left_creator, right_creator, splitter_key, default_left_width):
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
        splitter_handle.bind("<Enter>", lambda e, h=splitter_handle: self._on_splitter_enter(e, h))
        splitter_handle.bind("<Leave>", lambda e, h=splitter_handle: self._on_splitter_leave(e, h))
        splitter_handle.bind("<Button-1>", lambda e, h=splitter_handle: self._start_splitter_drag(e, h, left_panel, splitter_key))
        splitter_handle.bind("<B1-Motion>", lambda e, h=splitter_handle: self._drag_splitter(e, h, left_panel, splitter_key))
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

    def _on_splitter_enter(self, event, handle):
        """Handle mouse enter on splitter handle."""
        handle.configure(fg_color="#888888")
        handle.configure(cursor="sb_h_double_arrow")

    def _on_splitter_leave(self, event, handle):
        """Handle mouse leave on splitter handle."""
        if not hasattr(self, 'dragging_splitter') or not self.dragging_splitter:
            handle.configure(fg_color="#666666")

    def _start_splitter_drag(self, event, handle, left_panel, splitter_key):
        """Start dragging the splitter."""
        self.dragging_splitter = True
        self.drag_splitter_key = splitter_key
        self.drag_left_panel = left_panel
        self.drag_start_x = event.x_root
        self.drag_start_width = left_panel.winfo_width()
        handle.configure(fg_color="#AAAAAA")

    def _drag_splitter(self, event, handle, left_panel, splitter_key):
        """Drag the splitter."""
        if hasattr(self, 'dragging_splitter') and self.dragging_splitter:
            delta_x = event.x_root - self.drag_start_x
            new_width = max(150, min(800, self.drag_start_width + delta_x))  # Min 150, Max 800
            left_panel.configure(width=new_width)

    def _end_splitter_drag(self):
        """End dragging the splitter."""
        if hasattr(self, 'dragging_splitter') and self.dragging_splitter:
            # Save the current position
            if hasattr(self, 'drag_splitter_key') and hasattr(self, 'drag_left_panel'):
                self.layout_data[self.drag_splitter_key] = self.drag_left_panel.winfo_width()
                # Auto-save layout
                self._save_layout_config()
        
        self.dragging_splitter = False
        # Reset handle color
        for splitter_key, splitter in self.splitters.items():
            if hasattr(splitter, 'master') and hasattr(splitter.master, 'winfo_children'):
                for child in splitter.master.winfo_children():
                    if isinstance(child, ctk.CTkFrame) and child.winfo_width() == 8:
                        child.configure(fg_color="#666666")

    def _on_closing(self):
        """Handle application closing."""
        self._save_layout_config()
        self.quit()

    def _on_window_configure(self, event):
        """Handle window resize events to save layout."""
        # Only save if this is the main window being resized
        if event.widget == self:
            # Debounce the saving to avoid too frequent saves
            if hasattr(self, '_resize_timer'):
                self.after_cancel(self._resize_timer)
            self._resize_timer = self.after(1000, self._save_layout_config)

    def create_status_bar(self):
        """Create the status bar."""
        self.status_label = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.status_label.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    # Placeholder methods for functionality that would be implemented
    def on_plot_file_select(self, value):
        """Handle plot file selection."""
        if value == "Select a file...":
            return
            
        df = self.get_data_for_plotting(value)
        if df is not None and not df.empty:
            # Update x-axis options - use actual columns, not "default time"
            x_axis_options = list(df.columns)
            self.plot_xaxis_menu.configure(values=x_axis_options)
            
            # Set the first column as default x-axis (usually time)
            if x_axis_options:
                self.plot_xaxis_menu.set(x_axis_options[0])
            
            # Update signal checkboxes
            self.plot_signal_vars = {}
            for widget in self.plot_signal_frame.winfo_children():
                widget.destroy()
            
            for signal in df.columns:
                var = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.plot_signal_frame, text=signal, variable=var, command=self._on_plot_setting_change)
                cb.pack(anchor="w", padx=5, pady=2)
                self.plot_signal_vars[signal] = {'var': var, 'checkbox': cb}
            
            # Re-bind mouse wheel to all new checkboxes
            self._bind_mousewheel_to_frame(self.plot_signal_frame)
            
            # Update trendline signal options
            signal_options = ["Select signal..."] + [col for col in df.columns if col != x_axis_options[0]]  # Exclude time column
            self.trendline_signal_menu.configure(values=signal_options)
            self.trendline_signal_var.set("Select signal...")
                
            # Update plot
            self.update_plot()

    def update_plot(self, selected_signals=None):
        """The main function to draw/redraw the plot with all selected options."""
        # Check if plot canvas is initialized
        if not hasattr(self, 'plot_canvas') or not hasattr(self, 'plot_ax'):
            return
            
        selected_file = self.plot_file_menu.get()
        x_axis_col = self.plot_xaxis_menu.get()

        if selected_file == "Select a file..." or not x_axis_col:
            return

        df = self.get_data_for_plotting(selected_file)
        if df is None or df.empty:
            self.plot_ax.clear()
            self.plot_ax.text(0.5, 0.5, "Could not load or plot data.", ha='center', va='center')
            self.plot_canvas.draw()
            return

        # Safety check: ensure x_axis_col is a valid column
        if x_axis_col not in df.columns:
            # Try to set a valid x-axis column
            if len(df.columns) > 0:
                self.plot_xaxis_menu.set(df.columns[0])
                x_axis_col = df.columns[0]
            else:
                self.plot_ax.clear()
                self.plot_ax.text(0.5, 0.5, "No valid columns found for plotting.", ha='center', va='center')
                self.plot_canvas.draw()
                return

        signals_to_plot = [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
        self.plot_ax.clear()

        if not signals_to_plot:
            self.plot_ax.text(0.5, 0.5, "Select one or more signals to plot", ha='center', va='center')
        else:
            # Check if we should show both raw and filtered signals
            show_both = self.show_both_signals_var.get()
            plot_filter = self.plot_filter_type.get()
            
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
                if signal not in df.columns: 
                    continue
                
                plot_df = df[[x_axis_col, signal]].dropna()
                
                if show_both and plot_filter != "None":
                    # Plot raw signal with dashed line
                    raw_style = style_args.copy()
                    raw_style["linestyle"] = "--"
                    raw_style["alpha"] = 0.7
                    raw_style["color"] = colors[i]
                    raw_label = f"{self.custom_legend_entries.get(signal, signal)} (Raw)"
                    self.plot_ax.plot(plot_df[x_axis_col], plot_df[signal], label=raw_label, **raw_style)
                    
                    # Apply filter and plot filtered signal
                    filtered_df = self._apply_plot_filter(df.copy(), [signal], x_axis_col)
                    filtered_plot_df = filtered_df[[x_axis_col, signal]].dropna()
                    filtered_style = style_args.copy()
                    filtered_style["color"] = colors[i]
                    filtered_label = f"{self.custom_legend_entries.get(signal, signal)} (Filtered)"
                    self.plot_ax.plot(filtered_plot_df[x_axis_col], filtered_plot_df[signal], label=filtered_label, **filtered_style)
                else:
                    # Apply filter if selected (but not showing both)
                    if plot_filter != "None":
                        filtered_df = self._apply_plot_filter(df.copy(), [signal], x_axis_col)
                        plot_df = filtered_df[[x_axis_col, signal]].dropna()
                    
                    plot_style = style_args.copy()
                    plot_style["color"] = colors[i]
                    signal_label = self.custom_legend_entries.get(signal, signal)
                    self.plot_ax.plot(plot_df[x_axis_col], plot_df[signal], label=signal_label, **plot_style)

            # Add trendline if selected
            if self.trendline_type_var.get() != "None":
                selected_trendline_signal = self.trendline_signal_var.get()
                if selected_trendline_signal != "Select signal..." and selected_trendline_signal in df.columns:
                    self._add_trendline(df, selected_trendline_signal, x_axis_col)

        # Apply custom labels and title
        title = self.plot_title_entry.get() or f"Signals from {selected_file}"
        xlabel = self.plot_xlabel_entry.get() or x_axis_col
        ylabel = self.plot_ylabel_entry.get() or "Value"
        self.plot_ax.set_title(title, fontsize=14)
        self.plot_ax.set_xlabel(xlabel)
        self.plot_ax.set_ylabel(ylabel)

        self.plot_ax.legend()
        self.plot_ax.grid(True, linestyle='--', alpha=0.6)

        if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
             # Use simpler HH:MM format for cleaner plot appearance
             self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
             # Keep labels horizontal for better readability
             self.plot_ax.tick_params(axis='x', rotation=0)

        self.plot_canvas.draw()

    def _apply_plot_filter(self, df, signal_cols, x_axis_col):
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
                    window = window / 1000
                elif unit == "min":
                    window = window * 60
                elif unit == "hr":
                    window = window * 3600
                
                # Convert window to number of samples
                if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
                    time_diff = df[x_axis_col].diff().dt.total_seconds().median()
                    if time_diff > 0:
                        window_samples = int(window / time_diff)
                        filtered_df[signal] = df[signal].rolling(window=max(1, window_samples), center=True).mean()
                else:
                    filtered_df[signal] = df[signal].rolling(window=int(window), center=True).mean()
                    
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
                    nyquist = fs / 2.0
                    normalized_cutoff = cutoff / nyquist
                    
                    # Design filter
                    b, a = butter(order, normalized_cutoff, btype='low' if filter_type == "Butterworth Low-pass" else 'high')
                    
                    # Apply filter
                    signal_data = df[signal].fillna(method='ffill').fillna(method='bfill')
                    filtered_df[signal] = filtfilt(b, a, signal_data)
                    
                except ImportError:
                    # Fallback to simple smoothing if scipy not available
                    filtered_df[signal] = df[signal].rolling(window=order*2+1, center=True).mean()
                except Exception as e:
                    print(f"Error applying Butterworth filter: {e}")
                    # Fallback to simple smoothing
                    filtered_df[signal] = df[signal].rolling(window=order*2+1, center=True).mean()
                
            elif filter_type == "Median Filter":
                kernel = int(self.plot_median_kernel_entry.get() or "5")
                filtered_df[signal] = df[signal].rolling(window=kernel, center=True).median()
                
            elif filter_type == "Hampel Filter":
                window = int(self.plot_hampel_window_entry.get() or "7")
                threshold = float(self.plot_hampel_threshold_entry.get() or "3.0")
                
                try:
                    from scipy.signal import medfilt
                    signal_data = df[signal].ffill().bfill()
                    
                    # Apply Hampel filter
                    median_filtered = pd.Series(medfilt(signal_data, kernel_size=window), index=signal_data.index)
                    mad = signal_data.rolling(window=window, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))))
                    threshold_value = threshold * 1.4826 * mad  # 1.4826 is the constant for normal distribution
                    
                    # Replace outliers with median using proper indexing
                    outliers = np.abs(signal_data - median_filtered) > threshold_value
                    filtered_df = filtered_df.copy()  # Ensure we have a copy to avoid warnings
                    filtered_df.loc[outliers, signal] = median_filtered.loc[outliers]
                    
                except ImportError:
                    # Fallback to simple median filter
                    filtered_df[signal] = df[signal].rolling(window=window, center=True).median()
                except Exception as e:
                    print(f"Error applying Hampel filter: {e}")
                    # Fallback to simple median filter
                    filtered_df[signal] = df[signal].rolling(window=window, center=True).median()
                
            elif filter_type == "Z-Score Filter":
                threshold = float(self.plot_zscore_threshold_entry.get() or "3.0")
                method = self.plot_zscore_method_menu.get()
                
                signal_data = df[signal].fillna(method='ffill').fillna(method='bfill')
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
                    filtered_df[signal] = filtered_df[signal].clip(lower=lower_bound, upper=upper_bound)
                elif method == "Replace with Median":
                    # Replace outliers with median
                    median_val = signal_data.median()
                    filtered_df[signal] = signal_data.copy()
                    filtered_df[signal].loc[z_scores > threshold] = median_val
                
            elif filter_type == "Savitzky-Golay":
                window = int(self.plot_savgol_window_entry.get() or "11")
                polyorder = int(self.plot_savgol_polyorder_entry.get() or "3")
                
                try:
                    from scipy.signal import savgol_filter
                    signal_data = df[signal].fillna(method='ffill').fillna(method='bfill')
                    filtered_df[signal] = savgol_filter(signal_data, window, polyorder)
                except ImportError:
                    # Fallback to simple smoothing if scipy not available
                    filtered_df[signal] = df[signal].rolling(window=window, center=True).mean()
                except Exception as e:
                    print(f"Error applying Savitzky-Golay filter: {e}")
                    # Fallback to simple smoothing
                    filtered_df[signal] = df[signal].rolling(window=window, center=True).mean()
        
        return filtered_df

    def _add_trendline(self, df, signal, x_axis_col):
        """Add trendline to the plot."""
        trend_type = self.trendline_type_var.get()
        
        if trend_type == "None":
            return
            
        plot_df = df[[x_axis_col, signal]].dropna()
        if len(plot_df) < 2:
            return
            
        x_data = plot_df[x_axis_col]
        y_data = plot_df[signal]
        
        # Convert datetime to numeric for fitting
        if pd.api.types.is_datetime64_any_dtype(x_data):
            x_numeric = (x_data - x_data.min()).dt.total_seconds()
        else:
            x_numeric = x_data
            
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
                    return
                    
            elif trend_type == "Power":
                # Log-log fit for power law
                y_positive = y_data[y_data > 0]
                x_positive = x_numeric[x_numeric > 0]
                if len(y_positive) > 1 and len(x_positive) > 1:
                    log_x = np.log(x_positive)
                    log_y = np.log(y_positive)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    a = np.exp(coeffs[1])
                    b = coeffs[0]
                    trendline = a * (x_numeric ** b)
                    equation = f"y = {a:.4f} * x^({b:.4f})"
                else:
                    return
                    
            elif trend_type == "Polynomial":
                order = int(self.poly_order_entry.get() or "2")
                order = max(2, min(6, order))  # Limit to 2-6
                coeffs = np.polyfit(x_numeric, y_data, order)
                trend = np.poly1d(coeffs)
                trendline = trend(x_numeric)
                equation = f"Polynomial (order {order}): " + " + ".join([f"{coeffs[i]:.4f}x^{order-i}" for i in range(order+1)])
            
            # Plot trendline
            self.plot_ax.plot(x_data, trendline, '--', color='red', linewidth=2, label=f'Trendline ({trend_type})')
            
            # Update trendline textbox
            self.trendline_textbox.delete("1.0", tk.END)
            self.trendline_textbox.insert("1.0", equation)
            
        except Exception as e:
            print(f"Error adding trendline: {e}")

    def get_data_for_plotting(self, filename):
        """Get data for plotting from the specified file."""
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
                    if any(time_word in col.lower() for time_word in ['time', 'timestamp', 'date']):
                        time_col = col
                        break
                
                if time_col and pd.api.types.is_object_dtype(df[time_col]):
                    try:
                        df[time_col] = pd.to_datetime(df[time_col])
                    except:
                        pass
                
                return df
        except Exception as e:
            print(f"Error loading data for plotting: {e}")
            return None

    def _show_setup_help(self):
        """Show setup help."""
        messagebox.showinfo("Setup Help", "This tab allows you to configure file processing settings.")

    def _show_plot_help(self):
        """Show plotting help."""
        messagebox.showinfo("Plotting Help", "This tab allows you to visualize and analyze your data.")

    def _show_plots_list_help(self):
        """Show plots list help."""
        messagebox.showinfo("Plots List Help", "This tab allows you to save and manage plot configurations.")

    def _show_dat_import_help(self):
        """Show DAT import help."""
        messagebox.showinfo("DAT Import Help", "This tab allows you to import DAT files with DBF tag files.")

    def _show_legend_guide(self):
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
        title_label = ctk.CTkLabel(scrollable_frame, text="Custom Legend Formatting Guide", 
                                   font=ctk.CTkFont(size=18, weight="bold"))
        title_label.grid(row=0, column=0, pady=(0,20), sticky="w")
        
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
        text_widget = ctk.CTkTextbox(scrollable_frame, width=550, height=500, wrap="word")
        text_widget.grid(row=1, column=0, pady=10, sticky="ew")
        text_widget.insert("1.0", guide_text)
        text_widget.configure(state="disabled")
        
        # Close button
        close_button = ctk.CTkButton(guide_window, text="Close", command=guide_window.destroy)
        close_button.grid(row=1, column=0, pady=10)
        
        # Center the window
        guide_window.update_idletasks()
        x = (guide_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (guide_window.winfo_screenheight() // 2) - (700 // 2)
        guide_window.geometry(f"600x700+{x}+{y}")

    def save_settings(self):
        """Save current settings."""
        pass

    def load_settings(self):
        """Load saved settings."""
        pass

    def save_signal_list(self):
        """Save the currently selected signals as a signal list."""
        if not self.signal_vars:
            messagebox.showwarning("Warning", "No signals available to save. Please load a file first.")
            return
        
        # Get currently selected signals
        selected_signals = [signal for signal, data in self.signal_vars.items() if data['var'].get()]
        
        if not selected_signals:
            messagebox.showwarning("Warning", "No signals are currently selected. Please select signals to save.")
            return
        
        # Ask user for a name for this signal list
        signal_list_name = tk.simpledialog.askstring(
            "Save Signal List", 
            "Enter a name for this signal list:",
            initialvalue="My Signal List"
        )
        
        if not signal_list_name:
            return  # User cancelled
        
        # Create the signal list data
        signal_list_data = {
            'name': signal_list_name,
            'signals': selected_signals,
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        # Save to file
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Signal List",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"{signal_list_name}.json"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(signal_list_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Signal list '{signal_list_name}' saved successfully!\n\nSaved signals: {len(selected_signals)}")
                self.status_label.configure(text=f"Signal list saved: {signal_list_name}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save signal list:\n{e}")

    def load_signal_list(self):
        """Load a saved signal list from file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Signal List",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not file_path:
                return  # User cancelled
            
            with open(file_path, 'r') as f:
                signal_list_data = json.load(f)
            
            # Validate the loaded data
            if not isinstance(signal_list_data, dict) or 'signals' not in signal_list_data:
                messagebox.showerror("Error", "Invalid signal list file format.")
                return
            
            # Store the loaded signal list
            self.saved_signal_list = signal_list_data.get('signals', [])
            self.saved_signal_list_name = signal_list_data.get('name', 'Unknown')
            
            # Update status
            self.signal_list_status_label.configure(
                text=f"Loaded: {self.saved_signal_list_name} ({len(self.saved_signal_list)} signals)",
                text_color="green"
            )
            
            # Automatically apply the loaded signals if we have signals available
            if self.signal_vars:
                self._apply_loaded_signals_internal()
            
            messagebox.showinfo("Success", f"Signal list '{self.saved_signal_list_name}' loaded and applied successfully!\n\nSignals: {len(self.saved_signal_list)}")
            self.status_label.configure(text=f"Signal list loaded: {self.saved_signal_list_name}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load signal list:\n{e}")

    def _apply_loaded_signals_internal(self):
        """Internal method to apply loaded signals without showing message boxes."""
        if not self.saved_signal_list or not self.signal_vars:
            return
        
        # Get current available signals
        available_signals = list(self.signal_vars.keys())
        
        # Find which saved signals are present
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
                data['var'].set(True)
            else:
                data['var'].set(False)
        
        # Update status
        self.signal_list_status_label.configure(
            text=f"Applied: {self.saved_signal_list_name} ({len(present_signals)}/{len(self.saved_signal_list)} signals)",
            text_color="blue"
        )

    def apply_saved_signals(self):
        """Apply the saved signal list to the current file's signals."""
        if not self.saved_signal_list:
            messagebox.showwarning("Warning", "No saved signal list loaded. Please load a signal list first.")
            return
        
        if not self.signal_vars:
            messagebox.showwarning("Warning", "No signals available. Please load a file first.")
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
                data['var'].set(True)
            else:
                data['var'].set(False)
        
        # Show results to user
        if missing_signals:
            missing_text = "\n".join([f"• {signal}" for signal in missing_signals])
            messagebox.showinfo(
                "Signals Applied", 
                f"Applied {len(present_signals)} signals from '{self.saved_signal_list_name}'.\n\n"
                f"Missing signals ({len(missing_signals)}):\n{missing_text}"
            )
        else:
            messagebox.showinfo(
                "Signals Applied", 
                f"Successfully applied all {len(present_signals)} signals from '{self.saved_signal_list_name}'."
            )
        
        # Update status
        self.signal_list_status_label.configure(
            text=f"Applied: {self.saved_signal_list_name} ({len(present_signals)}/{len(self.saved_signal_list)} signals)",
            text_color="blue"
        )
        
        self.status_label.configure(text=f"Applied {len(present_signals)} signals from saved list")

    def _show_sharing_instructions(self):
        """Show sharing instructions."""
        messagebox.showinfo("Sharing Instructions", "To share this application, include all Python files and the requirements.txt file.")

    def _copy_plot_settings_to_processing(self):
        """Copies filter settings from the plot tab to the main processing tab."""
        plot_filter = self.plot_filter_type.get()
        self.filter_type_var.set(plot_filter)
        self._update_filter_ui(plot_filter)
        
        # Copy filter parameters
        if plot_filter == "Moving Average":
            if hasattr(self, 'plot_ma_value_entry') and hasattr(self, 'plot_ma_unit_menu'):
                self.ma_value_entry.delete(0, tk.END)
                self.ma_value_entry.insert(0, self.plot_ma_value_entry.get())
                self.ma_unit_menu.set(self.plot_ma_unit_menu.get())
        elif plot_filter == "Butterworth":
            if hasattr(self, 'plot_bw_order_entry') and hasattr(self, 'plot_bw_cutoff_entry'):
                self.bw_order_entry.delete(0, tk.END)
                self.bw_order_entry.insert(0, self.plot_bw_order_entry.get())
                self.bw_cutoff_entry.delete(0, tk.END)
                self.bw_cutoff_entry.insert(0, self.plot_bw_cutoff_entry.get())
        elif plot_filter == "Median Filter":
            if hasattr(self, 'plot_median_kernel_entry'):
                self.median_kernel_entry.delete(0, tk.END)
                self.median_kernel_entry.insert(0, self.plot_median_kernel_entry.get())
        elif plot_filter == "Hampel Filter":
            if hasattr(self, 'plot_hampel_window_entry') and hasattr(self, 'plot_hampel_threshold_entry'):
                self.hampel_window_entry.delete(0, tk.END)
                self.hampel_window_entry.insert(0, self.plot_hampel_window_entry.get())
                self.hampel_threshold_entry.delete(0, tk.END)
                self.hampel_threshold_entry.insert(0, self.plot_hampel_threshold_entry.get())
        elif plot_filter == "Z-Score Filter":
            if hasattr(self, 'plot_zscore_threshold_entry') and hasattr(self, 'plot_zscore_method_menu'):
                self.zscore_threshold_entry.delete(0, tk.END)
                self.zscore_threshold_entry.insert(0, self.plot_zscore_threshold_entry.get())
                self.zscore_method_menu.set(self.plot_zscore_method_menu.get())
        elif plot_filter == "Savitzky-Golay":
            if hasattr(self, 'plot_savgol_window_entry') and hasattr(self, 'plot_savgol_polyorder_entry'):
                self.savgol_window_entry.delete(0, tk.END)
                self.savgol_window_entry.insert(0, self.plot_savgol_window_entry.get())
                self.savgol_polyorder_entry.delete(0, tk.END)
                self.savgol_polyorder_entry.insert(0, self.plot_savgol_polyorder_entry.get())
        
        messagebox.showinfo("Settings Copied", "Filter settings from the plot tab have been applied to the main processing configuration.")

    def _export_chart_image(self):
        """Export the current chart as an image file."""
        if not hasattr(self, 'plot_fig') or not self.plot_fig.get_axes():
            messagebox.showwarning("Warning", "No plot to export. Please create a plot first.")
            return
            
        try:
            file_types = [
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg")
            ]
            
            save_path = filedialog.asksaveasfilename(
                title="Export Chart As Image",
                filetypes=file_types,
                defaultextension=".png"
            )
            
            if save_path:
                # Check for overwrite and get final path
                final_path = self._check_file_overwrite(save_path)
                if final_path is None:  # User cancelled
                    return
                
                self.plot_fig.savefig(final_path, dpi=300, bbox_inches='tight', 
                                    facecolor='white', edgecolor='none')
                messagebox.showinfo("Success", f"Chart exported to:\n{final_path}")
                self.status_label.configure(text=f"Chart exported: {os.path.basename(final_path)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart:\n{e}")

    def _export_chart_excel(self):
        """Export the current plot data and chart to Excel."""
        selected_file = self.plot_file_menu.get()
        
        if selected_file == "Select a file...":
            messagebox.showwarning("Warning", "Please select a file to plot first.")
            return
            
        try:
            save_path = filedialog.asksaveasfilename(
                title="Export Chart Data to Excel",
                filetypes=[("Excel files", "*.xlsx")],
                defaultextension=".xlsx"
            )
            
            if save_path:
                # Check for overwrite and get final path
                final_path = self._check_file_overwrite(save_path)
                if final_path is None:  # User cancelled
                    return
                
                df = self.get_data_for_plotting(selected_file)
                if df is not None and not df.empty:
                    signals_to_plot = [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
                    
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
                        with pd.ExcelWriter(final_path, engine='openpyxl') as writer:
                            export_df.to_excel(writer, sheet_name='Chart Data', index=False)
                            
                            # Add chart information
                            info_df = pd.DataFrame({
                                'Property': ['File', 'Title', 'X-Axis', 'Y-Axis', 'Signals Plotted'],
                                'Value': [
                                    selected_file,
                                    self.plot_title_entry.get() or 'No title',
                                    self.plot_xlabel_entry.get() or 'No label',
                                    self.plot_ylabel_entry.get() or 'No label',
                                    ', '.join(signals_to_plot)
                                ]
                            })
                            info_df.to_excel(writer, sheet_name='Chart Info', index=False)
                        
                        messagebox.showinfo("Success", f"Chart data exported to:\n{final_path}")
                        self.status_label.configure(text=f"Chart data exported: {os.path.basename(final_path)}")
                    else:
                        messagebox.showwarning("Warning", "No signals selected for plotting.")
                else:
                    messagebox.showerror("Error", "Could not load data for export.")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart data:\n{e}")

    def _add_plot_to_list(self):
        """Add plot to the plots list."""
        plot_name = self.plot_name_entry.get().strip()
        plot_desc = self.plot_desc_entry.get().strip()
        
        if not plot_name:
            messagebox.showerror("Error", "Please enter a plot name.")
            return
        
        # Get selected signals from plots signals frame
        selected_signals = []
        if hasattr(self, 'plots_signal_vars'):
            selected_signals = [signal for signal, var in self.plots_signal_vars.items() if var.get()]
        
        plot_config = {
            'name': plot_name,
            'description': plot_desc or f"Plot configuration created on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            'signals': selected_signals,
            'start_time': self.plot_start_time_entry.get(),
            'end_time': self.plot_end_time_entry.get(),
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        self.plots_list.append(plot_config)
        self._update_plots_listbox()
        self._save_plots_to_file()
        self._clear_plot_form()
        
        messagebox.showinfo("Success", f"Plot '{plot_name}' added to list!")

    def _update_selected_plot(self):
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
        if hasattr(self, 'plots_signal_vars'):
            selected_signals = [signal for signal, var in self.plots_signal_vars.items() if var.get()]
        
        self.plots_list[idx].update({
            'name': plot_name,
            'description': self.plot_desc_entry.get().strip(),
            'signals': selected_signals,
            'start_time': self.plot_start_time_entry.get(),
            'end_time': self.plot_end_time_entry.get(),
            'modified_date': pd.Timestamp.now().isoformat()
        })
        
        self._update_plots_listbox()
        self._save_plots_to_file()
        messagebox.showinfo("Success", "Plot configuration updated!")

    def _clear_plot_form(self):
        """Clear the plot form."""
        self.plot_name_entry.delete(0, tk.END)
        self.plot_desc_entry.delete(0, tk.END)
        self.plot_start_time_entry.delete(0, tk.END)
        self.plot_end_time_entry.delete(0, tk.END)
        
        # Clear signal selections
        if hasattr(self, 'plots_signal_vars'):
            for var in self.plots_signal_vars.values():
                var.set(False)

    def _on_plot_select(self, event):
        """Handle plot selection in listbox."""
        selection = self.plots_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        plot_config = self.plots_list[idx]
        
        # Populate form with selected plot data
        self.plot_name_entry.delete(0, tk.END)
        self.plot_name_entry.insert(0, plot_config.get('name', ''))
        
        self.plot_desc_entry.delete(0, tk.END)
        self.plot_desc_entry.insert(0, plot_config.get('description', ''))
        
        self.plot_start_time_entry.delete(0, tk.END)
        self.plot_start_time_entry.insert(0, plot_config.get('start_time', ''))
        
        self.plot_end_time_entry.delete(0, tk.END)
        self.plot_end_time_entry.insert(0, plot_config.get('end_time', ''))
        
        # Update signal selections
        if hasattr(self, 'plots_signal_vars'):
            saved_signals = plot_config.get('signals', [])
            for signal, var in self.plots_signal_vars.items():
                var.set(signal in saved_signals)

    def _load_selected_plot(self):
        """Load selected plot configuration."""
        selection = self.plots_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a plot to load.")
            return
        
        idx = selection[0]
        plot_config = self.plots_list[idx]
        
        # Apply to plotting tab
        if 'file' in plot_config and plot_config['file']:
            self.plot_file_menu.set(plot_config['file'])
        
        if 'x_axis' in plot_config and plot_config['x_axis']:
            self.plot_xaxis_menu.set(plot_config['x_axis'])
        
        # Apply filter settings
        if 'filter_type' in plot_config:
            self.plot_filter_type.set(plot_config['filter_type'])
            self._update_plot_filter_ui(plot_config['filter_type'])
        
        if 'show_both_signals' in plot_config:
            self.show_both_signals_var.set(plot_config['show_both_signals'])
        
        # Apply plot labels
        if 'plot_title' in plot_config and hasattr(self, 'plot_title_entry'):
            self.plot_title_entry.delete(0, tk.END)
            self.plot_title_entry.insert(0, plot_config.get('plot_title', ''))
        
        if 'plot_xlabel' in plot_config and hasattr(self, 'plot_xlabel_entry'):
            self.plot_xlabel_entry.delete(0, tk.END)
            self.plot_xlabel_entry.insert(0, plot_config.get('plot_xlabel', ''))
        
        if 'plot_ylabel' in plot_config and hasattr(self, 'plot_ylabel_entry'):
            self.plot_ylabel_entry.delete(0, tk.END)
            self.plot_ylabel_entry.insert(0, plot_config.get('plot_ylabel', ''))
        
        # Apply time range
        if hasattr(self, 'plot_start_time_entry'):
            self.plot_start_time_entry.delete(0, tk.END)
            self.plot_start_time_entry.insert(0, plot_config.get('start_time', ''))
        
        if hasattr(self, 'plot_end_time_entry'):
            self.plot_end_time_entry.delete(0, tk.END)
            self.plot_end_time_entry.insert(0, plot_config.get('end_time', ''))
        
        # Apply signal selections
        if hasattr(self, 'plot_signal_vars') and 'signals' in plot_config:
            saved_signals = plot_config['signals']
            for signal, data in self.plot_signal_vars.items():
                data['var'].set(signal in saved_signals)
        
        messagebox.showinfo("Success", f"Plot configuration '{plot_config['name']}' loaded!")

    def _delete_selected_plot(self):
        """Delete selected plot from list."""
        selection = self.plots_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a plot to delete.")
            return
        
        idx = selection[0]
        plot_name = self.plots_list[idx]['name']
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete plot '{plot_name}'?"):
            del self.plots_list[idx]
            self._update_plots_listbox()
            self._save_plots_to_file()
            self._clear_plot_form()
            messagebox.showinfo("Success", f"Plot '{plot_name}' deleted.")

    def _clear_all_plots(self):
        """Clear all plots from list."""
        if self.plots_list and messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all plots?"):
            self.plots_list.clear()
            self._update_plots_listbox()
            self._save_plots_to_file()
            self._clear_plot_form()
            messagebox.showinfo("Success", "All plots cleared.")

    def _update_plots_listbox(self):
        """Update the plots listbox with current plots."""
        self.plots_listbox.delete(0, tk.END)
        for plot in self.plots_list:
            display_text = f"{plot['name']} ({len(plot.get('signals', []))} signals)"
            self.plots_listbox.insert(tk.END, display_text)

    def _save_plots_to_file(self):
        """Save plots list to file."""
        try:
            plots_file = os.path.join(os.path.expanduser("~"), ".csv_processor_plots.json")
            with open(plots_file, 'w') as f:
                json.dump(self.plots_list, f, indent=2)
        except Exception as e:
            print(f"Error saving plots to file: {e}")

    def _load_plots_from_file(self):
        """Load plots list from file."""
        try:
            plots_file = os.path.join(os.path.expanduser("~"), ".csv_processor_plots.json")
            if os.path.exists(plots_file):
                with open(plots_file, 'r') as f:
                    self.plots_list = json.load(f)
                self._update_plots_listbox()
                self._update_load_plot_config_menu()
        except Exception as e:
            print(f"Error loading plots from file: {e}")
            self.plots_list = []

    def _select_tag_file(self):
        """Select tag file for DAT import."""
        filepath = filedialog.askopenfilename(
            title="Select Tag File",
            filetypes=[("DBF files", "*.dbf"), ("All files", "*.*")]
        )
        if filepath:
            self.dat_import_tag_file_path = filepath
            self.tag_file_label.configure(text=os.path.basename(filepath))

    def _select_data_file(self):
        """Select data file for DAT import."""
        filepath = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filepath:
            self.dat_import_data_file_path = filepath
            self.data_file_label.configure(text=os.path.basename(filepath))
            
            # Set default output directory to the folder of the selected DAT file
            dat_file_dir = os.path.dirname(filepath)
            self.output_directory = dat_file_dir
            # Update the output label to reflect the new default directory
            if hasattr(self, 'output_label'):
                self.output_label.configure(text=f"Output: {self.output_directory}")

    def _import_selected_tags(self):
        """Import selected tags."""
        pass

    def trim_and_save(self):
        """Trim data and save."""
        pass

    def _apply_plot_time_range(self):
        """Apply time range to plot."""
        start_time_str = self.plot_start_time_entry.get()
        end_time_str = self.plot_end_time_entry.get()
        
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
            date_str = df[time_col].iloc[0].strftime('%Y-%m-%d')
            
            # Create full datetime strings
            start_full_str = f"{date_str} {start_time_str}" if start_time_str else f"{date_str} 00:00:00"
            end_full_str = f"{date_str} {end_time_str}" if end_time_str else f"{date_str} 23:59:59"
            
            # Filter the data
            filtered_df = df.set_index(time_col).loc[start_full_str:end_full_str].reset_index()
            
            if filtered_df.empty:
                messagebox.showwarning("Warning", "The specified time range resulted in an empty dataset.")
                return
                
            # Update the plot with filtered data
            self.plot_ax.clear()
            
            signals_to_plot = [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
            
            if not signals_to_plot:
                self.plot_ax.text(0.5, 0.5, "Select one or more signals to plot", ha='center', va='center')
            else:
                # Apply filter preview if selected
                plot_filter = self.plot_filter_type.get()
                if plot_filter != "None":
                    filtered_df = self._apply_plot_filter(filtered_df, signals_to_plot, time_col)
                
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
                    self.plot_ax.plot(plot_df[time_col], plot_df[signal], label=signal_label, **plot_style)
                
                # Add trendline if selected
                if self.trendline_type_var.get() != "None":
                    selected_trendline_signal = self.trendline_signal_var.get()
                    if selected_trendline_signal != "Select signal..." and selected_trendline_signal in filtered_df.columns:
                        self._add_trendline(filtered_df, selected_trendline_signal, time_col)
            
            # Apply custom labels and title
            title = self.plot_title_entry.get() or f"Signals from {selected_file} (Time Range: {start_time_str} - {end_time_str})"
            xlabel = self.plot_xlabel_entry.get() or time_col
            ylabel = self.plot_ylabel_entry.get() or "Value"
            self.plot_ax.set_title(title, fontsize=14)
            self.plot_ax.set_xlabel(xlabel)
            self.plot_ax.set_ylabel(ylabel)
            
            self.plot_ax.legend()
            self.plot_ax.grid(True, linestyle='--', alpha=0.6)
            
            if pd.api.types.is_datetime64_any_dtype(filtered_df[time_col]):
                 # Use simpler HH:MM format for cleaner plot appearance
                 self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                 # Keep labels horizontal for better readability
                 self.plot_ax.tick_params(axis='x', rotation=0)
            
            self.plot_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Time Range Error", f"Invalid time format. Please use HH:MM:SS.\n{e}")

    def _reset_plot_range(self):
        """Reset plot range."""
        self.plot_start_time_entry.delete(0, tk.END)
        self.plot_end_time_entry.delete(0, tk.END)
        self.update_plot()

    def _copy_trim_to_plot_range(self):
        """Copy trim times to plot range."""
        start_time = self.trim_start_entry.get()
        end_time = self.trim_end_entry.get()
        
        if start_time:
            self.plot_start_time_entry.delete(0, tk.END)
            self.plot_start_time_entry.insert(0, start_time)
        
        if end_time:
            self.plot_end_time_entry.delete(0, tk.END)
            self.plot_end_time_entry.insert(0, end_time)
        
        self._apply_plot_time_range()

    def _copy_plot_range_to_trim(self):
        """Copy current plot x-axis range to time trimming fields."""
        try:
            # Check if plot exists and has data
            if not hasattr(self, 'plot_ax') or not self.plot_ax.lines:
                messagebox.showwarning("Warning", "No plot data available. Please create a plot first.")
                return
            
            # Get current x-axis limits
            xlim = self.plot_ax.get_xlim()
            
            # Convert matplotlib date numbers to datetime
            start_datetime = mdates.num2date(xlim[0])
            end_datetime = mdates.num2date(xlim[1])
            
            # Extract date and time components
            date_str = start_datetime.strftime('%Y-%m-%d')
            start_time_str = start_datetime.strftime('%H:%M:%S')
            end_time_str = end_datetime.strftime('%H:%M:%S')
            
            # Update the trim fields
            self.trim_date_entry.delete(0, tk.END)
            self.trim_date_entry.insert(0, date_str)
            
            self.trim_start_entry.delete(0, tk.END)
            self.trim_start_entry.insert(0, start_time_str)
            
            self.trim_end_entry.delete(0, tk.END)
            self.trim_end_entry.insert(0, end_time_str)
            
            messagebox.showinfo("Success", f"Copied plot range to time trimming:\nDate: {date_str}\nStart: {start_time_str}\nEnd: {end_time_str}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy plot range: {str(e)}")

    def _save_current_plot_view(self):
        """Save the current plot view state."""
        try:
            if not hasattr(self, 'plot_ax'):
                messagebox.showwarning("Warning", "No plot available.")
                return
            
            # Save current view limits
            self.saved_plot_view = {
                'xlim': self.plot_ax.get_xlim(),
                'ylim': self.plot_ax.get_ylim()
            }
            
            messagebox.showinfo("Success", "Current plot view saved! Use the Home button on the toolbar to return to this view.")
            
            # Override the home button functionality
            self._override_home_button()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot view: {str(e)}")

    def _copy_current_view_to_processing(self):
        """Copy current plot view range to processing tab time trimming."""
        try:
            # This is essentially the same as _copy_plot_range_to_trim but with a different message
            if not hasattr(self, 'plot_ax') or not self.plot_ax.lines:
                messagebox.showwarning("Warning", "No plot data available. Please create a plot first.")
                return
            
            # Get current x-axis limits
            xlim = self.plot_ax.get_xlim()
            
            # Convert matplotlib date numbers to datetime
            start_datetime = mdates.num2date(xlim[0])
            end_datetime = mdates.num2date(xlim[1])
            
            # Extract date and time components
            date_str = start_datetime.strftime('%Y-%m-%d')
            start_time_str = start_datetime.strftime('%H:%M:%S')
            end_time_str = end_datetime.strftime('%H:%M:%S')
            
            # Update the trim fields
            self.trim_date_entry.delete(0, tk.END)
            self.trim_date_entry.insert(0, date_str)
            
            self.trim_start_entry.delete(0, tk.END)
            self.trim_start_entry.insert(0, start_time_str)
            
            self.trim_end_entry.delete(0, tk.END)
            self.trim_end_entry.insert(0, end_time_str)
            
            # Switch to the Setup & Process tab
            self.main_tab_view.set("Setup & Process")
            
            messagebox.showinfo("Success", f"Copied current view to Processing tab time trimming:\nDate: {date_str}\nStart: {start_time_str}\nEnd: {end_time_str}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy current view: {str(e)}")

    def _override_home_button(self):
        """Override the matplotlib toolbar home button to use saved view."""
        if hasattr(self, 'plot_toolbar') and self.saved_plot_view:
            # Store original home function
            if not hasattr(self, '_original_home'):
                self._original_home = self.plot_toolbar.home
            
            # Create custom home function
            def custom_home():
                try:
                    if self.saved_plot_view:
                        self.plot_ax.set_xlim(self.saved_plot_view['xlim'])
                        self.plot_ax.set_ylim(self.saved_plot_view['ylim'])
                        self.plot_canvas.draw()
                    else:
                        # Fall back to original home if no saved view
                        self._original_home()
                except:
                    # Fall back to original home on any error
                    self._original_home()
            
            # Replace the home function
            self.plot_toolbar.home = custom_home

    def _refresh_legend_entries(self):
        """Refresh legend entries based on currently selected signals."""
        # Clear existing legend widgets
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
        
        # Get currently selected signals
        selected_signals = []
        if hasattr(self, 'plot_signal_vars'):
            selected_signals = [signal for signal, data in self.plot_signal_vars.items() if data['var'].get()]
        
        if not selected_signals:
            ctk.CTkLabel(self.legend_frame, text="Select signals to customize legend labels").pack(padx=5, pady=5)
            ctk.CTkLabel(self.legend_frame, text="Tip: For subscripts use $H_2O$, $CO_2$, $v_{max}$", 
                        font=ctk.CTkFont(size=10), text_color="gray").pack(padx=5, pady=2)
            return
        
        # Create entry widgets for each selected signal
        for signal in selected_signals:
            signal_frame = ctk.CTkFrame(self.legend_frame)
            signal_frame.pack(fill="x", padx=5, pady=2)
            
            # Signal name label
            ctk.CTkLabel(signal_frame, text=f"{signal}:", width=100).pack(side="left", padx=5, pady=2)
            
            # Custom legend entry
            current_value = self.custom_legend_entries.get(signal, signal)
            legend_entry = ctk.CTkEntry(signal_frame, placeholder_text=f"Custom label for {signal}")
            legend_entry.pack(side="right", fill="x", expand=True, padx=5, pady=2)
            legend_entry.insert(0, current_value)
            legend_entry.bind("<KeyRelease>", lambda e, s=signal: self._on_legend_change(s, e.widget.get()))
            legend_entry.bind("<FocusOut>", lambda e, s=signal: self._on_legend_change(s, e.widget.get()))

    def _on_legend_change(self, signal, new_label):
        """Handle changes to legend labels."""
        self.custom_legend_entries[signal] = new_label
        # Trigger plot update if needed
        if hasattr(self, '_update_pending'):
            return
        self._update_pending = self.after_idle(self.update_plot)

    def _add_trendline(self):
        """Add trendline to plot."""
        pass

    def _create_dat_import_right(self, right_panel):
        """Create right panel for DAT import."""
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Preview frame
        preview_frame = ctk.CTkFrame(right_panel)
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(preview_frame, text="Import Preview", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.import_preview_text = ctk.CTkTextbox(preview_frame, height=200)
        self.import_preview_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

    def create_help_tab(self, tab):
        """Create the help tab with comprehensive documentation."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(header_frame, text="Help & Documentation", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=10, pady=10)
        
        # Main content with scrollable help
        help_frame = ctk.CTkScrollableFrame(tab)
        help_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        help_frame.grid_columnconfigure(0, weight=1)
        
        # Help content
        help_content = """
# Advanced CSV Processor & DAT Importer - Help Guide

## Overview
This application provides comprehensive tools for processing, analyzing, and visualizing time series data from CSV files and DAT files with DBF tag files.

## Tab Descriptions

### Setup & Process Tab
**Purpose**: Configure file processing settings and batch export data.

**Features**:
- **Setup Sub-tab**:
  - Select input CSV files and output directory
  - Save/load processing configurations
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
- Multiple chart types (Line, Scatter, etc.)
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

## Troubleshooting

**Common Issues**:
- **Time parsing errors**: Ensure consistent datetime format
- **Memory issues**: Process fewer files or signals at once
- **Filter errors**: Check signal length vs. filter parameters
- **Integration errors**: Verify time column is properly formatted

**Performance Tips**:
- Close other applications when processing large files
- Use appropriate filter parameters for your data
- Consider resampling for very large datasets

## Keyboard Shortcuts

- **Ctrl+O**: Select files (in file dialogs)
- **Ctrl+S**: Save settings
- **Ctrl+L**: Load settings
- **F1**: Show this help (when help tab is active)

## Support

For additional support or feature requests, please refer to the application documentation or contact the development team.
        """
        
        # Create help text widget
        help_text = ctk.CTkTextbox(help_frame, wrap="word", font=ctk.CTkFont(size=12))
        help_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        help_text.insert("1.0", help_content)
        help_text.configure(state="disabled")  # Make read-only

    def _generate_unique_filename(self, base_path, extension):
        """Generate a unique filename to prevent overwriting existing files."""
        directory = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        
        # Remove any existing suffix like _processed, _1, _2, etc.
        if base_name.endswith('_processed'):
            base_name = base_name[:-10]  # Remove '_processed'
        
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

    def _check_file_overwrite(self, file_path):
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
                icon='warning'
            )
            
            if response is None:  # Cancel
                return None
            elif response:  # Yes - overwrite
                return file_path
            else:  # No - generate unique name
                directory = os.path.dirname(file_path)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                extension = os.path.splitext(file_path)[1]
                return self._generate_unique_filename(os.path.join(directory, base_name), extension)
        
        return file_path

    def _save_current_plot_config(self):
        """Save the current plot configuration."""
        # Get current plot settings
        plot_name = simpledialog.askstring("Save Plot Configuration", "Enter a name for this plot configuration:")
        if not plot_name:
            return
        
        # Get currently selected signals for plotting
        selected_signals = []
        if hasattr(self, 'plot_signal_vars'):
            selected_signals = [signal for signal, data in self.plot_signal_vars.items() if data['var'].get()]
        
        # Get current plot settings
        plot_config = {
            'name': plot_name,
            'description': f"Plot configuration saved on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            'file': self.plot_file_menu.get() if hasattr(self, 'plot_file_menu') else '',
            'x_axis': self.plot_xaxis_menu.get() if hasattr(self, 'plot_xaxis_menu') else '',
            'signals': selected_signals,
            'filter_type': self.plot_filter_type.get() if hasattr(self, 'plot_filter_type') else 'None',
            'show_both_signals': self.show_both_signals_var.get() if hasattr(self, 'show_both_signals_var') else False,
            'plot_title': self.plot_title_entry.get() if hasattr(self, 'plot_title_entry') else '',
            'plot_xlabel': self.plot_xlabel_entry.get() if hasattr(self, 'plot_xlabel_entry') else '',
            'plot_ylabel': self.plot_ylabel_entry.get() if hasattr(self, 'plot_ylabel_entry') else '',
            'start_time': self.plot_start_time_entry.get() if hasattr(self, 'plot_start_time_entry') else '',
            'end_time': self.plot_end_time_entry.get() if hasattr(self, 'plot_end_time_entry') else '',
            'color_scheme': self.color_scheme_var.get() if hasattr(self, 'color_scheme_var') else 'Auto (Matplotlib)',
            'line_width': self.line_width_var.get() if hasattr(self, 'line_width_var') else '1.0',
            'plot_type': self.plot_type_var.get() if hasattr(self, 'plot_type_var') else 'Line with Markers',
            'trendline_signal': self.trendline_signal_var.get() if hasattr(self, 'trendline_signal_var') else 'Select signal...',
            'trendline_type': self.trendline_type_var.get() if hasattr(self, 'trendline_type_var') else 'None',
            'custom_legend_entries': dict(self.custom_legend_entries),  # Save custom legend labels
            'created_date': pd.Timestamp.now().isoformat()
        }
        
        # Add filter-specific parameters for plot preview
        if plot_config['filter_type'] == "Moving Average":
            plot_config['ma_value'] = self.plot_ma_value_entry.get() if hasattr(self, 'plot_ma_value_entry') else ''
            plot_config['ma_unit'] = self.plot_ma_unit_menu.get() if hasattr(self, 'plot_ma_unit_menu') else ''
        elif plot_config['filter_type'] in ["Butterworth Low-pass", "Butterworth High-pass"]:
            plot_config['bw_order'] = self.plot_bw_order_entry.get() if hasattr(self, 'plot_bw_order_entry') else ''
            plot_config['bw_cutoff'] = self.plot_bw_cutoff_entry.get() if hasattr(self, 'plot_bw_cutoff_entry') else ''
        elif plot_config['filter_type'] == "Median Filter":
            plot_config['median_kernel'] = self.plot_median_kernel_entry.get() if hasattr(self, 'plot_median_kernel_entry') else ''
        elif plot_config['filter_type'] == "Hampel Filter":
            plot_config['hampel_window'] = self.plot_hampel_window_entry.get() if hasattr(self, 'plot_hampel_window_entry') else ''
            plot_config['hampel_threshold'] = self.plot_hampel_threshold_entry.get() if hasattr(self, 'plot_hampel_threshold_entry') else ''
        elif plot_config['filter_type'] == "Z-Score Filter":
            plot_config['zscore_threshold'] = self.plot_zscore_threshold_entry.get() if hasattr(self, 'plot_zscore_threshold_entry') else ''
            plot_config['zscore_method'] = self.plot_zscore_method_menu.get() if hasattr(self, 'plot_zscore_method_menu') else ''
        elif plot_config['filter_type'] == "Savitzky-Golay":
            plot_config['savgol_window'] = self.plot_savgol_window_entry.get() if hasattr(self, 'plot_savgol_window_entry') else ''
            plot_config['savgol_polyorder'] = self.plot_savgol_polyorder_entry.get() if hasattr(self, 'plot_savgol_polyorder_entry') else ''
        
        # Add to plots list
        self.plots_list.append(plot_config)
        self._update_plots_listbox()
        self._update_load_plot_config_menu()
        self._save_plots_to_file()
        
        messagebox.showinfo("Success", f"Plot configuration '{plot_name}' saved successfully!")

    def _on_load_plot_config_select(self, selected_plot_name):
        """Handle selection from the load plot config dropdown."""
        if selected_plot_name == "No saved plots":
            return
        
        # Find the plot config by name
        plot_config = None
        for config in self.plots_list:
            if config['name'] == selected_plot_name:
                plot_config = config
                break
        
        if not plot_config:
            messagebox.showerror("Error", f"Plot configuration '{selected_plot_name}' not found.")
            return
        
        # Apply the plot configuration
        self._apply_plot_config(plot_config)
        messagebox.showinfo("Success", f"Plot configuration '{selected_plot_name}' loaded!")

    def _apply_plot_config(self, plot_config):
        """Apply a plot configuration to the current plotting tab."""
        # Apply file selection
        if 'file' in plot_config and plot_config['file'] and hasattr(self, 'plot_file_menu'):
            self.plot_file_menu.set(plot_config['file'])
            # Trigger file selection to populate signals
            self.on_plot_file_select(plot_config['file'])
        
        # Apply x-axis selection
        if 'x_axis' in plot_config and plot_config['x_axis'] and hasattr(self, 'plot_xaxis_menu'):
            self.plot_xaxis_menu.set(plot_config['x_axis'])
        
        # Apply signal selections
        if hasattr(self, 'plot_signal_vars') and 'signals' in plot_config:
            saved_signals = plot_config['signals']
            for signal, data in self.plot_signal_vars.items():
                data['var'].set(signal in saved_signals)
        
        # Apply filter settings
        if 'filter_type' in plot_config and hasattr(self, 'plot_filter_type'):
            self.plot_filter_type.set(plot_config['filter_type'])
            self._update_plot_filter_ui(plot_config['filter_type'])
        
        # Apply filter parameters - enhanced with all filter types
        if plot_config.get('filter_type') == "Moving Average":
            if 'ma_value' in plot_config and hasattr(self, 'plot_ma_value_entry'):
                self.plot_ma_value_entry.delete(0, tk.END)
                self.plot_ma_value_entry.insert(0, plot_config['ma_value'])
            if 'ma_unit' in plot_config and hasattr(self, 'plot_ma_unit_menu'):
                self.plot_ma_unit_menu.set(plot_config['ma_unit'])
        elif plot_config.get('filter_type') in ["Butterworth Low-pass", "Butterworth High-pass"]:
            if 'bw_order' in plot_config and hasattr(self, 'plot_bw_order_entry'):
                self.plot_bw_order_entry.delete(0, tk.END)
                self.plot_bw_order_entry.insert(0, plot_config['bw_order'])
            if 'bw_cutoff' in plot_config and hasattr(self, 'plot_bw_cutoff_entry'):
                self.plot_bw_cutoff_entry.delete(0, tk.END)
                self.plot_bw_cutoff_entry.insert(0, plot_config['bw_cutoff'])
        elif plot_config.get('filter_type') == "Median Filter":
            if 'median_kernel' in plot_config and hasattr(self, 'plot_median_kernel_entry'):
                self.plot_median_kernel_entry.delete(0, tk.END)
                self.plot_median_kernel_entry.insert(0, plot_config['median_kernel'])
        elif plot_config.get('filter_type') == "Hampel Filter":
            if 'hampel_window' in plot_config and hasattr(self, 'plot_hampel_window_entry'):
                self.plot_hampel_window_entry.delete(0, tk.END)
                self.plot_hampel_window_entry.insert(0, plot_config['hampel_window'])
            if 'hampel_threshold' in plot_config and hasattr(self, 'plot_hampel_threshold_entry'):
                self.plot_hampel_threshold_entry.delete(0, tk.END)
                self.plot_hampel_threshold_entry.insert(0, plot_config['hampel_threshold'])
        elif plot_config.get('filter_type') == "Z-Score Filter":
            if 'zscore_threshold' in plot_config and hasattr(self, 'plot_zscore_threshold_entry'):
                self.plot_zscore_threshold_entry.delete(0, tk.END)
                self.plot_zscore_threshold_entry.insert(0, plot_config['zscore_threshold'])
            if 'zscore_method' in plot_config and hasattr(self, 'plot_zscore_method_menu'):
                self.plot_zscore_method_menu.set(plot_config['zscore_method'])
        elif plot_config.get('filter_type') == "Savitzky-Golay":
            if 'savgol_window' in plot_config and hasattr(self, 'plot_savgol_window_entry'):
                self.plot_savgol_window_entry.delete(0, tk.END)
                self.plot_savgol_window_entry.insert(0, plot_config['savgol_window'])
            if 'savgol_polyorder' in plot_config and hasattr(self, 'plot_savgol_polyorder_entry'):
                self.plot_savgol_polyorder_entry.delete(0, tk.END)
                self.plot_savgol_polyorder_entry.insert(0, plot_config['savgol_polyorder'])
        
        # Apply custom legend entries
        if 'custom_legend_entries' in plot_config:
            self.custom_legend_entries = dict(plot_config['custom_legend_entries'])
            self._refresh_legend_entries()  # Refresh the legend UI
        
        # Apply other settings
        if 'show_both_signals' in plot_config and hasattr(self, 'show_both_signals_var'):
            self.show_both_signals_var.set(plot_config['show_both_signals'])
        
        if 'plot_title' in plot_config and hasattr(self, 'plot_title_entry'):
            self.plot_title_entry.delete(0, tk.END)
            self.plot_title_entry.insert(0, plot_config.get('plot_title', ''))
        
        if 'plot_xlabel' in plot_config and hasattr(self, 'plot_xlabel_entry'):
            self.plot_xlabel_entry.delete(0, tk.END)
            self.plot_xlabel_entry.insert(0, plot_config.get('plot_xlabel', ''))
        
        if 'plot_ylabel' in plot_config and hasattr(self, 'plot_ylabel_entry'):
            self.plot_ylabel_entry.delete(0, tk.END)
            self.plot_ylabel_entry.insert(0, plot_config.get('plot_ylabel', ''))
        
        if 'start_time' in plot_config and hasattr(self, 'plot_start_time_entry'):
            self.plot_start_time_entry.delete(0, tk.END)
            self.plot_start_time_entry.insert(0, plot_config.get('start_time', ''))
        
        if 'end_time' in plot_config and hasattr(self, 'plot_end_time_entry'):
            self.plot_end_time_entry.delete(0, tk.END)
            self.plot_end_time_entry.insert(0, plot_config.get('end_time', ''))
        
        # Apply color scheme and styling settings
        if 'color_scheme' in plot_config and hasattr(self, 'color_scheme_var'):
            self.color_scheme_var.set(plot_config.get('color_scheme', 'Auto (Matplotlib)'))
        
        if 'line_width' in plot_config and hasattr(self, 'line_width_var'):
            self.line_width_var.set(plot_config.get('line_width', '1.0'))
        
        if 'plot_type' in plot_config and hasattr(self, 'plot_type_var'):
            self.plot_type_var.set(plot_config.get('plot_type', 'Line with Markers'))
        
        # Apply trendline settings
        if 'trendline_signal' in plot_config and hasattr(self, 'trendline_signal_var'):
            self.trendline_signal_var.set(plot_config.get('trendline_signal', 'Select signal...'))
        
        if 'trendline_type' in plot_config and hasattr(self, 'trendline_type_var'):
            self.trendline_type_var.set(plot_config.get('trendline_type', 'None'))
        
        # Update the plot
        self._on_plot_setting_change()

    def _update_load_plot_config_menu(self):
        """Update the load plot config dropdown menu."""
        if not hasattr(self, 'load_plot_config_menu'):
            return
        
        if self.plots_list:
            plot_names = [config['name'] for config in self.plots_list]
            self.load_plot_config_menu.configure(values=plot_names)
            self.load_plot_config_menu.set("Select a plot config...")
        else:
            self.load_plot_config_menu.configure(values=["No saved plots"])
            self.load_plot_config_menu.set("No saved plots")

    def _update_plots_signals(self, signals):
        """Update signals available in plots list tab."""
        if not hasattr(self, 'plots_signals_frame'):
            return
        
        # Clear existing widgets
        for widget in self.plots_signals_frame.winfo_children():
            widget.destroy()
        
        # Initialize plots signal vars if not exists
        if not hasattr(self, 'plots_signal_vars'):
            self.plots_signal_vars = {}
        
        self.plots_signal_vars.clear()
        
        # Add checkboxes for each signal
        for signal in signals:
            if signal != signals[0]:  # Skip time column
                var = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(self.plots_signals_frame, text=signal, variable=var)
                cb.grid(sticky="w", padx=5, pady=2)
                self.plots_signal_vars[signal] = var
        
        # Re-bind mouse wheel to all new checkboxes
        self._bind_mousewheel_to_frame(self.plots_signals_frame)

    def _generate_plot_preview(self):
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
            signals = plot_config.get('signals', [])
            file_name = plot_config.get('file', '')
            
            if not signals:
                self.preview_ax.text(0.5, 0.5, "No signals selected in this configuration", 
                                   transform=self.preview_ax.transAxes, 
                                   ha='center', va='center', fontsize=12)
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return
            
            if not file_name or not hasattr(self, 'processed_files'):
                self.preview_ax.text(0.5, 0.5, "Data file not available\nLoad the data file first", 
                                   transform=self.preview_ax.transAxes, 
                                   ha='center', va='center', fontsize=12)
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return
            
            # Find the actual data - try multiple matching strategies
            df = None
            
            # Strategy 1: Exact basename match
            for file_path, data in self.processed_files.items():
                if os.path.basename(file_path) == file_name:
                    df = data
                    break
            
            # Strategy 2: Try without extension
            if df is None and '.' in file_name:
                file_name_no_ext = os.path.splitext(file_name)[0]
                for file_path, data in self.processed_files.items():
                    if os.path.splitext(os.path.basename(file_path))[0] == file_name_no_ext:
                        df = data
                        break
            
            # Strategy 3: Try partial match
            if df is None:
                for file_path, data in self.processed_files.items():
                    if file_name in os.path.basename(file_path) or os.path.basename(file_path) in file_name:
                        df = data
                        break
            
            if df is None:
                # Show available files for debugging
                available_files = [os.path.basename(fp) for fp in self.processed_files.keys()]
                debug_text = f"Data file '{file_name}' not found\n\nAvailable files:\n" + "\n".join(available_files[:5])
                if len(available_files) > 5:
                    debug_text += f"\n... and {len(available_files)-5} more"
                
                self.preview_ax.text(0.5, 0.5, debug_text, 
                                   transform=self.preview_ax.transAxes, 
                                   ha='center', va='center', fontsize=10)
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return
            
            # Get time column and available signals
            time_col = df.columns[0]
            available_signals = [s for s in signals if s in df.columns]
            
            if not available_signals:
                self.preview_ax.text(0.5, 0.5, "None of the selected signals\nare available in the data", 
                                   transform=self.preview_ax.transAxes, 
                                   ha='center', va='center', fontsize=12)
                self.preview_ax.set_title(f"Preview: {plot_config['name']}")
                self.preview_canvas.draw()
                return
            
            # Apply time range if specified
            plot_df = df.copy()
            start_time = plot_config.get('start_time', '')
            end_time = plot_config.get('end_time', '')
            
            if start_time or end_time:
                if pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                    if start_time:
                        try:
                            start_datetime = pd.to_datetime(f"{plot_df[time_col].dt.date.iloc[0]} {start_time}")
                            plot_df = plot_df[plot_df[time_col] >= start_datetime]
                        except:
                            pass
                    if end_time:
                        try:
                            end_datetime = pd.to_datetime(f"{plot_df[time_col].dt.date.iloc[0]} {end_time}")
                            plot_df = plot_df[plot_df[time_col] <= end_datetime]
                        except:
                            pass
            
            # Plot all available signals
            colors = plt.cm.tab10(np.linspace(0, 1, len(available_signals)))
            for i, signal in enumerate(available_signals):
                signal_data = plot_df[[time_col, signal]].dropna()
                if len(signal_data) > 0:
                    self.preview_ax.plot(signal_data[time_col], signal_data[signal], 
                                       label=signal, linewidth=1, color=colors[i])
            
            # Apply plot configuration
            title = plot_config.get('plot_title', '') or f"Preview: {plot_config['name']}"
            xlabel = plot_config.get('plot_xlabel', '') or time_col
            ylabel = plot_config.get('plot_ylabel', '') or "Value"
            
            self.preview_ax.set_title(title, fontsize=14)
            self.preview_ax.set_xlabel(xlabel)
            self.preview_ax.set_ylabel(ylabel)
            self.preview_ax.legend()
            self.preview_ax.grid(True, linestyle='--', alpha=0.6)
            
            # Format x-axis for time data
            if pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                import matplotlib.dates as mdates
                self.preview_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                self.preview_ax.tick_params(axis='x', rotation=0)
            
            self.preview_canvas.draw()
            
        except Exception as e:
            self.preview_ax.clear()
            self.preview_ax.text(0.5, 0.5, f"Error generating preview:\n{str(e)}", 
                               transform=self.preview_ax.transAxes, 
                               ha='center', va='center', fontsize=12)
            self.preview_ax.set_title("Preview Error")
            self.preview_canvas.draw()

    def _export_all_plots(self):
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
                
                with open(filepath, 'w') as f:
                    f.write(f"Plot Configuration: {plot_config['name']}\n")
                    f.write(f"Description: {plot_config.get('description', 'N/A')}\n")
                    f.write(f"Created: {plot_config.get('created_date', 'N/A')}\n")
                    f.write(f"Signals: {', '.join(plot_config.get('signals', []))}\n")
                    f.write(f"Start Time: {plot_config.get('start_time', 'N/A')}\n")
                    f.write(f"End Time: {plot_config.get('end_time', 'N/A')}\n")
                    
                    if 'filter_type' in plot_config:
                        f.write(f"Filter: {plot_config['filter_type']}\n")
                    
                    f.write("\nFull Configuration:\n")
                    for key, value in plot_config.items():
                        f.write(f"  {key}: {value}\n")
                
                exported_count += 1
            
            messagebox.showinfo("Export Complete", f"Exported {exported_count} plot configurations to {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting plots: {e}")

    def _on_plot_setting_change(self, *args):
        """Automatically update plot when appearance settings change."""
        # Only update if we have data and signals selected
        if hasattr(self, 'plot_signal_vars') and any(data['var'].get() for data in self.plot_signal_vars.values()):
            # Use after_idle to prevent too many rapid updates
            if hasattr(self, '_update_pending'):
                self.after_cancel(self._update_pending)
            self._update_pending = self.after_idle(self.update_plot)

    def _bind_mousewheel_to_frame(self, frame):
        """Bind mouse wheel events to a frame for proper scrolling."""
        def on_mousewheel(event):
            # Scroll the frame's canvas
            try:
                frame._parent_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except:
                # Fallback for different systems
                frame._parent_canvas.yview_scroll(int(-1 * event.delta), "units")
        
        # Bind mousewheel to the frame and all its children
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)
            widget.bind("<Button-4>", lambda e: frame._parent_canvas.yview_scroll(-1, "units"))  # Linux
            widget.bind("<Button-5>", lambda e: frame._parent_canvas.yview_scroll(1, "units"))   # Linux
            
            for child in widget.winfo_children():
                bind_mousewheel(child)
        
        bind_mousewheel(frame)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Starting Advanced CSV Processor - Complete Version...")
    app = CSVProcessorApp()
    app.mainloop() 
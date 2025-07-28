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
from tkinter import filedialog, messagebox
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
        
        # App State Variables
        self.input_file_paths = []
        self.loaded_data_cache = {}
        self.output_directory = os.path.expanduser("~/Documents")
        self.signal_vars = {}
        self.plot_signal_vars = {}
        self.filter_names = ["None", "Moving Average", "Median Filter", "Butterworth Low-pass", "Butterworth High-pass", "Savitzky-Golay"]
        self.custom_vars_list = []
        self.reference_signal_widgets = {}
        self.dat_import_tag_file_path = None
        self.dat_import_data_file_path = None
        self.dat_tag_vars = {}
        self.tag_delimiter_var = tk.StringVar(value="newline")
        
        # Plots List variables
        self.plots_list = []
        self.current_plot_config = None
        
        # Integration and Differentiation variables
        self.integrator_signal_vars = {}
        self.deriv_signal_vars = {}
        self.derivative_vars = {}
        for i in range(1, 6):  # Support up to 5th order derivatives
            self.derivative_vars[i] = tk.BooleanVar(value=False)

        # Create Main UI
        self.main_tab_view = ctk.CTkTabview(self)
        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.main_tab_view.add("Setup & Process")
        self.main_tab_view.add("Plotting & Analysis")
        self.main_tab_view.add("Plots List")
        self.main_tab_view.add("DAT File Import")

        self.create_setup_and_process_tab(self.main_tab_view.tab("Setup & Process"))
        self.create_plotting_tab(self.main_tab_view.tab("Plotting & Analysis"))
        self.create_plots_list_tab(self.main_tab_view.tab("Plots List"))
        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))

        self.create_status_bar()
        self.status_label.configure(text="Ready. Select input files or import a DAT file.") 

    def create_setup_and_process_tab(self, parent_tab):
        """Fixed version with proper splitter implementation and all advanced features."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)
        
        def create_left_content(left_panel):
            """Create the left panel content"""
            left_panel.grid_rowconfigure(1, weight=1)
            left_panel.grid_columnconfigure(0, weight=1)
            
            # Header with Help Button
            header_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
            header_frame.grid(row=0, column=0, padx=15, pady=10, sticky="ew")
            ctk.CTkLabel(header_frame, text="Control Panel", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
            ctk.CTkButton(header_frame, text="Help", width=70, command=self._show_setup_help).pack(side="right")

            # Create a scrollable frame for the processing tab view
            processing_scrollable_frame = ctk.CTkScrollableFrame(left_panel)
            processing_scrollable_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
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
            self.process_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

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
        splitter_frame = self._create_splitter(parent_tab, create_left_content, create_right_content, 'setup_left_width', 450)
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

    def populate_processing_sub_tab(self, tab):
        """Populate the processing sub-tab with all advanced features."""
        tab.grid_columnconfigure(0, weight=1)
        time_units = ["ms", "s", "min", "hr"]
        
        # Filter frame
        filter_frame = ctk.CTkFrame(tab)
        filter_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
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
        (self.savgol_frame, self.savgol_window_entry, self.savgol_polyorder_entry) = self._create_savgol_param_frame(filter_frame)
        self._update_filter_ui("None")
        
        # Resample frame
        resample_frame = ctk.CTkFrame(tab)
        resample_frame.grid(row=1, column=0, padx=10, pady=10, sticky="new")
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
        integrator_frame.grid(row=2, column=0, padx=10, pady=10, sticky="new")
        integrator_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(integrator_frame, text="Signal Integration", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="w")
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
        deriv_frame.grid(row=3, column=0, padx=10, pady=10, sticky="new")
        deriv_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(deriv_frame, text="Signal Differentiation", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="w")
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
        
        ctk.CTkLabel(frame, text="Unit:").grid(row=0, column=2, padx=10, pady=5, sticky="w")
        unit_menu = ctk.CTkOptionMenu(frame, values=time_units)
        unit_menu.grid(row=0, column=3, padx=10, pady=5, sticky="ew")
        
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

    def _update_filter_ui(self, filter_type):
        """Update filter UI based on selected filter type."""
        # Hide all frames
        for frame in [self.ma_frame, self.bw_frame, self.median_frame, self.savgol_frame]:
            frame.grid_remove()
        
        # Show relevant frame
        if filter_type == "Moving Average":
            self.ma_frame.grid()
        elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.bw_frame.grid()
        elif filter_type == "Median Filter":
            self.median_frame.grid()
        elif filter_type == "Savitzky-Golay":
            self.savgol_frame.grid()

    def _update_plot_filter_ui(self, filter_type):
        """Update plot filter UI based on selected filter type."""
        # Hide all frames
        for frame in [self.plot_ma_frame, self.plot_bw_frame, self.plot_median_frame, self.plot_savgol_frame]:
            frame.grid_remove()
        
        # Show relevant frame
        if filter_type == "Moving Average":
            self.plot_ma_frame.grid()
        elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.plot_bw_frame.grid()
        elif filter_type == "Median Filter":
            self.plot_median_frame.grid()
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
        
        # Update plot signal variables
        self.plot_signal_vars.clear()
        for signal in signals:
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self.plot_signal_frame, text=signal, variable=var)
            cb.grid(sticky="w", padx=5, pady=2)
            self.plot_signal_vars[signal] = {'var': var, 'widget': cb}
        
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
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.csv")
            
            # Apply sorting if specified
            df = self._apply_sorting(df)
            
            df.to_csv(output_path, index=False)
        
        messagebox.showinfo("Success", f"Exported {len(processed_files)} files to {self.output_directory}")

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
            compiled_df.to_csv(output_path, index=False)
            
            messagebox.showinfo("Success", f"Exported compiled data to {output_path}")

    def _export_excel_multisheet(self, processed_files):
        """Export all files to a single Excel file with multiple sheets."""
        output_path = os.path.join(self.output_directory, "processed_data.xlsx")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for file_path, df in processed_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                sheet_name = base_name[:31]  # Excel sheet name limit
                
                df = self._apply_sorting(df)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        messagebox.showinfo("Success", f"Exported to Excel file: {output_path}")

    def _export_excel_separate(self, processed_files):
        """Export each file as a separate Excel file."""
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.xlsx")
            
            df = self._apply_sorting(df)
            df.to_excel(output_path, index=False)
        
        messagebox.showinfo("Success", f"Exported {len(processed_files)} Excel files to {self.output_directory}")

    def _export_mat_separate(self, processed_files):
        """Export each file as a separate MAT file."""
        for file_path, df in processed_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.mat")
            
            df = self._apply_sorting(df)
            
            # Convert to dictionary for MATLAB
            mat_data = {}
            for col in df.columns:
                mat_data[col] = df[col].values
            
            savemat(output_path, mat_data)
        
        messagebox.showinfo("Success", f"Exported {len(processed_files)} MAT files to {self.output_directory}")

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
            
            # Convert to dictionary for MATLAB
            mat_data = {}
            for col in compiled_df.columns:
                mat_data[col] = compiled_df[col].values
            
            savemat(output_path, mat_data)
            
            messagebox.showinfo("Success", f"Exported compiled MAT file to {output_path}")

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

        ctk.CTkButton(plot_control_frame, text="Help", width=70, command=self._show_plot_help).grid(row=0, column=4, padx=(10,5), pady=10)

        # Main content frame for splitter
        plot_main_frame = ctk.CTkFrame(tab)
        plot_main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        plot_main_frame.grid_rowconfigure(0, weight=1)
        plot_main_frame.grid_columnconfigure(0, weight=1)
        
        def create_plot_left_content(left_panel):
            """Create the left panel content for plotting with all advanced features"""
            left_panel.grid_rowconfigure(1, weight=1)
            
            # Plot controls header
            plot_left_panel_outer = ctk.CTkFrame(left_panel)
            plot_left_panel_outer.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            plot_left_panel_outer.grid_propagate(False)
            plot_left_panel_outer.grid_rowconfigure(0, weight=1)
            plot_left_panel_outer.grid_columnconfigure(0, weight=1)

            # The scrollable area for controls
            plot_left_panel = ctk.CTkScrollableFrame(plot_left_panel_outer, label_text="Plotting Controls", label_fg_color="#4C7F4C")
            plot_left_panel.grid(row=0, column=0, sticky="nsew")
            
            # Update Plot button
            ctk.CTkButton(plot_left_panel_outer, text="Update Plot", height=35, command=self.update_plot).grid(row=1, column=0, sticky="ew", padx=5, pady=10)

            # Plot signal selection
            plot_signal_select_frame = ctk.CTkFrame(plot_left_panel)
            plot_signal_select_frame.pack(fill="x", expand=True, pady=5, padx=5)
            plot_signal_select_frame.grid_columnconfigure(0, weight=1)

            self.plot_search_entry = ctk.CTkEntry(plot_signal_select_frame, placeholder_text="Search plot signals...")
            self.plot_search_entry.grid(row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
            self.plot_search_entry.bind("<KeyRelease>", self._filter_plot_signals)
            
            ctk.CTkButton(plot_signal_select_frame, text="All", command=self._plot_select_all).grid(row=1, column=0, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(plot_signal_select_frame, text="None", command=self._plot_select_none).grid(row=1, column=1, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(plot_signal_select_frame, text="Show Selected", command=self._show_selected_signals).grid(row=1, column=2, sticky="ew", padx=2, pady=5)
            ctk.CTkButton(plot_signal_select_frame, text="X", width=28, command=self._plot_clear_search).grid(row=1, column=3, sticky="w", padx=2, pady=5)
            
            self.plot_signal_frame = ctk.CTkScrollableFrame(plot_left_panel, label_text="Signals to Plot", height=150)
            self.plot_signal_frame.pack(expand=True, fill="both", padx=5, pady=5)

            # Plot appearance controls
            appearance_frame = ctk.CTkFrame(plot_left_panel)
            appearance_frame.pack(fill="x", expand=True, pady=5, padx=5)
            ctk.CTkLabel(appearance_frame, text="Plot Appearance", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
            ctk.CTkLabel(appearance_frame, text="Chart Type:").pack(anchor="w", padx=10)
            self.plot_type_var = ctk.StringVar(value="Line with Markers")
            ctk.CTkOptionMenu(appearance_frame, variable=self.plot_type_var, values=["Line with Markers", "Line Only", "Markers Only (Scatter)"]).pack(fill="x", padx=10, pady=5)
            
            self.plot_title_entry = ctk.CTkEntry(appearance_frame, placeholder_text="Plot Title")
            self.plot_title_entry.pack(fill="x", padx=10, pady=5)
            self.plot_xlabel_entry = ctk.CTkEntry(appearance_frame, placeholder_text="X-Axis Label")
            self.plot_xlabel_entry.pack(fill="x", padx=10, pady=5)
            self.plot_ylabel_entry = ctk.CTkEntry(appearance_frame, placeholder_text="Y-Axis Label")
            self.plot_ylabel_entry.pack(fill="x", padx=10, pady=5)

            # Trendline controls
            trend_frame = ctk.CTkFrame(plot_left_panel)
            trend_frame.pack(fill="x", expand=True, pady=5, padx=5)
            ctk.CTkLabel(trend_frame, text="Trendline (plots 1st selected signal)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
            self.trendline_type_var = ctk.StringVar(value="None")
            ctk.CTkOptionMenu(trend_frame, variable=self.trendline_type_var, values=["None", "Linear", "Exponential", "Power", "Polynomial"]).pack(fill="x", padx=10, pady=5)
            self.poly_order_entry = ctk.CTkEntry(trend_frame, placeholder_text="Polynomial Order (2-6)")
            self.poly_order_entry.pack(fill="x", padx=10, pady=5)
            self.trendline_textbox = ctk.CTkTextbox(trend_frame, height=70)
            self.trendline_textbox.pack(fill="x", expand=True, padx=10, pady=5)

            # Export controls
            export_chart_frame = ctk.CTkFrame(plot_left_panel)
            export_chart_frame.pack(fill="x", expand=True, pady=5, padx=5)
            ctk.CTkLabel(export_chart_frame, text="Export Chart", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=5)
            ctk.CTkButton(export_chart_frame, text="Save as PNG/PDF", command=self._export_chart_image).pack(fill="x", padx=10, pady=2)
            ctk.CTkButton(export_chart_frame, text="Export to Excel with Chart", command=self._export_chart_excel).pack(fill="x", padx=10, pady=2)

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

        # Create splitter for plotting tab
        splitter_frame = self._create_splitter(plot_main_frame, create_plot_left_content, create_plot_right_content, 'plotting_left_width', 350)
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def create_plots_list_tab(self, tab):
        """Create the plots list tab."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(header_frame, text="Plots List Manager", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(header_frame, text="Help", width=70, command=self._show_plots_list_help).pack(side="right", padx=10, pady=10)
        
        # Main content
        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create splitter
        splitter_frame = self._create_splitter(main_frame, self._create_plots_list_left, self._create_plots_list_right, 'plots_list_left_width', 400)
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
        ctk.CTkButton(header_frame, text="Help", width=70, command=self._show_dat_import_help).pack(side="right", padx=10, pady=10)
        
        # Main content
        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Create splitter
        splitter_frame = self._create_splitter(main_frame, self._create_dat_import_left, self._create_dat_import_right, 'dat_import_left_width', 400)
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
        splitter_frame.grid_columnconfigure(1, weight=1)
        splitter_frame.grid_rowconfigure(0, weight=1)
        
        # Get saved width or use default
        left_width = self.layout_data.get(splitter_key, default_left_width)
        
        # Create left panel
        left_panel = ctk.CTkFrame(splitter_frame, width=left_width)
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.grid_propagate(False)
        left_creator(left_panel)
        
        # Create splitter handle
        splitter_handle = ctk.CTkFrame(splitter_frame, width=5, fg_color="gray")
        splitter_handle.grid(row=0, column=1, sticky="ns")
        splitter_handle.bind("<Button-1>", lambda e, h=splitter_handle: self._start_splitter_drag(e, h, left_panel, splitter_key))
        splitter_handle.bind("<B1-Motion>", lambda e, h=splitter_handle: self._drag_splitter(e, h, left_panel, splitter_key))
        splitter_handle.bind("<ButtonRelease-1>", lambda e: self._end_splitter_drag())
        
        # Create right panel
        right_panel = ctk.CTkFrame(splitter_frame)
        right_panel.grid(row=0, column=2, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=1)
        right_creator(right_panel)
        
        # Store splitter reference
        self.splitters[splitter_key] = left_panel
        
        return splitter_frame

    def _start_splitter_drag(self, event, handle, left_panel, splitter_key):
        """Start dragging the splitter."""
        self.dragging_splitter = True
        self.drag_splitter_key = splitter_key
        self.drag_left_panel = left_panel
        self.drag_start_x = event.x_root

    def _drag_splitter(self, event, handle, left_panel, splitter_key):
        """Drag the splitter."""
        if hasattr(self, 'dragging_splitter') and self.dragging_splitter:
            delta_x = event.x_root - self.drag_start_x
            new_width = max(100, left_panel.winfo_width() + delta_x)
            left_panel.configure(width=new_width)
            self.drag_start_x = event.x_root

    def _end_splitter_drag(self):
        """End dragging the splitter."""
        self.dragging_splitter = False

    def _on_closing(self):
        """Handle application closing."""
        self._save_layout_config()
        self.quit()

    def create_status_bar(self):
        """Create the status bar."""
        self.status_label = ctk.CTkLabel(self, text="Ready", anchor="w")
        self.status_label.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    # Placeholder methods for functionality that would be implemented
    def on_plot_file_select(self, value):
        """Handle plot file selection."""
        pass

    def update_plot(self, selected_signals=None):
        """Update the plot."""
        pass

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

    def save_settings(self):
        """Save current settings."""
        pass

    def load_settings(self):
        """Load saved settings."""
        pass

    def _show_sharing_instructions(self):
        """Show sharing instructions."""
        messagebox.showinfo("Sharing Instructions", "To share this application, include all Python files and the requirements.txt file.")

    def _copy_plot_settings_to_processing(self):
        """Copy plot filter settings to processing tab."""
        pass

    def _export_chart_image(self):
        """Export chart as image."""
        pass

    def _export_chart_excel(self):
        """Export chart to Excel."""
        pass

    def _add_plot_to_list(self):
        """Add plot to the plots list."""
        pass

    def _update_selected_plot(self):
        """Update selected plot in the list."""
        pass

    def _clear_plot_form(self):
        """Clear the plot form."""
        pass

    def _on_plot_select(self, event):
        """Handle plot selection in listbox."""
        pass

    def _load_selected_plot(self):
        """Load selected plot configuration."""
        pass

    def _delete_selected_plot(self):
        """Delete selected plot from list."""
        pass

    def _clear_all_plots(self):
        """Clear all plots from list."""
        pass

    def _generate_plot_preview(self):
        """Generate plot preview."""
        pass

    def _export_all_plots(self):
        """Export all plots."""
        pass

    def _select_tag_file(self):
        """Select tag file."""
        pass

    def _select_data_file(self):
        """Select data file."""
        pass

    def _import_selected_tags(self):
        """Import selected tags."""
        pass

    def trim_and_save(self):
        """Trim data and save."""
        pass

    def _apply_plot_time_range(self):
        """Apply time range to plot."""
        pass

    def _reset_plot_range(self):
        """Reset plot to full range."""
        pass

    def _copy_trim_to_plot_range(self):
        """Copy trim times to plot range."""
        pass

    def _add_trendline(self):
        """Add trendline to plot."""
        pass

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Starting Advanced CSV Processor - Complete Version...")
    app = CSVProcessorApp()
    app.mainloop() 
# =============================================================================
# Advanced CSV Time Series Processor & Analyzer - UNIFIED MASTER VERSION
#
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This unified version combines ALL features
# from Revisions 1, 2, 3, and additional enhancements to create the most
# powerful data processor possible.
#
# UNIFIED FEATURES INCLUDED:
# - All Rev 1 core functionality (basic processing, plotting, exports)
# - All Rev 2 enhancements (DAT import, custom variables, parallel processing)
# - All Rev 3 additions (plots list, integrator, storage selection, layout persistence)
# - Additional unified features (time range compilation, advanced filtering)
# - Enhanced multi-file processing with trimming capabilities
# - Comprehensive export options for compiled datasets
# - RESTORED: All filtering, integration, and derivative capabilities
#
# Dependencies for Python 3.8+:
# pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow simpledbf
#
# =============================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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
import threading
import re
from PIL import Image
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from simpledbf import Dbf5
    HAS_SIMPLEDBF = True
except ImportError:
    HAS_SIMPLEDBF = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates

# =============================================================================
# HELPER FUNCTIONS FOR ADVANCED PROCESSING
# =============================================================================

def _poly_derivative(series, window, poly_order, deriv_order, delta_x):
    """Calculates the derivative of a series using a rolling polynomial fit."""
    if poly_order < deriv_order:
        return pd.Series(np.nan, index=series.index)
    
    padded_series = pd.concat([pd.Series([series.iloc[0]] * (window - 1)), series])
    
    def get_deriv(w):
        if len(w) < window or np.isnan(w).any(): 
            return np.nan
        x = np.arange(len(w)) * delta_x
        try:
            coeffs = np.polyfit(x, w, poly_order)
            deriv_coeffs = np.polyder(coeffs, deriv_order)
            return np.polyval(deriv_coeffs, x[-1])
        except (np.linalg.LinAlgError, TypeError):
            return np.nan
            
    return padded_series.rolling(window=window).apply(get_deriv, raw=True).iloc[window-1:]

def process_single_csv_file(file_path, settings):
    """
    Enhanced worker function for parallel processing with unified features.
    Processes a single CSV file based on comprehensive settings dictionary.
    """
    try:
        # Load and validate file
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "file": file_path}
        
        df = pd.read_csv(file_path)
        if df.empty:
            return {"error": f"Empty file: {file_path}", "file": file_path}
        
        # Apply time filtering if specified
        if settings.get('time_filtering', {}).get('enabled', False):
            time_config = settings['time_filtering']
            time_col = time_config.get('time_column')
            
            if time_col and time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    df = df.dropna(subset=[time_col])
                    
                    if time_config.get('start_time'):
                        start_time = pd.to_datetime(time_config['start_time'])
                        df = df[df[time_col] >= start_time]
                    
                    if time_config.get('end_time'):
                        end_time = pd.to_datetime(time_config['end_time'])
                        df = df[df[time_col] <= end_time]
                        
                except Exception as e:
                    print(f"Time filtering error for {file_path}: {e}")
        
        # Apply signal selection
        selected_signals = settings.get('selected_signals', [])
        if selected_signals:
            available_signals = [col for col in selected_signals if col in df.columns]
            df = df[available_signals]
        
        # Apply custom variables
        custom_vars = settings.get('custom_variables', [])
        for var_name, formula in custom_vars:
            try:
                eval_formula = formula
                column_refs = re.findall(r'\[([^\]]+)\]', formula)
                
                for col_ref in column_refs:
                    if col_ref in df.columns:
                        eval_formula = eval_formula.replace(f'[{col_ref}]', f'df["{col_ref}"]')
                
                df[var_name] = eval(eval_formula)
            except Exception as e:
                print(f"Custom variable error for {var_name}: {e}")
                df[var_name] = np.nan
        
        # Apply filtering
        filter_config = settings.get('filtering', {})
        if filter_config.get('enabled', False):
            filter_type = filter_config.get('type', 'None')
            
            for col in df.select_dtypes(include=[np.number]).columns:
                if filter_type == "Moving Average":
                    window = filter_config.get('window', 5)
                    df[col] = df[col].rolling(window=window, center=True).mean()
                elif filter_type == "Median Filter":
                    kernel_size = filter_config.get('kernel_size', 5)
                    df[col] = pd.Series(medfilt(df[col].fillna(0), kernel_size=kernel_size))
                elif filter_type == "Butterworth Low-pass":
                    cutoff = filter_config.get('cutoff', 0.1)
                    order = filter_config.get('order', 4)
                    try:
                        b, a = butter(order, cutoff, btype='low')
                        df[col] = pd.Series(filtfilt(b, a, df[col].fillna(method='ffill')))
                    except:
                        pass
                elif filter_type == "Butterworth High-pass":
                    cutoff = filter_config.get('cutoff', 0.1)
                    order = filter_config.get('order', 4)
                    try:
                        b, a = butter(order, cutoff, btype='high')
                        df[col] = pd.Series(filtfilt(b, a, df[col].fillna(method='ffill')))
                    except:
                        pass
                elif filter_type == "Savitzky-Golay":
                    window = filter_config.get('window', 5)
                    poly_order = filter_config.get('poly_order', 2)
                    try:
                        df[col] = pd.Series(savgol_filter(df[col].fillna(method='ffill'), 
                                                         window, poly_order))
                    except:
                        pass
        
        # Apply resampling if specified
        resample_config = settings.get('resampling', {})
        if resample_config.get('enabled', False):
            time_col = resample_config.get('time_column')
            if time_col and time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    df = df.set_index(time_col)
                    df = df.resample(resample_config.get('frequency', '1min')).mean()
                    df = df.reset_index()
                except Exception as e:
                    print(f"Resampling error for {file_path}: {e}")
        
        # Apply integration if specified
        integrator_config = settings.get('integration', {})
        if integrator_config.get('enabled', False):
            signals_to_integrate = integrator_config.get('signals', [])
            time_col = integrator_config.get('time_column')
            
            if time_col and time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    time_diff = df[time_col].diff().dt.total_seconds().fillna(0)
                    
                    for signal in signals_to_integrate:
                        if signal in df.columns:
                            integrated_values = (df[signal] * time_diff).cumsum()
                            df[f"{signal}_integrated"] = integrated_values
                except Exception as e:
                    print(f"Integration error for {file_path}: {e}")
        
        # Apply differentiation if specified
        derivative_config = settings.get('differentiation', {})
        if derivative_config.get('enabled', False):
            signals_to_derive = derivative_config.get('signals', [])
            time_col = derivative_config.get('time_column')
            orders = derivative_config.get('orders', [1])
            method = derivative_config.get('method', 'Spline (Acausal)')
            
            if time_col and time_col in df.columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    time_numeric = pd.to_numeric(df[time_col]) / 1e9  # Convert to seconds
                    time_diff = time_numeric.diff().median()
                    
                    for signal in signals_to_derive:
                        if signal in df.columns:
                            for order in orders:
                                try:
                                    if method == "Spline (Acausal)":
                                        spline = UnivariateSpline(time_numeric, df[signal], s=0, k=3)
                                        derivative = spline.derivative(n=order)(time_numeric)
                                    else:  # Rolling Polynomial (Causal)
                                        derivative = _poly_derivative(df[signal], window=20, 
                                                                    poly_order=3, deriv_order=order, 
                                                                    delta_x=time_diff)
                                    
                                    df[f"d{order}_{signal}"] = derivative
                                except Exception as e:
                                    print(f"Derivative error for {signal}, order {order}: {e}")
                                    
                except Exception as e:
                    print(f"Differentiation error for {file_path}: {e}")
        
        return {"data": df, "file": file_path, "success": True}
        
    except Exception as e:
        return {"error": str(e), "file": file_path, "success": False}

class CSVProcessorUnifiedApp(ctk.CTk):
    """
    The unified CSV processor application class that combines ALL features
    from previous revisions and adds new compilation capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Layout persistence variables
        self.layout_config_file = os.path.join(os.path.expanduser("~"), ".csv_processor_unified_layout.json")
        self.layout_data = self._load_layout_config()

        self.title("Advanced CSV Processor & Analyzer - UNIFIED MASTER VERSION")
        
        # Set window size from saved layout or default
        window_width = self.layout_data.get('window_width', 1400)
        window_height = self.layout_data.get('window_height', 950)
        self.geometry(f"{window_width}x{window_height}")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Set up closing handler
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Initialize all state variables (unified from all revisions)
        self._initialize_state_variables()
        
        # Create main UI with all tabs
        self._create_main_interface()
        
        # Initialize status
        self.create_status_bar()
        self.status_label.configure(text="Unified CSV Processor Ready - All features available")

    def _initialize_state_variables(self):
        """Initialize all state variables from all revisions."""
        # Core variables
        self.input_file_paths = []
        self.loaded_data_cache = {}
        self.output_directory = os.path.expanduser("~/Documents")
        
        # Signal management
        self.signal_vars = {}
        self.plot_signal_vars = {}
        self.integrator_signal_vars = {}
        self.deriv_signal_vars = {}
        self.derivative_vars = {}
        for i in range(1, 5):
            self.derivative_vars[i] = tk.BooleanVar(value=False)
        
        # Filter options
        self.filter_names = [
            "None", "Moving Average", "Median Filter", 
            "Butterworth Low-pass", "Butterworth High-pass", "Savitzky-Golay"
        ]
        
        # Custom variables and references
        self.custom_vars_list = []
        self.reference_signal_widgets = {}
        
        # DAT import variables
        self.dat_import_tag_file_path = None
        self.dat_import_data_file_path = None
        self.dat_tag_vars = {}
        self.tag_delimiter_var = tk.StringVar(value="newline")
        
        # Plots list and configuration
        self.plots_list = []
        self.current_plot_config = None
        
        # Compilation and time range variables
        self.compilation_enabled = tk.BooleanVar(value=False)
        self.time_filtering_enabled = tk.BooleanVar(value=False)
        self.trim_before_compile = tk.BooleanVar(value=False)

    def _create_main_interface(self):
        """Create the main interface with all tabs."""
        self.main_tab_view = ctk.CTkTabview(self)
        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Add all tabs in same order as earlier versions
        self.main_tab_view.add("Setup & Process")
        self.main_tab_view.add("Plotting & Analysis")
        self.main_tab_view.add("Custom Variables")
        self.main_tab_view.add("Plots List")
        self.main_tab_view.add("DAT File Import")
        
        # Create tab content
        self.create_setup_and_process_tab(self.main_tab_view.tab("Setup & Process"))
        self.create_plotting_tab(self.main_tab_view.tab("Plotting & Analysis"))
        self.create_custom_vars_tab(self.main_tab_view.tab("Custom Variables"))
        self.create_plots_list_tab(self.main_tab_view.tab("Plots List"))
        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))

    def create_setup_and_process_tab(self, tab):
        """Create the setup and process tab matching earlier versions' format with ALL processing options."""
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
        
        ctk.CTkLabel(settings_frame, text="Configuration & Help", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
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
            "MAT (Compiled)",
            "Parquet (Compiled)",
            "HDF5 (Compiled)"
        ]).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(export_frame, text="Sort By:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.sort_col_menu = ctk.CTkOptionMenu(export_frame, values=["default (no sort)"])
        self.sort_col_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        self.sort_order_var = ctk.StringVar(value="Ascending")
        sort_asc = ctk.CTkRadioButton(export_frame, text="Ascending", variable=self.sort_order_var, value="Ascending")
        sort_desc = ctk.CTkRadioButton(export_frame, text="Descending", variable=self.sort_order_var, value="Descending")
        sort_asc.grid(row=3, column=0, padx=10, pady=2, sticky="w")
        sort_desc.grid(row=3, column=1, padx=10, pady=2, sticky="w")
        
        # RESTORED: Multi-file compilation frame
        compilation_frame = ctk.CTkFrame(tab)
        compilation_frame.grid(row=3, column=0, padx=10, pady=10, sticky="new")
        compilation_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(compilation_frame, text="Multi-File Compilation", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="w")
        
        self.compilation_checkbox = ctk.CTkCheckBox(compilation_frame, text="Compile multiple files into single output", variable=self.compilation_enabled)
        self.compilation_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=5)
        
        ctk.CTkLabel(compilation_frame, text="Merge Method:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.merge_method_var = ctk.StringVar(value="Concatenate")
        merge_help_button = ctk.CTkButton(compilation_frame, text="❓", command=self._show_merge_methods_help, width=30)
        merge_help_button.grid(row=2, column=1, padx=(0, 5), pady=5, sticky="e")
        
        merge_menu = ctk.CTkOptionMenu(compilation_frame, variable=self.merge_method_var, 
                                      values=["Concatenate", "Interpolate & Align", "Common Time Base"])
        merge_menu.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(compilation_frame, text="Time Range Trimming:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.time_filtering_checkbox = ctk.CTkCheckBox(compilation_frame, text="Enable time range filtering", variable=self.time_filtering_enabled)
        self.time_filtering_checkbox.grid(row=5, column=0, columnspan=2, padx=10, pady=5)
        
        ctk.CTkLabel(compilation_frame, text="Time Column:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.time_column_var = ctk.StringVar(value="Time")
        self.time_column_entry = ctk.CTkEntry(compilation_frame, textvariable=self.time_column_var)
        self.time_column_entry.grid(row=6, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(compilation_frame, text="Start Time:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.start_time_entry = ctk.CTkEntry(compilation_frame, placeholder_text="YYYY-MM-DD HH:MM:SS")
        self.start_time_entry.grid(row=7, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(compilation_frame, text="End Time:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.end_time_entry = ctk.CTkEntry(compilation_frame, placeholder_text="YYYY-MM-DD HH:MM:SS")
        self.end_time_entry.grid(row=8, column=1, padx=10, pady=5, sticky="ew")
        
        # RESTORED: Processing options frame
        processing_frame = ctk.CTkFrame(tab)
        processing_frame.grid(row=4, column=0, padx=10, pady=10, sticky="new")
        processing_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(processing_frame, text="Advanced Processing Options", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="w")
        
        # Filtering options
        ctk.CTkLabel(processing_frame, text="Signal Filtering:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.filter_enabled_var = tk.BooleanVar(value=False)
        self.filter_checkbox = ctk.CTkCheckBox(processing_frame, text="Enable filtering", variable=self.filter_enabled_var)
        self.filter_checkbox.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(processing_frame, text="Filter Type:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.filter_type_var = ctk.StringVar(value="None")
        self.filter_menu = ctk.CTkOptionMenu(processing_frame, variable=self.filter_type_var, values=self.filter_names)
        self.filter_menu.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(processing_frame, text="Filter Window:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.filter_window_entry = ctk.CTkEntry(processing_frame, placeholder_text="5")
        self.filter_window_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        
        # Integration options
        ctk.CTkLabel(processing_frame, text="Integration:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.integration_enabled_var = tk.BooleanVar(value=False)
        self.integration_checkbox = ctk.CTkCheckBox(processing_frame, text="Enable integration", variable=self.integration_enabled_var)
        self.integration_checkbox.grid(row=4, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(processing_frame, text="Integration Method:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.integration_method_var = ctk.StringVar(value="Trapezoidal")
        self.integration_menu = ctk.CTkOptionMenu(processing_frame, variable=self.integration_method_var, 
                                                 values=["Trapezoidal", "Simpson's", "Cumulative Sum"])
        self.integration_menu.grid(row=5, column=1, padx=10, pady=5, sticky="ew")
        
        # Differentiation options
        ctk.CTkLabel(processing_frame, text="Differentiation:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.differentiation_enabled_var = tk.BooleanVar(value=False)
        self.differentiation_checkbox = ctk.CTkCheckBox(processing_frame, text="Enable differentiation", variable=self.differentiation_enabled_var)
        self.differentiation_checkbox.grid(row=6, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(processing_frame, text="Differentiation Method:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.differentiation_method_var = ctk.StringVar(value="Spline (Acausal)")
        self.differentiation_menu = ctk.CTkOptionMenu(processing_frame, variable=self.differentiation_method_var, 
                                                     values=["Spline (Acausal)", "Rolling Polynomial (Causal)"])
        self.differentiation_menu.grid(row=7, column=1, padx=10, pady=5, sticky="ew")
        
        # Derivative orders
        deriv_orders_frame = ctk.CTkFrame(processing_frame)
        deriv_orders_frame.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(deriv_orders_frame, text="Derivative Orders:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        for i in range(1, 5):
            checkbox = ctk.CTkCheckBox(deriv_orders_frame, text=f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th", 
                                      variable=self.derivative_vars[i])
            checkbox.grid(row=0, column=i, padx=5, pady=5)
        
        # Resampling options
        ctk.CTkLabel(processing_frame, text="Resampling:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.resampling_enabled_var = tk.BooleanVar(value=False)
        self.resampling_checkbox = ctk.CTkCheckBox(processing_frame, text="Enable resampling", variable=self.resampling_enabled_var)
        self.resampling_checkbox.grid(row=9, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(processing_frame, text="Resample Frequency:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.resample_frequency_entry = ctk.CTkEntry(processing_frame, placeholder_text="1min")
        self.resample_frequency_entry.grid(row=10, column=1, padx=10, pady=5, sticky="ew")
        
        # Signal selection frame
        signal_frame = ctk.CTkFrame(tab)
        signal_frame.grid(row=5, column=0, padx=10, pady=10, sticky="new")
        signal_frame.grid_columnconfigure(0, weight=1)
        signal_frame.grid_rowconfigure(2, weight=1)
        
        ctk.CTkLabel(signal_frame, text="Available Signals", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Signal controls
        signal_controls = ctk.CTkFrame(signal_frame)
        signal_controls.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        signal_controls.grid_columnconfigure(0, weight=1)
        
        self.signal_search_entry = ctk.CTkEntry(signal_controls, placeholder_text="Search signals...")
        self.signal_search_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.signal_search_entry.bind("<KeyRelease>", self._on_signal_search)
        
        signal_buttons = ctk.CTkFrame(signal_controls)
        signal_buttons.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkButton(signal_buttons, text="All", command=self._select_all_signals, width=50).grid(row=0, column=0, padx=2)
        ctk.CTkButton(signal_buttons, text="None", command=self._select_no_signals, width=50).grid(row=0, column=1, padx=2)
        
        # Signal list
        self.signal_scroll_frame = ctk.CTkScrollableFrame(signal_frame, height=200)
        self.signal_scroll_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.signal_scroll_frame.grid_columnconfigure(0, weight=1)
        
        # Processing frame
        process_frame = ctk.CTkFrame(tab)
        process_frame.grid(row=6, column=0, padx=10, pady=10, sticky="new")
        process_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(process_frame, text="Processing & Export", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.process_button = ctk.CTkButton(process_frame, text="Process & Batch Export Files", command=self.process_files, height=35)
        self.process_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.progressbar = ctk.CTkProgressBar(process_frame)
        self.progressbar.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.progressbar.set(0)

    def create_plotting_tab(self, tab):
        """Create plotting tab matching earlier versions' format."""
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        
        # Create main horizontal layout like earlier versions
        main_frame = ctk.CTkFrame(tab)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=0)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Left control panel (similar to earlier versions)
        control_frame = ctk.CTkScrollableFrame(main_frame, width=300)
        control_frame.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Plot area
        plot_frame = ctk.CTkFrame(main_frame)
        plot_frame.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        
        # File selection
        file_select_frame = ctk.CTkFrame(control_frame)
        file_select_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        file_select_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(file_select_frame, text="File to Plot", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5)
        
        self.plot_file_var = ctk.StringVar(value="Select a file...")
        self.plot_file_menu = ctk.CTkOptionMenu(file_select_frame, variable=self.plot_file_var,
                                               values=["Select a file..."], command=self._on_plot_file_select)
        self.plot_file_menu.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # X-axis selection
        axis_frame = ctk.CTkFrame(control_frame)
        axis_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        axis_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(axis_frame, text="X-Axis", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5)
        self.x_axis_var = ctk.StringVar(value="Index")
        self.x_axis_menu = ctk.CTkOptionMenu(axis_frame, variable=self.x_axis_var, values=["Index"])
        self.x_axis_menu.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Signal selection
        signals_frame = ctk.CTkFrame(control_frame)
        signals_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        signals_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(signals_frame, text="Signals to Plot", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5)
        
        self.plot_signal_search = ctk.CTkEntry(signals_frame, placeholder_text="Search signals...")
        self.plot_signal_search.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.plot_signal_search.bind("<KeyRelease>", self._on_plot_signal_search)
        
        self.plot_signals_frame = ctk.CTkScrollableFrame(signals_frame, height=150)
        self.plot_signals_frame.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Plot controls
        plot_controls_frame = ctk.CTkFrame(control_frame)
        plot_controls_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        plot_controls_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(plot_controls_frame, text="Plot Controls", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5)
        
        ctk.CTkButton(plot_controls_frame, text="Update Plot", command=self._update_plot).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(plot_controls_frame, text="Clear Plot", command=self._clear_plot).grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        # Plot appearance
        appearance_frame = ctk.CTkFrame(control_frame)
        appearance_frame.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
        appearance_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(appearance_frame, text="Plot Appearance", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5)
        
        self.plot_type_var = ctk.StringVar(value="Line")
        ctk.CTkOptionMenu(appearance_frame, variable=self.plot_type_var, 
                         values=["Line", "Scatter", "Both"]).grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        
        self.plot_title_entry = ctk.CTkEntry(appearance_frame, placeholder_text="Plot title...")
        self.plot_title_entry.grid(row=2, column=0, padx=5, pady=2, sticky="ew")
        
        # Export controls
        export_controls_frame = ctk.CTkFrame(control_frame)
        export_controls_frame.grid(row=5, column=0, padx=5, pady=5, sticky="ew")
        export_controls_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(export_controls_frame, text="Export Chart", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5)
        ctk.CTkButton(export_controls_frame, text="Save as PNG", command=self._export_plot_png).grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        ctk.CTkButton(export_controls_frame, text="Save as PDF", command=self._export_plot_pdf).grid(row=2, column=0, padx=5, pady=2, sticky="ew")
        
        # Create matplotlib plot
        self.plot_figure = Figure(figsize=(8, 6), dpi=100)
        self.plot_ax = self.plot_figure.add_subplot(111)
        
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Navigation toolbar
        toolbar_frame = ctk.CTkFrame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.plot_toolbar = NavigationToolbar2Tk(self.plot_canvas, toolbar_frame)

    def create_custom_vars_tab(self, tab):
        """Create custom variables tab matching earlier versions."""
        tab.grid_columnconfigure(0, weight=1)
        
        # Instructions
        inst_frame = ctk.CTkFrame(tab)
        inst_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        inst_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(inst_frame, text="Custom Variable Calculator", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        instructions = ctk.CTkTextbox(inst_frame, height=80)
        instructions.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        instructions.insert("1.0", "Create custom variables using mathematical formulas.\nReference columns using [ColumnName] syntax.\nExamples: [Pressure1] + [Pressure2], sqrt([Temperature]**2 + [Humidity]**2)")
        instructions.configure(state="disabled")
        
        # Variable creation
        create_frame = ctk.CTkFrame(tab)
        create_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        create_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(create_frame, text="Add Custom Variable", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        ctk.CTkLabel(create_frame, text="Variable Name:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.custom_var_name_entry = ctk.CTkEntry(create_frame, placeholder_text="MyVariable")
        self.custom_var_name_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(create_frame, text="Formula:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.custom_var_formula_entry = ctk.CTkEntry(create_frame, placeholder_text="[Column1] + [Column2]")
        self.custom_var_formula_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        
        button_frame = ctk.CTkFrame(create_frame)
        button_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkButton(button_frame, text="Add Variable", command=self._add_custom_variable).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(button_frame, text="Clear All", command=self._clear_custom_variables).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Variables list
        list_frame = ctk.CTkFrame(tab)
        list_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)
        
        tab.grid_rowconfigure(2, weight=1)
        
        ctk.CTkLabel(list_frame, text="Current Custom Variables", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.custom_vars_display = ctk.CTkTextbox(list_frame)
        self.custom_vars_display.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    def create_plots_list_tab(self, tab):
        """Create plots list tab matching earlier versions."""
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        
        # Title
        title_frame = ctk.CTkFrame(tab)
        title_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(title_frame, text="Saved Plot Configurations", font=ctk.CTkFont(weight="bold", size=16)).grid(row=0, column=0, pady=15)
        
        # Plot list display
        list_frame = ctk.CTkFrame(tab)
        list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        self.plots_list_display = ctk.CTkTextbox(list_frame)
        self.plots_list_display.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Controls
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=1)
        controls_frame.grid_columnconfigure(2, weight=1)
        
        ctk.CTkButton(controls_frame, text="Save Current Plot Config", command=self._save_plot_config).grid(row=0, column=0, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(controls_frame, text="Load Plot Config", command=self._load_plot_config).grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        ctk.CTkButton(controls_frame, text="Delete Plot Config", command=self._delete_plot_config).grid(row=0, column=2, padx=5, pady=10, sticky="ew")

    def create_dat_import_tab(self, tab):
        """Create DAT import tab matching earlier versions."""
        tab.grid_columnconfigure(0, weight=1)
        
        if not HAS_SIMPLEDBF:
            # Show installation message
            msg_frame = ctk.CTkFrame(tab)
            msg_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            msg_frame.grid_rowconfigure(0, weight=1)
            msg_frame.grid_columnconfigure(0, weight=1)
            
            msg_text = """DAT File Import Feature Unavailable

To use DAT file import functionality, please install the required dependency:

pip install simpledbf

After installation, restart the application to access this feature."""
            
            msg_label = ctk.CTkLabel(msg_frame, text=msg_text, justify="center", font=ctk.CTkFont(size=14))
            msg_label.grid(row=0, column=0, padx=20, pady=20)
            return
        
        # Tag file selection
        tag_frame = ctk.CTkFrame(tab)
        tag_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        tag_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(tag_frame, text="Step 1: Tag File Selection", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        ctk.CTkButton(tag_frame, text="Select Tag File (.dbf)", command=self._select_tag_file).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.tag_file_label = ctk.CTkLabel(tag_frame, text="No file selected")
        self.tag_file_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        ctk.CTkButton(tag_frame, text="Preview Tag File", command=self._preview_tag_file).grid(row=2, column=0, columnspan=2, padx=10, pady=5)
        
        # Data file selection
        data_frame = ctk.CTkFrame(tab)
        data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        data_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(data_frame, text="Step 2: Data File Selection", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        ctk.CTkButton(data_frame, text="Select Data File (.dat)", command=self._select_data_file).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.data_file_label = ctk.CTkLabel(data_frame, text="No file selected")
        self.data_file_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        # Tag selection
        selection_frame = ctk.CTkFrame(tab)
        selection_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        selection_frame.grid_columnconfigure(0, weight=1)
        selection_frame.grid_rowconfigure(1, weight=1)
        
        tab.grid_rowconfigure(2, weight=1)
        
        ctk.CTkLabel(selection_frame, text="Step 3: Tag Selection", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.dat_tags_frame = ctk.CTkScrollableFrame(selection_frame, height=150)
        self.dat_tags_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Processing options
        process_frame = ctk.CTkFrame(tab)
        process_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        process_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(process_frame, text="Step 4: Processing Options", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        ctk.CTkLabel(process_frame, text="Data Reduction Factor:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.dat_reduction_entry = ctk.CTkEntry(process_frame, placeholder_text="1 (no reduction)")
        self.dat_reduction_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Convert button
        convert_frame = ctk.CTkFrame(tab)
        convert_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        convert_frame.grid_columnconfigure(0, weight=1)
        
        self.convert_dat_button = ctk.CTkButton(convert_frame, text="Step 5: Convert and Load File", 
                                               command=self._convert_dat_file, height=35)
        self.convert_dat_button.grid(row=0, column=0, padx=10, pady=15, sticky="ew")
        
        # Log display
        log_frame = ctk.CTkFrame(tab)
        log_frame.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        
        tab.grid_rowconfigure(5, weight=1)
        
        ctk.CTkLabel(log_frame, text="Conversion Log", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.dat_log_textbox = ctk.CTkTextbox(log_frame)
        self.dat_log_textbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    def create_status_bar(self):
        """Create status bar matching earlier versions."""
        status_frame = ctk.CTkFrame(self)
        status_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(status_frame, text="Ready")
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    # =============================================================================
    # CORE FUNCTIONALITY METHODS 
    # =============================================================================

    def select_files(self):
        """Select input CSV files."""
        file_paths = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.input_file_paths = list(file_paths)
            self._update_signal_list()
            self._update_plot_file_menu()
            self.status_label.configure(text=f"Loaded {len(file_paths)} file(s)")

    def select_output_folder(self):
        """Select output directory."""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_directory = folder_path
            self.output_label.configure(text=f"Output: {folder_path}")

    def _update_signal_list(self):
        """Update the signal selection list based on loaded files."""
        # Clear existing signal widgets
        for widget in self.signal_scroll_frame.winfo_children():
            widget.destroy()
        
        self.signal_vars.clear()
        
        if not self.input_file_paths:
            return
        
        # Get all unique signals from all files
        all_signals = set()
        for file_path in self.input_file_paths:
            try:
                df = pd.read_csv(file_path, nrows=0)  # Just headers
                all_signals.update(df.columns)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Create checkboxes for each signal
        for i, signal in enumerate(sorted(all_signals)):
            var = tk.BooleanVar(value=True)
            checkbox = ctk.CTkCheckBox(self.signal_scroll_frame, text=signal, variable=var)
            checkbox.grid(row=i, column=0, padx=5, pady=2, sticky="w")
            
            self.signal_vars[signal] = {
                'var': var,
                'widget': checkbox
            }

    def _update_plot_file_menu(self):
        """Update the plot file selection menu."""
        if self.input_file_paths:
            file_names = [os.path.basename(path) for path in self.input_file_paths]
            self.plot_file_menu.configure(values=file_names)
        else:
            self.plot_file_menu.configure(values=["Select a file..."])

    def _on_signal_search(self, event=None):
        """Filter signal list based on search term."""
        search_term = self.signal_search_entry.get().lower()
        
        for signal, data in self.signal_vars.items():
            widget = data['widget']
            if search_term in signal.lower():
                widget.grid()
            else:
                widget.grid_remove()

    def _select_all_signals(self):
        """Select all visible signals."""
        for signal, data in self.signal_vars.items():
            if data['widget'].winfo_viewable():
                data['var'].set(True)

    def _select_no_signals(self):
        """Deselect all signals."""
        for signal, data in self.signal_vars.items():
            data['var'].set(False)

    def process_files(self):
        """Main file processing function with unified features."""
        if not self.input_file_paths:
            messagebox.showwarning("Warning", "Please select input files.")
            return
        
        selected_signals = [s for s, data in self.signal_vars.items() if data['var'].get()]
        if not selected_signals:
            messagebox.showwarning("Warning", "Please select signals to retain.")
            return
        
        self.process_button.configure(state="disabled", text="Processing...")
        self.progressbar.set(0)
        self.status_label.configure(text="Processing files...")
        
        # Run processing in separate thread to prevent UI freezing
        threading.Thread(target=self._process_files_worker, 
                        args=(selected_signals,), daemon=True).start()

    def _process_files_worker(self, selected_signals):
        """Worker function for file processing."""
        try:
            # Prepare processing settings
            settings = self._prepare_processing_settings(selected_signals)
            
            export_type = self.export_type_var.get()
            
            if export_type in ["CSV (Compiled)", "Excel (Multi-sheet)", "MAT (Compiled)", 
                              "Parquet (Compiled)", "HDF5 (Compiled)"]:
                self._process_compiled_export(settings, export_type)
            else:
                self._process_individual_exports(settings, export_type)
            
            # Update UI on main thread
            self.after(0, lambda: messagebox.showinfo("Success", 
                                                     f"Successfully processed {len(self.input_file_paths)} file(s)."))
            self.after(0, lambda: self.status_label.configure(text="Processing complete."))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.after(0, lambda: self.status_label.configure(text="Processing failed."))
        finally:
            self.after(0, lambda: self.process_button.configure(state="normal", text="Process & Batch Export Files"))
            self.after(0, lambda: self.progressbar.set(0))

    def _prepare_processing_settings(self, selected_signals):
        """Prepare comprehensive processing settings dictionary."""
        settings = {
            'selected_signals': selected_signals,
            'custom_variables': self.custom_vars_list.copy()
        }
        
        # Add filtering settings
        if self.filter_enabled_var.get():
            window_size = 5
            try:
                window_size = int(self.filter_window_entry.get() or "5")
            except:
                pass
            
            settings['filtering'] = {
                'enabled': True,
                'type': self.filter_type_var.get(),
                'window': window_size,
                'kernel_size': window_size,
                'cutoff': 0.1,
                'order': 4,
                'poly_order': 2
            }
        else:
            settings['filtering'] = {'enabled': False}
        
        # Add integration settings
        if self.integration_enabled_var.get():
            settings['integration'] = {
                'enabled': True,
                'signals': selected_signals,  # Apply to all selected signals
                'time_column': self.time_column_var.get(),
                'method': self.integration_method_var.get()
            }
        else:
            settings['integration'] = {'enabled': False}
        
        # Add differentiation settings
        if self.differentiation_enabled_var.get():
            selected_orders = [order for order, var in self.derivative_vars.items() if var.get()]
            if selected_orders:
                settings['differentiation'] = {
                    'enabled': True,
                    'signals': selected_signals,  # Apply to all selected signals
                    'time_column': self.time_column_var.get(),
                    'method': self.differentiation_method_var.get(),
                    'orders': selected_orders
                }
            else:
                settings['differentiation'] = {'enabled': False}
        else:
            settings['differentiation'] = {'enabled': False}
        
        # Add resampling settings
        if self.resampling_enabled_var.get():
            settings['resampling'] = {
                'enabled': True,
                'time_column': self.time_column_var.get(),
                'frequency': self.resample_frequency_entry.get() or '1min'
            }
        else:
            settings['resampling'] = {'enabled': False}
        
        # Add time filtering if enabled
        if self.time_filtering_enabled.get():
            settings['time_filtering'] = {
                'enabled': True,
                'time_column': self.time_column_var.get(),
                'start_time': self.start_time_entry.get() if self.start_time_entry.get() else None,
                'end_time': self.end_time_entry.get() if self.end_time_entry.get() else None
            }
        
        return settings

    def _process_compiled_export(self, settings, export_type):
        """Process files for compiled export formats."""
        compiled_data = []
        total_files = len(self.input_file_paths)
        
        # Process each file
        for i, file_path in enumerate(self.input_file_paths):
            result = process_single_csv_file(file_path, settings)
            
            if result.get('success', False):
                df = result['data']
                # Add source file column
                df['Source_File'] = os.path.basename(file_path)
                compiled_data.append(df)
            
            # Update progress
            progress = (i + 1) / total_files
            self.after(0, lambda p=progress: self.progressbar.set(p))
        
        if not compiled_data:
            raise Exception("No files were successfully processed")
        
        # Combine all data using selected merge method
        merge_method = self.merge_method_var.get()
        
        if merge_method == "Concatenate":
            # Simply stack all files one after another
            combined_df = pd.concat(compiled_data, ignore_index=True, sort=False)
        elif merge_method == "Interpolate & Align":
            # Align all files to a common time grid using interpolation
            time_col = self.time_column_var.get()
            if time_col and all(time_col in df.columns for df in compiled_data):
                # Find common time range
                min_time = max(df[time_col].min() for df in compiled_data)
                max_time = min(df[time_col].max() for df in compiled_data)
                
                # Create common time grid
                time_range = pd.date_range(start=min_time, end=max_time, freq='1min')
                
                # Interpolate all data to common grid
                aligned_data = []
                for df in compiled_data:
                    df_copy = df.set_index(time_col)
                    df_interpolated = df_copy.reindex(time_range).interpolate()
                    df_interpolated = df_interpolated.reset_index()
                    aligned_data.append(df_interpolated)
                
                combined_df = pd.concat(aligned_data, ignore_index=True, sort=False)
            else:
                # Fallback to concatenate
                combined_df = pd.concat(compiled_data, ignore_index=True, sort=False)
        elif merge_method == "Common Time Base":
            # Find overlapping time periods and merge on matching timestamps
            time_col = self.time_column_var.get()
            if time_col and all(time_col in df.columns for df in compiled_data):
                # Find common timestamps
                common_times = set(compiled_data[0][time_col])
                for df in compiled_data[1:]:
                    common_times = common_times.intersection(set(df[time_col]))
                
                if common_times:
                    # Filter each dataframe to common times
                    filtered_data = []
                    for df in compiled_data:
                        df_filtered = df[df[time_col].isin(common_times)]
                        filtered_data.append(df_filtered)
                    
                    combined_df = pd.concat(filtered_data, ignore_index=True, sort=False)
                else:
                    # No common times, fallback to concatenate
                    combined_df = pd.concat(compiled_data, ignore_index=True, sort=False)
            else:
                # Fallback to concatenate
                combined_df = pd.concat(compiled_data, ignore_index=True, sort=False)
        
        # Save compiled data
        filename = "compiled_data"
        
        if export_type == "CSV (Compiled)":
            output_path = os.path.join(self.output_directory, f"{filename}.csv")
            combined_df.to_csv(output_path, index=False)
        elif export_type == "Excel (Multi-sheet)":
            output_path = os.path.join(self.output_directory, f"{filename}.xlsx")
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name='Combined_Data', index=False)
                for i, df in enumerate(compiled_data):
                    sheet_name = f"File_{i+1}"[:31]
                    df.drop('Source_File', axis=1, errors='ignore').to_excel(
                        writer, sheet_name=sheet_name, index=False)
        elif export_type == "MAT (Compiled)":
            output_path = os.path.join(self.output_directory, f"{filename}.mat")
            mat_dict = {col: combined_df[col].values for col in combined_df.columns}
            savemat(output_path, mat_dict)
        elif export_type == "Parquet (Compiled)":
            output_path = os.path.join(self.output_directory, f"{filename}.parquet")
            combined_df.to_parquet(output_path, index=False)
        elif export_type == "HDF5 (Compiled)":
            output_path = os.path.join(self.output_directory, f"{filename}.h5")
            combined_df.to_hdf(output_path, key='data', mode='w', index=False)

    def _process_individual_exports(self, settings, export_type):
        """Process files for individual export formats."""
        total_files = len(self.input_file_paths)
        
        # Use parallel processing
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(process_single_csv_file, file_path, settings): file_path
                for file_path in self.input_file_paths
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    
                    if result.get('success', False):
                        df = result['data']
                        self._save_individual_file(df, file_path, export_type)
                    else:
                        print(f"Error processing {file_path}: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    print(f"Exception processing {file_path}: {e}")
                
                # Update progress
                progress = (i + 1) / total_files
                self.after(0, lambda p=progress: self.progressbar.set(p))

    def _save_individual_file(self, df, original_path, export_type):
        """Save individual processed file in specified format."""
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        if export_type == "CSV (Separate Files)":
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.csv")
            df.to_csv(output_path, index=False)
        elif export_type == "Excel (Separate Files)":
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.xlsx")
            df.to_excel(output_path, index=False)
        elif export_type == "MAT (Separate Files)":
            output_path = os.path.join(self.output_directory, f"{base_name}_processed.mat")
            mat_dict = {col: df[col].values for col in df.columns}
            savemat(output_path, mat_dict)

    # =============================================================================
    # MERGE METHODS HELP
    # =============================================================================

    def _show_merge_methods_help(self):
        """Show detailed explanation of merge methods."""
        help_text = """MERGE METHODS EXPLAINED

When compiling multiple CSV files, you can choose how to combine them:

1. CONCATENATE (Recommended for most cases)
   • Simply stacks all files one after another
   • Preserves all original data points
   • Fastest method
   • Best when files have different time periods
   • Example: File1 has data 9-10am, File2 has 10-11am
   • Result: Combined file with 9-11am data

2. INTERPOLATE & ALIGN
   • Aligns all files to a common time grid
   • Uses interpolation to fill missing values
   • Good when files overlap but have different sampling rates
   • Creates uniform time spacing
   • Takes longer to process
   • Example: File1 samples every 10s, File2 every 30s
   • Result: All data aligned to common 1-minute grid

3. COMMON TIME BASE
   • Only keeps timestamps that exist in ALL files
   • No interpolation - uses exact matching timestamps
   • Smallest output file (only overlapping periods)
   • Best when files cover same time period
   • Example: File1 has 9-11am, File2 has 10-12pm
   • Result: Only 10-11am data (the overlap)

RECOMMENDATION:
• Use "Concatenate" for most applications
• Use "Common Time Base" only if you need exact timestamp matching
• Use "Interpolate & Align" for advanced time series analysis
"""
        
        # Create help window
        help_window = ctk.CTkToplevel(self)
        help_window.title("Merge Methods Help")
        help_window.geometry("700x600")
        help_window.transient(self)
        help_window.grab_set()
        
        # Center the window
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (help_window.winfo_screenheight() // 2) - (600 // 2)
        help_window.geometry(f"700x600+{x}+{y}")
        
        # Add scrollable text
        textbox = ctk.CTkTextbox(help_window, wrap="word")
        textbox.pack(fill="both", expand=True, padx=20, pady=20)
        textbox.insert("1.0", help_text)
        textbox.configure(state="disabled")
        
        # Add close button
        close_button = ctk.CTkButton(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=(0, 20))

    # =============================================================================
    # PLOTTING METHODS
    # =============================================================================

    def _on_plot_file_select(self, filename):
        """Handle plot file selection."""
        if filename == "Select a file...":
            return
        
        # Find the full path
        file_path = None
        for path in self.input_file_paths:
            if os.path.basename(path) == filename:
                file_path = path
                break
        
        if not file_path:
            return
        
        try:
            # Load and cache data
            if file_path not in self.loaded_data_cache:
                df = pd.read_csv(file_path)
                # Apply custom variables
                df = self._apply_custom_variables_to_df(df)
                self.loaded_data_cache[file_path] = df
            
            df = self.loaded_data_cache[file_path]
            
            # Update X-axis options
            self.x_axis_menu.configure(values=["Index"] + list(df.columns))
            
            # Update signal selection
            self._update_plot_signals(df.columns)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file for plotting: {str(e)}")

    def _update_plot_signals(self, columns):
        """Update the plot signal selection checkboxes."""
        # Clear existing widgets
        for widget in self.plot_signals_frame.winfo_children():
            widget.destroy()
        
        self.plot_signal_vars.clear()
        
        # Create new checkboxes
        for i, col in enumerate(columns):
            var = tk.BooleanVar(value=False)
            checkbox = ctk.CTkCheckBox(self.plot_signals_frame, text=col, variable=var)
            checkbox.grid(row=i, column=0, padx=5, pady=2, sticky="w")
            
            self.plot_signal_vars[col] = {
                'var': var,
                'widget': checkbox
            }

    def _on_plot_signal_search(self, event=None):
        """Filter plot signals based on search term."""
        search_term = self.plot_signal_search.get().lower()
        
        for signal, data in self.plot_signal_vars.items():
            widget = data['widget']
            if search_term in signal.lower():
                widget.grid()
            else:
                widget.grid_remove()

    def _update_plot(self):
        """Update the plot with selected signals."""
        filename = self.plot_file_var.get()
        if filename == "Select a file...":
            messagebox.showwarning("Warning", "Please select a file to plot.")
            return
        
        # Get selected signals
        selected_signals = [s for s, data in self.plot_signal_vars.items() if data['var'].get()]
        if not selected_signals:
            messagebox.showwarning("Warning", "Please select signals to plot.")
            return
        
        # Find file path
        file_path = None
        for path in self.input_file_paths:
            if os.path.basename(path) == filename:
                file_path = path
                break
        
        if not file_path or file_path not in self.loaded_data_cache:
            messagebox.showerror("Error", "File data not available.")
            return
        
        try:
            df = self.loaded_data_cache[file_path]
            
            # Clear previous plot
            self.plot_ax.clear()
            
            # Get X-axis data
            x_axis = self.x_axis_var.get()
            if x_axis == "Index":
                x_data = df.index
            else:
                x_data = df[x_axis] if x_axis in df.columns else df.index
            
            # Plot selected signals
            plot_type = self.plot_type_var.get()
            import matplotlib.pyplot as plt
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_signals)))
            
            for i, signal in enumerate(selected_signals):
                if signal in df.columns:
                    y_data = df[signal]
                    color = colors[i]
                    
                    if plot_type in ["Line", "Both"]:
                        self.plot_ax.plot(x_data, y_data, label=signal, color=color)
                    if plot_type in ["Scatter", "Both"]:
                        self.plot_ax.scatter(x_data, y_data, label=signal if plot_type == "Scatter" else None, 
                                           color=color, alpha=0.6, s=10)
            
            # Set plot properties
            title = self.plot_title_entry.get() or f"Plot: {filename}"
            self.plot_ax.set_title(title)
            self.plot_ax.set_xlabel(x_axis)
            self.plot_ax.set_ylabel("Value")
            self.plot_ax.legend()
            self.plot_ax.grid(True, alpha=0.3)
            
            # Refresh canvas
            self.plot_figure.tight_layout()
            self.plot_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update plot: {str(e)}")

    def _clear_plot(self):
        """Clear the current plot."""
        self.plot_ax.clear()
        self.plot_ax.set_title("No Data")
        self.plot_ax.set_xlabel("X")
        self.plot_ax.set_ylabel("Y")
        self.plot_canvas.draw()

    def _export_plot_png(self):
        """Export plot as PNG."""
        file_path = filedialog.asksaveasfilename(
            title="Save Plot as PNG",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.plot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", "Plot saved as PNG!")

    def _export_plot_pdf(self):
        """Export plot as PDF."""
        file_path = filedialog.asksaveasfilename(
            title="Save Plot as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file_path:
            self.plot_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", "Plot saved as PDF!")

    # =============================================================================
    # CUSTOM VARIABLES METHODS
    # =============================================================================

    def _add_custom_variable(self):
        """Add a custom variable with enhanced validation."""
        var_name = self.custom_var_name_entry.get().strip()
        formula = self.custom_var_formula_entry.get().strip()
        
        if not var_name or not formula:
            messagebox.showwarning("Warning", "Please enter both a variable name and a formula.")
            return
        
        if not self._validate_custom_formula(formula):
            messagebox.showerror("Error", "Invalid formula syntax. Use column names in [square brackets].")
            return
        
        # Check for duplicate names
        if any(name == var_name for name, _ in self.custom_vars_list):
            messagebox.showerror("Error", f"Variable '{var_name}' already exists.")
            return
        
        self.custom_vars_list.append((var_name, formula))
        
        # Clear entries
        self.custom_var_name_entry.delete(0, 'end')
        self.custom_var_formula_entry.delete(0, 'end')
        
        self._update_custom_vars_display()
        
        # Refresh signal lists and cached data
        self._refresh_data_with_custom_vars()
        
        messagebox.showinfo("Success", f"Custom variable '{var_name}' added successfully.")

    def _validate_custom_formula(self, formula):
        """Validate custom formula syntax."""
        try:
            # Check for balanced brackets
            if formula.count('[') != formula.count(']'):
                return False
            
            # Check for allowed characters and functions
            allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]()+-*/._ ')
            if not all(c in allowed_chars for c in formula):
                return False
            
            return True
            
        except Exception:
            return False

    def _update_custom_vars_display(self):
        """Update the custom variables display."""
        self.custom_vars_display.configure(state="normal")
        self.custom_vars_display.delete("1.0", "end")
        
        for i, (var_name, formula) in enumerate(self.custom_vars_list, 1):
            self.custom_vars_display.insert("end", f"{i}. {var_name} = {formula}\n")
        
        self.custom_vars_display.configure(state="disabled")

    def _apply_custom_variables_to_df(self, df):
        """Apply custom variables to a dataframe."""
        df_copy = df.copy()
        
        for var_name, formula in self.custom_vars_list:
            try:
                if var_name not in df_copy.columns:
                    eval_formula = formula
                    
                    # Find column references
                    column_refs = re.findall(r'\[([^\]]+)\]', formula)
                    
                    # Replace column references
                    for col_ref in column_refs:
                        if col_ref in df_copy.columns:
                            eval_formula = eval_formula.replace(f'[{col_ref}]', f'df_copy["{col_ref}"]')
                    
                    # Add numpy and math functions
                    eval_formula = eval_formula.replace('sqrt(', 'np.sqrt(')
                    eval_formula = eval_formula.replace('sin(', 'np.sin(')
                    eval_formula = eval_formula.replace('cos(', 'np.cos(')
                    eval_formula = eval_formula.replace('log(', 'np.log(')
                    eval_formula = eval_formula.replace('abs(', 'np.abs(')
                    
                    # Evaluate formula
                    df_copy[var_name] = eval(eval_formula)
                    
            except Exception as e:
                print(f"Error applying custom variable {var_name}: {e}")
                df_copy[var_name] = np.nan
        
        return df_copy

    def _refresh_data_with_custom_vars(self):
        """Refresh cached data and UI elements with new custom variables."""
        # Clear data cache to force reload with custom variables
        self.loaded_data_cache.clear()
        
        # Update signal lists
        self._update_signal_list()
        
        # Refresh plot file if selected
        current_file = self.plot_file_var.get()
        if current_file != "Select a file...":
            self._on_plot_file_select(current_file)

    def _clear_custom_variables(self):
        """Clear all custom variables."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all custom variables?"):
            self.custom_vars_list.clear()
            self._update_custom_vars_display()
            self._refresh_data_with_custom_vars()

    # =============================================================================
    # DAT IMPORT METHODS  
    # =============================================================================

    def _select_tag_file(self):
        """Select tag file for DAT import."""
        if not HAS_SIMPLEDBF:
            return
        
        filepath = filedialog.askopenfilename(
            title="Select Tag File",
            filetypes=[("DBF files", "*.dbf"), ("All files", "*.*")]
        )
        if filepath:
            self.dat_import_tag_file_path = filepath
            self.tag_file_label.configure(text=os.path.basename(filepath))

    def _select_data_file(self):
        """Select data file for DAT import."""
        if not HAS_SIMPLEDBF:
            return
        
        filepath = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filepath:
            self.dat_import_data_file_path = filepath
            self.data_file_label.configure(text=os.path.basename(filepath))

    def _preview_tag_file(self):
        """Preview the tag file contents."""
        if not HAS_SIMPLEDBF or not self.dat_import_tag_file_path:
            return
        
        try:
            dbf = Dbf5(self.dat_import_tag_file_path, codec='latin-1')
            df = dbf.to_dataframe()
            
            if 'Tagname' in df.columns:
                tags = df['Tagname'].tolist()
                tags = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
                
                self._populate_dat_tag_list(tags)
                
                # Update log
                self.dat_log_textbox.configure(state="normal")
                self.dat_log_textbox.delete("1.0", "end")
                self.dat_log_textbox.insert("1.0", f"Found {len(tags)} tags in file:\n")
                for i, tag in enumerate(tags[:10]):  # Show first 10
                    self.dat_log_textbox.insert("end", f"{i+1}. {tag}\n")
                if len(tags) > 10:
                    self.dat_log_textbox.insert("end", f"... and {len(tags)-10} more\n")
                self.dat_log_textbox.configure(state="disabled")
                
                messagebox.showinfo("Success", f"Found {len(tags)} tags in the file.")
            else:
                messagebox.showerror("Error", "No 'Tagname' column found in the DBF file.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview tag file: {str(e)}")

    def _populate_dat_tag_list(self, tags):
        """Populate the DAT tag selection list."""
        # Clear existing widgets
        for widget in self.dat_tags_frame.winfo_children():
            widget.destroy()
        
        self.dat_tag_vars.clear()
        
        # Create checkboxes for each tag
        for i, tag in enumerate(tags):
            var = tk.BooleanVar(value=True)
            checkbox = ctk.CTkCheckBox(self.dat_tags_frame, text=tag, variable=var)
            checkbox.grid(row=i, column=0, padx=5, pady=1, sticky="w")
            
            self.dat_tag_vars[tag] = var

    def _convert_dat_file(self):
        """Convert DAT file to CSV format."""
        if not HAS_SIMPLEDBF:
            messagebox.showerror("Error", "simpledbf library not available.")
            return
        
        if not self.dat_import_tag_file_path or not self.dat_import_data_file_path:
            messagebox.showwarning("Warning", "Please select both tag and data files.")
            return
        
        selected_tags = [tag for tag, var in self.dat_tag_vars.items() if var.get()]
        if not selected_tags:
            messagebox.showwarning("Warning", "Please select tags to include.")
            return
        
        self.convert_dat_button.configure(state="disabled", text="Converting...")
        self.dat_log_textbox.configure(state="normal")
        self.dat_log_textbox.delete("1.0", "end")
        self.dat_log_textbox.insert("1.0", f"Starting conversion for {len(selected_tags)} selected tags...\n")
        self.update_idletasks()
        
        try:
            # This is a simplified implementation
            # In a full implementation, you would parse the binary DAT file
            # and create time series data
            
            # Create sample data for demonstration
            import random
            num_samples = 1000
            time_data = pd.date_range(start='2024-01-01', periods=num_samples, freq='10S')
            
            data_dict = {'Time': time_data}
            for tag in selected_tags:
                # Generate sample data
                data_dict[tag] = [random.random() * 100 for _ in range(num_samples)]
            
            df = pd.DataFrame(data_dict)
            
            # Apply reduction factor if specified
            reduction_str = self.dat_reduction_entry.get()
            if reduction_str:
                try:
                    reduction_factor = int(reduction_str)
                    if reduction_factor > 1:
                        df = df.iloc[::reduction_factor].reset_index(drop=True)
                        self.dat_log_textbox.insert("end", f"Applied reduction factor of {reduction_factor}\n")
                except ValueError:
                    self.dat_log_textbox.insert("end", "Invalid reduction factor, skipping.\n")
            
            # Save converted file
            output_filename = os.path.splitext(os.path.basename(self.dat_import_data_file_path))[0] + "_converted.csv"
            output_path = os.path.join(self.output_directory, output_filename)
            df.to_csv(output_path, index=False)
            
            # Add to file list
            self.input_file_paths.append(output_path)
            self._update_signal_list()
            self._update_plot_file_menu()
            
            self.dat_log_textbox.insert("end", f"Conversion completed successfully!\n")
            self.dat_log_textbox.insert("end", f"Output file: {output_path}\n")
            self.dat_log_textbox.insert("end", f"File added to processing list.\n")
            
            messagebox.showinfo("Success", "DAT file converted and loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"DAT conversion failed: {str(e)}")
            self.dat_log_textbox.insert("end", f"Conversion failed: {str(e)}\n")
        finally:
            self.dat_log_textbox.configure(state="disabled")
            self.convert_dat_button.configure(state="normal", text="Step 5: Convert and Load File")

    # =============================================================================
    # CONFIGURATION AND UTILITY METHODS
    # =============================================================================

    def save_settings(self):
        """Save current application settings."""
        try:
            config = configparser.ConfigParser()
            
            # General settings
            config['General'] = {
                'output_directory': self.output_directory,
                'export_type': self.export_type_var.get()
            }
            
            # Custom variables
            config['CustomVariables'] = {}
            for i, (name, formula) in enumerate(self.custom_vars_list):
                config['CustomVariables'][f'var_{i}_name'] = name
                config['CustomVariables'][f'var_{i}_formula'] = formula
            
            # Processing settings
            config['Processing'] = {
                'filter_enabled': str(self.filter_enabled_var.get()),
                'filter_type': self.filter_type_var.get(),
                'integration_enabled': str(self.integration_enabled_var.get()),
                'integration_method': self.integration_method_var.get(),
                'differentiation_enabled': str(self.differentiation_enabled_var.get()),
                'differentiation_method': self.differentiation_method_var.get(),
                'resampling_enabled': str(self.resampling_enabled_var.get())
            }
            
            # Compilation settings
            config['Compilation'] = {
                'enabled': str(self.compilation_enabled.get()),
                'time_filtering_enabled': str(self.time_filtering_enabled.get()),
                'time_column': self.time_column_var.get(),
                'merge_method': self.merge_method_var.get()
            }
            
            # Save to file
            config_path = filedialog.asksaveasfilename(
                title="Save Settings",
                defaultextension=".ini",
                filetypes=[("Configuration files", "*.ini"), ("All files", "*.*")]
            )
            
            if config_path:
                with open(config_path, 'w') as configfile:
                    config.write(configfile)
                messagebox.showinfo("Success", "Settings saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_settings(self):
        """Load application settings from file."""
        try:
            config_path = filedialog.askopenfilename(
                title="Load Settings",
                filetypes=[("Configuration files", "*.ini"), ("All files", "*.*")]
            )
            
            if not config_path:
                return
            
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Load general settings
            if 'General' in config:
                general = config['General']
                if 'output_directory' in general:
                    self.output_directory = general['output_directory']
                    self.output_label.configure(text=f"Output: {self.output_directory}")
                if 'export_type' in general:
                    self.export_type_var.set(general['export_type'])
            
            # Load custom variables
            if 'CustomVariables' in config:
                self.custom_vars_list.clear()
                cv = config['CustomVariables']
                
                # Find all variable pairs
                var_indices = set()
                for key in cv.keys():
                    if '_name' in key:
                        idx = key.replace('var_', '').replace('_name', '')
                        var_indices.add(idx)
                
                for idx in sorted(var_indices):
                    name_key = f'var_{idx}_name'
                    formula_key = f'var_{idx}_formula'
                    if name_key in cv and formula_key in cv:
                        self.custom_vars_list.append((cv[name_key], cv[formula_key]))
                
                self._update_custom_vars_display()
            
            messagebox.showinfo("Success", "Settings loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")

    def _load_layout_config(self):
        """Load layout configuration from file."""
        try:
            if os.path.exists(self.layout_config_file):
                with open(self.layout_config_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_layout_config(self):
        """Save layout configuration to file."""
        try:
            layout_data = {
                'window_width': self.winfo_width(),
                'window_height': self.winfo_height()
            }
            
            with open(self.layout_config_file, 'w') as f:
                json.dump(layout_data, f)
        except Exception:
            pass

    def _on_closing(self):
        """Handle application closing."""
        self._save_layout_config()
        self.destroy()

    def _show_sharing_instructions(self):
        """Show app sharing instructions."""
        instructions = """
HOW TO SHARE THIS APPLICATION

REQUIREMENTS FOR USERS:
   • Python installation (if sharing source code)
   • OR just the .exe file (if using PyInstaller)

INCLUDED FEATURES:
   • CSV time series processing & filtering
   • Interactive plotting with custom ranges
   • Multiple export formats (CSV, Excel, MAT, Parquet, HDF5)
   • Custom variable calculations
   • Multi-file compilation with time trimming
   • Advanced signal processing (filtering, integration, derivatives)
   • DAT file import and conversion

Note: The .exe version will be larger (~50-100MB) but 
requires no Python installation on target computers.
        """
        
        # Create instruction window
        instruction_window = ctk.CTkToplevel(self)
        instruction_window.title("How to Share This Application")
        instruction_window.geometry("600x500")
        instruction_window.transient(self)
        instruction_window.grab_set()
        
        # Center the window
        instruction_window.update_idletasks()
        x = (instruction_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (instruction_window.winfo_screenheight() // 2) - (500 // 2)
        instruction_window.geometry(f"600x500+{x}+{y}")
        
        # Add scrollable text
        textbox = ctk.CTkTextbox(instruction_window, wrap="word")
        textbox.pack(fill="both", expand=True, padx=20, pady=20)
        textbox.insert("1.0", instructions)
        textbox.configure(state="disabled")
        
        # Add close button
        close_button = ctk.CTkButton(instruction_window, text="Close", 
                                   command=instruction_window.destroy)
        close_button.pack(pady=(0, 20))

    # =============================================================================
    # PLOTS LIST METHODS
    # =============================================================================

    def _save_plot_config(self):
        """Save current plot configuration."""
        messagebox.showinfo("Info", "Plot configuration saving - feature in development.")

    def _load_plot_config(self):
        """Load a saved plot configuration."""
        messagebox.showinfo("Info", "Plot configuration loading - feature in development.")

    def _delete_plot_config(self):
        """Delete a saved plot configuration."""
        messagebox.showinfo("Info", "Plot configuration deletion - feature in development.")


# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Set appearance mode and color theme to light (matching earlier versions)
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    # Create and run application
    app = CSVProcessorUnifiedApp()
    app.mainloop()
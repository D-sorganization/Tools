# =============================================================================
# Advanced CSV Time Series Processor & Analyzer
#
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This version includes all 11 requested
# advanced features for professional time series analysis.

# Dependencies for Python 3.8+:
# pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow
#
# =============================================================================

import configparser
import io
import json
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

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

    return (
        padded_series.rolling(window=window)
        .apply(get_deriv, raw=True)
        .iloc[window - 1 :]
    )


class CSVProcessorApp(ctk.CTk):
    """The main application class that encapsulates the entire GUI and processing logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Advanced CSV Time Series Processor v10.4")
        self.geometry("1350x900")

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

        # --- Create Main UI ---
        self.main_tab_view = ctk.CTkTabview(self)
        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_tab_view.add("1. Setup & Process")
        self.main_tab_view.add("2. Plotting & Analysis")
        self.create_setup_and_process_tab(self.main_tab_view.tab("1. Setup & Process"))
        self.create_plotting_tab(self.main_tab_view.tab("2. Plotting & Analysis"))
        self.create_status_bar()
        self.status_label.configure(
            text="Ready. Select input files on the 'Setup & Process' tab to begin."
        )

    def create_setup_and_process_tab(self, parent_tab):
        parent_tab.grid_columnconfigure(1, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)
        left_panel = ctk.CTkFrame(parent_tab, width=350)
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            left_panel, text="Control Panel", font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, padx=15, pady=10, sticky="w")
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

    def populate_setup_sub_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        file_frame = ctk.CTkFrame(tab)
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="new")
        file_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(file_frame, text="File I/O", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(10, 5), sticky="w"
        )
        ctk.CTkButton(
            file_frame, text="Select Input CSV(s)", command=self.select_files
        ).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            file_frame, text="Select Output Folder", command=self.select_output_folder
        ).grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.output_label = ctk.CTkLabel(
            file_frame,
            text=f"Output: {self.output_directory}",
            wraplength=300,
            justify="left",
            font=ctk.CTkFont(size=11),
        )
        self.output_label.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky="new")
        settings_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            settings_frame, text="Configuration & Help", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkButton(
            settings_frame, text="Save Settings", command=self.save_settings
        ).grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            settings_frame, text="Load Settings", command=self.load_settings
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(
            settings_frame,
            text="How to Share App",
            command=self._show_sharing_instructions,
        ).grid(
            row=1, column=2, padx=10, pady=5, sticky="ew"
        )  # FEATURE 5: Enhanced export options including multi-sheet Excel
        export_frame = ctk.CTkFrame(tab)
        export_frame.grid(row=2, column=0, padx=10, pady=10, sticky="new")
        export_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            export_frame, text="Export Options", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(export_frame, text="Format:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
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
            ],
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(export_frame, text="Sort By:").grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )
        self.sort_col_menu = ctk.CTkOptionMenu(
            export_frame, values=["default (no sort)"]
        )
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

    def populate_processing_sub_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        time_units = ["ms", "s", "min", "hr"]
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
        self.filter_menu = ctk.CTkOptionMenu(
            filter_frame,
            variable=self.filter_type_var,
            values=self.filter_names,
            command=self._update_filter_ui,
        )
        self.filter_menu.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        (self.ma_frame, self.ma_value_entry, self.ma_unit_menu) = (
            self._create_ma_param_frame(filter_frame, time_units)
        )
        (self.bw_frame, self.bw_order_entry, self.bw_cutoff_entry) = (
            self._create_bw_param_frame(filter_frame)
        )
        (self.median_frame, self.median_kernel_entry) = self._create_median_param_frame(
            filter_frame
        )
        (self.savgol_frame, self.savgol_window_entry, self.savgol_polyorder_entry) = (
            self._create_savgol_param_frame(filter_frame)
        )
        self._update_filter_ui("None")
        resample_frame = ctk.CTkFrame(tab)
        resample_frame.grid(row=3, column=0, padx=10, pady=10, sticky="new")
        resample_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            resample_frame, text="Time Resampling", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        self.resample_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            resample_frame, text="Enable Resampling", variable=self.resample_var
        ).grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(resample_frame, text="Time Gap:").grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )
        resample_time_frame = ctk.CTkFrame(resample_frame, fg_color="transparent")
        resample_time_frame.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        resample_time_frame.grid_columnconfigure(0, weight=2)
        resample_time_frame.grid_columnconfigure(1, weight=1)
        self.resample_value_entry = ctk.CTkEntry(
            resample_time_frame, placeholder_text="e.g., 10"
        )
        self.resample_value_entry.grid(row=0, column=0, sticky="ew")
        self.resample_unit_menu = ctk.CTkOptionMenu(
            resample_time_frame, values=time_units
        )
        self.resample_unit_menu.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        deriv_frame = ctk.CTkFrame(tab)
        deriv_frame.grid(row=4, column=0, padx=10, pady=10, sticky="new")
        deriv_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            deriv_frame, text="Derivatives", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        ctk.CTkLabel(deriv_frame, text="Method:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        self.deriv_method_var = ctk.StringVar(value="Spline (Acausal)")
        ctk.CTkOptionMenu(
            deriv_frame,
            variable=self.deriv_method_var,
            values=["Spline (Acausal)", "Rolling Polynomial (Causal)"],
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.derivative_vars = {}
        for i in range(1, 5):
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(deriv_frame, text=f"Order {i}", variable=var)
            cb.grid(row=i + 1, column=0, columnspan=2, padx=10, pady=2, sticky="w")
            self.derivative_vars[i] = var

    def populate_custom_var_sub_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(8, weight=1)  # Allow reference list to expand

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
            row=2, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        self.custom_var_name_entry = ctk.CTkEntry(
            tab, placeholder_text="e.g., Power_Ratio"
        )
        self.custom_var_name_entry.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(tab, text="Formula:").grid(
            row=4, column=0, padx=10, pady=(5, 0), sticky="w"
        )
        self.custom_var_formula_entry = ctk.CTkEntry(
            tab, placeholder_text="e.g., ( [SignalA] + [SignalB] ) / 2"
        )
        self.custom_var_formula_entry.grid(
            row=5, column=0, padx=10, pady=5, sticky="ew"
        )

        ctk.CTkButton(
            tab, text="Add Custom Variable", command=self._add_custom_variable
        ).grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        # --- NEW SEARCHABLE REFERENCE LIST ---
        reference_frame = ctk.CTkFrame(tab)
        reference_frame.grid(row=7, column=0, rowspan=2, padx=10, pady=5, sticky="nsew")
        reference_frame.grid_columnconfigure(0, weight=1)
        reference_frame.grid_rowconfigure(1, weight=1)

        search_bar_frame = ctk.CTkFrame(reference_frame, fg_color="transparent")
        search_bar_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        search_bar_frame.grid_columnconfigure(0, weight=1)

        self.custom_var_search_entry = ctk.CTkEntry(
            search_bar_frame, placeholder_text="Search available signals..."
        )
        self.custom_var_search_entry.grid(row=0, column=0, sticky="ew")
        self.custom_var_search_entry.bind(
            "<KeyRelease>", self._filter_reference_signals
        )

        self.custom_var_clear_button = ctk.CTkButton(
            search_bar_frame, text="X", width=28, command=self._clear_reference_search
        )
        self.custom_var_clear_button.grid(row=0, column=1, padx=(5, 0))

        self.signal_reference_frame = ctk.CTkScrollableFrame(
            reference_frame, label_text="Available Signals Reference"
        )
        self.signal_reference_frame.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")

    def _create_ma_param_frame(self, parent, time_units):
        """Creates the parameter frame for Moving Average using .pack()"""
        frame = ctk.CTkFrame(parent)

        inner_frame = ctk.CTkFrame(frame, fg_color="transparent")
        inner_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(inner_frame, text="Time Window:").pack(side="left")

        entry = ctk.CTkEntry(inner_frame, placeholder_text="e.g., 30")
        entry.pack(side="left", fill="x", expand=True, padx=5)

        menu = ctk.CTkOptionMenu(inner_frame, values=time_units)
        menu.set(time_units[1])
        menu.pack(side="left")

        return frame, entry, menu

    def _create_bw_param_frame(self, parent):
        """Creates the parameter frame for Butterworth filter using .pack()"""
        frame = ctk.CTkFrame(parent)

        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row1, text="Filter Order:", width=110, anchor="w").pack(
            side="left"
        )
        entry_ord = ctk.CTkEntry(row1, placeholder_text="e.g., 3")
        entry_ord.pack(side="left", fill="x", expand=True)

        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row2, text="Cutoff Freq (Hz):", width=110, anchor="w").pack(
            side="left"
        )
        entry_cut = ctk.CTkEntry(row2, placeholder_text="e.g., 0.1")
        entry_cut.pack(side="left", fill="x", expand=True)

        return frame, entry_ord, entry_cut

    def _create_median_param_frame(self, parent):
        """Creates the parameter frame for Median filter using .pack()"""
        frame = ctk.CTkFrame(parent)

        inner_frame = ctk.CTkFrame(frame, fg_color="transparent")
        inner_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(inner_frame, text="Kernel Size:", width=110, anchor="w").pack(
            side="left"
        )
        entry = ctk.CTkEntry(inner_frame, placeholder_text="Odd integer, e.g., 5")
        entry.pack(side="left", fill="x", expand=True)

        return frame, entry

    def _create_savgol_param_frame(self, parent):
        """Creates the parameter frame for Savitzky-Golay filter using .pack()"""
        frame = ctk.CTkFrame(parent)

        row1 = ctk.CTkFrame(frame, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row1, text="Window Len:", width=110, anchor="w").pack(side="left")
        entry_win = ctk.CTkEntry(row1, placeholder_text="Odd integer, e.g., 11")
        entry_win.pack(side="left", fill="x", expand=True)

        # This row had the typo 'cтk' which is now corrected to 'ctk'
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(row2, text="Poly Order:", width=110, anchor="w").pack(side="left")
        entry_poly = ctk.CTkEntry(row2, placeholder_text="e.g., 2 (< Window Len)")
        entry_poly.pack(side="left", fill="x", expand=True)

        return frame, entry_win, entry_poly

    def _update_filter_ui(self, choice):
        """Update the filter UI to show appropriate parameters."""
        self.ma_frame.grid_remove()
        self.bw_frame.grid_remove()
        self.median_frame.grid_remove()
        self.savgol_frame.grid_remove()

        if choice == "Moving Average":
            self.ma_frame.grid(
                row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
            )
        elif choice in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.bw_frame.grid(
                row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
            )
        elif choice == "Median Filter":
            self.median_frame.grid(
                row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
            )
        elif choice == "Savitzky-Golay":
            self.savgol_frame.grid(
                row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
            )

    def create_plotting_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        plot_control_frame = ctk.CTkFrame(tab)
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
        plot_main_frame = ctk.CTkFrame(tab)
        plot_main_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        plot_main_frame.grid_rowconfigure(0, weight=1)
        plot_main_frame.grid_columnconfigure(1, weight=1)
        # FEATURE 1: Scrollable plot controls panel with proper sizing
        plot_left_panel_outer = ctk.CTkFrame(plot_main_frame, width=350)
        plot_left_panel_outer.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        plot_left_panel_outer.grid_rowconfigure(0, weight=1)
        plot_left_panel_outer.grid_propagate(False)
        plot_left_panel = ctk.CTkScrollableFrame(
            plot_left_panel_outer, label_text="Plotting Controls", height=600
        )
        plot_left_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        plot_left_panel_outer.grid_columnconfigure(0, weight=1)
        # FEATURE 3: Enhanced signal selection controls for plotting
        plot_signal_select_frame = ctk.CTkFrame(plot_left_panel)
        plot_signal_select_frame.pack(fill="x", expand=False, pady=5)
        plot_signal_select_frame.grid_columnconfigure(0, weight=1)
        self.plot_search_entry = ctk.CTkEntry(
            plot_signal_select_frame, placeholder_text="Search plot signals..."
        )
        self.plot_search_entry.grid(
            row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5
        )
        self.plot_search_entry.bind("<KeyRelease>", self._filter_plot_signals)
        ctk.CTkButton(
            plot_signal_select_frame, text="All", command=self._plot_select_all
        ).grid(row=1, column=0, sticky="ew", padx=2, pady=5)
        ctk.CTkButton(
            plot_signal_select_frame, text="None", command=self._plot_select_none
        ).grid(row=1, column=1, sticky="ew", padx=2, pady=5)
        ctk.CTkButton(
            plot_signal_select_frame,
            text="Show Selected",
            command=self._show_selected_signals,
        ).grid(row=1, column=2, sticky="ew", padx=2, pady=5)
        ctk.CTkButton(
            plot_signal_select_frame,
            text="X",
            width=28,
            command=self._plot_clear_search,
        ).grid(row=1, column=3, sticky="w", padx=2, pady=5)
        self.plot_signal_frame = ctk.CTkScrollableFrame(
            plot_left_panel, label_text="Signals to Plot", height=150
        )
        self.plot_signal_frame.pack(expand=True, fill="both", padx=5, pady=5)

        appearance_frame = ctk.CTkFrame(plot_left_panel)
        appearance_frame.pack(fill="x", expand=False, pady=5)
        ctk.CTkLabel(
            appearance_frame, text="Plot Appearance", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        ctk.CTkLabel(appearance_frame, text="Chart Type:").pack(anchor="w", padx=10)
        self.plot_type_var = ctk.StringVar(value="Line with Markers")
        ctk.CTkOptionMenu(
            appearance_frame,
            variable=self.plot_type_var,
            values=["Line with Markers", "Line Only", "Markers Only (Scatter)"],
        ).pack(fill="x", padx=10, pady=5)
        self.plot_title_entry = ctk.CTkEntry(
            appearance_frame, placeholder_text="Plot Title"
        )
        self.plot_title_entry.pack(fill="x", padx=10, pady=5)
        self.plot_xlabel_entry = ctk.CTkEntry(
            appearance_frame, placeholder_text="X-Axis Label"
        )
        self.plot_xlabel_entry.pack(fill="x", padx=10, pady=5)
        self.plot_ylabel_entry = ctk.CTkEntry(
            appearance_frame, placeholder_text="Y-Axis Label"
        )
        self.plot_ylabel_entry.pack(fill="x", padx=10, pady=5)

        plot_filter_frame = ctk.CTkFrame(plot_left_panel)
        plot_filter_frame.pack(fill="x", expand=False, pady=5)
        ctk.CTkLabel(
            plot_filter_frame, text="Filter Preview", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        self.plot_filter_type = ctk.StringVar(value="None")
        self.plot_filter_menu = ctk.CTkOptionMenu(
            plot_left_panel,
            variable=self.plot_filter_type,
            values=self.filter_names,
            command=self._update_plot_filter_ui,
        )
        self.plot_filter_menu.pack(fill="x", padx=5, pady=5)
        time_units = ["ms", "s", "min", "hr"]
        (self.plot_ma_frame, self.plot_ma_value_entry, self.plot_ma_unit_menu) = (
            self._create_ma_param_frame(plot_filter_frame, time_units)
        )
        (self.plot_bw_frame, self.plot_bw_order_entry, self.plot_bw_cutoff_entry) = (
            self._create_bw_param_frame(plot_filter_frame)
        )
        (self.plot_median_frame, self.plot_median_kernel_entry) = (
            self._create_median_param_frame(plot_filter_frame)
        )
        (
            self.plot_savgol_frame,
            self.plot_savgol_window_entry,
            self.plot_savgol_polyorder_entry,
        ) = self._create_savgol_param_frame(plot_filter_frame)
        self._update_plot_filter_ui("None")
        ctk.CTkButton(
            plot_filter_frame,
            text="Copy Settings to Processing Tab",
            command=self._copy_plot_settings_to_processing,
        ).pack(fill="x", padx=10, pady=5)

        trend_frame = ctk.CTkFrame(plot_left_panel)
        trend_frame.pack(fill="x", expand=False, pady=5)
        ctk.CTkLabel(
            trend_frame,
            text="Trendline (plots 1st selected signal)",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=10, pady=5)
        self.trendline_type_var = ctk.StringVar(value="None")
        ctk.CTkOptionMenu(
            trend_frame,
            variable=self.trendline_type_var,
            values=["None", "Linear", "Exponential", "Power", "Polynomial"],
        ).pack(fill="x", padx=10, pady=5)
        self.poly_order_entry = ctk.CTkEntry(
            trend_frame, placeholder_text="Polynomial Order (2-6)"
        )
        self.poly_order_entry.pack(fill="x", padx=10, pady=5)
        self.trendline_textbox = ctk.CTkTextbox(trend_frame, height=70)
        self.trendline_textbox.pack(fill="x", expand=True, padx=10, pady=5)

        ctk.CTkButton(
            plot_left_panel, text="Update Plot", height=35, command=self.update_plot
        ).pack(fill="x", padx=5, pady=10)

        # FEATURE 11: Chart export options
        export_chart_frame = ctk.CTkFrame(plot_left_panel)
        export_chart_frame.pack(fill="x", expand=False, pady=5)
        ctk.CTkLabel(
            export_chart_frame, text="Export Chart", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        ctk.CTkButton(
            export_chart_frame, text="Save as PNG/PDF", command=self._export_chart_image
        ).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(
            export_chart_frame,
            text="Export to Excel with Chart",
            command=self._export_chart_excel,
        ).pack(fill="x", padx=10, pady=2)

        # Plot Range Controls
        plot_range_frame = ctk.CTkFrame(plot_left_panel)
        plot_range_frame.pack(fill="x", expand=False, pady=5)
        ctk.CTkLabel(
            plot_range_frame, text="Plot Time Range", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        ctk.CTkLabel(plot_range_frame, text="Start Time (HH:MM:SS):").pack(
            fill="x", padx=10
        )
        self.plot_start_entry = ctk.CTkEntry(
            plot_range_frame, placeholder_text="e.g., 09:30:00"
        )
        self.plot_start_entry.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkLabel(plot_range_frame, text="End Time (HH:MM:SS):").pack(
            fill="x", padx=10
        )
        self.plot_end_entry = ctk.CTkEntry(
            plot_range_frame, placeholder_text="e.g., 17:00:00"
        )
        self.plot_end_entry.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkButton(
            plot_range_frame,
            text="Apply Time Range to Plot",
            command=self._apply_plot_time_range,
        ).pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(
            plot_range_frame, text="Reset to Full Range", command=self._reset_plot_range
        ).pack(fill="x", padx=10, pady=(0, 10))

        trim_frame = ctk.CTkFrame(plot_left_panel)
        trim_frame.pack(fill="x", expand=False, pady=5)
        ctk.CTkLabel(
            trim_frame, text="Trim & Export", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        self.trim_date_range_label = ctk.CTkLabel(
            trim_frame, text="Data on date:", text_color="gray"
        )
        self.trim_date_range_label.pack(fill="x", padx=10)
        self.trim_date_entry = ctk.CTkEntry(
            trim_frame, placeholder_text="Date (YYYY-MM-DD)"
        )
        self.trim_date_entry.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(trim_frame, text="Start Time (HH:MM:SS):").pack(fill="x", padx=10)
        self.trim_start_entry = ctk.CTkEntry(
            trim_frame, placeholder_text="e.g., 09:30:00"
        )
        self.trim_start_entry.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkLabel(trim_frame, text="End Time (HH:MM:SS):").pack(fill="x", padx=10)
        self.trim_end_entry = ctk.CTkEntry(
            trim_frame, placeholder_text="e.g., 17:00:00"
        )
        self.trim_end_entry.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkButton(
            trim_frame,
            text="Copy Times to Plot Range",
            command=self._copy_trim_to_plot_range,
        ).pack(fill="x", padx=10, pady=5)
        self.trim_resample_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            trim_frame, text="Resample on Save", variable=self.trim_resample_var
        ).pack(anchor="w", padx=10, pady=5)
        ctk.CTkButton(
            trim_frame, text="Trim & Save As...", command=self.trim_and_save
        ).pack(fill="x", padx=10, pady=(0, 10))

        plot_canvas_frame = ctk.CTkFrame(plot_main_frame)
        plot_canvas_frame.grid(
            row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 10), pady=10
        )
        plot_canvas_frame.grid_rowconfigure(1, weight=1)
        plot_canvas_frame.grid_columnconfigure(0, weight=1)
        self.plot_fig = Figure(figsize=(5, 4), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_fig.tight_layout()
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=plot_canvas_frame)
        self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(
            self.plot_canvas, plot_canvas_frame, pack_toolbar=False
        )
        toolbar.grid(row=0, column=0, sticky="ew")

    def update_plot(self, event=None):
        """The main function to draw/redraw the plot with all selected options."""
        selected_file = self.plot_file_menu.get()
        x_axis_col = self.plot_xaxis_menu.get()

        if selected_file == "Select a file..." or not x_axis_col:
            return

        df = self.get_data_for_plotting(selected_file)
        if df is None or df.empty:
            self.plot_ax.clear()
            self.plot_ax.text(
                0.5, 0.5, "Could not load or plot data.", ha="center", va="center"
            )
            self.plot_canvas.draw()
            return

        signals_to_plot = [
            s for s, data in self.plot_signal_vars.items() if data["var"].get()
        ]
        self.plot_ax.clear()
        self.trendline_textbox.delete("1.0", "end")  # Clear trendline info

        if not signals_to_plot:
            self.plot_ax.text(
                0.5, 0.5, "Select one or more signals to plot", ha="center", va="center"
            )
        else:
            # FEATURE 9: Chart Customization
            plot_style = self.plot_type_var.get()
            style_args = {"linestyle": "-", "marker": ""}
            if plot_style == "Line with Markers":
                style_args = {"linestyle": "-", "marker": ".", "markersize": 4}
            elif plot_style == "Markers Only (Scatter)":
                style_args = {"linestyle": "None", "marker": ".", "markersize": 5}

            # Plot each selected signal
            for signal in signals_to_plot:
                if signal not in df.columns:
                    continue

                plot_df = df[[x_axis_col, signal]].dropna()
                self.plot_ax.plot(
                    plot_df[x_axis_col],
                    plot_df[signal],
                    label=f"{signal} (Raw)",
                    alpha=0.7,
                    **style_args,
                )

                # Apply and plot the preview filter
                filtered_series = self._apply_plot_filter(
                    plot_df.copy(), signal, x_axis_col
                )
                if filtered_series is not None:
                    self.plot_ax.plot(
                        filtered_series.index,
                        filtered_series.values,
                        label=f"{signal} (Filtered)",
                        lw=2,
                    )

            # FEATURE 4: Trendlines
            trend_type = self.trendline_type_var.get()
            if trend_type != "None" and signals_to_plot:
                self._add_trendline(df, x_axis_col, signals_to_plot[0], trend_type)

        # Apply custom labels and title
        title = self.plot_title_entry.get() or f"Signals from {selected_file}"
        xlabel = self.plot_xlabel_entry.get() or x_axis_col
        ylabel = self.plot_ylabel_entry.get() or "Value"
        self.plot_ax.set_title(title, fontsize=14)
        self.plot_ax.set_xlabel(xlabel)
        self.plot_ax.set_ylabel(ylabel)

        self.plot_ax.legend()
        self.plot_ax.grid(True, linestyle="--", alpha=0.6)

        if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
            # Set the format to show only Hour:Minute
            self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # Ensure the labels are not rotated (horizontal)
            self.plot_ax.tick_params(axis="x", rotation=0)

        self.plot_canvas.draw()

    def _apply_plot_filter(self, df, signal_col, x_axis_col):
        """Applies the selected filter from the plot tab to a given signal."""
        filter_type = self.plot_filter_type.get()
        if filter_type == "None" or df.empty:
            return None

        df_indexed = df.set_index(x_axis_col)
        signal_data = df_indexed[signal_col].dropna()
        if len(signal_data) < 2:
            return None

        try:
            if filter_type == "Moving Average":
                val = self.plot_ma_value_entry.get()
                unit = self.plot_ma_unit_menu.get()
                if val:
                    return signal_data.rolling(
                        window=f"{val}{unit}", min_periods=1
                    ).mean()
            elif filter_type in ["Butterworth Low-pass", "Butterworth High-pass"]:
                sr = (
                    1.0
                    / pd.to_numeric(
                        signal_data.index.to_series().diff().dt.total_seconds()
                    ).mean()
                )
                if (
                    pd.notna(sr)
                    and len(signal_data) > int(self.plot_bw_order_entry.get()) * 3
                ):
                    order = int(self.plot_bw_order_entry.get())
                    cutoff = float(self.plot_bw_cutoff_entry.get())
                    btype = "low" if filter_type == "Butterworth Low-pass" else "high"
                    b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sr)
                    return pd.Series(
                        filtfilt(b, a, signal_data), index=signal_data.index
                    )
            elif filter_type == "Median Filter":
                kernel = int(self.plot_median_kernel_entry.get())
                if kernel % 2 == 0:
                    kernel += 1  # Kernel must be odd
                if len(signal_data) > kernel:
                    return pd.Series(
                        medfilt(signal_data, kernel_size=kernel),
                        index=signal_data.index,
                    )
            elif filter_type == "Savitzky-Golay":
                win = int(self.plot_savgol_window_entry.get())
                poly = int(self.plot_savgol_polyorder_entry.get())
                if win % 2 == 0:
                    win += 1  # Window must be odd
                if poly >= win:
                    poly = win - 1 if win > 1 else 0
                if len(signal_data) > win:
                    return pd.Series(
                        savgol_filter(signal_data, win, poly), index=signal_data.index
                    )
        except Exception as e:
            print(f"Could not apply plot filter '{filter_type}': {e}")
        return None

    def _add_trendline(self, df, x_col, y_col, trend_type):
        """Calculates and plots a trendline for the first selected signal."""
        trend_df = df[[x_col, y_col]].dropna()

        # Convert x-axis to numeric for regression
        if pd.api.types.is_datetime64_any_dtype(trend_df[x_col]):
            x_numeric = mdates.date2num(trend_df[x_col])
        else:
            x_numeric = trend_df[x_col]
        y_numeric = trend_df[y_col]

        if len(x_numeric) < 3:
            return  # Need at least 3 points for a meaningful regression

        try:
            y_fit = None
            equation = ""
            r_squared_text = ""

            # Manual R-squared calculation function
            def calculate_r_squared(y_true, y_pred):
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                if ss_tot == 0:
                    return 1.0  # Perfect fit if all y values are the same
                return 1 - (ss_res / ss_tot)

            if trend_type == "Linear":
                coeffs = np.polyfit(x_numeric, y_numeric, 1)
                p = np.poly1d(coeffs)
                y_fit = p(x_numeric)
                equation = f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"
                r_squared_text = f"R² = {calculate_r_squared(y_numeric, y_fit):.4f}"

            elif trend_type == "Polynomial":
                order = int(self.poly_order_entry.get())
                if not (2 <= order <= 6):
                    order = 2
                coeffs = np.polyfit(x_numeric, y_numeric, order)
                p = np.poly1d(coeffs)
                y_fit = p(x_numeric)
                equation = "y = " + " + ".join(
                    [f"{c:.2f}x^{order-i}" for i, c in enumerate(coeffs)]
                ).replace("x^1", "x").replace("x^0", "")
                r_squared_text = f"R² = {calculate_r_squared(y_numeric, y_fit):.4f}"

            elif trend_type == "Exponential" and (y_numeric > 0).all():
                coeffs = np.polyfit(x_numeric, np.log(y_numeric), 1)
                y_fit = np.exp(coeffs[1]) * np.exp(coeffs[0] * x_numeric)
                equation = f"y = {np.exp(coeffs[1]):.3f} * e^({coeffs[0]:.3f}x)"
                r_squared_text = f"R² = {calculate_r_squared(y_numeric, y_fit):.4f}"

            elif (
                trend_type == "Power"
                and (y_numeric > 0).all()
                and (x_numeric > 0).all()
            ):
                coeffs = np.polyfit(np.log(x_numeric), np.log(y_numeric), 1)
                y_fit = np.exp(coeffs[1]) * (x_numeric ** coeffs[0])
                equation = f"y = {np.exp(coeffs[1]):.3f} * x^{coeffs[0]:.3f}"
                r_squared_text = f"R² = {calculate_r_squared(y_numeric, y_fit):.4f}"

            if y_fit is not None:
                self.plot_ax.plot(
                    trend_df[x_col],
                    y_fit,
                    linestyle="--",
                    lw=2,
                    color="red",
                    label=f"{trend_type} Trend",
                )
                self.trendline_textbox.insert("1.0", f"{equation}\n{r_squared_text}")

        except Exception as e:
            self.trendline_textbox.insert(
                "1.0", f"Could not fit {trend_type} trendline.\nError: {e}"
            )

    def _copy_trim_to_plot_range(self):
        """Copies the start/end times from the trim section to the plot range section."""
        start_trim = self.trim_start_entry.get()
        end_trim = self.trim_end_entry.get()

        self.plot_start_entry.delete(0, "end")
        self.plot_start_entry.insert(0, start_trim)

        self.plot_end_entry.delete(0, "end")
        self.plot_end_entry.insert(0, end_trim)

        # Automatically apply the new range to the plot
        self._apply_plot_time_range()

    def trim_and_save(self):
        """Trims the data of the currently selected file and saves it to a new CSV."""
        selected_file = self.plot_file_menu.get()
        if selected_file == "Select a file...":
            messagebox.showwarning("Warning", "Please select a file to trim.")
            return

        df = self.get_data_for_plotting(selected_file)
        if df is None:
            return

        try:
            date_str = self.trim_date_entry.get()
            start_time_str = self.trim_start_entry.get() or "00:00:00"
            end_time_str = self.trim_end_entry.get() or "23:59:59"
            time_col = df.columns[0]

            start_full_str = f"{date_str} {start_time_str}"
            end_full_str = f"{date_str} {end_time_str}"

            # Set time column as index to perform time-based slicing
            trimmed_df = df.set_index(time_col).loc[start_full_str:end_full_str]

            # Resample if the user has checked the box
            if self.trim_resample_var.get():
                resample_value = self.resample_value_entry.get()
                resample_unit = self.resample_unit_menu.get()
                if resample_value:
                    rule = f"{resample_value}{resample_unit}"
                    trimmed_df = trimmed_df.resample(rule).mean().dropna(how="all")

            trimmed_df.reset_index(inplace=True)

            if trimmed_df.empty:
                messagebox.showwarning(
                    "Warning", "The specified time range resulted in an empty dataset."
                )
                return

            save_path = filedialog.asksaveasfilename(
                title="Save Trimmed File As...",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=f"{os.path.splitext(selected_file)[0]}_trimmed.csv",
            )

            if save_path:
                trimmed_df.to_csv(save_path, index=False)
                self.status_label.configure(
                    text=f"Trimmed file saved to {os.path.basename(save_path)}"
                )
                messagebox.showinfo("Success", "Trimmed file saved successfully.")
        except Exception as e:
            messagebox.showerror(
                "Trimming Error",
                f"An error occurred.\nEnsure Date is YYYY-MM-DD and Time is HH:MM:SS.\n\nError: {e}",
            )

    def _apply_plot_time_range(self):
        """Applies a time filter to the x-axis of the plot."""
        start_time_str = self.plot_start_entry.get()
        end_time_str = self.plot_end_entry.get()
        if not start_time_str and not end_time_str:
            return

        try:
            xmin, xmax = self.plot_ax.get_xlim()
            start_num = (
                mdates.datestr2num(f"1900-01-01 {start_time_str}")
                if start_time_str
                else xmin
            )
            end_num = (
                mdates.datestr2num(f"1900-01-01 {end_time_str}")
                if end_time_str
                else xmax
            )

            # Use only the time part for setting limits
            self.plot_ax.set_xlim(left=start_num, right=end_num)
            self.plot_canvas.draw()
        except Exception as e:
            messagebox.showerror(
                "Time Range Error", f"Invalid time format. Please use HH:MM:SS.\n{e}"
            )

    def _reset_plot_range(self):
        """Resets the plot view to its full default range."""
        self.plot_start_entry.delete(0, "end")
        self.plot_end_entry.delete(0, "end")
        self.update_plot()

    def _update_plot_filter_ui(self, choice):
        self.plot_ma_frame.pack_forget()
        self.plot_bw_frame.pack_forget()
        self.plot_median_frame.pack_forget()
        self.plot_savgol_frame.pack_forget()
        if choice == "Moving Average":
            self.plot_ma_frame.pack(fill="x", expand=True, padx=5, pady=2)
        elif choice in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.plot_bw_frame.pack(fill="x", expand=True, padx=5, pady=2)
        elif choice == "Median Filter":
            self.plot_median_frame.pack(fill="x", expand=True, padx=5, pady=2)
        elif choice == "Savitzky-Golay":
            self.plot_savgol_frame.pack(fill="x", expand=True, padx=5, pady=2)

    def create_status_bar(self):
        status_frame = ctk.CTkFrame(self, height=30)
        status_frame.grid(
            row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="sew"
        )
        status_frame.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(status_frame, text="", anchor="w")
        self.status_label.grid(row=0, column=0, padx=10, sticky="ew")
        self.progressbar = ctk.CTkProgressBar(status_frame, orientation="horizontal")
        self.progressbar.set(0)
        self.progressbar.grid(row=0, column=1, padx=10, sticky="e")

    def select_files(self):
        """Opens a dialog to select multiple CSV files and updates the UI."""
        self.input_file_paths = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )

        # Clear the initial "files will be listed here" label
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()

        if self.input_file_paths:
            self.loaded_data_cache.clear()  # Clear cache on new file selection

            # Update the file list frame
            for f_path in self.input_file_paths:
                label = ctk.CTkLabel(
                    self.file_list_frame, text=os.path.basename(f_path)
                )
                label.pack(anchor="w", padx=5)

            # Update the signal list based on all selected files
            self.update_signal_list()

            # Update the plotting tab's file dropdown
            file_names = [os.path.basename(p) for p in self.input_file_paths]
            self.plot_file_menu.configure(values=file_names)
            if file_names:
                self.plot_file_menu.set(file_names[0])
                self.on_plot_file_select(file_names[0])  # Load data for the first file

            self.status_label.configure(
                text=f"Loaded {len(self.input_file_paths)} files. Ready."
            )
        else:
            # If no files are selected, show the initial message again
            self.initial_file_label = ctk.CTkLabel(
                self.file_list_frame, text="No files selected."
            )
            self.initial_file_label.pack(padx=5, pady=5)
            self.status_label.configure(text="File selection cancelled.")

    def update_signal_list(self):
        """Reads headers from all selected CSVs and populates ALL signal lists."""
        # --- Clear all relevant widgets ---
        for widget in self.signal_list_frame.winfo_children():
            widget.destroy()
        self.signal_vars.clear()

        for widget in self.signal_reference_frame.winfo_children():
            widget.destroy()
        self.reference_signal_widgets.clear()

        all_columns = set()
        if not self.input_file_paths:
            label = ctk.CTkLabel(
                self.signal_reference_frame, text="Load a file to see signals..."
            )
            label.pack(padx=5, pady=5)
            return

        try:
            for f in self.input_file_paths:
                df = pd.read_csv(f, nrows=0)
                all_columns.update(df.columns)
        except Exception as e:
            messagebox.showerror(
                "Error Reading Files", f"Could not read headers from files.\nError: {e}"
            )
            return

        all_columns.update([var[0] for var in self.custom_vars_list])
        sorted_columns = sorted(list(all_columns))

        # --- Populate all three signal-related areas ---
        self.search_entry.delete(0, "end")
        self.custom_var_search_entry.delete(0, "end")

        for signal in sorted_columns:
            # 1. Populate Processing Tab Signal List
            if signal not in self.signal_vars:
                var = tk.BooleanVar(value=True)
                cb = ctk.CTkCheckBox(self.signal_list_frame, text=signal, variable=var)
                cb.pack(anchor="w", padx=10, pady=2)
                self.signal_vars[signal] = {"var": var, "widget": cb}

            # 2. Populate Custom Vars Reference List
            if signal not in self.reference_signal_widgets:
                label = ctk.CTkLabel(
                    self.signal_reference_frame, text=signal, anchor="w"
                )
                label.pack(anchor="w", padx=5)
                self.reference_signal_widgets[signal] = label

        # 3. Update the sorting dropdown menu
        self.sort_col_menu.configure(values=["default (no sort)"] + sorted_columns)

    def get_data_for_plotting(self, filename):
        """Loads a CSV into a pandas DataFrame and caches it."""
        if filename in self.loaded_data_cache:
            return self.loaded_data_cache[filename]

        filepath = next(
            (p for p in self.input_file_paths if os.path.basename(p) == filename), None
        )
        if not filepath:
            return None

        try:
            df = pd.read_csv(filepath, low_memory=False)
            time_col = df.columns[0]

            # Ensure first column is datetime
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df.dropna(subset=[time_col], inplace=True)

            # Ensure other columns are numeric where possible
            for col in df.columns:
                if col != time_col:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Apply custom variables
            df = self._apply_custom_variables(df)

            self.loaded_data_cache[filename] = df.copy()
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data for {filename}.\n{e}")
            return None

    def on_plot_file_select(self, filename):
        """Called when a file is selected from the plotting tab's dropdown menu."""
        if filename == "Select a file...":
            return

        df = self.get_data_for_plotting(filename)
        if df is None:
            return

        self.plot_signal_vars = {}
        for widget in self.plot_signal_frame.winfo_children():
            widget.destroy()

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        # Populate X-Axis menu and signal list
        self.plot_xaxis_menu.configure(values=all_cols)
        self.plot_xaxis_menu.set(df.columns[0])

        for signal in numeric_cols:
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                self.plot_signal_frame,
                text=signal,
                variable=var,
                command=self.update_plot,
            )  # Update plot on check
            cb.pack(anchor="w", padx=5, pady=2)
            self.plot_signal_vars[signal] = {"var": var, "widget": cb}

        # Update the date range display for the trim UI
        time_col = df.columns[0]
        if not df.empty and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            min_date = df[time_col].min()
            max_date = df[time_col].max()
            date_str = min_date.strftime("%Y-%m-%d")
            if min_date.date() != max_date.date():
                date_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

            self.trim_date_range_label.configure(text=f"Data on date: {date_str}")
            self.trim_date_entry.delete(0, "end")
            self.trim_date_entry.insert(0, min_date.strftime("%Y-%m-%d"))

        self.update_plot()

    def select_output_folder(self):
        """Opens a dialog to select the output directory."""
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_directory = path
            self.output_label.configure(text=f"Output: {self.output_directory}")

    def select_all(self):
        """Selects all signals in the main processing list."""
        for data in self.signal_vars.values():
            data["var"].set(True)

    def deselect_all(self):
        """Deselects all signals in the main processing list."""
        for data in self.signal_vars.values():
            data["var"].set(False)

    def _filter_signals(self, event=None):
        """Filters the main signal list based on the search entry."""
        search_term = self.search_entry.get().lower()
        for signal_name, data in self.signal_vars.items():
            widget = data["widget"]
            if search_term in signal_name.lower():
                widget.pack(anchor="w", padx=10, pady=2)
            else:
                widget.pack_forget()

    def _filter_plot_signals(self, event=None):
        """Filters the plot signal list based on the plot search entry."""
        search_term = self.plot_search_entry.get().lower()
        for signal_name, data in self.plot_signal_vars.items():
            widget = data["widget"]
            # Show if search term matches or if we're showing only selected
            if search_term in signal_name.lower():
                widget.pack(anchor="w", padx=5, pady=2)
            else:
                widget.pack_forget()

    def _plot_clear_search(self):
        """Clears the plot search entry and shows all signals."""
        self.plot_search_entry.delete(0, "end")
        self._filter_plot_signals()

    def _plot_select_all(self):
        """Selects all signals in the plot list and updates the plot."""
        for data in self.plot_signal_vars.values():
            data["var"].set(True)
        self.update_plot()

    def _plot_select_none(self):
        """Deselects all signals in the plot list and updates the plot."""
        for data in self.plot_signal_vars.values():
            data["var"].set(False)
        self.update_plot()

    def _show_selected_signals(self, event=None):
        """Special filter to only show signals that are currently checked."""
        self.plot_search_entry.delete(0, "end")  # Clear search bar
        for signal_name, data in self.plot_signal_vars.items():
            widget = data["widget"]
            if data["var"].get():  # If the checkbox is selected
                widget.pack(anchor="w", padx=5, pady=2)
            else:
                widget.pack_forget()

    def _clear_search(self):
        """Clears the search entry and shows all signals."""
        self.search_entry.delete(0, "end")
        self._filter_signals()

    def get_unique_filepath(self, filepath):
        """Ensures a filepath is unique by appending a number if it already exists."""
        base, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(filepath):
            filepath = f"{base}({counter}){ext}"
            counter += 1
        return filepath

    def _copy_plot_settings_to_processing(self):
        """Copies filter settings from the plot tab to the main processing tab."""
        plot_filter = self.plot_filter_type.get()
        self.filter_type_var.set(plot_filter)
        self._update_filter_ui(plot_filter)  # Update main UI to show correct frame

        def set_entry(entry, value):
            entry.delete(0, "end")
            entry.insert(0, value)

        if plot_filter == "Moving Average":
            set_entry(self.ma_value_entry, self.plot_ma_value_entry.get())
            self.ma_unit_menu.set(self.plot_ma_unit_menu.get())
        elif plot_filter in ["Butterworth Low-pass", "Butterworth High-pass"]:
            set_entry(self.bw_order_entry, self.plot_bw_order_entry.get())
            set_entry(self.bw_cutoff_entry, self.plot_bw_cutoff_entry.get())
        elif plot_filter == "Median Filter":
            set_entry(self.median_kernel_entry, self.plot_median_kernel_entry.get())
        elif plot_filter == "Savitzky-Golay":
            set_entry(self.savgol_window_entry, self.plot_savgol_window_entry.get())
            set_entry(
                self.savgol_polyorder_entry, self.plot_savgol_polyorder_entry.get()
            )

        self.status_label.configure(
            text="Plot filter settings copied to Processing tab."
        )
        messagebox.showinfo(
            "Settings Copied",
            "Filter settings from the plot tab have been applied to the main processing configuration.",
        )

    def _filter_reference_signals(self, event=None):
        """Filters the reference signal list on the Custom Vars tab."""
        search_term = self.custom_var_search_entry.get().lower()
        for signal_name, widget in self.reference_signal_widgets.items():
            if search_term in signal_name.lower():
                widget.pack(anchor="w", padx=5)
            else:
                widget.pack_forget()

    def _clear_reference_search(self):
        """Clears the reference search and shows all signals."""
        self.custom_var_search_entry.delete(0, "end")
        self._filter_reference_signals()

    def save_settings(self):
        """Save current application settings to a configuration file."""
        try:
            config = configparser.ConfigParser()

            # General settings
            config["General"] = {
                "output_directory": self.output_directory,
                "export_type": self.export_type_var.get(),
                "sort_order": self.sort_order_var.get(),
            }

            # Filter settings
            config["Filters"] = {
                "filter_type": self.filter_type_var.get(),
                "ma_value": self.ma_value_entry.get(),
                "ma_unit": self.ma_unit_menu.get(),
                "bw_order": self.bw_order_entry.get(),
                "bw_cutoff": self.bw_cutoff_entry.get(),
                "median_kernel": self.median_kernel_entry.get(),
                "savgol_window": self.savgol_window_entry.get(),
                "savgol_polyorder": self.savgol_polyorder_entry.get(),
            }

            # Resample settings
            config["Resample"] = {
                "enable_resample": str(self.resample_var.get()),
                "resample_value": self.resample_value_entry.get(),
                "resample_unit": self.resample_unit_menu.get(),
            }

            # Derivative settings
            config["Derivatives"] = {"method": self.deriv_method_var.get()}
            for i, var in self.derivative_vars.items():
                config["Derivatives"][f"order_{i}"] = str(var.get())

            # Custom variables
            config["CustomVariables"] = {}
            for i, (name, formula) in enumerate(self.custom_vars_list):
                config["CustomVariables"][f"var_{i}_name"] = name
                config["CustomVariables"][f"var_{i}_formula"] = formula

            # Save to file
            save_path = filedialog.asksaveasfilename(
                title="Save Settings",
                defaultextension=".ini",
                filetypes=[("Configuration files", "*.ini"), ("All files", "*.*")],
            )

            if save_path:
                with open(save_path, "w") as configfile:
                    config.write(configfile)
                messagebox.showinfo("Success", f"Settings saved to:\n{save_path}")
                self.status_label.configure(
                    text=f"Settings saved: {os.path.basename(save_path)}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{e}")

    def load_settings(self):
        """Load application settings from a configuration file."""
        try:
            load_path = filedialog.askopenfilename(
                title="Load Settings",
                filetypes=[("Configuration files", "*.ini"), ("All files", "*.*")],
            )

            if not load_path:
                return

            config = configparser.ConfigParser()
            config.read(load_path)

            # Load general settings
            if "General" in config:
                general = config["General"]
                if "output_directory" in general:
                    self.output_directory = general["output_directory"]
                    self.output_label.configure(text=f"Output: {self.output_directory}")
                if "export_type" in general:
                    self.export_type_var.set(general["export_type"])
                if "sort_order" in general:
                    self.sort_order_var.set(general["sort_order"])

            # Load filter settings
            if "Filters" in config:
                filters = config["Filters"]
                if "filter_type" in filters:
                    self.filter_type_var.set(filters["filter_type"])
                    self._update_filter_ui(filters["filter_type"])
                if "ma_value" in filters:
                    self.ma_value_entry.delete(0, "end")
                    self.ma_value_entry.insert(0, filters["ma_value"])
                if "ma_unit" in filters:
                    self.ma_unit_menu.set(filters["ma_unit"])
                if "bw_order" in filters:
                    self.bw_order_entry.delete(0, "end")
                    self.bw_order_entry.insert(0, filters["bw_order"])
                if "bw_cutoff" in filters:
                    self.bw_cutoff_entry.delete(0, "end")
                    self.bw_cutoff_entry.insert(0, filters["bw_cutoff"])
                if "median_kernel" in filters:
                    self.median_kernel_entry.delete(0, "end")
                    self.median_kernel_entry.insert(0, filters["median_kernel"])
                if "savgol_window" in filters:
                    self.savgol_window_entry.delete(0, "end")
                    self.savgol_window_entry.insert(0, filters["savgol_window"])
                if "savgol_polyorder" in filters:
                    self.savgol_polyorder_entry.delete(0, "end")
                    self.savgol_polyorder_entry.insert(0, filters["savgol_polyorder"])

            # Load resample settings
            if "Resample" in config:
                resample = config["Resample"]
                if "enable_resample" in resample:
                    self.resample_var.set(resample.getboolean("enable_resample"))
                if "resample_value" in resample:
                    self.resample_value_entry.delete(0, "end")
                    self.resample_value_entry.insert(0, resample["resample_value"])
                if "resample_unit" in resample:
                    self.resample_unit_menu.set(resample["resample_unit"])

            # Load derivative settings
            if "Derivatives" in config:
                derivatives = config["Derivatives"]
                if "method" in derivatives:
                    self.deriv_method_var.set(derivatives["method"])
                for i in range(1, 5):
                    key = f"order_{i}"
                    if key in derivatives:
                        self.derivative_vars[i].set(derivatives.getboolean(key))

            # Load custom variables
            if "CustomVariables" in config:
                custom_vars = config["CustomVariables"]
                self.custom_vars_list.clear()

                # Group by variable index
                var_dict = {}
                for key, value in custom_vars.items():
                    if "_name" in key:
                        var_idx = key.split("_")[1]
                        if var_idx not in var_dict:
                            var_dict[var_idx] = {}
                        var_dict[var_idx]["name"] = value
                    elif "_formula" in key:
                        var_idx = key.split("_")[1]
                        if var_idx not in var_dict:
                            var_dict[var_idx] = {}
                        var_dict[var_idx]["formula"] = value

                # Reconstruct custom variables list
                for var_idx, var_data in var_dict.items():
                    if "name" in var_data and "formula" in var_data:
                        self.custom_vars_list.append(
                            (var_data["name"], var_data["formula"])
                        )

                self._update_custom_vars_listbox()

            messagebox.showinfo("Success", f"Settings loaded from:\n{load_path}")
            self.status_label.configure(
                text=f"Settings loaded: {os.path.basename(load_path)}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings:\n{e}")

    def _show_sharing_instructions(self):
        """Show instructions for sharing the application."""
        instructions = """
How to Share This Application:

1. SHARE THE SOURCE CODE:
   • Copy the entire Python file (TryAgain.py)
   • Recipients need Python 3.8+ with these packages:
     - customtkinter, pandas, numpy, scipy
     - matplotlib, openpyxl, Pillow

2. CREATE AN EXECUTABLE:
   • Install PyInstaller: pip install pyinstaller
   • Run: pyinstaller --onefile --windowed TryAgain.py
   • Share the generated .exe file (in 'dist' folder)

3. REQUIREMENTS FOR USERS:
   • Python installation (if sharing source code)
   • OR just the .exe file (if using PyInstaller)

4. INCLUDED FEATURES:
   • CSV time series processing & filtering
   • Interactive plotting with custom ranges
   • Multiple export formats (CSV, Excel, MAT)
   • Custom variable calculations
   • Trim & save functionality

Note: The .exe version will be larger (~50-100MB) but 
requires no Python installation on target computers.
        """

        # Create a new window for instructions
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
        close_button = ctk.CTkButton(
            instruction_window, text="Close", command=instruction_window.destroy
        )
        close_button.pack(pady=(0, 20))

    def _apply_custom_variables(self, df):
        """Apply custom variable calculations to the dataframe."""
        # FEATURE 10: Enhanced custom variable engine
        for var_name, formula in self.custom_vars_list:
            try:
                if var_name not in df.columns:
                    # Parse the formula and substitute column names
                    eval_formula = formula

                    # Find all column references in square brackets
                    column_refs = re.findall(r"\[([^\]]+)\]", formula)

                    # Replace column references with dataframe column access
                    for col_ref in column_refs:
                        if col_ref in df.columns:
                            eval_formula = eval_formula.replace(
                                f"[{col_ref}]", f'df["{col_ref}"]'
                            )

                    # Evaluate the formula safely
                    try:
                        df[var_name] = eval(eval_formula)
                    except Exception as eval_error:
                        print(f"Error evaluating formula for {var_name}: {eval_error}")
                        df[var_name] = np.nan  # Create column with NaN values

            except Exception as e:
                print(f"Error applying custom variable {var_name}: {e}")

        return df

    def _add_custom_variable(self):
        """Add a custom variable with formula engine."""
        var_name = self.custom_var_name_entry.get().strip()
        formula = self.custom_var_formula_entry.get().strip()

        if not var_name or not formula:
            messagebox.showwarning(
                "Warning", "Please enter both a variable name and a formula."
            )
            return

        if not self._validate_custom_formula(formula):
            messagebox.showerror(
                "Error",
                "Invalid formula syntax. Use column names in [square brackets].",
            )
            return

        self.custom_vars_list.append((var_name, formula))

        # Clear the entries
        self.custom_var_name_entry.delete(0, "end")
        self.custom_var_formula_entry.delete(0, "end")

        self._update_custom_vars_listbox()

        # --- NEW LOGIC TO REFRESH PLOTTING TAB ---
        current_plot_file = self.plot_file_menu.get()
        if current_plot_file != "Select a file...":
            # Force the data to be re-loaded and re-processed with the new variable
            if current_plot_file in self.loaded_data_cache:
                del self.loaded_data_cache[current_plot_file]  # Delete cached version

            # Re-trigger the function that populates the plot signal list
            self.on_plot_file_select(current_plot_file)

        # Also update the signal list on the processing tab
        self.update_signal_list()

        messagebox.showinfo(
            "Success",
            f"Custom variable '{var_name}' added and is now available for plotting.",
        )

    def _validate_custom_formula(self, formula):
        """Validate the custom formula syntax."""
        try:
            # Check for basic mathematical operations and column references
            allowed_chars = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789[]()+-*/. _"
            )
            if not all(c in allowed_chars for c in formula):
                return False

            # Check for balanced brackets
            if formula.count("[") != formula.count("]"):
                return False

            return True
        except:
            return False

    def _update_custom_vars_listbox(self):
        """Update the custom variables listbox display."""
        # Clear existing content
        self.custom_vars_listbox.configure(state="normal")
        self.custom_vars_listbox.delete(1.0, tk.END)

        # Add each custom variable
        for i, (var_name, formula) in enumerate(self.custom_vars_list):
            self.custom_vars_listbox.insert(tk.END, f"{i+1}. {var_name} = {formula}\n")

        self.custom_vars_listbox.configure(state="disabled")

    def process_files(self):
        """Process the selected files with the configured settings."""
        if not self.input_file_paths:
            messagebox.showwarning("Warning", "Please select input files.")
            return

        selected_signals = [
            s for s, data in self.signal_vars.items() if data["var"].get()
        ]
        if not selected_signals:
            messagebox.showwarning("Warning", "Please select signals to retain.")
            return

        self.process_button.configure(state="disabled", text="Processing...")
        self.progressbar.set(0)

        try:
            export_type = self.export_type_var.get()

            # FEATURE 5 & 6: Enhanced export functionality
            if export_type == "Excel (Multi-sheet)":
                self._export_excel_multisheet(selected_signals)
            elif export_type == "MAT (Compiled)":
                self._export_mat_compiled(selected_signals)
            elif export_type == "CSV (Compiled)":
                self._export_csv_compiled(selected_signals)
            else:
                # Standard individual file processing
                self._export_individual_files(selected_signals, export_type)

            messagebox.showinfo(
                "Success",
                f"Successfully processed and saved {len(self.input_file_paths)} file(s).",
            )
            self.status_label.configure(text="Processing complete.")

        except Exception as e:
            messagebox.showerror(
                "Processing Error", f"An error occurred during processing:\n{e}"
            )
        finally:
            self.process_button.configure(
                state="normal", text="Process & Batch Export Files"
            )
            self.progressbar.set(0)

    def _export_individual_files(self, selected_signals, export_type):
        """Export each file individually."""
        for i, file_path in enumerate(self.input_file_paths):
            self.status_label.configure(
                text=f"Processing [{i+1}/{len(self.input_file_paths)}]: {os.path.basename(file_path)}"
            )
            self.progressbar.set((i + 1) / len(self.input_file_paths))
            self.update_idletasks()

            # Process the file
            df = self._process_single_file(file_path, selected_signals)

            # Save based on export type
            original_filename = os.path.basename(file_path)
            name, _ = os.path.splitext(original_filename)

            if export_type == "CSV (Separate Files)":
                output_filename = f"{name}_processed.csv"
                output_path = os.path.join(self.output_directory, output_filename)
                unique_output_path = self.get_unique_filepath(output_path)
                df.to_csv(unique_output_path, index=False)

            elif export_type == "Excel (Separate Files)":
                output_filename = f"{name}_processed.xlsx"
                output_path = os.path.join(self.output_directory, output_filename)
                unique_output_path = self.get_unique_filepath(output_path)
                with pd.ExcelWriter(unique_output_path, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Data", index=False)

            elif export_type == "MAT (Separate Files)":
                output_filename = f"{name}_processed.mat"
                output_path = os.path.join(self.output_directory, output_filename)
                unique_output_path = self.get_unique_filepath(output_path)

                # Convert to MATLAB format
                mat_dict = {}
                for col in df.columns:
                    mat_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
                    mat_dict[mat_col] = df[col].values
                savemat(unique_output_path, mat_dict)

    def _export_excel_multisheet(self, selected_signals):
        """Export all files to a single Excel workbook with multiple sheets."""
        output_filename = "processed_data_multisheet.xlsx"
        output_path = os.path.join(self.output_directory, output_filename)
        unique_output_path = self.get_unique_filepath(output_path)

        with pd.ExcelWriter(unique_output_path, engine="openpyxl") as writer:
            for i, file_path in enumerate(self.input_file_paths):
                self.status_label.configure(
                    text=f"Processing sheet [{i+1}/{len(self.input_file_paths)}]: {os.path.basename(file_path)}"
                )
                self.progressbar.set((i + 1) / len(self.input_file_paths))
                self.update_idletasks()

                df = self._process_single_file(file_path, selected_signals)

                # Create a valid sheet name
                sheet_name = os.path.splitext(os.path.basename(file_path))[0]
                sheet_name = re.sub(
                    r'[\\/*?:"<>|]', "_", sheet_name
                )  # Remove invalid characters
                sheet_name = sheet_name[:31]  # Excel sheet name limit

                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _export_csv_compiled(self, selected_signals):
        """Export all files to a single compiled CSV."""
        output_filename = "processed_data_compiled.csv"
        output_path = os.path.join(self.output_directory, output_filename)
        unique_output_path = self.get_unique_filepath(output_path)

        compiled_data = []
        for i, file_path in enumerate(self.input_file_paths):
            self.status_label.configure(
                text=f"Processing file [{i+1}/{len(self.input_file_paths)}]: {os.path.basename(file_path)}"
            )
            self.progressbar.set((i + 1) / len(self.input_file_paths))
            self.update_idletasks()

            df = self._process_single_file(file_path, selected_signals)
            df["Source_File"] = os.path.basename(file_path)
            compiled_data.append(df)

        # Combine all data
        final_df = pd.concat(compiled_data, ignore_index=True)
        final_df.to_csv(unique_output_path, index=False)

    def _export_mat_compiled(self, selected_signals):
        """Export all files to a single compiled MAT file."""
        output_filename = "processed_data_compiled.mat"
        output_path = os.path.join(self.output_directory, output_filename)
        unique_output_path = self.get_unique_filepath(output_path)

        mat_dict = {}
        for i, file_path in enumerate(self.input_file_paths):
            self.status_label.configure(
                text=f"Processing file [{i+1}/{len(self.input_file_paths)}]: {os.path.basename(file_path)}"
            )
            self.progressbar.set((i + 1) / len(self.input_file_paths))
            self.update_idletasks()

            df = self._process_single_file(file_path, selected_signals)

            # Add to MAT dictionary with file prefix
            file_prefix = re.sub(
                r"[^a-zA-Z0-9_]", "_", os.path.splitext(os.path.basename(file_path))[0]
            )
            for col in df.columns:
                mat_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
                mat_dict[f"{file_prefix}_{mat_col}"] = df[col].values

        savemat(unique_output_path, mat_dict)

    def _process_single_file(self, file_path, selected_signals):
        """Process a single file with all configured settings."""
        # Load and apply custom variables
        df = pd.read_csv(file_path, low_memory=False)
        df = self._apply_custom_variables(df)

        # FEATURE 7: Add date/time columns as default
        time_col = df.columns[0]
        if pd.api.types.is_datetime64_any_dtype(
            df[time_col]
        ) or self._can_convert_to_datetime(df[time_col]):
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

            # Add separate date and time columns
            df.insert(1, "Date", df[time_col].dt.date)
            df.insert(2, "Time_HH_MM_SS", df[time_col].dt.time)

        # Filter signals
        signals_in_file = [s for s in selected_signals if s in df.columns]
        if time_col not in signals_in_file:
            signals_in_file.insert(0, time_col)

        # Add date and time columns if they exist
        if "Date" in df.columns and "Date" not in signals_in_file:
            signals_in_file.insert(1, "Date")
        if "Time_HH_MM_SS" in df.columns and "Time_HH_MM_SS" not in signals_in_file:
            signals_in_file.insert(2, "Time_HH_MM_SS")

        processed_df = df[signals_in_file].copy()

        # FEATURE 8: Apply sorting if specified
        sort_col = self.sort_col_menu.get()
        if sort_col != "default (no sort)" and sort_col in processed_df.columns:
            ascending = self.sort_order_var.get() == "Ascending"
            processed_df = processed_df.sort_values(by=sort_col, ascending=ascending)

        return processed_df

    def _can_convert_to_datetime(self, series):
        """Check if a series can be converted to datetime."""
        try:
            pd.to_datetime(series.iloc[: min(100, len(series))], errors="raise")
            return True
        except:
            return False

    def _export_chart_image(self):
        """Export the current chart as an image file."""
        # FEATURE 11: Chart image export
        if not hasattr(self, "plot_fig") or not self.plot_fig.get_axes():
            messagebox.showwarning(
                "Warning", "No plot to export. Please create a plot first."
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
                self.plot_fig.savefig(
                    save_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor="white",
                    edgecolor="none",
                )
                messagebox.showinfo("Success", f"Chart exported to:\n{save_path}")
                self.status_label.configure(
                    text=f"Chart exported: {os.path.basename(save_path)}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart:\n{e}")

    def _export_chart_excel(self):
        """Export the current plot data and chart to Excel."""
        # FEATURE 11: Excel chart export
        selected_file = self.plot_file_menu.get()

        if selected_file == "Select a file...":
            messagebox.showwarning("Warning", "Please select a file to plot first.")
            return

        df = self.get_data_for_plotting(selected_file)
        if df is None:
            return

        signals_to_plot = [
            s for s, data in self.plot_signal_vars.items() if data["var"].get()
        ]

        if not signals_to_plot:
            messagebox.showwarning("Warning", "Please select signals to plot first.")
            return

        try:
            save_path = filedialog.asksaveasfilename(
                title="Export Data and Chart to Excel",
                filetypes=[("Excel files", "*.xlsx")],
                defaultextension=".xlsx",
            )

            if save_path:
                # Prepare data for export
                x_axis_col = self.plot_xaxis_menu.get()
                export_columns = [x_axis_col] + signals_to_plot
                export_df = df[export_columns].dropna()

                # Apply time range filtering if specified
                start_time = self.plot_start_entry.get()
                end_time = self.plot_end_entry.get()

                if start_time or end_time:
                    try:
                        time_col = df.columns[0]
                        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                            date_str = self.trim_date_entry.get() or df[time_col].iloc[
                                0
                            ].strftime("%Y-%m-%d")
                            if start_time:
                                start_datetime = pd.to_datetime(
                                    f"{date_str} {start_time}"
                                )
                                export_df = export_df[
                                    export_df[time_col] >= start_datetime
                                ]
                            if end_time:
                                end_datetime = pd.to_datetime(f"{date_str} {end_time}")
                                export_df = export_df[
                                    export_df[time_col] <= end_datetime
                                ]
                    except Exception as e:
                        print(f"Error applying time range: {e}")

                # Write to Excel with chart
                with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                    export_df.to_excel(writer, sheet_name="Data", index=False)

                    # Get the workbook and worksheet
                    workbook = writer.book
                    worksheet = writer.sheets["Data"]

                    # Add a chart (basic line chart)
                    from openpyxl.chart import LineChart, Reference

                    chart = LineChart()
                    chart.title = (
                        self.plot_title_entry.get()
                        or f"Chart of {', '.join(signals_to_plot)}"
                    )
                    chart.x_axis.title = self.plot_xlabel_entry.get() or x_axis_col
                    chart.y_axis.title = self.plot_ylabel_entry.get() or "Value"

                    # Define data ranges
                    data_rows = len(export_df) + 1  # +1 for header
                    x_values = Reference(
                        worksheet, min_col=1, min_row=2, max_row=data_rows
                    )

                    for i, signal in enumerate(signals_to_plot):
                        col_index = export_df.columns.get_loc(signal) + 1
                        y_values = Reference(
                            worksheet,
                            min_col=col_index + 1,
                            min_row=2,
                            max_row=data_rows,
                        )
                        series = chart.add_data(y_values, titles_from_data=False)
                        chart.series[i].title = signal

                    chart.set_categories(x_values)

                    # Add the chart to the worksheet
                    worksheet.add_chart(chart, "H2")

                messagebox.showinfo(
                    "Success", f"Data and chart exported to:\n{save_path}"
                )
                self.status_label.configure(
                    text=f"Excel chart exported: {os.path.basename(save_path)}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to Excel:\n{e}")


if __name__ == "__main__":
    # Set the appearance mode and default color theme
    ctk.set_appearance_mode("system")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme(
        "blue"
    )  # Themes: "blue" (standard), "green", "dark-blue"

    print("Starting application...")

    # Create and run the application
    app = CSVProcessorApp()
    app.mainloop()

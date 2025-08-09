# =============================================================================
# Advanced CSV Time Series Processor & Analyzer
#
# Author: Gemini AI
# Version: 9.2 (Definitive & Complete Release)
#
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This version includes a full-featured
# plotting/analysis tab and implements all requested UI/UX improvements.
# This is a complete, functional script with no placeholders.
#
# Dependencies:
# - customtkinter, pandas, numpy, scipy, Pillow, matplotlib
# =============================================================================

import configparser
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, filtfilt, medfilt, savgol_filter


# Helper function for causal derivative calculation
def _poly_derivative(series, window, poly_order, deriv_order, delta_x):
    """
    Calculates the derivative of a series using a rolling polynomial fit.
    This is a causal method, only using past data.
    """
    if poly_order < deriv_order:
        # Cannot find a derivative of a higher order than the polynomial
        return pd.Series(np.nan, index=series.index)

    # Pad the series at the beginning to get derivatives for the initial points
    padded_series = pd.concat([pd.Series([series.iloc[0]] * (window - 1)), series])

    # The lambda function will be applied to each window
    def get_deriv(w):
        # Can't compute if the window is not full or has NaNs
        if len(w) < window or np.isnan(w).any():
            return np.nan

        # Create an x-axis for the window (0, delta_x, 2*delta_x, ...)
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

    # Apply the function over a rolling window
    return (
        padded_series.rolling(window=window)
        .apply(get_deriv, raw=True)
        .iloc[window - 1 :]
    )


class CSVProcessorApp(ctk.CTk):
    """The main application class that encapsulates the entire GUI and processing logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Advanced CSV Time Series Processor v9.2")
        self.geometry("1200x850")

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
        """Creates all widgets for the main processing workflow tab."""
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

        self.populate_setup_sub_tab(processing_tab_view.tab("Setup"))
        self.populate_processing_sub_tab(processing_tab_view.tab("Processing"))

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
            right_panel, label_text="Selected Input Files", height=100
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
        ).grid(row=1, column=2, padx=10, pady=5, sticky="ew")
        output_mode_frame = ctk.CTkFrame(tab)
        output_mode_frame.grid(row=2, column=0, padx=10, pady=10, sticky="new")
        ctk.CTkLabel(
            output_mode_frame, text="Batch Output Mode", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        self.output_mode = tk.StringVar(value="separate")
        ctk.CTkRadioButton(
            output_mode_frame,
            text="Save as Separate Files",
            variable=self.output_mode,
            value="separate",
        ).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkRadioButton(
            output_mode_frame,
            text="Compile into a Single File",
            variable=self.output_mode,
            value="compile",
        ).grid(row=2, column=0, padx=10, pady=5, sticky="w")

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

    def _create_ma_param_frame(self, parent, time_units):
        frame = ctk.CTkFrame(parent)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="Time Window:").grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )
        time_entry_frame = ctk.CTkFrame(frame, fg_color="transparent")
        time_entry_frame.grid(row=0, column=1, sticky="ew", padx=10)
        time_entry_frame.grid_columnconfigure(0, weight=2)
        time_entry_frame.grid_columnconfigure(1, weight=1)
        entry = ctk.CTkEntry(time_entry_frame, placeholder_text="e.g., 30")
        entry.grid(row=0, column=0, sticky="ew")
        menu = ctk.CTkOptionMenu(time_entry_frame, values=time_units)
        menu.set(time_units[1])
        menu.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        return frame, entry, menu

    def _create_bw_param_frame(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="Filter Order:").grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )
        entry_ord = ctk.CTkEntry(frame, placeholder_text="e.g., 3")
        entry_ord.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(frame, text="Cutoff Freq (Hz):").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        entry_cut = ctk.CTkEntry(frame, placeholder_text="e.g., 0.1")
        entry_cut.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        return frame, entry_ord, entry_cut

    def _create_median_param_frame(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="Kernel Size:").grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )
        entry = ctk.CTkEntry(frame, placeholder_text="Odd integer, e.g., 5")
        entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        return frame, entry

    def _create_savgol_param_frame(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="Window Len:").grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )
        entry_win = ctk.CTkEntry(frame, placeholder_text="Odd integer, e.g., 11")
        entry_win.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(frame, text="Poly Order:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        entry_poly = ctk.CTkEntry(frame, placeholder_text="e.g., 2 (< Window Len)")
        entry_poly.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        return frame, entry_win, entry_poly

    def _update_filter_ui(self, choice):
        self.ma_frame.grid_remove()
        self.bw_frame.grid_remove()
        self.median_frame.grid_remove()
        self.savgol_frame.grid_remove()
        if choice == "Moving Average":
            self.ma_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
        elif choice in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.bw_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
        elif choice == "Median Filter":
            self.median_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)
        elif choice == "Savitzky-Golay":
            self.savgol_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)

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
        plot_left_panel = ctk.CTkFrame(plot_main_frame, width=280)
        plot_left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        plot_left_panel.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(
            plot_left_panel, text="Plot Controls", font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        self.plot_signal_frame = ctk.CTkScrollableFrame(
            plot_left_panel, label_text="Signals to Plot"
        )
        self.plot_signal_frame.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5
        )
        ctk.CTkLabel(
            plot_left_panel, text="Filter Preview:", font=ctk.CTkFont(weight="bold")
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=(10, 0))
        self.plot_filter_type = ctk.StringVar(value="None")
        self.plot_filter_menu = ctk.CTkOptionMenu(
            plot_left_panel,
            variable=self.plot_filter_type,
            values=self.filter_names,
            command=self._update_plot_filter_ui,
        )
        self.plot_filter_menu.grid(
            row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )
        time_units = ["ms", "s", "min", "hr"]
        (self.plot_ma_frame, self.plot_ma_value_entry, self.plot_ma_unit_menu) = (
            self._create_ma_param_frame(plot_left_panel, time_units)
        )
        (self.plot_bw_frame, self.plot_bw_order_entry, self.plot_bw_cutoff_entry) = (
            self._create_bw_param_frame(plot_left_panel)
        )
        (self.plot_median_frame, self.plot_median_kernel_entry) = (
            self._create_median_param_frame(plot_left_panel)
        )
        (
            self.plot_savgol_frame,
            self.plot_savgol_window_entry,
            self.plot_savgol_polyorder_entry,
        ) = self._create_savgol_param_frame(plot_left_panel)
        self._update_plot_filter_ui("None")
        ctk.CTkButton(
            plot_left_panel,
            text="Copy Settings to Processing Tab",
            command=self._copy_plot_settings_to_processing,
        ).grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(
            plot_left_panel, text="Update Plot", command=self.update_plot
        ).grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        trim_frame = ctk.CTkFrame(plot_left_panel)
        trim_frame.grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=(10, 5)
        )
        ctk.CTkLabel(
            trim_frame, text="Trim & Export", font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=5)
        self.trim_date_range_label = ctk.CTkLabel(
            trim_frame, text="Date Range: (load file)", text_color="gray"
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
        self.trim_end_entry.pack(fill="x", padx=10, pady=(0, 10))
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

    def _update_plot_filter_ui(self, choice):
        # BUG FIX: This function now correctly references the plot-specific frames
        self.plot_ma_frame.grid_remove()
        self.plot_bw_frame.grid_remove()
        self.plot_median_frame.grid_remove()
        self.plot_savgol_frame.grid_remove()
        if choice == "Moving Average":
            self.plot_ma_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5)
        elif choice in ["Butterworth Low-pass", "Butterworth High-pass"]:
            self.plot_bw_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5)
        elif choice == "Median Filter":
            self.plot_median_frame.grid(
                row=5, column=0, columnspan=2, sticky="ew", padx=5
            )
        elif choice == "Savitzky-Golay":
            self.plot_savgol_frame.grid(
                row=5, column=0, columnspan=2, sticky="ew", padx=5
            )

    def _copy_plot_settings_to_processing(self):
        """Copies filter settings from the plot tab to the main processing tab."""
        plot_filter = self.plot_filter_type.get()
        self.filter_type_var.set(plot_filter)
        self._update_filter_ui(plot_filter)

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

    def _show_sharing_instructions(self):
        instructions_window = ctk.CTkToplevel(self)
        instructions_window.title("How to Share This Application")
        instructions_window.geometry("650x500")
        instructions_window.grab_set()
        textbox = ctk.CTkTextbox(
            instructions_window, wrap="word", font=("Courier New", 12)
        )
        textbox.pack(expand=True, fill="both", padx=15, pady=15)
        instructions = """Sharing this Application with Colleagues\n======================================\n\nTo share this app with non-programmers, package it into a standalone executable (.exe for Windows) using PyInstaller.\n\n--- Step-by-Step ---\n\n1. Install PyInstaller in your Python environment:\n   (Open PowerShell and run)\n   python -m pip install pyinstaller\n\n2. Navigate to this script's directory in PowerShell:\n   cd path\\to\\your\\script\n\n3. Find the path to customtkinter library:\n   python -c "import customtkinter; print(customtkinter.__path__[0])"\n   (Copy the full path that this command prints)\n\n4. Run the PyInstaller command:\n   pyinstaller --name "CSV_Processor" --onefile --windowed --add-data "PASTE_PATH_HERE;customtkinter/" "your_script_name.py"\n\n   (Replace 'PASTE_PATH_HERE' with the path you copied, and replace 'your_script_name.py' with the real name of this file)\n\n5. Find your .exe file in the new 'dist' folder."""
        textbox.insert("0.0", instructions)
        textbox.configure(state="disabled")

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
        self.input_file_paths = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
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
                text=f"Loaded {len(self.input_file_paths)} files. Ready."
            )
        else:
            self.initial_file_label = ctk.CTkLabel(
                self.file_list_frame, text="No files selected."
            )
            self.initial_file_label.pack(padx=5, pady=5)
            self.status_label.configure(text="File selection cancelled.")

    def update_signal_list(self):
        for widget in self.signal_list_frame.winfo_children():
            widget.destroy()
        self.signal_vars.clear()
        all_columns = set()
        if not self.input_file_paths:
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
        sorted_columns = sorted(list(all_columns))
        self.search_entry.delete(0, "end")
        for signal in sorted_columns:
            var = tk.BooleanVar(value=True)
            cb = ctk.CTkCheckBox(self.signal_list_frame, text=signal, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)
            self.signal_vars[signal] = {"var": var, "widget": cb}

    def _filter_signals(self, event=None):
        search_term = self.search_entry.get().lower()
        for signal_name, data in self.signal_vars.items():
            widget = data["widget"]
            if search_term in signal_name.lower():
                widget.pack(anchor="w", padx=10, pady=2)
            else:
                widget.pack_forget()

    def _clear_search(self):
        self.search_entry.delete(0, "end")
        self._filter_signals()

    def on_plot_file_select(self, filename):
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
        self.plot_xaxis_menu.configure(values=all_cols)
        self.plot_xaxis_menu.set(df.columns[0])
        for signal in numeric_cols:
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(self.plot_signal_frame, text=signal, variable=var)
            cb.pack(anchor="w", padx=5, pady=2)
            self.plot_signal_vars[signal] = var
        time_col = df.columns[0]
        if not df.empty and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            min_date = df[time_col].min()
            max_date = df[time_col].max()
            date_str = min_date.strftime("%Y-%m-%d")
            if min_date.date() != max_date.date():
                date_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            self.trim_date_range_label.configure(text=f"Data Date(s): {date_str}")
            self.trim_date_entry.delete(0, "end")
            self.trim_date_entry.insert(0, min_date.strftime("%Y-%m-%d"))
        self.update_plot()

    def get_data_for_plotting(self, filename):
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
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df.dropna(subset=[time_col], inplace=True)
            for col in df.columns:
                if col != time_col:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            self.loaded_data_cache[filename] = df.copy()
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data for {filename}.\n{e}")
            return None

    def update_plot(self):
        selected_file = self.plot_file_menu.get()
        x_axis_col = self.plot_xaxis_menu.get()
        if selected_file == "Select a file..." or not x_axis_col:
            return
        df = self.get_data_for_plotting(selected_file)
        if df is None:
            return
        signals_to_plot = [s for s, v in self.plot_signal_vars.items() if v.get()]
        self.plot_ax.clear()
        if not signals_to_plot:
            self.plot_ax.text(
                0.5,
                0.5,
                "Select one or more signals to plot",
                ha="center",
                va="center",
                transform=self.plot_ax.transAxes,
            )
            self.plot_canvas.draw()
            return
        for signal in signals_to_plot:
            plot_df = df[[x_axis_col, signal]].dropna()
            self.plot_ax.plot(
                plot_df[x_axis_col],
                plot_df[signal],
                label=f"{signal} (Raw)",
                alpha=0.6,
                marker=".",
                markersize=2,
                linestyle="None",
            )
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
        self.plot_ax.set_xlabel(x_axis_col)
        self.plot_ax.set_ylabel("Value")
        self.plot_ax.set_title(f"Signals from {selected_file}")
        self.plot_ax.legend()
        self.plot_ax.grid(True, linestyle="--", alpha=0.6)
        if pd.api.types.is_datetime64_any_dtype(df[x_axis_col]):
            self.plot_fig.autofmt_xdate()
            self.plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.plot_canvas.draw()

    def _apply_plot_filter(self, df, signal_col, x_axis_col):
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
                        df_indexed.index.to_series().diff().dt.total_seconds()
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
                    kernel += 1
                if len(signal_data) > kernel:
                    return pd.Series(
                        medfilt(signal_data, kernel_size=kernel),
                        index=signal_data.index,
                    )
            elif filter_type == "Savitzky-Golay":
                win = int(self.plot_savgol_window_entry.get())
                poly = int(self.plot_savgol_polyorder_entry.get())
                if win % 2 == 0:
                    win += 1
                if poly >= win:
                    poly = win - 1 if win > 1 else 0
                if len(signal_data) > win:
                    return pd.Series(
                        savgol_filter(signal_data, win, poly), index=signal_data.index
                    )
        except Exception as e:
            print(f"Could not apply plot filter '{filter_type}': {e}")
        return None

    def trim_and_save(self):
        selected_file = self.plot_file_menu.get()
        if selected_file == "Select a file...":
            messagebox.showwarning("Warning", "Please select a file to trim.")
            return
        df = self.get_data_for_plotting(selected_file).copy()
        if df is None:
            return
        try:
            date_str = self.trim_date_entry.get()
            start_time_str = self.trim_start_entry.get() or "00:00:00"
            end_time_str = self.trim_end_entry.get() or "23:59:59"
            time_col = df.columns[0]
            start_full_str = f"{date_str} {start_time_str}"
            end_full_str = f"{date_str} {end_time_str}"
            trimmed_df = df.set_index(time_col).loc[start_full_str:end_full_str]
            if self.trim_resample_var.get():
                resample_value = self.resample_value_entry.get()
                resample_unit = self.resample_unit_menu.get()
                if resample_value:
                    trimmed_df = (
                        trimmed_df.resample(f"{resample_value}{resample_unit}")
                        .mean()
                        .dropna(how="all")
                    )
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

    def save_settings(self):
        save_path = filedialog.asksaveasfilename(
            title="Save Settings File",
            defaultextension=".ini",
            filetypes=[("Settings files", "*.ini")],
        )
        if not save_path:
            return
        config = configparser.ConfigParser()
        config["SIGNALS"] = {
            "selected": json.dumps(
                [s for s, data in self.signal_vars.items() if data["var"].get()]
            )
        }
        config["PROCESSING"] = {
            "filter_type": self.filter_type_var.get(),
            "ma_value": self.ma_value_entry.get(),
            "ma_unit": self.ma_unit_menu.get(),
            "bw_order": self.bw_order_entry.get(),
            "bw_cutoff_hz": self.bw_cutoff_entry.get(),
            "median_kernel": self.median_kernel_entry.get(),
            "savgol_window": self.savgol_window_entry.get(),
            "savgol_polyorder": self.savgol_polyorder_entry.get(),
            "resample_enabled": self.resample_var.get(),
            "resample_value": self.resample_value_entry.get(),
            "resample_unit": self.resample_unit_menu.get(),
            "deriv_method": self.deriv_method_var.get(),
            "derivatives": json.dumps(
                [i for i, v in self.derivative_vars.items() if v.get()]
            ),
        }
        config["OUTPUT"] = {"mode": self.output_mode.get()}
        try:
            with open(save_path, "w") as f:
                config.write(f)
            self.status_label.configure(text="Settings saved successfully.")
        except Exception as e:
            messagebox.showerror(
                "Error Saving", f"Could not save settings file.\nError: {e}"
            )

    def load_settings(self):
        if not self.signal_vars:
            messagebox.showwarning(
                "No Signals", "Please load CSV files first before loading settings."
            )
            return
        load_path = filedialog.askopenfilename(
            title="Load Settings File", filetypes=[("Settings files", "*.ini")]
        )
        if not load_path:
            return
        config = configparser.ConfigParser(allow_no_value=True)
        try:
            config.read(load_path)
            if "SIGNALS" in config and "selected" in config["SIGNALS"]:
                signals_to_load = set(json.loads(config["SIGNALS"]["selected"]))
                [
                    data["var"].set(s in signals_to_load)
                    for s, data in self.signal_vars.items()
                ]
            if "PROCESSING" in config:
                proc = config["PROCESSING"]
                time_units = ["ms", "s", "min", "hr"]

                def set_entry(entry, value):
                    entry.delete(0, "end")
                    entry.insert(0, value)

                self.filter_type_var.set(proc.get("filter_type", "None"))
                self._update_filter_ui(self.filter_type_var.get())
                set_entry(self.ma_value_entry, proc.get("ma_value", ""))
                self.ma_unit_menu.set(proc.get("ma_unit", "s"))
                set_entry(self.bw_order_entry, proc.get("bw_order", ""))
                set_entry(self.bw_cutoff_entry, proc.get("bw_cutoff_hz", ""))
                set_entry(self.median_kernel_entry, proc.get("median_kernel", ""))
                set_entry(self.savgol_window_entry, proc.get("savgol_window", ""))
                set_entry(self.savgol_polyorder_entry, proc.get("savgol_polyorder", ""))
                self.resample_var.set(proc.getboolean("resample_enabled", False))
                set_entry(self.resample_value_entry, proc.get("resample_value", ""))
                self.resample_unit_menu.set(proc.get("resample_unit", "s"))
                self.deriv_method_var.set(proc.get("deriv_method", "Spline (Acausal)"))
                derivatives_to_load = json.loads(proc.get("derivatives", "[]"))
                [
                    var.set(i in derivatives_to_load)
                    for i, var in self.derivative_vars.items()
                ]
            if "OUTPUT" in config:
                self.output_mode.set(config["OUTPUT"].get("mode", "separate"))
            self.status_label.configure(text="Settings loaded successfully.")
        except Exception as e:
            messagebox.showerror(
                "Error Loading",
                f"Could not load or parse the settings file.\nError: {e}",
            )

    def select_all(self):
        for data in self.signal_vars.values():
            data["var"].set(True)

    def deselect_all(self):
        for data in self.signal_vars.values():
            data["var"].set(False)

    def select_output_folder(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_directory = path
            self.output_label.configure(text=f"Output: {self.output_directory}")

    def get_unique_filepath(self, filepath):
        base, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(filepath):
            filepath = f"{base}({counter}){ext}"
            counter += 1
        return filepath

    def process_files(self):
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

        derivatives_to_compute = [i for i, v in self.derivative_vars.items() if v.get()]
        filter_type = self.filter_type_var.get()
        resample_enabled = self.resample_var.get()
        deriv_method = self.deriv_method_var.get()
        all_processed_dfs, skipped_files = [], []

        try:
            for i, file_path in enumerate(self.input_file_paths):
                self.status_label.configure(
                    text=f"Processing [{i+1}/{len(self.input_file_paths)}]: {os.path.basename(file_path)}"
                )
                self.progressbar.set((i + 1) / len(self.input_file_paths))
                self.update_idletasks()
                df = pd.read_csv(file_path, low_memory=False)
                signals_in_this_file = [s for s in selected_signals if s in df.columns]
                time_col = df.columns[0]
                if time_col not in signals_in_this_file:
                    signals_in_this_file.insert(0, time_col)
                processed_df = df[signals_in_this_file].copy()
                processed_df[time_col] = pd.to_datetime(
                    processed_df[time_col], errors="coerce"
                )
                processed_df.dropna(subset=[time_col], inplace=True)
                for col in processed_df.columns:
                    if col != time_col:
                        processed_df[col] = pd.to_numeric(
                            processed_df[col], errors="coerce"
                        )
                if processed_df.empty:
                    skipped_files.append(os.path.basename(file_path))
                    continue
                processed_df.set_index(time_col, inplace=True)
                if filter_type != "None":
                    numeric_cols = processed_df.select_dtypes(
                        include=np.number
                    ).columns.tolist()
                    for col in numeric_cols:
                        signal_data = processed_df[col].dropna()
                        if len(signal_data) < 2:
                            continue
                        try:
                            if filter_type == "Moving Average":
                                ma_value = self.ma_value_entry.get()
                                ma_unit = self.ma_unit_menu.get()
                                if ma_value:
                                    processed_df[col] = (
                                        processed_df[col]
                                        .rolling(
                                            window=f"{ma_value}{ma_unit}", min_periods=1
                                        )
                                        .mean()
                                    )
                            elif filter_type in [
                                "Butterworth Low-pass",
                                "Butterworth High-pass",
                            ]:
                                sampling_rate = (
                                    1.0
                                    / processed_df.index.to_series()
                                    .diff()
                                    .dt.total_seconds()
                                    .mean()
                                )
                                if pd.isna(sampling_rate):
                                    sampling_rate = 1.0
                                bw_order = int(self.bw_order_entry.get())
                                bw_cutoff = float(self.bw_cutoff_entry.get())
                                btype = (
                                    "low"
                                    if filter_type == "Butterworth Low-pass"
                                    else "high"
                                )
                                if len(signal_data) > bw_order * 3:
                                    b, a = butter(
                                        N=bw_order,
                                        Wn=bw_cutoff,
                                        btype=btype,
                                        fs=sampling_rate,
                                    )
                                    processed_df.loc[signal_data.index, col] = filtfilt(
                                        b, a, signal_data
                                    )
                            elif filter_type == "Median Filter":
                                kernel = int(self.median_kernel_entry.get())
                                if kernel % 2 == 0:
                                    kernel += 1
                                if len(signal_data) > kernel:
                                    processed_df.loc[signal_data.index, col] = medfilt(
                                        signal_data, kernel_size=kernel
                                    )
                            elif filter_type == "Savitzky-Golay":
                                window = int(self.savgol_window_entry.get())
                                poly = int(self.savgol_polyorder_entry.get())
                                if window % 2 == 0:
                                    window += 1
                                if poly >= window:
                                    poly = window - 1 if window > 1 else 0
                                if len(signal_data) > window and poly >= 0:
                                    processed_df.loc[signal_data.index, col] = (
                                        savgol_filter(signal_data, window, poly)
                                    )
                        except Exception as filter_e:
                            print(
                                f"Could not apply filter '{filter_type}' to column '{col}'. Error: {filter_e}"
                            )
                if resample_enabled:
                    resample_value = self.resample_value_entry.get()
                    resample_unit = self.resample_unit_menu.get()
                    if resample_value:
                        processed_df = (
                            processed_df.resample(f"{resample_value}{resample_unit}")
                            .mean()
                            .dropna(how="all")
                        )
                if processed_df.empty:
                    skipped_files.append(os.path.basename(file_path))
                    continue
                processed_df.reset_index(inplace=True)
                if derivatives_to_compute and len(processed_df) > 10:
                    first_time = processed_df[time_col].iloc[0]
                    time_numeric_col = (
                        processed_df[time_col] - first_time
                    ).dt.total_seconds()
                    for signal in signals_in_this_file:
                        if signal == time_col:
                            continue
                        if deriv_method == "Spline (Acausal)":
                            temp_df = pd.concat(
                                [time_numeric_col, processed_df[signal]], axis=1
                            ).dropna()
                            if (
                                len(temp_df) < 10
                                or len(np.unique(temp_df.iloc[:, 0])) < 5
                            ):
                                continue
                            x, y = temp_df.iloc[:, 0].values, temp_df.iloc[:, 1].values
                            spl = UnivariateSpline(x, y, s=0, k=5)
                            for d_order in derivatives_to_compute:
                                d_spl, suffix = (
                                    spl.derivative(n=d_order),
                                    "_" + "d" * d_order,
                                )
                                processed_df[f"{signal}{suffix}"] = pd.Series(
                                    d_spl(x), index=temp_df.index
                                )
                        elif deriv_method == "Rolling Polynomial (Causal)":
                            signal_data = processed_df.set_index(time_numeric_col)[
                                signal
                            ].dropna()
                            if len(signal_data) < 11:
                                continue
                            dx = time_numeric_col.diff().mean()
                            for d_order in derivatives_to_compute:
                                suffix = "_" + "d" * d_order
                                poly_order = max(d_order, 2)
                                window = poly_order * 2 + 1
                                if len(signal_data) > window:
                                    deriv_series = _poly_derivative(
                                        signal_data,
                                        window,
                                        poly_order,
                                        d_order,
                                        dx if pd.notna(dx) else 1.0,
                                    )
                                    processed_df[f"{signal}{suffix}"] = (
                                        deriv_series.values
                                    )
                    if deriv_method == "Spline (Acausal)":
                        trim_size = 5
                        if len(processed_df) > trim_size * 2:
                            processed_df = processed_df.iloc[
                                trim_size:-trim_size
                            ].reset_index(drop=True)
                if processed_df.empty:
                    if os.path.basename(file_path) not in skipped_files:
                        skipped_files.append(os.path.basename(file_path))
                    continue
                if self.output_mode.get() == "compile":
                    processed_df["source_file"] = os.path.basename(file_path)
                all_processed_dfs.append(processed_df)

            if not all_processed_dfs:
                messagebox.showwarning(
                    "Processing Complete",
                    "No data was left to save after processing all files.",
                )
                return
            final_message = ""
            if self.output_mode.get() == "separate":
                valid_input_paths = [
                    p
                    for p in self.input_file_paths
                    if os.path.basename(p) not in skipped_files
                ]
                for i, df_to_save in enumerate(all_processed_dfs):
                    original_filename = os.path.basename(valid_input_paths[i])
                    name, ext = os.path.splitext(original_filename)
                    output_filename = f"{name}_processed{ext}"
                    output_path = os.path.join(self.output_directory, output_filename)
                    unique_output_path = self.get_unique_filepath(output_path)
                    df_to_save.to_csv(unique_output_path, index=False)
                final_message = f"Successfully processed and saved {len(all_processed_dfs)} file(s)."
            elif self.output_mode.get() == "compile":
                compiled_df = pd.concat(all_processed_dfs, ignore_index=True)
                save_path = filedialog.asksaveasfilename(
                    initialdir=self.output_directory,
                    title="Save Compiled File",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                )
                if save_path:
                    compiled_df.to_csv(save_path, index=False)
                    final_message = f"Successfully compiled {len(all_processed_dfs)} file(s) into one."
                else:
                    final_message = "Compile cancelled by user."
            if skipped_files:
                final_message += f"\n\nNote: {len(skipped_files)} file(s) were skipped."
            messagebox.showinfo("Success", final_message)
        except Exception as e:
            messagebox.showerror(
                "Processing Error", f"An error occurred during processing:\n{e}"
            )
        finally:
            self.status_label.configure(text="Ready.")
            self.process_button.configure(state="normal", text="Process Files")
            self.progressbar.set(0)


if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = CSVProcessorApp()
    app.mainloop()

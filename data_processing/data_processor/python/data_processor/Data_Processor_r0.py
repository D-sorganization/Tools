# =============================================================================
# Advanced CSV Time Series Processor & Analyzer - Complete Version
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This version combines all advanced features
# from Rev2 with the UI fixes from Rev4_Claude, ensuring complete functionality.
# Dependencies for Python 3.8+:
# pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow
# simpledbf pyarrow tables feather-format
# =============================================================================

import json
import os
import threading
import tkinter as tk
import traceback
from tkinter import colorchooser, filedialog, messagebox, simpledialog
from typing import Any

import customtkinter as ctk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, medfilt

# Optional Savitzky-Golay import with guard
try:
    from scipy.signal import savgol_filter as _savgol_filter
except Exception:  # pragma: no cover - optional dependency
    _savgol_filter = None

# Import vectorized filter engine
# Import constants
from constants import (
    DEFAULT_ALPHA,
    DEFAULT_BW_CUTOFF,
    DEFAULT_BW_NYQUIST,
    DEFAULT_BW_ORDER,
    DEFAULT_DPI,
    DEFAULT_END_TIME,
    DEFAULT_GAUSSIAN_MODE,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_HAMPEL_THRESHOLD,
    DEFAULT_HAMPEL_WINDOW,
    DEFAULT_MA_WINDOW,
    DEFAULT_MEDIAN_KERNEL,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW,
    DEFAULT_START_TIME,
    DEFAULT_WINDOW_HEIGHT,
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_ZSCORE_METHOD,
    DEFAULT_ZSCORE_THRESHOLD,
    EXCEL_SHEET_NAME_MAX_LENGTH,
    LARGE_SIGNAL_THRESHOLD,
    LAYOUT_SAVE_DELAY_MS,
    MAX_DERIVATIVE_ORDER,
    MILLISECONDS_PER_SECOND,
    MIN_BUTTERWORTH_DATA_MULTIPLIER,
    MIN_PERIODS_DEFAULT,
    MIN_SIGNAL_DATA_POINTS,
    NORMAL_DISTRIBUTION_CONSTANT,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    SIGNAL_BATCH_SIZE,
    UI_UPDATE_DELAY_MS,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
)
from high_performance_loader import HighPerformanceDataLoader, LoadingConfig
from vectorized_filter_engine import VectorizedFilterEngine

# =============================================================================
# WORKER FUNCTION FOR PARALLEL PROCESSING
# =============================================================================


def process_single_csv_file(
    file_path: str,
    settings: dict[str, Any],
) -> pd.DataFrame | None:
    """Processes a single CSV file based on a dictionary of settings.

    This function is designed to be run in a separate process.

    Args:
        file_path: Path to the CSV file to process
        settings: Dictionary containing processing settings

    Returns:
        Processed DataFrame or None if processing fails
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

        # Apply Filtering using VectorizedFilterEngine
        filter_type = settings.get("filter_type")
        if filter_type and filter_type != "None":
            numeric_cols = processed_df.select_dtypes(
                include=np.number,
            ).columns.tolist()

            # Use VectorizedFilterEngine for faster processing
            filter_engine = VectorizedFilterEngine()
            processed_df[numeric_cols] = filter_engine.apply_filter_batch(
                processed_df, filter_type, settings, numeric_cols

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
        print(f"Error processing file: {e!s}")
        return None

class SimpleProgressDialog:
    """Simple progress dialog with cancellation support."""

    def __init__(self, parent, title: str, total: int):
        """Initialize the progress dialog."""
        self.parent = parent
        self.total = total
        self.cancel_event = threading.Event()

        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x200")
        self.dialog.resizable(False, False)

        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Create UI components
        self.title_label = ctk.CTkLabel(
            self.dialog, text=title, font=ctk.CTkFont(size=16, weight="bold")
        self.title_label.pack(pady=20)

        self.status_label = ctk.CTkLabel(
            self.dialog, text="Starting...", font=ctk.CTkFont(size=12)
        self.status_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.dialog)
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.set(0)

        self.cancel_button = ctk.CTkButton(
            self.dialog,
            text="Cancel",
            command=self._on_cancel,
            fg_color="red",
            hover_color="darkred",
        self.cancel_button.pack(pady=10)

    def _on_cancel(self):
        """Handle cancel button click."""
        self.cancel_event.set()
        self.dialog.destroy()

    def update(self, completed: int, total: int, message: str):
        """Update progress."""
        if self.dialog.winfo_exists():
            self.status_label.configure(text=f"{message} ({completed}/{total}))self.progress_bar.set(completed / total)
            self.dialog.update()

    def is_cancelled(self) -> bool:
        """Check if operation was cancelled."""
        return self.cancel_event.is_set()

When enabled (default):

When disabled:

This mode only affects signal detection, not data processing.

1. Select Input Files  Choose your CSV files
2. Configure Bulk Mode  Enable/disable based on your dataset
3. Select Output Folder  Choose where to save results
4. Load Signals  Use Signal List Management to load available signals
5. Process Data  Apply filters, export, etc.

 TIPS:

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
        close_button.pack(pady=10)

    def _show_signal_list_help(self) -> None:
        """Show comprehensive help for Signal List Management section."""
        help_text = """Signal List Management - Complete Guide

This section helps you manage which signals (columns) to process from your files.

1. Select Files  Click "Select Input Files"
2. Load Signals  Click "Load from Files" (or "Load from First File" for bulk mode)
3. Select Signals  Choose which signals to process
4. Save List  Optionally save your selection for future use
5. Process Data  Apply filters, export, etc.

 TIPS:

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
            cancel_button.pack(pady=10)

            # Store reference for cleanup
            self.current_progress_window = progress_window

        # Check if bulk processing mode is enabled
        bulk_mode = getattr(self, "bulk_mode_var", None) and self.bulk_mode_var.get()
        print(
            f"DEBUG: bulk_mode_var exists: {getattr(self,'bulk_mode_var'," \
                "None) is not None}",
        bulk_mode_value = (
            getattr(self, "bulk_mode_var", None).get()
            if getattr(self, "bulk_mode_var", None)
            else "N/A"
        print(f"DEBUG: bulk_mode_var" \
            "value: {bulk_mode_value})print(f"DEBUG: bulk_mode result: {bulk_mode}")

        # Check for cancellation
        if hasattr(self, "signal_loading_cancelled") and self.signal_loading_cancelled:
            self.signal_loading_cancelled = False
            if progress_window:
                try:
                    progress_window.destroy()
                except Exception as e:
                    # Log progress window destruction errors for debugging
                    print(f"Warning: Failed to destroy progress window: {e})return

        try:
            if bulk_mode and total_files > 1:
                # Check if first file only option is enabled
                first_file_only = (
                    getattr(self, "first_file_only_var", None)
                    and self.first_file_only_var.get()

                if first_file_only:
                    # First file only mode: most conservative approach
                    print(
                        "DEBUG: Using bulk processing mode -" \
                            "reading headers from first file only",

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text="Bulk mode: Reading headers from first file only...",
                        progress_window.update()
                    else:
                        self.update_status(
                            "Bulk mode: Reading headers from first file only...",
                            show_progress=True,
                            progress_value=0.1,
                            progress_text="Reading file headers...",

                    # Read headers from first file only
                    sample_files = self.input_file_paths[:1]
                    all_signals: set[str] = set()
                else:
                    # Standard bulk mode: read headers from first few files
                    print(
                        "DEBUG: Using bulk processing mode - " "reading headers from sample files only",

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text="Bulk mode: Reading headers from sample files...",
                        progress_window.update()
                    else:
                        self.update_status(
                            "Bulk mode: Reading headers from sample files...",
                            show_progress=True,
                            progress_value=0.1,
                            progress_text="Reading file headers...",

                    # Read headers from first 3 files only
                    sample_files = self.input_file_paths[:3]
                    all_signals: set[str] = set()

                for i, file_path in enumerate(sample_files):
                    # Check for cancellation
                    if (
                        hasattr(self, "signal_loading_cancelled")
                        and self.signal_loading_cancelled
                        print(
                            "DEBUG: Signal loading cancelled" \
                                "during bulk mode processing",
                        return

                    try:
                        if total_files > 100:
                            status_label.configure(
                                text=f"Reading sample file {i+1}/3:f"{os.path.basename(file_path)}",
                            if progress_bar:
                                progress = (i + 1) / len(sample_files)
                                progress_bar.set(progress)
                            progress_window.update()
                        elif hasattr(self, "status_label"):
                            self.status_label.configure(
                                text=f"Reading sample file {i+1}/3:f"{os.path.basename(file_path)}",
                            self.update()

                        df = pd.read_csv(file_path, nrows=1)
                        signals = df.columns.tolist()
                        all_signals.update(signals)if first_file_only:
                    print(
                        f"DEBUG: Bulk mode (first file only) - " f"signals from 1 file: {len(all_signals)} unique signals",

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text=(
                                f"Bulk mode: Using {len(all_signals)}signals" \
                                    "from first file only f"(assumed same for all {total_files} files)"
                        progress_window.update()
                    elif hasattr(self, "status_label"):
                        self.status_label.configure(
                            text=(
                                f"Bulk mode: Using {len(all_signals)}signals" \
                                    "from first file only f"(assumed same for all {total_files} files)"
                        self.update()
                else:
                    print(
                        f"DEBUG: Bulk mode -" \
                            "signals from" \
                                "{len(sample_files)}sample files: f"{len(all_signals)} unique signals",

                    # Update status
                    if total_files > 100:
                        status_label.configure(
                            text=f"Bulk mode:" \
                                "Using {len(all_signals)}signals fromf"sample files " f"(assumed same for all {total_files} files)",
                        progress_window.update()
                    elif hasattr(self, "status_label"):
                        self.status_label.configure(
                            text=f"Bulk mode:" \
                                "Using {len(all_signals)}signals fromf"sample files " f"(assumed same for all {total_files} files)",
                        self.update()

            else:
                # High-performance mode: use HighPerformanceDataLoader
                print("DEBUG: Using high-performance mode with parallel loading")

                # Configure high-performance loader
                config = LoadingConfig(
                    max_workers=8,  # Use 8 threads for parallel processing
                    cache_enabled=True,
                    parallel_loading=True,
                    lazy_loading=True,
                    max_files_per_batch=20,

                loader = HighPerformanceDataLoader(config)

                # Progress callback for UI updates
                def progress_callback(completed, total, message):
                    try:
                        if total_files > 100 and status_label:
                        try:
                            status_label.configure(
                                text=f"{message} ({completed}/{total})if progress_bar:
                                progress_bar.set(completed / total)
                            progress_window.update()
                        except Exception as e:
                            print(f"Progress update error (ignoring): {e}")
                    elif hasattr(self, "status_label"):
                        try:
                            self.status_label.configure(
                                text=f"{message} ({completed}/{total})self.update()
                        except Exception as e:
                            print(f"Status update error (ignoring): {e}")

                # Cancel flag
                cancel_event = threading.Event()
                if hasattr(self, "signal_loading_cancelled"):
                    if self.signal_loading_cancelled:
                        cancel_event.set()

                # Load signals using high-performance loader
                all_signals, file_metadata = loader.load_signals_from_files(
                    self.input_file_paths,
                    progress_callback=progress_callback,
                    cancel_flag=cancel_event,

                if cancel_event.is_set():
                    print("DEBUG: Signal loading cancelled")
                    return self.cancel_event.is_set()
BASIC SUBSCRIPTS:

MULTI-CHARACTER SUBSCRIPTS:

SUPERSCRIPTS:

COMBINED SUB & SUPERSCRIPTS:

GREEK LETTERS:

ENGINEERING EXAMPLES:

FRACTIONS & MATH:

TIPS:

COMMON MISTAKES TO AVOID:

        # Create text widget for the guide
        text_widget = ctk.CTkTextbox(
            scrollable_frame,
            width=550,
            height=500,
            wrap="word",
        text_widget.grid(row=1, column=0, pady=10, sticky="ew")
        text_widget.insert("1.0", guide_text)
        text_widget.configure(state="disabled")

        # Close button
        close_button = ctk.CTkButton(
            guide_window,
            text="Close",
            command=guide_window.destroy,
        close_button.grid(row=1, column=0, pady=10)

        # Center the window
        guide_window.update_idletasks()
        x = (guide_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (guide_window.winfo_screenheight() // 2) - (700 // 2)
        guide_window.geometry(f"600x700+{x}+{y})def save_settings(self) -> None:
        """Save current settings to a configuration file."""
        try:
            # Collect all current settings
            settings = {
                "filter_settings": {
                    "filter_type": (
                        self.filter_type_var.get()
                        if hasattr(self, "filter_type_var")
                        else "None"
                    "ma_window": (
                        self.ma_value_entry.get()
                        if hasattr(self, "ma_value_entry")
                        else "10"
                    "ma_unit": (
                        self.ma_unit_menu.get()
                        if hasattr(self, "ma_unit_menu")
                        else "s"
                    "bw_order": (
                        self.bw_order_entry.get()
                        if hasattr(self, "bw_order_entry")
                        else "3"
                    "bw_cutoff": (
                        self.bw_cutoff_entry.get()
                        if hasattr(self, "bw_cutoff_entry")
                        else "0.1"
                    "median_kernel": (
                        self.median_kernel_entry.get()
                        if hasattr(self, "median_kernel_entry")
                        else "5"
                    "hampel_window": (
                        self.hampel_window_entry.get()
                        if hasattr(self, "hampel_window_entry")
                        else "7"
                    "hampel_threshold": (
                        self.hampel_threshold_entry.get()
                        if hasattr(self, "hampel_threshold_entry")
                        else "3.0"
                    "zscore_threshold": (
                        self.zscore_threshold_entry.get()
                        if hasattr(self, "zscore_threshold_entry")
                        else "3.0"
                    "zscore_method": (
                        self.zscore_method_menu.get()
                        if hasattr(self, "zscore_method_menu")
                        else "Remove Outliers"
                    "savgol_window": (
                        self.savgol_window_entry.get()
                        if hasattr(self, "savgol_window_entry")
                        else "11"
                    "savgol_polyorder": (
                        self.savgol_polyorder_entry.get()
                        if hasattr(self, "savgol_polyorder_entry")
                        else "2"
                    "fft_window_shape": (
                        self.fft_window_shape_menu.get()
                        if hasattr(self, "fft_window_shape_menu")
                        else "Gaussian"
                    "fft_freq_unit": (
                        self.fft_freq_unit_menu.get()
                        if hasattr(self, "fft_freq_unit_menu")
                        else "normalized"
                    "fft_freq_low": (
                        self.fft_freq_low_entry.get()
                        if hasattr(self, "fft_freq_low_entry")
                        else "0.1"
                    "fft_freq_high": (
                        self.fft_freq_high_entry.get()
                        if hasattr(self, "fft_freq_high_entry")
                        else "0.3"
                    "fft_transition_bw": (
                        self.fft_transition_bw_entry.get()
                        if hasattr(self, "fft_transition_bw_entry")
                        else "0.05"
                    "fft_zero_phase": (
                        self.fft_zero_phase_checkbox.get()
                        if hasattr(self, "fft_zero_phase_checkbox")
                        else True
                "resample_settings": {
                    "enabled": (
                        self.resample_var.get()
                        if hasattr(self, "resample_var")
                        else False
                    "value": (
                        self.resample_value_entry.get()
                        if hasattr(self, "resample_value_entry")
                        else "10"
                    "unit": (
                        self.resample_unit_menu.get()
                        if hasattr(self, "resample_unit_menu")
                        else "s"
                "trim_settings": {
                    "date": (
                        self.trim_date_entry.get()
                        if hasattr(self, "trim_date_entry")
                        else ""
                    "start_time": (
                        self.trim_start_entry.get()
                        if hasattr(self, "trim_start_entry")
                        else ""
                    "end_time": (
                        self.trim_end_entry.get()
                        if hasattr(self, "trim_end_entry")
                        else ""
                "integration_settings": {
                    "method": (
                        self.integrator_method_var.get()
                        if hasattr(self, "integrator_method_var")
                        else "Trapezoidal"
                "differentiation_settings": {
                    "method": (
                        self.deriv_method_var.get()
                        if hasattr(self, "deriv_method_var")
                        else "Spline (Acausal)"
                    "orders": (
                        {str(i): var.get() for i, var in self.derivative_vars.items()}
                        if hasattr(self, "derivative_vars")
                        else {}
                "export_settings": {
                    "type": (
                        self.export_type_var.get()
                        if hasattr(self, "export_type_var")
                        else "CSV (Separate Files)"
                    "sort_column": (
                        self.sort_col_menu.get()
                        if hasattr(self, "sort_col_menu")
                        else "No Sorting"
                    "sort_order": (
                        self.sort_order_var.get()
                        if hasattr(self, "sort_order_var")
                        else "Ascending"
                "dataset_naming": {
                    "mode": (
                        self.dataset_naming_var.get()
                        if hasattr(self, "dataset_naming_var")
                        else "auto"
                    "custom_name": (
                        self.custom_dataset_entry.get()
                        if hasattr(self, "custom_dataset_entry")
                        else ""
                "custom_variables": (
                    self.custom_vars_list if hasattr(self, "custom_vars_list") else []
                "output_directory": self.output_directory,
                "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),

            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                title="Save Configuration Settings",
                defaultextension=".json",
                filetypes=[("JSON Configuration", "*.json"), ("All files", "*.*")],
                initialfile="csv_processor_config.json",

            if file_path:
                with open(file_path, "w") as f:
                    json.dump(settings, f, indent=2)
                messagebox.showinfo(
                    "Success",
                    f"Settings saved to:\n{file_path})        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings:\n{e!s})def load_settings(self) -> None:
        """Load settings from a configuration file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Configuration Settings",
                filetypes=[("JSON Configuration", "*.json"), ("All files", "*.*")],

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
                if hasattr(self, "zscore_threshold_entry"):
                    self.zscore_threshold_entry.delete(0, tk.END)
                    self.zscore_threshold_entry.insert(
                        0,
                        fs.get("zscore_threshold", "3.0"),
                if hasattr(self, "zscore_method_menu"):
                    self.zscore_method_menu.set(
                        fs.get("zscore_method", "Remove Outliers"),
                if hasattr(self, "savgol_window_entry"):
                    self.savgol_window_entry.delete(0, tk.END)
                    self.savgol_window_entry.insert(0, fs.get("savgol_window", "11"))
                if hasattr(self, "savgol_polyorder_entry"):
                    self.savgol_polyorder_entry.delete(0, tk.END)
                    self.savgol_polyorder_entry.insert(
                        0,
                        fs.get("savgol_polyorder", "2"),

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
                    self.output_label.configure(text=f"Output: {self.output_directory})saved_at" \
                        "= settings.get("saved_at", "Unknown time")
            messagebox.showinfo(
                "Success",
                f"Settings loaded successfully!\n\nConfiguration saved at: {saved_at}",
            )        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings:\n{e!s})def manage_configurations(self) -> None:
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
            config_scrollbar = tk.Scrollbar(
                list_frame,
                orient="vertical",
                command=self.config_listbox.yview,
            self.config_listbox.configure(yscrollcommand=config_scrollbar.set)

            self.config_listbox.pack(
                side="left",
                fill="both",
                expand=True,
                padx=(5, 0),
                pady=5,
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
            self.config_status_label.pack(pady=5)

            # Store the window reference
            self.config_management_window = config_window

            # Load initial list
            self._refresh_config_list()        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open configuration manager:\n{e!s}",

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
                            # Check if it has the expected structure (processing configs
                            # have 'saved_at',
                            # plotting configs have 'plot_name')
                            if isinstance(data, dict) and (
                                "saved_at" in data or "plot_name" in data
                                if "saved_at" in data:
                                    config_files.append(
                                            file,
                                            file_path,
                                            data.get("saved_at", "Unknown"),
                                            "Processing Config",
                                elif "plot_name" in data:
                                    config_files.append(
                                            file,
                                            file_path,
                                            data.get("created_date", "Unknown"),
                                            "Plot Config",
                    except Exception:
                        # Skip files that can't be read as JSON or
                        # don't have the right structure
                        continue

            # Sort by creation date (newest first)
            config_files.sort(key=lambda x: x[2], reverse=True)

            # Add to listbox
            for filename, _filepath, saved_at, config_type in config_files:
                display_text = f"{filename} ({config_type} - {saved_at})self.config_listbox.insert(tk.END, display_text)
                # Store the filepath as item data
                self.config_listbox.itemconfig(
                    tk.END,
                        "bg": (
                            "lightgray"
                            if self.config_listbox.size() % 2 == 0
                            else "white"

            self.config_status_label.configure(
                text=f"Found {len(config_files)} configuration file(s)",
            )        except Exception as e:
            self.config_status_label.configure(text=f"Error refreshing list: {e!s})def _load_selected_config(self) -> None:
        """Load the selected configuration file."""
        try:
            selection = self.config_listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "WarningPlease select a configuration file to load.",
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

            self.config_status_label.configure(text=f"Loaded configuration: {filename})messagebox.showinfo(
                "Success",
                f"Configuration loaded successfully:\n{filename}",
            )        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e!s})def _delete_selected_config(self) -> None:
        """Delete the selected configuration file."""
        try:
            selection = self.config_listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "WarningPlease select a configuration file to delete.",
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
                f"Are you sure you want to delete this configuration file?\n\n"
                f"{filename}\n\nThis action cannot be undone.",
            if result:
                os.remove(filepath)
                self.config_status_label.configure(
                    text=f"Deleted configuration: {filename}",
                self._refresh_config_list()
                messagebox.showinfo(
                    "Success",
                    f"Configuration deleted successfully:\n{filename}",
                )        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete configuration:\n{e!s})def _open_config_location(self) -> None:
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
# Advanced CSV Processor & DAT Importer - Help Guide

## Overview
This application provides comprehensive tools for processing, analyzing, and
visualizing time series data from CSV files and DAT files with DBF tag files.

## New Features (Latest Update)

###

###

###

## Tab Descriptions

### Processing Tab
**Purpose**: Configure file processing settings and batch export data.

**Features**:
      Butterworth,
      Median,
      Savitzky-Golay)

**Usage**:
1. Select input CSV files
2. Choose output directory
3. Configure processing options
4. Select signals to process
5. Click "Process & Batch Export Files"

### Plotting & Analysis Tab
**Purpose**: Visualize and analyze processed data.

**Features**:

**Usage**:
1. Select file to plot from dropdown
2. Choose signals to display
3. Configure plot appearance
4. Add trendlines if needed
5. Export results

### Plots List Tab
**Purpose**: Save and manage plot configurations for batch processing.

**Features**:

**Usage**:
1. Configure a plot in Plotting & Analysis tab
2. Add plot configuration to list
3. Generate previews
4. Export all plots at once

### DAT File Import Tab
**Purpose**: Import data from DAT files with DBF tag files.

**Features**:

**Usage**:
1. Select DBF tag file (.dbf)
2. Select DAT data file (.dat)
3. Choose tags to import
4. Preview and import data

## Advanced Features

### Smart Auto-Zoom System

### Configuration Management

### Signal Filtering

### Signal Integration

### Signal Differentiation

### Custom Variables
Use mathematical formulas with signal references:

## Tips & Best Practices

1. **File Selection**: Use consistent time formats across files
2. **Signal Selection**: Only select signals you need to reduce processing time
3. **Filtering**: Start with "None" and add filters as needed
4. **Integration**: Use Trapezoidal method for most accurate results
5. **Custom Variables**: Test formulas with simple calculations first
7. **Auto-Zoom**: Disable for stable filter comparison, enable for exploration
8. **Configuration Management**: Regularly clean up old configurations

## Troubleshooting

**Common Issues**:

**Performance Tips**:

## Keyboard Shortcuts

## Support

For additional support or feature requests, please refer to the application documentation or contact the development team.

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
                filename = f"{base_name}_processed{extension}else:
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
                f"The file '{filename}' already exists.\n\nf"Would you like to:\n"
                f" Yes: Overwrite the existing file\n"
                f" No: Generate a unique filename\n"
                f" Cancel: Cancel the operation",
                icon="warning",

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

        return file_path

    def _save_current_plot_config(self) -> None:
        """Save the current plot configuration."""
        # Get current plot settings
        plot_name = simpledialog.askstring(
            "Save Plot ConfigurationEnter a name for this plot configuration:",
        if not plot_name:
            return

        # Get currently selected signals for plotting
        selected_signals = []
        if hasattr(self, "plot_signal_vars"):
            selected_signals = [
                signal
                for signal, data in self.plot_signal_vars.items()
                if data["var"].get()

        # Get current plot settings
        plot_config = {
            "name": plot_name,
            "description": f"Plot configuration saved on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}file: (self.plot_file_menu.get() if hasattr(self, "plot_file_menu") else ""
            "x_axis": (
                self.plot_xaxis_menu.get() if hasattr(self, "plot_xaxis_menu") else ""
            "signals": selected_signals,
            "filter_type": (
                self.plot_filter_type.get()
                if hasattr(self, "plot_filter_type")
                else "None"
            "show_both_signals": (
                self.show_both_signals_var.get()
                if hasattr(self, "show_both_signals_var")
                else False
            "plot_title": (
                self.plot_title_entry.get() if hasattr(self, "plot_title_entry") else ""
            "plot_xlabel": (
                self.plot_xlabel_entry.get()
                if hasattr(self, "plot_xlabel_entry")
                else ""
            "plot_ylabel": (
                self.plot_ylabel_entry.get()
                if hasattr(self, "plot_ylabel_entry")
                else ""
            "start_time": (
                self.plotting_start_time_entry.get()
                if hasattr(self, "plotting_start_time_entry")
                else ""
            "end_time": (
                self.plotting_end_time_entry.get()
                if hasattr(self, "plotting_end_time_entry")
                else ""
            "color_scheme": (
                self.color_scheme_var.get()
                if hasattr(self, "color_scheme_var")
                else "Auto (Matplotlib)"
            "line_width": (
                self.line_width_var.get() if hasattr(self, "line_width_var") else "1.0"
            "legend_position": (
                self.legend_position_var.get()
                if hasattr(self, "legend_position_var")
                else "best"
            "plot_type": (
                self.plot_type_var.get()
                if hasattr(self, "plot_type_var")
                else "Line with Markers"
            "trendline_signal": (
                self.trendline_signal_var.get()
                if hasattr(self, "trendline_signal_var")
                else "Select signal..."
            "trendline_type": (
                self.trendline_type_var.get()
                if hasattr(self, "trendline_type_var")
                else "None"
            "custom_legend_entries": dict(
                self.custom_legend_entries,
            ),  # Save custom legend labels
            "custom_colors": list(self.custom_colors),  # Save custom colors
            "created_date": pd.Timestamp.now().isoformat(),

        # Add filter-specific parameters for plot preview
        if plot_config["filter_type"] == "Moving Average":
            plot_config["ma_value"] = (
                self.plot_ma_value_entry.get()
                if hasattr(self, "plot_ma_value_entry")
                else ""
            plot_config["ma_unit"] = (
                self.plot_ma_unit_menu.get()
                if hasattr(self, "plot_ma_unit_menu")
                else ""
        elif plot_config["filter_type"] in [
            "Butterworth Low-passButterworth High-pass",
            plot_config["bw_order"] = (
                self.plot_bw_order_entry.get()
                if hasattr(self, "plot_bw_order_entry")
                else ""
            plot_config["bw_cutoff"] = (
                self.plot_bw_cutoff_entry.get()
                if hasattr(self, "plot_bw_cutoff_entry")
                else ""
        elif plot_config["filter_type"] == "Median Filter":
            plot_config["median_kernel"] = (
                self.plot_median_kernel_entry.get()
                if hasattr(self, "plot_median_kernel_entry")
                else ""
        elif plot_config["filter_type"] == "Hampel Filter":
            plot_config["hampel_window"] = (
                self.plot_hampel_window_entry.get()
                if hasattr(self, "plot_hampel_window_entry")
                else ""
            plot_config["hampel_threshold"] = (
                self.plot_hampel_threshold_entry.get()
                if hasattr(self, "plot_hampel_threshold_entry")
                else ""
        elif plot_config["filter_type"] == "Z-Score Filter":
            plot_config["zscore_threshold"] = (
                self.plot_zscore_threshold_entry.get()
                if hasattr(self, "plot_zscore_threshold_entry")
                else ""
            plot_config["zscore_method"] = (
                self.plot_zscore_method_menu.get()
                if hasattr(self, "plot_zscore_method_menu")
                else ""
        elif plot_config["filter_type"] == "Savitzky-Golay":
            plot_config["savgol_window"] = (
                self.plot_savgol_window_entry.get()
                if hasattr(self, "plot_savgol_window_entry")
                else ""
            plot_config["savgol_polyorder"] = (
                self.plot_savgol_polyorder_entry.get()
                if hasattr(self, "plot_savgol_polyorder_entry")
                else ""
        elif plot_config["filter_type"] in [
            "FFT Low-passFFT High-pass",
            "FFT Band-passFFT Band-stop",
            plot_config["fft_window_shape"] = (
                self.plot_fft_window_shape_menu.get()
                if hasattr(self, "plot_fft_window_shape_menu")
                else ""
            plot_config["fft_freq_unit"] = (
                self.plot_fft_freq_unit_menu.get()
                if hasattr(self, "plot_fft_freq_unit_menu")
                else ""
            plot_config["fft_freq_low"] = (
                self.plot_fft_freq_low_entry.get()
                if hasattr(self, "plot_fft_freq_low_entry")
                else ""
            plot_config["fft_freq_high"] = (
                self.plot_fft_freq_high_entry.get()
                if hasattr(self, "plot_fft_freq_high_entry")
                else ""
            plot_config["fft_transition_bw"] = (
                self.plot_fft_transition_bw_entry.get()
                if hasattr(self, "plot_fft_transition_bw_entry")
                else ""
            plot_config["fft_zero_phase"] = (
                self.plot_fft_zero_phase_checkbox.get()
                if hasattr(self, "plot_fft_zero_phase_checkbox")
                else ""

        # Add to plots list
        self.plots_list.append(plot_config)
        self._update_plots_listbox()
        self._update_load_plot_config_menu()
        self._save_plots_to_file()

        messagebox.showinfo(
            "Success",
            f"Plot configuration '{plot_name}' saved successfully!",

    def _modify_plot_config(self) -> None:
        """Modify an existing plot configuration."""
        if not hasattr(self, "plots_list") or not self.plots_list:
            messagebox.showwarning(
                "No ConfigurationsNo saved plot configurations" \
                    "found. Please save a configuration first.",
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
        dialog.geometry(f"400x300+{x}+{y})# Create listbox for configurations
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
        for _i, config in enumerate(self.plots_list):
            listbox.insert(
                tk.END,
                f"{config['name']} ({config.get('created_date', 'Unknown date')})",

        # Buttons frame
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill="x", padx=20, pady=10)

        def on_modify() -> None:
            """Modify the selected plot configuration."""
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "No SelectionPlease select a configuration to modify.",
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
                f"Configuration '{selected_config['name']}' has" \
                    "been updated withcurrent settings!",

        def on_delete() -> None:
            """Delete the selected plot configuration."""
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning(
                    "No SelectionPlease select a configuration to delete.",
                return

            selected_index = selection[0]
            selected_config = self.plots_list[selected_index]

            # Ask for confirmation
            result = messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete" \
                    "the configuration '{selected_config['name']}'?\n\nThis" \
                        "action cannot be undone.",
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

        def on_cancel() -> None:
            """Cancel the configuration modification dialog."""
            dialog.destroy()

        ctk.CTkButton(button_frame, text="Modify Selected", command=on_modify).pack(
            side="left",
            padx=5,
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

        # Update the configuration with current settings
        self.plots_list[config_index].update(
                "file": (
                    self.plot_file_menu.get() if hasattr(self, "plot_file_menu") else ""
                "x_axis": (
                    self.plot_xaxis_menu.get()
                    if hasattr(self, "plot_xaxis_menu")
                    else ""
                "signals": selected_signals,
                "filter_type": (
                    self.plot_filter_type.get()
                    if hasattr(self, "plot_filter_type")
                    else "None"
                "show_both_signals": (
                    self.show_both_signals_var.get()
                    if hasattr(self, "show_both_signals_var")
                    else False
                "compare_filters": (
                    self.compare_filters_var.get()
                    if hasattr(self, "compare_filters_var")
                    else False
                "plot_title": (
                    self.plot_title_entry.get()
                    if hasattr(self, "plot_title_entry")
                    else ""
                "plot_xlabel": (
                    self.plot_xlabel_entry.get()
                    if hasattr(self, "plot_xlabel_entry")
                    else ""
                "plot_ylabel": (
                    self.plot_ylabel_entry.get()
                    if hasattr(self, "plot_ylabel_entry")
                    else ""
                "start_time": (
                    self.plotting_start_time_entry.get()
                    if hasattr(self, "plotting_start_time_entry")
                    else ""
                "end_time": (
                    self.plotting_end_time_entry.get()
                    if hasattr(self, "plotting_end_time_entry")
                    else ""
                "color_scheme": (
                    self.color_scheme_var.get()
                    if hasattr(self, "color_scheme_var")
                    else "Auto (Matplotlib)"
                "line_width": (
                    self.line_width_var.get()
                    if hasattr(self, "line_width_var")
                    else "1.0"
                "legend_position": (
                    self.legend_position_var.get()
                    if hasattr(self, "legend_position_var")
                    else "best"
                "plot_type": (
                    self.plot_type_var.get()
                    if hasattr(self, "plot_type_var")
                    else "Line with Markers"
                "trendline_signal": (
                    self.trendline_signal_var.get()
                    if hasattr(self, "trendline_signal_var")
                    else "Select signal..."
                "trendline_type": (
                    self.trendline_type_var.get()
                    if hasattr(self, "trendline_type_var")
                    else "None"
                "custom_legend_entries": dict(self.custom_legend_entries),
                "custom_colors": list(self.custom_colors),
                "modified_date": pd.Timestamp.now().isoformat(),

        # Add filter-specific parameters
        if self.plots_list[config_index]["filter_type"] == "Moving Average":
            self.plots_list[config_index]["ma_value"] = (
                self.plot_ma_value_entry.get()
                if hasattr(self, "plot_ma_value_entry")
                else ""
            self.plots_list[config_index]["ma_unit"] = (
                self.plot_ma_unit_menu.get()
                if hasattr(self, "plot_ma_unit_menu")
                else ""
        elif self.plots_list[config_index]["filter_type"] in [
            "Butterworth Low-passButterworth High-pass",
            self.plots_list[config_index]["bw_order"] = (
                self.plot_bw_order_entry.get()
                if hasattr(self, "plot_bw_order_entry")
                else ""
            self.plots_list[config_index]["bw_cutoff"] = (
                self.plot_bw_cutoff_entry.get()
                if hasattr(self, "plot_bw_cutoff_entry")
                else ""
        elif self.plots_list[config_index]["filter_type"] == "Median Filter":
            self.plots_list[config_index]["median_kernel"] = (
                self.plot_median_kernel_entry.get()
                if hasattr(self, "plot_median_kernel_entry")
                else ""
        elif self.plots_list[config_index]["filter_type"] == "Hampel Filter":
            self.plots_list[config_index]["hampel_window"] = (
                self.plot_hampel_window_entry.get()
                if hasattr(self, "plot_hampel_window_entry")
                else ""
            self.plots_list[config_index]["hampel_threshold"] = (
                self.plot_hampel_threshold_entry.get()
                if hasattr(self, "plot_hampel_threshold_entry")
                else ""
        elif self.plots_list[config_index]["filter_type"] == "Z-Score Filter":
            self.plots_list[config_index]["zscore_threshold"] = (
                self.plot_zscore_threshold_entry.get()
                if hasattr(self, "plot_zscore_threshold_entry")
                else ""
            self.plots_list[config_index]["zscore_method"] = (
                self.plot_zscore_method_menu.get()
                if hasattr(self, "plot_zscore_method_menu")
                else ""
        elif self.plots_list[config_index]["filter_type"] == "Savitzky-Golay":
            self.plots_list[config_index]["savgol_window"] = (
                self.plot_savgol_window_entry.get()
                if hasattr(self, "plot_savgol_window_entry")
                else ""
            self.plots_list[config_index]["savgol_polyorder"] = (
                self.plot_savgol_polyorder_entry.get()
                if hasattr(self, "plot_savgol_polyorder_entry")
                else ""
        elif self.plots_list[config_index]["filter_type"] in [
            "FFT Low-passFFT High-pass",
            "FFT Band-passFFT Band-stop",
            self.plots_list[config_index]["fft_window_shape"] = (
                self.plot_fft_window_shape_menu.get()
                if hasattr(self, "plot_fft_window_shape_menu")
                else ""
            self.plots_list[config_index]["fft_freq_unit"] = (
                self.plot_fft_freq_unit_menu.get()
                if hasattr(self, "plot_fft_freq_unit_menu")
                else ""
            self.plots_list[config_index]["fft_freq_low"] = (
                self.plot_fft_freq_low_entry.get()
                if hasattr(self, "plot_fft_freq_low_entry")
                else ""
            self.plots_list[config_index]["fft_freq_high"] = (
                self.plot_fft_freq_high_entry.get()
                if hasattr(self, "plot_fft_freq_high_entry")
                else ""
            self.plots_list[config_index]["fft_transition_bw"] = (
                self.plot_fft_transition_bw_entry.get()
                if hasattr(self, "plot_fft_transition_bw_entry")
                else ""
            self.plots_list[config_index]["fft_zero_phase"] = (
                self.plot_fft_zero_phase_checkbox.get()
                if hasattr(self, "plot_fft_zero_phase_checkbox")
                else ""

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
            return

        # Apply the plot configuration
        self._apply_plot_config(plot_config)
        messagebox.showinfo(
            "Success",
            f"Plot configuration '{selected_plot_name}' loaded!",

    def _apply_plot_config(self, plot_config: dict[str, Any]) -> None:
        """Apply a plot configuration to the current plotting tab."""
        # Apply file selection first
        if (
            "file" in plot_config
            and plot_config["file"]
            and hasattr(self, "plot_file_menu")
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
            "Butterworth Low-passButterworth High-pass",
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
                self.plot_median_kernel_entry.delete(0, tk.END)
                self.plot_median_kernel_entry.insert(0, plot_config["median_kernel"])
        elif plot_config.get("filter_type") == "Hampel Filter":
            if "hampel_window" in plot_config and hasattr(
                self,
                "plot_hampel_window_entry",
                self.plot_hampel_window_entry.delete(0, tk.END)
                self.plot_hampel_window_entry.insert(0, plot_config["hampel_window"])
            if "hampel_threshold" in plot_config and hasattr(
                self,
                "plot_hampel_threshold_entry",
                self.plot_hampel_threshold_entry.delete(0, tk.END)
                self.plot_hampel_threshold_entry.insert(
                    0,
                    plot_config["hampel_threshold"],
        elif plot_config.get("filter_type") == "Z-Score Filter":
            if "zscore_threshold" in plot_config and hasattr(
                self,
                "plot_zscore_threshold_entry",
                self.plot_zscore_threshold_entry.delete(0, tk.END)
                self.plot_zscore_threshold_entry.insert(
                    0,
                    plot_config["zscore_threshold"],
            if "zscore_method" in plot_config and hasattr(
                self,
                "plot_zscore_method_menu",
                self.plot_zscore_method_menu.set(plot_config["zscore_method"])
        elif plot_config.get("filter_type") == "Savitzky-Golay":
            if "savgol_window" in plot_config and hasattr(
                self,
                "plot_savgol_window_entry",
                self.plot_savgol_window_entry.delete(0, tk.END)
                self.plot_savgol_window_entry.insert(0, plot_config["savgol_window"])
            if "savgol_polyorder" in plot_config and hasattr(
                self,
                "plot_savgol_polyorder_entry",
                self.plot_savgol_polyorder_entry.delete(0, tk.END)
                self.plot_savgol_polyorder_entry.insert(
                    0,
                    plot_config["savgol_polyorder"],
        elif plot_config.get("filter_type") in [
            "FFT Low-passFFT High-pass",
            "FFT Band-passFFT Band-stop",
            if "fft_window_shape" in plot_config and hasattr(
                self, "plot_fft_window_shape_menu"
                self.plot_fft_window_shape_menu.set(plot_config["fft_window_shape"])
            if "fft_freq_unit" in plot_config and hasattr(
                self, "plot_fft_freq_unit_menu"
                self.plot_fft_freq_unit_menu.set(plot_config["fft_freq_unit"])
            if "fft_freq_low" in plot_config and hasattr(
                self, "plot_fft_freq_low_entry"
                self.plot_fft_freq_low_entry.delete(0, tk.END)
                self.plot_fft_freq_low_entry.insert(0, plot_config["fft_freq_low"])
            if "fft_freq_high" in plot_config and hasattr(
                self, "plot_fft_freq_high_entry"
                self.plot_fft_freq_high_entry.delete(0, tk.END)
                self.plot_fft_freq_high_entry.insert(0, plot_config["fft_freq_high"])
            if "fft_transition_bw" in plot_config and hasattr(
                self, "plot_fft_transition_bw_entry"
                self.plot_fft_transition_bw_entry.delete(0, tk.END)
                self.plot_fft_transition_bw_entry.insert(
                    0, plot_config["fft_transition_bw"]
            if "fft_zero_phase" in plot_config and hasattr(
                self, "plot_fft_zero_phase_checkbox"
                    self.plot_fft_zero_phase_checkbox.select()
                    if plot_config["fft_zero_phase"]
                    else self.plot_fft_zero_phase_checkbox.deselect()

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
            self.plots_signal_vars: dict[str, dict[str, Any]] = {}

        self.plots_signal_vars.clear()

        # Add checkboxes for each signal
        for signal in signals:
            if signal != signals[0]:  # Skip time column
                var = tk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(
                    self.plots_signals_frame,
                    text=signal,
                    variable=var,
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

            if not signals:
                self.preview_ax.text(
                    0.5,
                    0.5,
                    "No signals selected in this configuration",
                    transform=self.preview_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                self.preview_ax.set_title(f"Preview: {plot_config['name']})self.preview_canvas.draw()
                return

            if not file_name or file_name == "Select a file...":
                # Show available files for debugging
                available_files = []
                if hasattr(self, "plot_file_menu") and hasattr(
                    self.plot_file_menu,
                    "_values",
                    available_files = [f
                        for f in self.plot_file_menu._values
                        if f != "Select a file..."

                debug_tex]

                    t = f"No data file specified in plot configuration

Saved file: '{file_name}' + "\n".join(available_files[:3],
                    if len(available_files) > 3:
                        debug_text += f"\n... and {len(available_files)-3} more"
                else:
                    debug_text += "\n\nNo files currently loaded.\nPlease load" \
                        "files on Setup tab first."

                self.preview_ax.text(
                    0.5,
                    0.5,
                    debug_text,
                    transform=self.preview_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                self.preview_ax.set_title(f"Preview: {plot_config['name']})self.preview_canvas.draw()
                return

            # Get the actual data using the same method as main plotting

            try:
            df = self.get_data_for_plotting(file_name)

            if df is None or df.empty:
                # Show available files for debugging
                available_files = []
                if hasattr(self, "processed_files") and self.processed_files:
                    available_files.extend(
                        [os.path.basename(fp) for fp in self.processed_files.keys()],
                if hasattr(self, "input_file_paths") and self.input_file_paths:
                    available_files.extend(
                        [os.path.basename(fp) for fp in self.input_file_paths],

                if available_files:
                    debug_text = (
                        f"Data file '{file_name}'" \
                            "not found\n\nAvailable files:\n+"\n".join(set(available_files)[:5])
                    if len(set(available_files)) > 5:
                        debug_text += f"\n... and {len(set(available_files))-5}moreelse:
                    debug_text = (
                        "No data files loaded\n\n"
                        "Please:\n1. Select CSV files on Setup" \
                            "tab\n2. Process files or plot directly"

                self.preview_ax.text(
                    0.5,
                    0.5,
                    debug_text,
                    transform=self.preview_ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                self.preview_ax.set_title(f"Preview: {plot_config['name']})self.preview_canvas.draw()
                return

            # Get time column and available signals
            time_col = df.columns[0]
            # Try to find a better time column if the first column doesn't look like
            # time
            for col in df.columns:
                if any(
                    time_word in col.lower()
                    for time_word in ["time", "timestamp", "date"]
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
                self.preview_ax.set_title(f"Preview: {plot_config['name']})self.preview_canvas.draw()
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
                            plot_df = plot_df[plot_df[time_col] >= start_datetime]
                        except Exception as e:
                            # Log time range filtering errors for debugging
                            print(f"Warning: Failed to apply start time filter: {e})if end_time:
                        try:
                            end_datetime = pd.to_datetime(
                                f"{plot_df[time_col].dt.date.iloc[0]} {end_time}",
                            plot_df = plot_df[plot_df[time_col] <= end_datetime]
                        except Exception as e:
                            # Log time range filtering errors for debugging
                            print(f"Warning: Failed to apply end time filter: {e})# Plot all available signals
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

            # Apply plot configuration
            title = (
                plot_config.get("plot_title", "") or f"Preview: {plot_config['name']}xlabel = plot_config.get("plot_xlabel", "") or time_col
            ylabel = plot_config.get("plot_ylabel", "") or "Value"

            self.preview_ax.set_title(title, fontsize=14)
            self.preview_ax.set_xlabel(xlabel)
            self.preview_ax.set_ylabel(ylabel)

            # Use legend position from plot config if available,
            # otherwise default to 'best'
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

            self.preview_canvas.draw()        except Exception as e:
            self.preview_ax.clear()
            self.preview_ax.text(
                0.5,
                0.5,
                f"Error generating preview:\n{e!s}",
                transform=self.preview_ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
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
                filename = f"{plot_config['name'].replace(' ', '_')}_config.txtfilepath = os.path.join(export_dir, filename)

                with open(filepath, "w") as f:
                    f.write(f"Plot Configuration: {plot_config['name']}\n)f.write(f"Description: {plot_config.get('description', 'N/A')}\n")
                    f.write(f"Created: {plot_config.get('created_date', 'N/A')}\n)f.write(f"Signals: {', '.join(plot_config.get('signals', []))}\n")
                    f.write(f"Start Time:" \
                        "{plot_config.get('start_time', 'N/A')}\n)f.write(f"End Time: {plot_config.get('end_time', 'N/A')}\n")

                    if "filter_type" in plot_config:
                        f.write(f"Filter: {plot_config['filter_type']}\n)f.write("\nFull Configuration:\n")
                    f.writelines(
                        f"  {key}: {value}\n" for key, value in plot_config.items()

                exported_count += 1

            messagebox.showinfo(
                "Export Complete",
                f"Exported {exported_count} plot configurations to {export_dir}",
            )        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting plots: {e})def _on_plot_setting_change(self, *args: Any) -> None:
        """Automatically update plot when appearance settings change."""
        # Only update if we have data and signals selected
        if hasattr(self, "plot_signal_vars"):
            selected_count = sum(
                1 for data in self.plot_signal_vars.values() if data["var"].get()

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
            color_button.pack(side="left", padx=5, pady=5)

            # Color hex code label
            color_label = ctk.CTkLabel(
                color_frame,
                text=color,
                font=ctk.CTkFont(size=10),
            color_label.pack(side="left", padx=5, pady=5)

            # Remove button
            remove_button = ctk.CTkButton(
                color_frame,
                text="",
                width=30,
                height=30,
                command=lambda idx=i: self._remove_custom_color(idx),
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
            "#1f77b4#ff7f0e",
            "#2ca02c#d62728",
            "#9467bd#8c564b",
            "#e377c2#7f7f7f",
            "#bcbd22#17becf",
        self._update_custom_colors_display()
        if self.color_scheme_var.get() == "Custom Colors":
            self._on_plot_setting_change()

    def _bind_mousewheel_to_frame(self, frame: ctk.CTkFrame) -> None:
        """Bind mouse wheel events to a frame for proper scrolling."""

        def on_mousewheel(event: tk.Event) -> None:
            """Handle mouse wheel scrolling for the frame."""
            # Scroll the frame's canvas
            try:
                frame._parent_canvas.yview_scroll(
                    int(-1 * (event.delta / 120)),
                    "units",
            except Exception:
                # Fallback for different systems
                frame._parent_canvas.yview_scroll(int(-1 * event.delta), "units")

        # Bind mousewheel to the frame and all its children
        def bind_mousewheel(widget: tk.Widget) -> None:
            """Recursively bind mouse wheel events to a widget and its children."""
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
        self.plot_canvas.mpl_connect(
            "button_release_event",
            self._on_trendline_selection_end,

        # Update button text
        self.trendline_select_button.configure(
            text="Click and drag on plot to select range",
        self.trendline_selected_range.configure(text="Selection active...")

    def _on_trendline_selection_start(self, event: Any) -> None:
        """Handle start of trendline selection."""
        if (
            hasattr(self, "trendline_selection_active")
            and self.trendline_selection_active
            and event.inaxes
            self.trendline_selection_start = event.xdata

    def _on_trendline_selection_end(self, event: Any) -> None:
        """Handle end of trendline selection."""
        if (
            hasattr(self, "trendline_selection_active")
            and self.trendline_selection_active
            and event.inaxes
            if self.trendline_selection_start is not None:
                self.trendline_selection_end = event.xdata

                # Ensure start < end
                if self.trendline_selection_start > self.trendline_selection_end:
                    self.trendline_selection_start, self.trendline_selection_end = (
                        self.trendline_selection_end,
                        self.trendline_selection_start,

                # Update display
                start_str = f"{self.trendline_selection_start:.2f}end_str = f"{self.trendline_selection_end:.2f}"
                self.trendline_selected_range.configure(
                    text=f"Range: {start_str} to {end_str}",

                # Disable selection mode
                self.trendline_selection_active = False
                self.trendline_select_button.configure(
                    text="Select Time Window on Plot",

                # Update plot
                self._on_plot_setting_change()

    def _on_dataset_naming_change(self) -> None:
        """Handle changes to dataset naming mode."""
        if self.dataset_naming_var.get() == "custom":
            self.custom_dataset_entry.configure(state="normal")
            self.custom_dataset_entry.bind(
                "<KeyRelease>",
                self._check_custom_name_overwrite,
        else:
            self.custom_dataset_entry.configure(state="disabled")
            self.overwrite_warning_label.configure(text="")

    def _check_custom_name_overwrite(self, event: Any = None) -> None:
        """Check if custom dataset name will cause file overwrite."""
        if not hasattr(self, "custom_dataset_entry") or not hasattr(
            self,
            "output_directory",
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
                potential_file = os.path.join(output_dir, f"{custom_name}{ext})if os.path.exists(potential_file):
                    existing_files.append(f"{custom_name}{ext}")

            if existing_files:
                warning_text = f"WARNING Warning: Will overwrite" \
                    "existing files: {', '.join(existing_files)}",
            else:
                self.overwrite_warning_label.configure(
                    text=" No file conflicts found",
                    text_color="green",
        else:
            self.overwrite_warning_label.configure(text="")

    def _save_zoom_state(self) -> None:
        """Save current zoom/pan state of the plot."""
        if hasattr(self, "plot_ax"):
            self.saved_zoom_state = {
                "xlim": self.plot_ax.get_xlim(),
                "ylim": self.plot_ax.get_ylim(),
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
            self.plot_ax.set_ylim(
                y_center - new_y_range / 2,
                y_center + new_y_range / 2,
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
            self.plot_ax.set_ylim(
                y_center - new_y_range / 2,
                y_center + new_y_range / 2,
            self.plot_canvas.draw()

    def _preserve_zoom_during_update(self) -> None:
        """Store zoom state before plot update and restore after."""
        zoom_state = None
        if hasattr(self, "plot_ax"):
            zoom_state = {
                "xlim": self.plot_ax.get_xlim(),
                "ylim": self.plot_ax.get_ylim(),
        return zoom_state

    def _auto_fit_plot(self) -> None:
        """Auto-fit the plot to show all data."""
        if hasattr(self, "plot_ax"):
            try:
                self.plot_ax.autoscale_view()
                self.plot_canvas.draw()
                self.status_label.configure(text="Plot auto-fitted to data")
            except Exception as e:
                print(f"Error auto-fitting plot: {e})def" \
                    "_should_auto_zoom(self, reason: str ="filter_change") -> bool:
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
            self.last_plotted_signals: set[str] = set()
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
                print(f"Error restoring zoom state: {e})# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("Starting Advanced CSV Processor - Complete Version...")
    app = CSVProcessorApp()
    app.mainloop()

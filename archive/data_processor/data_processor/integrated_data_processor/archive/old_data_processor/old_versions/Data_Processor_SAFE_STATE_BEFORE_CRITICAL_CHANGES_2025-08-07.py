# =============================================================================
# Advanced CSV Time Series Processor & Analyzer - Integrated Version
#
# Description:
# A comprehensive GUI application for processing, analyzing, and visualizing
# time series data from CSV files. This version integrates the compiler converter
# functionality as an additional tab, along with a parquet file analyzer popup.
#
# Dependencies for Python 3.8+:
# pip install customtkinter pandas numpy scipy matplotlib openpyxl Pillow simpledbf pyarrow tables feather-format
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
import tempfile
import threading
import time
import tkinter as tk
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, simpledialog
from typing import Any, Dict, List, Optional, Set, Tuple

import customtkinter as ctk
import numpy as np
import pandas as pd
from PIL import Image
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

# Import the original data processor
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Data_Processor_r0 import CSVProcessorApp as OriginalCSVProcessorApp

# Import folder tool functionality
try:
    from Claude_Folders_Uno import \
        FolderProcessorApp as OriginalFolderProcessorApp

    FOLDER_TOOL_AVAILABLE = True
except ImportError:
    FOLDER_TOOL_AVAILABLE = False
    print("Warning: Folder tool not available. Folder Tool tab will be disabled.")

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

    enabled: bool = False
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


class FileFormatDetector:
    """Detects file format based on extension and content."""

    @staticmethod
    def detect_format(file_path: str) -> Optional[str]:
        """Detect file format from path and content."""
        if not os.path.exists(file_path):
            return None

        # Check by extension first
        ext = Path(file_path).suffix.lower()

        # Extension-based detection
        if ext in [".csv"]:
            return "csv"
        elif ext in [".tsv", ".txt"]:
            return "tsv"
        elif ext in [".parquet", ".pq"]:
            return "parquet"
        elif ext in [".xlsx", ".xls"]:
            return "excel"
        elif ext in [".json"]:
            return "json"
        elif ext in [".h5", ".hdf5"]:
            return "hdf5"
        elif ext in [".pkl", ".pickle"]:
            return "pickle"
        elif ext in [".npy"]:
            return "numpy"
        elif ext in [".mat"]:
            return "matlab"
        elif ext in [".feather"]:
            return "feather"
        elif ext in [".arrow"]:
            return "arrow"
        elif ext in [".db", ".sqlite"]:
            return "sqlite"

        # Content-based detection for ambiguous extensions
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)

            # Check for CSV/TSV
            if b"," in header and b"\n" in header:
                return "csv"
            elif b"\t" in header and b"\n" in header:
                return "tsv"
            elif header.startswith(b"{") or header.startswith(b"["):
                return "json"
            elif header.startswith(b"PK"):
                return "excel"  # ZIP-based format

        except Exception:
            pass

        return None


class DataReader:
    """Handles reading data from various file formats."""

    @staticmethod
    def read_file(file_path: str, format_type: str, **kwargs) -> pd.DataFrame:
        """Read file based on format type."""
        try:
            if format_type == "csv":
                return pd.read_csv(file_path, **kwargs)
            elif format_type == "tsv":
                return pd.read_csv(file_path, sep="\t", **kwargs)
            elif format_type == "parquet":
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for parquet files")
                return pd.read_parquet(file_path, **kwargs)
            elif format_type == "excel":
                return pd.read_excel(file_path, **kwargs)
            elif format_type == "json":
                return pd.read_json(file_path, **kwargs)
            elif format_type == "hdf5":
                return pd.read_hdf(file_path, **kwargs)
            elif format_type == "pickle":
                return pd.read_pickle(file_path)
            elif format_type == "numpy":
                data = np.load(file_path)
                if isinstance(data, np.ndarray):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame(data.item())
            elif format_type == "matlab":
                if not SCIPY_AVAILABLE:
                    raise ImportError("SciPy is required for MATLAB files")
                data = scipy.io.loadmat(file_path)
                # Convert MATLAB struct to DataFrame
                if len(data) == 1:
                    return pd.DataFrame(data[list(data.keys())[0]])
                else:
                    return pd.DataFrame(data)
            elif format_type == "feather":
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for feather files")
                return pd.read_feather(file_path, **kwargs)
            elif format_type == "arrow":
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for arrow files")
                table = pa.ipc.open_file(file_path).read_all()
                return table.to_pandas()
            elif format_type == "sqlite":
                return pd.read_sql_query("SELECT * FROM data", f"sqlite:///{file_path}")
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            raise Exception(f"Error reading {file_path}: {str(e)}")


class DataWriter:
    """Handles writing data to various file formats."""

    @staticmethod
    def write_file(
        df: pd.DataFrame, file_path: str, format_type: str, **kwargs
    ) -> None:
        """Write DataFrame to file based on format type."""
        try:
            if format_type == "csv":
                df.to_csv(file_path, index=False, **kwargs)
            elif format_type == "tsv":
                df.to_csv(file_path, sep="\t", index=False, **kwargs)
            elif format_type == "parquet":
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for parquet files")
                df.to_parquet(file_path, **kwargs)
            elif format_type == "excel":
                df.to_excel(file_path, index=False, **kwargs)
            elif format_type == "json":
                df.to_json(file_path, orient="records", **kwargs)
            elif format_type == "hdf5":
                df.to_hdf(file_path, key="data", **kwargs)
            elif format_type == "pickle":
                df.to_pickle(file_path)
            elif format_type == "numpy":
                np.save(file_path, df.values)
            elif format_type == "matlab":
                if not SCIPY_AVAILABLE:
                    raise ImportError("SciPy is required for MATLAB files")
                scipy.io.savemat(
                    file_path, {"data": df.values, "columns": df.columns.tolist()}
                )
            elif format_type == "feather":
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for feather files")
                df.to_feather(file_path, **kwargs)
            elif format_type == "arrow":
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for arrow files")
                table = pa.Table.from_pandas(df)
                with pa.ipc.open_file(file_path, "w") as writer:
                    writer.write(table)
            elif format_type == "sqlite":
                import sqlite3

                conn = sqlite3.connect(file_path)
                df.to_sql("data", conn, if_exists="replace", index=False)
                conn.close()
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            raise Exception(f"Error writing {file_path}: {str(e)}")


# =============================================================================
# PARQUET ANALYZER DIALOG
# =============================================================================


class ParquetAnalyzerDialog(ctk.CTkToplevel):
    """Dialog for analyzing parquet file metadata."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title("Parquet File Analyzer")
        self.geometry("600x500")
        self.resizable(True, True)

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title = ctk.CTkLabel(
            main_frame,
            text="Parquet File Metadata Analyzer",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        title.pack(pady=(10, 20))

        # Select file button
        self.select_btn = ctk.CTkButton(
            main_frame, text="Select Parquet File", command=self.select_file, height=40
        )
        self.select_btn.pack(pady=(0, 20))

        # Results display
        self.results_text = ctk.CTkTextbox(main_frame, height=300)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Close button
        close_btn = ctk.CTkButton(
            main_frame, text="Close", command=self.destroy, height=35
        )
        close_btn.pack(pady=(0, 10))

    def select_file(self):
        """Open file dialog to select a parquet file."""
        file_path = filedialog.askopenfilename(
            title="Select Parquet File",
            filetypes=[("Parquet Files", "*.parquet *.pq"), ("All Files", "*.*")],
        )

        if file_path:
            self.analyze_parquet_file(file_path)

    def analyze_parquet_file(self, file_path):
        """Analyze the selected parquet file."""
        try:
            if not PYARROW_AVAILABLE:
                self.results_text.insert(
                    "end", "Error: PyArrow is required for parquet analysis.\n"
                )
                return

            # Read just the metadata
            parquet_file = pq.ParquetFile(file_path)

            # Get file size
            file_size = Path(file_path).stat().st_size

            # Format results
            results = f"=== Parquet File Analysis ===\n"
            results += f"File: {Path(file_path).name}\n"
            results += f"Path: {file_path}\n"
            results += f"Size: {self.format_file_size(file_size)}\n\n"

            results += f"=== Metadata ===\n"
            results += f"Rows: {parquet_file.metadata.num_rows:,}\n"
            results += f"Columns: {parquet_file.metadata.num_columns}\n"
            results += f"Row Groups: {parquet_file.metadata.num_row_groups}\n\n"

            results += f"=== Schema ===\n"
            schema = parquet_file.schema_arrow
            for field in schema:
                results += f"{field.name}: {field.type}\n"

            results += f"\n=== Row Group Details ===\n"
            for i, row_group in enumerate(parquet_file.metadata.row_group_metadata):
                results += f"Row Group {i}:\n"
                results += f"  Rows: {row_group.num_rows:,}\n"
                results += (
                    f"  Size: {self.format_file_size(row_group.total_byte_size)}\n"
                )
                results += f"  Columns: {row_group.num_columns}\n"

                # Column details
                for j, col in enumerate(row_group.column_metadata):
                    results += f"    Column {j}: {col.path_in_schema[0]}\n"
                    results += f"      Values: {col.num_values:,}\n"
                    results += f"      Size: {self.format_file_size(col.total_uncompressed_size)}\n"
                    results += f"      Compressed: {self.format_file_size(col.total_compressed_size)}\n"
                    if col.statistics:
                        stats = col.statistics
                        if hasattr(stats, "min") and hasattr(stats, "max"):
                            results += f"      Min: {stats.min}\n"
                            results += f"      Max: {stats.max}\n"
                    results += f"\n"

            # Clear and insert results
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", results)

        except Exception as e:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Error analyzing file: {str(e)}")

    def format_file_size(self, size_bytes):
        """Format file size in human readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


# =============================================================================
# INTEGRATED APPLICATION CLASS
# =============================================================================


class IntegratedCSVProcessorApp(OriginalCSVProcessorApp):
    """Extended application class with integrated compiler converter functionality."""

    def __init__(self, *args, **kwargs):
        # Initialize converter variables BEFORE calling parent class
        # This ensures they exist when parent class methods are called
        self.converter_input_files = []
        self.converter_output_path = ""
        self.converter_output_format = "parquet"
        self.converter_combine_files = True
        self.converter_selected_columns = set()
        self.converter_use_all_columns = True
        self.converter_split_config = SplitConfig()
        self.converter_batch_processing = False
        self.converter_batch_size = 10
        self.converter_chunk_size = 10000

        # Initialize folder tool variables BEFORE calling parent class
        # This ensures they exist when parent class methods are called
        self.folder_source_folders = []
        self.folder_destination = ""
        self.folder_operation_mode = ctk.StringVar(value="combine")
        self.folder_filter_extensions = ctk.StringVar(value="")
        self.folder_min_file_size = ctk.StringVar(value="0")
        self.folder_max_file_size = ctk.StringVar(value="1000")
        self.folder_organize_by_type_var = ctk.BooleanVar(value=False)
        self.folder_organize_by_date_var = ctk.BooleanVar(value=False)
        self.folder_deduplicate_var = ctk.BooleanVar(value=False)
        self.folder_zip_output_var = ctk.BooleanVar(value=False)
        self.folder_preview_mode_var = ctk.BooleanVar(value=False)
        self.folder_backup_before_var = ctk.BooleanVar(value=False)
        self.folder_progress_var = ctk.DoubleVar(value=0)
        self.folder_status_var = ctk.StringVar(value="Ready")
        self.folder_cancel_flag = False  # For cancelling processing

        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Update title to reflect integration
        self.title("Advanced CSV Time Series Processor & Analyzer - Integrated")

        # Remove the Help tab that was added by parent class
        # We'll add it back at the end to ensure it's the rightmost tab
        self.main_tab_view.delete("Help")

        # Remove the DAT File Import tab to reorder it
        self.main_tab_view.delete("DAT File Import")

        # Add the Format Converter tab
        self.main_tab_view.add("Format Converter")
        self.create_format_converter_tab(self.main_tab_view.tab("Format Converter"))

        # Add DAT File Import tab back (now it will be before Help)
        self.main_tab_view.add("DAT File Import")
        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))

        # Add Folder Tool tab (if available)
        if FOLDER_TOOL_AVAILABLE:
            self.main_tab_view.add("Folder Tool")
            self.create_folder_tool_tab(self.main_tab_view.tab("Folder Tool"))

        # Add Help tab back as the rightmost tab
        self.main_tab_view.add("Help")
        self.create_help_tab(self.main_tab_view.tab("Help"))

    # Compiler converter methods - Define these BEFORE creating the UI
    def converter_browse_files(self):
        """Browse for input files."""
        try:
            files = filedialog.askopenfilenames(
                title="Select Input Files",
                filetypes=[
                    (
                        "All Supported",
                        "*.csv *.tsv *.txt *.parquet *.pq *.xlsx *.xls *.json *.h5 *.hdf5 *.pkl *.pickle *.npy *.mat *.feather *.arrow *.db *.sqlite",
                    ),
                    ("CSV Files", "*.csv"),
                    ("TSV Files", "*.tsv *.txt"),
                    ("Parquet Files", "*.parquet *.pq"),
                    ("Excel Files", "*.xlsx *.xls"),
                    ("JSON Files", "*.json"),
                    ("HDF5 Files", "*.h5 *.hdf5"),
                    ("Pickle Files", "*.pkl *.pickle"),
                    ("NumPy Files", "*.npy"),
                    ("MATLAB Files", "*.mat"),
                    ("Feather Files", "*.feather"),
                    ("Arrow Files", "*.arrow"),
                    ("SQLite Files", "*.db *.sqlite"),
                    ("All Files", "*.*"),
                ],
            )

            if files:
                self.converter_input_files = list(files)
                self.converter_update_file_list()
                self.converter_input_label.configure(
                    text=f"{len(files)} files selected"
                )
                self.converter_update_convert_button()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to browse files: {str(e)}")

    def converter_browse_folder(self):
        """Browse for input folder."""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            # Scan for supported files
            supported_extensions = {
                ".csv",
                ".tsv",
                ".txt",
                ".parquet",
                ".pq",
                ".xlsx",
                ".xls",
                ".json",
                ".h5",
                ".hdf5",
                ".pkl",
                ".pickle",
                ".npy",
                ".mat",
                ".feather",
                ".arrow",
                ".db",
                ".sqlite",
            }

            files = []
            for root, dirs, filenames in os.walk(folder):
                for filename in filenames:
                    if Path(filename).suffix.lower() in supported_extensions:
                        files.append(os.path.join(root, filename))

            if files:
                self.converter_input_files = files
                self.converter_update_file_list()
                self.converter_input_label.configure(
                    text=f"{len(files)} files found in folder"
                )
                self.converter_update_convert_button()
            else:
                messagebox.showwarning(
                    "No Files Found", "No supported files found in the selected folder."
                )

    def converter_clear_files(self):
        """Clear all selected files."""
        self.converter_input_files = []
        self.converter_update_file_list()
        self.converter_input_label.configure(text="No files selected")
        self.converter_update_convert_button()

    def converter_update_file_list(self):
        """Update the file list display."""
        # Clear existing widgets
        for widget in self.converter_file_list_frame.winfo_children():
            widget.destroy()

        if not self.converter_input_files:
            ctk.CTkLabel(self.converter_file_list_frame, text="No files selected").pack(
                padx=5, pady=5
            )
            return

        for i, file_path in enumerate(self.converter_input_files):
            file_frame = ctk.CTkFrame(self.converter_file_list_frame)
            file_frame.pack(fill="x", padx=5, pady=2)

            # File name (truncated if too long)
            filename = os.path.basename(file_path)
            if len(filename) > 40:
                filename = filename[:37] + "..."

            ctk.CTkLabel(file_frame, text=filename).pack(side="left", padx=5, pady=2)

            # Remove button
            ctk.CTkButton(
                file_frame,
                text="X",
                width=30,
                command=lambda fp=file_path: self.converter_remove_file(fp),
            ).pack(side="right", padx=5, pady=2)

    def converter_remove_file(self, file_path):
        """Remove a specific file from the list."""
        if file_path in self.converter_input_files:
            self.converter_input_files.remove(file_path)
            self.converter_update_file_list()
            self.converter_input_label.configure(
                text=f"{len(self.converter_input_files)} files selected"
            )
            self.converter_update_convert_button()

    def converter_browse_output(self):
        """Browse for output directory."""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.converter_output_path = folder
            self.converter_output_label.configure(text=folder)
            self.converter_update_convert_button()

    def converter_select_columns(self):
        """Open column selection dialog."""
        if not self.converter_input_files:
            messagebox.showwarning(
                "No Files", "Please select input files first to determine columns."
            )
            return

        try:
            first_file = self.converter_input_files[0]
            format_type = FileFormatDetector.detect_format(first_file)
            if not format_type:
                messagebox.showerror(
                    "Error", "Could not detect format for the first file."
                )
                return

            df = DataReader.read_file(first_file, format_type)
            columns = df.columns.tolist()

            dialog = ColumnSelectionDialog(self, columns)
            if dialog.result:
                self.converter_selected_columns = set(dialog.result)
                self.converter_use_all_columns_var.set(False)
                self.converter_columns_label.configure(
                    text=f"{len(dialog.result)} columns selected"
                )
                self._log_conversion_message(
                    f"Selected {len(dialog.result)} columns: {', '.join(dialog.result[:5])}{'...' if len(dialog.result) > 5 else ''}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Error reading file: {str(e)}")

    def converter_update_convert_button(self):
        """Update the convert button state."""
        if self.converter_input_files and self.converter_output_path:
            self.converter_convert_button.configure(state="normal")
        else:
            self.converter_convert_button.configure(state="disabled")

    def converter_start_conversion(self):
        """Start the file conversion process."""
        if not self.converter_input_files:
            messagebox.showwarning("No Files", "Please select input files first.")
            return

        if not self.converter_output_path:
            messagebox.showwarning("No Output", "Please select an output directory.")
            return

        # Get conversion parameters
        output_format = self.converter_format_var.get()
        combine_files = self.converter_combine_var.get()
        use_all_columns = self.converter_use_all_columns_var.get()
        batch_processing = self.converter_batch_var.get()
        split_files = self.converter_split_var.get()

        # Start conversion in background thread
        conversion_thread = threading.Thread(
            target=self._perform_conversion,
            args=(
                output_format,
                combine_files,
                use_all_columns,
                batch_processing,
                split_files,
            ),
        )
        conversion_thread.daemon = True
        conversion_thread.start()

    def _perform_conversion(
        self,
        output_format,
        combine_files,
        use_all_columns,
        batch_processing,
        split_files,
    ):
        """Perform the actual file conversion in a background thread."""
        try:
            self.converter_status_label.configure(text="Converting files...")
            self.converter_progress.set(0)
            self.converter_convert_button.configure(state="disabled")

            total_files = len(self.converter_input_files)
            processed_files = 0

            if combine_files:
                # Combine all files into one
                self._log_conversion_message(
                    f"Starting conversion: combining {total_files} files into {output_format.upper()}"
                )

                combined_data = []
                for i, file_path in enumerate(self.converter_input_files):
                    try:
                        format_type = FileFormatDetector.detect_format(file_path)
                        if not format_type:
                            self._log_conversion_message(
                                f"Warning: Could not detect format for {os.path.basename(file_path)}"
                            )
                            continue

                        df = DataReader.read_file(file_path, format_type)

                        # Apply column selection
                        if not use_all_columns and self.converter_selected_columns:
                            available_columns = [
                                col
                                for col in self.converter_selected_columns
                                if col in df.columns
                            ]
                            if available_columns:
                                df = df[available_columns]
                            else:
                                self._log_conversion_message(
                                    f"Warning: No selected columns found in {os.path.basename(file_path)}"
                                )
                                continue

                        combined_data.append(df)
                        self._log_conversion_message(
                            f"Loaded {os.path.basename(file_path)}: {len(df)} rows, {len(df.columns)} columns"
                        )

                        processed_files += 1
                        self.converter_progress.set(processed_files / total_files)

                    except Exception as e:
                        self._log_conversion_message(
                            f"Error reading {os.path.basename(file_path)}: {str(e)}"
                        )

                if combined_data:
                    try:
                        combined_df = pd.concat(combined_data, ignore_index=True)
                        output_filename = self._generate_output_filename(
                            output_format, "combined_data"
                        )
                        output_path = os.path.join(
                            self.converter_output_path, output_filename
                        )

                        DataWriter.write_file(combined_df, output_path, output_format)
                        self._log_conversion_message(
                            f"Successfully created: {output_filename}"
                        )
                        self._log_conversion_message(
                            f"Combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns"
                        )

                    except Exception as e:
                        self._log_conversion_message(
                            f"Error writing combined file: {str(e)}"
                        )
                else:
                    self._log_conversion_message("No valid data to combine")

            else:
                # Process files individually
                self._log_conversion_message(
                    f"Starting conversion: processing {total_files} files individually"
                )

                for i, file_path in enumerate(self.converter_input_files):
                    try:
                        format_type = FileFormatDetector.detect_format(file_path)
                        if not format_type:
                            self._log_conversion_message(
                                f"Warning: Could not detect format for {os.path.basename(file_path)}"
                            )
                            continue

                        df = DataReader.read_file(file_path, format_type)

                        # Apply column selection
                        if not use_all_columns and self.converter_selected_columns:
                            available_columns = [
                                col
                                for col in self.converter_selected_columns
                                if col in df.columns
                            ]
                            if available_columns:
                                df = df[available_columns]
                            else:
                                self._log_conversion_message(
                                    f"Warning: No selected columns found in {os.path.basename(file_path)}"
                                )
                                continue

                        # Generate output filename
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        output_filename = self._generate_output_filename(
                            output_format, base_name
                        )
                        output_path = os.path.join(
                            self.converter_output_path, output_filename
                        )

                        DataWriter.write_file(df, output_path, output_format)
                        self._log_conversion_message(
                            f"Converted {os.path.basename(file_path)} -> {output_filename}"
                        )

                        processed_files += 1
                        self.converter_progress.set(processed_files / total_files)

                    except Exception as e:
                        self._log_conversion_message(
                            f"Error converting {os.path.basename(file_path)}: {str(e)}"
                        )

            self.converter_status_label.configure(
                text=f"Conversion complete. {processed_files} files processed."
            )
            self.converter_progress.set(1.0)

        except Exception as e:
            self._log_conversion_message(f"Conversion error: {str(e)}")
            self.converter_status_label.configure(text="Conversion failed")
        finally:
            self.converter_convert_button.configure(state="normal")

    def _generate_output_filename(self, output_format, base_name=None):
        """Generate output filename with proper extension."""
        if not base_name:
            base_name = "converted_data"

        extensions = {
            "parquet": ".parquet",
            "csv": ".csv",
            "tsv": ".tsv",
            "excel": ".xlsx",
            "json": ".json",
            "hdf5": ".h5",
            "pickle": ".pkl",
            "numpy": ".npy",
            "matlab": ".mat",
            "feather": ".feather",
            "arrow": ".arrow",
            "sqlite": ".db",
        }

        extension = extensions.get(output_format, ".txt")
        return f"{base_name}{extension}"

    def _log_conversion_message(self, message):
        """Add a message to the conversion log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        # Update in main thread
        self.after(0, lambda: self.converter_log_text.insert("end", log_message))
        self.after(0, lambda: self.converter_log_text.see("end"))

    def converter_clear_log(self):
        """Clear the conversion log."""
        self.converter_log_text.delete("1.0", "end")

    def converter_save_log(self):
        """Save the conversion log to a file."""
        log_content = self.converter_log_text.get("1.0", "end")
        if log_content.strip():
            file_path = filedialog.asksaveasfilename(
                title="Save Conversion Log",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            )
            if file_path:
                try:
                    with open(file_path, "w") as f:
                        f.write(log_content)
                    messagebox.showinfo("Success", f"Log saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save log: {str(e)}")

    def show_parquet_analyzer(self):
        """Show the parquet analyzer dialog."""
        dialog = ParquetAnalyzerDialog(self)
        dialog.grab_set()  # Make dialog modal

    def create_format_converter_tab(self, parent_tab):
        """Create the format converter tab."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)

        def create_converter_left_content(left_panel):
            """Create the left panel content for converter"""
            left_panel.grid_rowconfigure(0, weight=1)
            left_panel.grid_columnconfigure(0, weight=1)

            # Create scrollable frame
            converter_scrollable_frame = ctk.CTkScrollableFrame(left_panel)
            converter_scrollable_frame.grid(
                row=0, column=0, padx=10, pady=10, sticky="nsew"
            )
            converter_scrollable_frame.grid_columnconfigure(0, weight=1)

            # Input section
            input_frame = ctk.CTkFrame(converter_scrollable_frame)
            input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
            input_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(input_frame, text="Input Files:").grid(
                row=0, column=0, padx=5, pady=5, sticky="w"
            )
            self.converter_input_label = ctk.CTkLabel(
                input_frame, text="No files selected"
            )
            self.converter_input_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

            input_buttons_frame = ctk.CTkFrame(input_frame)
            input_buttons_frame.grid(
                row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
            )

            ctk.CTkButton(
                input_buttons_frame,
                text="Browse Files",
                command=self.converter_browse_files,
            ).pack(side="left", padx=5)
            ctk.CTkButton(
                input_buttons_frame,
                text="Browse Folder",
                command=self.converter_browse_folder,
            ).pack(side="left", padx=5)
            ctk.CTkButton(
                input_buttons_frame,
                text="Clear Files",
                command=self.converter_clear_files,
            ).pack(side="left", padx=5)

            # Output section
            output_frame = ctk.CTkFrame(converter_scrollable_frame)
            output_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
            output_frame.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(output_frame, text="Output Format:").grid(
                row=0, column=0, padx=5, pady=5, sticky="w"
            )
            self.converter_format_var = ctk.StringVar(value="parquet")
            format_combo = ctk.CTkComboBox(
                output_frame,
                values=[
                    "parquet",
                    "csv",
                    "tsv",
                    "excel",
                    "json",
                    "hdf5",
                    "pickle",
                    "numpy",
                    "matlab",
                    "feather",
                    "arrow",
                    "sqlite",
                ],
                variable=self.converter_format_var,
            )
            format_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

            ctk.CTkLabel(output_frame, text="Output Path:").grid(
                row=1, column=0, padx=5, pady=5, sticky="w"
            )
            self.converter_output_label = ctk.CTkLabel(
                output_frame, text="No output path selected"
            )
            self.converter_output_label.grid(
                row=1, column=1, padx=5, pady=5, sticky="w"
            )

            ctk.CTkButton(
                output_frame, text="Browse Output", command=self.converter_browse_output
            ).grid(row=2, column=0, columnspan=2, padx=5, pady=5)

            # Options section
            options_frame = ctk.CTkFrame(converter_scrollable_frame)
            options_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

            self.converter_combine_var = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                options_frame,
                text="Combine all files into one",
                variable=self.converter_combine_var,
            ).pack(anchor="w", padx=5, pady=2)

            self.converter_use_all_columns_var = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                options_frame,
                text="Use all columns",
                variable=self.converter_use_all_columns_var,
            ).pack(anchor="w", padx=5, pady=2)

            self.converter_batch_var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                options_frame,
                text="Batch processing",
                variable=self.converter_batch_var,
            ).pack(anchor="w", padx=5, pady=2)

            self.converter_split_var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                options_frame,
                text="Split large files",
                variable=self.converter_split_var,
            ).pack(anchor="w", padx=5, pady=2)

            # Column selection
            column_frame = ctk.CTkFrame(converter_scrollable_frame)
            column_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

            ctk.CTkLabel(column_frame, text="Column Selection:").pack(
                anchor="w", padx=5, pady=2
            )
            self.converter_columns_label = ctk.CTkLabel(
                column_frame, text="All columns selected"
            )
            self.converter_columns_label.pack(anchor="w", padx=5, pady=2)

            ctk.CTkButton(
                column_frame,
                text="Select Columns",
                command=self.converter_select_columns,
            ).pack(anchor="w", padx=5, pady=5)

            # Convert button
            self.converter_convert_button = ctk.CTkButton(
                converter_scrollable_frame,
                text="Convert Files",
                command=lambda: self.converter_start_conversion(),
                height=40,
            )
            self.converter_convert_button.grid(
                row=4, column=0, sticky="ew", padx=5, pady=10
            )

            # Progress
            self.converter_progress = ctk.CTkProgressBar(converter_scrollable_frame)
            self.converter_progress.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
            self.converter_progress.set(0)

            # Status
            self.converter_status_label = ctk.CTkLabel(
                converter_scrollable_frame, text="Ready"
            )
            self.converter_status_label.grid(
                row=6, column=0, sticky="w", padx=5, pady=5
            )

        def create_converter_right_content(right_panel):
            """Create the right panel content for converter"""
            right_panel.grid_rowconfigure(1, weight=1)
            right_panel.grid_columnconfigure(0, weight=1)

            # File list
            self.converter_file_list_frame = ctk.CTkScrollableFrame(
                right_panel, label_text="Selected Files", height=200
            )
            self.converter_file_list_frame.grid(
                row=0, column=0, padx=10, pady=(0, 10), sticky="ew"
            )

            # Log area
            log_frame = ctk.CTkFrame(right_panel)
            log_frame.grid(row=1, column=0, padx=10, pady=0, sticky="nsew")
            log_frame.grid_columnconfigure(0, weight=1)
            log_frame.grid_rowconfigure(0, weight=1)

            ctk.CTkLabel(log_frame, text="Conversion Log:").pack(
                anchor="w", padx=5, pady=2
            )
            self.converter_log_text = ctk.CTkTextbox(log_frame, height=300)
            self.converter_log_text.pack(fill="both", expand=True, padx=5, pady=5)

            # Buttons
            button_frame = ctk.CTkFrame(right_panel)
            button_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

            ctk.CTkButton(
                button_frame, text="Analyze Parquet", command=self.show_parquet_analyzer
            ).pack(side="left", padx=5)
            ctk.CTkButton(
                button_frame, text="Clear Log", command=self.converter_clear_log
            ).pack(side="left", padx=5)
            ctk.CTkButton(
                button_frame, text="Save Log", command=self.converter_save_log
            ).pack(side="left", padx=5)

        # Create the splitter
        splitter_frame = self._create_splitter(
            parent_tab,
            create_converter_left_content,
            create_converter_right_content,
            "converter_left_width",
            400,
        )
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def create_folder_tool_tab(self, parent_tab):
        """Create the folder tool tab with integrated folder processor functionality."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)

        # Create scrollable frame for the folder tool
        folder_scrollable_frame = ctk.CTkScrollableFrame(parent_tab)
        folder_scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        folder_scrollable_frame.grid_columnconfigure(0, weight=1)

        # Folder tool variables are now initialized in __init__()

        # Create UI sections
        self._create_folder_source_section(folder_scrollable_frame)
        self._create_folder_destination_section(folder_scrollable_frame)
        self._create_folder_filtering_section(folder_scrollable_frame)
        self._create_folder_operation_section(folder_scrollable_frame)
        self._create_folder_organization_section(folder_scrollable_frame)
        self._create_folder_output_section(folder_scrollable_frame)
        self._create_folder_progress_section(folder_scrollable_frame)
        self._create_folder_run_section(folder_scrollable_frame)

        # Initialize mode description
        self._update_folder_mode_description()

    def _create_folder_source_section(self, parent):
        """Create the source folders section."""
        source_frame = ctk.CTkFrame(parent)
        source_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        source_frame.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(
            source_frame,
            text="1. Select Folder(s) to Process",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Source folder listbox
        self.folder_source_listbox = ctk.CTkTextbox(source_frame, height=120)
        self.folder_source_listbox.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Buttons
        button_frame = ctk.CTkFrame(source_frame)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkButton(
            button_frame,
            text="Add Folder(s)",
            command=self._folder_select_source_folders,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            button_frame,
            text="Remove Selected",
            command=self._folder_remove_selected_source,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            button_frame, text="Clear All", command=self._folder_clear_source_folders
        ).pack(side="left", padx=5)

        # Info label
        self.folder_source_info_label = ctk.CTkLabel(
            source_frame, text="No folders selected", text_color="gray"
        )
        self.folder_source_info_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)

    def _create_folder_destination_section(self, parent):
        """Create the destination folder section."""
        dest_frame = ctk.CTkFrame(parent)
        dest_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        dest_frame.grid_columnconfigure(1, weight=1)

        # Title
        ctk.CTkLabel(
            dest_frame,
            text="2. Select Final Destination Folder",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        # Destination label
        self.folder_dest_label = ctk.CTkLabel(
            dest_frame, text="No destination selected", text_color="gray"
        )
        self.folder_dest_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        # Set destination button
        ctk.CTkButton(
            dest_frame, text="Set Destination", command=self._folder_select_dest_folder
        ).grid(row=1, column=1, padx=10, pady=5)

    def _create_folder_filtering_section(self, parent):
        """Create the file filtering section."""
        filter_frame = ctk.CTkFrame(parent)
        filter_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        filter_frame.grid_columnconfigure(1, weight=1)

        # Title
        ctk.CTkLabel(
            filter_frame,
            text="3. File Filtering Options",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        # Extensions filter
        ctk.CTkLabel(
            filter_frame, text="Include only extensions (comma-separated):"
        ).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkEntry(
            filter_frame,
            textvariable=self.folder_filter_extensions,
            placeholder_text=".jpg,.png,.pdf",
        ).grid(row=1, column=1, sticky="ew", padx=10, pady=2)

        # File size filters
        ctk.CTkLabel(filter_frame, text="Min size (MB):").grid(
            row=2, column=0, sticky="w", padx=10, pady=2
        )
        ctk.CTkEntry(
            filter_frame, textvariable=self.folder_min_file_size, width=100
        ).grid(row=2, column=1, sticky="w", padx=10, pady=2)

        ctk.CTkLabel(filter_frame, text="Max size (MB):").grid(
            row=3, column=0, sticky="w", padx=10, pady=2
        )
        ctk.CTkEntry(
            filter_frame, textvariable=self.folder_max_file_size, width=100
        ).grid(row=3, column=1, sticky="w", padx=10, pady=2)

        # Help text
        ctk.CTkLabel(
            filter_frame,
            text="Example: .jpg,.png,.pdf (leave empty for all files)",
            text_color="gray",
            font=ctk.CTkFont(size=12),
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    def _create_folder_operation_section(self, parent):
        """Create the main operation section."""
        operation_frame = ctk.CTkFrame(parent)
        operation_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            operation_frame,
            text="4. Choose Main Operation",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Radio buttons
        operations = [
            ("Combine & Copy", "combine"),
            ("Flatten & Tidy", "flatten"),
            ("Copy & Prune Empty Folders", "prune"),
            ("Deduplicate Files (In-Place)", "deduplicate"),
            ("Analyze & Report Only", "analyze"),
        ]

        for i, (text, value) in enumerate(operations):
            ctk.CTkRadioButton(
                operation_frame,
                text=text,
                variable=self.folder_operation_mode,
                value=value,
                command=self._update_folder_mode_description,
            ).grid(row=i + 1, column=0, sticky="w", padx=10, pady=2)

        # Mode description
        self.folder_mode_description = ctk.CTkLabel(
            operation_frame, text="", wraplength=600, text_color="blue"
        )
        self.folder_mode_description.grid(
            row=len(operations) + 1, column=0, sticky="w", padx=10, pady=10
        )

    def _create_folder_organization_section(self, parent):
        """Create the organization options section."""
        org_frame = ctk.CTkFrame(parent)
        org_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            org_frame,
            text="5. File Organization Options",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Checkboxes
        ctk.CTkCheckBox(
            org_frame,
            text="Organize files by type (create subfolders)",
            variable=self.folder_organize_by_type_var,
        ).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            org_frame,
            text="Organize files by date (YYYY/MM folders)",
            variable=self.folder_organize_by_date_var,
        ).grid(row=2, column=0, sticky="w", padx=10, pady=2)

    def _create_folder_output_section(self, parent):
        """Create the output options section."""
        output_frame = ctk.CTkFrame(parent)
        output_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            output_frame,
            text="6. Output Options",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Checkboxes
        ctk.CTkCheckBox(
            output_frame,
            text="Deduplicate renamed files in destination folder after copy",
            variable=self.folder_deduplicate_var,
        ).grid(row=1, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            output_frame,
            text="Create ZIP archive of final result",
            variable=self.folder_zip_output_var,
        ).grid(row=2, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            output_frame,
            text="Preview mode (show what would be done without executing)",
            variable=self.folder_preview_mode_var,
        ).grid(row=3, column=0, sticky="w", padx=10, pady=2)
        ctk.CTkCheckBox(
            output_frame,
            text="Create backup before processing",
            variable=self.folder_backup_before_var,
        ).grid(row=4, column=0, sticky="w", padx=10, pady=2)

    def _create_folder_progress_section(self, parent):
        """Create the progress section."""
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)

        # Title
        ctk.CTkLabel(
            progress_frame, text="Progress", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Progress bar
        self.folder_progress_bar = ctk.CTkProgressBar(progress_frame)
        self.folder_progress_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.folder_progress_bar.set(0)

        # Status label
        self.folder_status_label = ctk.CTkLabel(
            progress_frame, textvariable=self.folder_status_var
        )
        self.folder_status_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)

    def _create_folder_run_section(self, parent):
        """Create the run button section."""
        run_frame = ctk.CTkFrame(parent)
        run_frame.grid(row=7, column=0, sticky="ew", padx=5, pady=5)

        # Buttons
        self.folder_run_button = ctk.CTkButton(
            run_frame,
            text="Run Folder Process",
            command=self._folder_run_processing,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.folder_run_button.grid(row=0, column=0, padx=10, pady=10)

        self.folder_cancel_button = ctk.CTkButton(
            run_frame,
            text="Cancel",
            command=self._folder_cancel_processing,
            state="disabled",
        )
        self.folder_cancel_button.grid(row=0, column=1, padx=10, pady=10)

    def _update_folder_mode_description(self):
        """Update the mode description based on selected operation."""
        mode = self.folder_operation_mode.get()

        descriptions = {
            "combine": "Copies all files from source folders into the single destination folder.",
            "flatten": "Finds deeply nested folders and copies them to the top level of the destination.",
            "prune": "Copies source folders to the destination, preserving structure but skipping empty sub-folders.",
            "deduplicate": "Deletes renamed duplicates like 'file (1).txt' within the source folder(s), keeping the newest version.",
            "analyze": "Analyzes folder contents and generates a detailed report without making changes.",
        }

        self.folder_mode_description.configure(text=descriptions.get(mode, ""))

    def _folder_select_source_folders(self):
        """Select source folders for processing."""
        try:
            folders = filedialog.askdirectory(
                title="Select Source Folders", multiple=True
            )
            if folders:
                self.folder_source_folders.extend(folders)
                self._folder_update_source_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select source folders: {str(e)}")

    def _folder_remove_selected_source(self):
        """Remove selected source folder from the list."""
        # For simplicity, remove the last added folder
        if self.folder_source_folders:
            self.folder_source_folders.pop()
            self._folder_update_source_display()

    def _folder_clear_source_folders(self):
        """Clear all source folders."""
        self.folder_source_folders = []
        self._folder_update_source_display()

    def _folder_update_source_display(self):
        """Update the source folders display."""
        self.folder_source_listbox.delete("1.0", "end")
        if self.folder_source_folders:
            for folder in self.folder_source_folders:
                self.folder_source_listbox.insert("end", f"{folder}\n")
            self.folder_source_info_label.configure(
                text=f"{len(self.folder_source_folders)} folder(s) selected"
            )
        else:
            self.folder_source_info_label.configure(text="No folders selected")

    def _folder_select_dest_folder(self):
        """Select destination folder."""
        try:
            folder = filedialog.askdirectory(title="Select Destination Folder")
            if folder:
                self.folder_destination = folder
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to select destination folder: {str(e)}"
            )
            self.folder_dest_label.configure(text=folder)

    def _folder_run_processing(self):
        """Start the folder processing operation."""
        if not self.folder_source_folders:
            messagebox.showwarning(
                "No Source Folders", "Please select at least one source folder."
            )
            return

        mode = self.folder_operation_mode.get()
        if mode not in ["deduplicate", "analyze"] and not self.folder_destination:
            messagebox.showwarning(
                "No Destination", "Please select a destination folder."
            )
            return

        # Reset cancel flag
        self.folder_cancel_flag = False

        # Start processing in background thread
        processing_thread = threading.Thread(target=self._folder_perform_processing)
        processing_thread.daemon = True
        processing_thread.start()

        # Update UI
        self.folder_run_button.configure(state="disabled")
        self.folder_cancel_button.configure(state="normal")
        self.folder_status_var.set("Processing...")

    def _folder_cancel_processing(self):
        """Cancel the folder processing operation."""
        self.folder_cancel_flag = True  # Set flag to signal cancellation
        self.folder_status_var.set("Cancelled")
        self.folder_progress_bar.set(0)
        self.folder_run_button.configure(state="normal")
        self.folder_cancel_button.configure(state="disabled")

    def _folder_perform_processing(self):
        """Perform the actual folder processing operation."""
        try:
            mode = self.folder_operation_mode.get()

            if mode == "combine":
                self._folder_combine_operation()
            elif mode == "flatten":
                self._folder_flatten_operation()
            elif mode == "prune":
                self._folder_prune_operation()
            elif mode == "deduplicate":
                self._folder_deduplicate_operation()
            elif mode == "analyze":
                self._folder_analyze_operation()

            # Complete
            self.after(0, lambda: self.folder_status_var.set("Processing complete"))
            self.after(0, lambda: self.folder_progress_bar.set(1.0))
            self.after(0, lambda: self.folder_run_button.configure(state="normal"))
            self.after(0, lambda: self.folder_cancel_button.configure(state="disabled"))

        except Exception as e:
            self.after(0, lambda: self.folder_status_var.set(f"Error: {str(e)}"))
            self.after(0, lambda: self.folder_run_button.configure(state="normal"))
            self.after(0, lambda: self.folder_cancel_button.configure(state="disabled"))

    def _folder_combine_operation(self):
        """Perform combine operation - copy all files from source folders to destination."""
        try:
            import os
            import shutil
            from datetime import datetime

            # Create destination directory
            os.makedirs(self.folder_destination, exist_ok=True)

            # Count total files for progress tracking
            total_files = 0
            for src in self.folder_source_folders:
                for root, dirs, files in os.walk(src):
                    total_files += len(files)

            if total_files == 0:
                self.after(
                    0,
                    lambda: self.folder_status_var.set(
                        "No files found in source folders"
                    ),
                )
                return

            processed_files = 0
            copied_count = 0
            renamed_count = 0
            skipped_count = 0

            for src in self.folder_source_folders:
                if self.folder_cancel_flag:
                    break

                for root, dirs, files in os.walk(src):
                    for file in files:
                        if self.folder_cancel_flag:
                            break

                        source_path = os.path.join(root, file)

                        # Apply file filters
                        if not self._folder_validate_file_filters(source_path):
                            skipped_count += 1
                            processed_files += 1
                            continue

                        # Get organized destination path
                        dest_path = self._folder_get_organized_path(
                            source_path, self.folder_destination
                        )
                        dest_dir = os.path.dirname(dest_path)

                        # Create destination directory if needed
                        os.makedirs(dest_dir, exist_ok=True)

                        # Handle naming conflicts
                        final_dest_path = self._folder_get_unique_path(dest_path)
                        if final_dest_path != dest_path:
                            renamed_count += 1

                        try:
                            if not self.folder_preview_mode_var.get():
                                shutil.copy2(source_path, final_dest_path)
                            copied_count += 1
                        except Exception as e:
                            print(f"Error copying '{file}': {e}")

                        processed_files += 1
                        if processed_files % 10 == 0:  # Update progress every 10 files
                            progress = processed_files / total_files
                            self.after(
                                0, lambda p=progress: self.folder_progress_bar.set(p)
                            )
                            self.after(
                                0,
                                lambda p=processed_files, t=total_files: self.folder_status_var.set(
                                    f"Processed {p}/{t} files"
                                ),
                            )

            # Final status
            if self.folder_preview_mode_var.get():
                status = f"PREVIEW: Would copy {copied_count} files, rename {renamed_count}, skip {skipped_count}"
            else:
                status = f"Copied {copied_count} files, renamed {renamed_count}, skipped {skipped_count}"

            self.after(0, lambda: self.folder_status_var.set(status))

        except Exception as e:
            self.after(0, lambda: self.folder_status_var.set(f"Error: {str(e)}"))

    def _folder_flatten_operation(self):
        """Perform flatten operation - copy files from nested folders to top level."""
        try:
            import os
            import shutil

            # Create destination directory
            os.makedirs(self.folder_destination, exist_ok=True)

            # Count total files for progress tracking
            total_files = 0
            for src in self.folder_source_folders:
                for root, dirs, files in os.walk(src):
                    total_files += len(files)

            if total_files == 0:
                self.after(
                    0,
                    lambda: self.folder_status_var.set(
                        "No files found in source folders"
                    ),
                )
                return

            processed_files = 0
            copied_count = 0
            renamed_count = 0
            skipped_count = 0

            for src in self.folder_source_folders:
                if self.folder_cancel_flag:
                    break

                for root, dirs, files in os.walk(src):
                    for file in files:
                        if self.folder_cancel_flag:
                            break

                        source_path = os.path.join(root, file)

                        # Apply file filters
                        if not self._folder_validate_file_filters(source_path):
                            skipped_count += 1
                            processed_files += 1
                            continue

                        # For flatten operation, files go directly to destination root
                        dest_path = os.path.join(self.folder_destination, file)

                        # Handle naming conflicts
                        final_dest_path = self._folder_get_unique_path(dest_path)
                        if final_dest_path != dest_path:
                            renamed_count += 1

                        try:
                            if not self.folder_preview_mode_var.get():
                                shutil.copy2(source_path, final_dest_path)
                            copied_count += 1
                        except Exception as e:
                            print(f"Error copying '{file}': {e}")

                        processed_files += 1
                        if processed_files % 10 == 0:
                            progress = processed_files / total_files
                            self.after(
                                0, lambda p=progress: self.folder_progress_bar.set(p)
                            )
                            self.after(
                                0,
                                lambda p=processed_files, t=total_files: self.folder_status_var.set(
                                    f"Processed {p}/{t} files"
                                ),
                            )

            # Final status
            if self.folder_preview_mode_var.get():
                status = f"PREVIEW: Would flatten {copied_count} files, rename {renamed_count}, skip {skipped_count}"
            else:
                status = f"Flattened {copied_count} files, renamed {renamed_count}, skipped {skipped_count}"

            self.after(0, lambda: self.folder_status_var.set(status))

        except Exception as e:
            self.after(0, lambda: self.folder_status_var.set(f"Error: {str(e)}"))

    def _folder_prune_operation(self):
        """Perform prune operation - copy folders but skip empty subfolders."""
        try:
            import os
            import shutil

            # Create destination directory
            os.makedirs(self.folder_destination, exist_ok=True)

            # Count total files for progress tracking
            total_files = 0
            for src in self.folder_source_folders:
                for root, dirs, files in os.walk(src):
                    total_files += len(files)

            if total_files == 0:
                self.after(
                    0,
                    lambda: self.folder_status_var.set(
                        "No files found in source folders"
                    ),
                )
                return

            processed_files = 0
            copied_count = 0
            skipped_count = 0

            for src in self.folder_source_folders:
                if self.folder_cancel_flag:
                    break

                # Get relative path from source
                src_name = os.path.basename(src)
                dest_src_path = os.path.join(self.folder_destination, src_name)

                for root, dirs, files in os.walk(src):
                    if self.folder_cancel_flag:
                        break

                    # Skip empty directories
                    if not files:
                        continue

                    # Calculate relative path
                    rel_path = os.path.relpath(root, src)
                    dest_dir = os.path.join(dest_src_path, rel_path)

                    # Create destination directory
                    if not self.folder_preview_mode_var.get():
                        os.makedirs(dest_dir, exist_ok=True)

                    for file in files:
                        if self.folder_cancel_flag:
                            break

                        source_path = os.path.join(root, file)

                        # Apply file filters
                        if not self._folder_validate_file_filters(source_path):
                            skipped_count += 1
                            processed_files += 1
                            continue

                        dest_path = os.path.join(dest_dir, file)

                        try:
                            if not self.folder_preview_mode_var.get():
                                shutil.copy2(source_path, dest_path)
                            copied_count += 1
                        except Exception as e:
                            print(f"Error copying '{file}': {e}")

                        processed_files += 1
                        if processed_files % 10 == 0:
                            progress = processed_files / total_files
                            self.after(
                                0, lambda p=progress: self.folder_progress_bar.set(p)
                            )
                            self.after(
                                0,
                                lambda p=processed_files, t=total_files: self.folder_status_var.set(
                                    f"Processed {p}/{t} files"
                                ),
                            )

            # Final status
            if self.folder_preview_mode_var.get():
                status = f"PREVIEW: Would copy {copied_count} files, skip {skipped_count} (pruned empty folders)"
            else:
                status = f"Copied {copied_count} files, skipped {skipped_count} (pruned empty folders)"

            self.after(0, lambda: self.folder_status_var.set(status))

        except Exception as e:
            self.after(0, lambda: self.folder_status_var.set(f"Error: {str(e)}"))

    def _folder_deduplicate_operation(self):
        """Perform deduplicate operation - remove renamed duplicates in source folders."""
        try:
            import os
            import re

            # Count total files for progress tracking
            total_files = 0
            for src in self.folder_source_folders:
                for root, dirs, files in os.walk(src):
                    total_files += len(files)

            if total_files == 0:
                self.after(
                    0,
                    lambda: self.folder_status_var.set(
                        "No files found in source folders"
                    ),
                )
                return

            processed_files = 0
            deleted_count = 0
            pattern = re.compile(r"(.+?)(?: \((\d+)\))?(\.\w+)$")

            for src in self.folder_source_folders:
                if self.folder_cancel_flag:
                    break

                for root, dirs, files in os.walk(src):
                    if self.folder_cancel_flag:
                        break

                    files_by_base_name = {}
                    for filename in files:
                        match = pattern.match(filename)
                        if match:
                            base, _, ext = match.groups()
                            base_name = f"{base}{ext}"
                            files_by_base_name.setdefault(base_name, []).append(
                                os.path.join(root, filename)
                            )

                    for base_name, file_list in files_by_base_name.items():
                        if len(file_list) > 1:
                            try:
                                # Keep the newest file
                                file_to_keep = max(
                                    file_list, key=lambda f: os.path.getmtime(f)
                                )
                            except (OSError, FileNotFoundError):
                                continue

                            for file_path in file_list:
                                if file_path != file_to_keep:
                                    try:
                                        if not self.folder_preview_mode_var.get():
                                            os.remove(file_path)
                                        deleted_count += 1
                                    except OSError as e:
                                        print(
                                            f"Failed to delete '{os.path.basename(file_path)}': {e}"
                                        )

                        processed_files += len(file_list)
                        if processed_files % 10 == 0:
                            progress = processed_files / total_files
                            self.after(
                                0, lambda p=progress: self.folder_progress_bar.set(p)
                            )
                            self.after(
                                0,
                                lambda p=processed_files, t=total_files: self.folder_status_var.set(
                                    f"Processed {p}/{t} files"
                                ),
                            )

            # Final status
            if self.folder_preview_mode_var.get():
                status = f"PREVIEW: Would delete {deleted_count} duplicate files"
            else:
                status = f"Deleted {deleted_count} duplicate files"

            self.after(0, lambda: self.folder_status_var.set(status))

        except Exception as e:
            self.after(0, lambda: self.folder_status_var.set(f"Error: {str(e)}"))

    def _folder_analyze_operation(self):
        """Perform analyze operation - generate detailed report of folder contents."""
        try:
            import os
            from collections import defaultdict
            from datetime import datetime

            # Count total files for progress tracking
            total_files = 0
            for src in self.folder_source_folders:
                for root, dirs, files in os.walk(src):
                    total_files += len(files)

            if total_files == 0:
                self.after(
                    0,
                    lambda: self.folder_status_var.set(
                        "No files found in source folders"
                    ),
                )
                return

            processed_files = 0
            total_size = 0
            file_types = defaultdict(int)
            size_by_type = defaultdict(int)
            largest_files = []

            report_lines = [
                "=== FOLDER ANALYSIS REPORT ===",
                f"Generated: {datetime.now()}",
                "",
            ]

            for src in self.folder_source_folders:
                if self.folder_cancel_flag:
                    break

                report_lines.append(f"Analyzing: {src}")
                folder_files = 0
                folder_size = 0

                for root, dirs, files in os.walk(src):
                    for file in files:
                        if self.folder_cancel_flag:
                            break

                        file_path = os.path.join(root, file)
                        try:
                            file_size = os.path.getsize(file_path)
                            file_ext = (
                                os.path.splitext(file)[1].lower() or "no_extension"
                            )

                            total_size += file_size
                            folder_files += 1
                            folder_size += file_size
                            file_types[file_ext] += 1
                            size_by_type[file_ext] += file_size

                            # Track largest files
                            largest_files.append((file_path, file_size))
                            if len(largest_files) > 10:
                                largest_files.sort(key=lambda x: x[1], reverse=True)
                                largest_files = largest_files[:10]

                        except OSError:
                            continue

                        processed_files += 1
                        if processed_files % 10 == 0:
                            progress = processed_files / total_files
                            self.after(
                                0, lambda p=progress: self.folder_progress_bar.set(p)
                            )
                            self.after(
                                0,
                                lambda p=processed_files, t=total_files: self.folder_status_var.set(
                                    f"Analyzed {p}/{t} files"
                                ),
                            )

                report_lines.append(
                    f"  Files: {folder_files}, Size: {folder_size/(1024*1024):.1f} MB"
                )

            report_lines.extend(
                [
                    "",
                    f"TOTAL FILES: {processed_files}",
                    f"TOTAL SIZE: {total_size/(1024*1024):.1f} MB",
                    "",
                    "FILE TYPES:",
                ]
            )

            for ext, count in sorted(
                file_types.items(), key=lambda x: x[1], reverse=True
            ):
                size_mb = size_by_type[ext] / (1024 * 1024)
                report_lines.append(f"  {ext}: {count} files, {size_mb:.1f} MB")

            report_lines.extend(["", "LARGEST FILES:"])
            for file_path, size in sorted(
                largest_files, key=lambda x: x[1], reverse=True
            ):
                size_mb = size / (1024 * 1024)
                report_lines.append(
                    f"  {os.path.basename(file_path)}: {size_mb:.1f} MB"
                )

            # Show report in a dialog
            report_text = "\n".join(report_lines)
            self.after(0, lambda: self._show_folder_analysis_report(report_text))

            self.after(
                0,
                lambda: self.folder_status_var.set(
                    f"Analysis complete: {processed_files} files analyzed"
                ),
            )

        except Exception as e:
            self.after(0, lambda: self.folder_status_var.set(f"Error: {str(e)}"))

    def _show_folder_analysis_report(self, report_text):
        """Show the analysis report in a dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Folder Analysis Report")
        dialog.geometry("800x600")

        # Make dialog modal
        dialog.transient(self)
        dialog.grab_set()

        # Create text widget
        text_widget = ctk.CTkTextbox(dialog)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # Insert report text
        text_widget.insert("1.0", report_text)

        # Add close button
        close_button = ctk.CTkButton(dialog, text="Close", command=dialog.destroy)
        close_button.pack(pady=10)

    def _folder_validate_file_filters(self, file_path):
        """Validate if a file meets the filtering criteria."""
        if self.folder_cancel_flag:
            return False

        # Extension filter
        extensions = self.folder_filter_extensions.get().strip()
        if extensions:
            ext_list = [ext.strip().lower() for ext in extensions.split(",")]
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in ext_list:
                return False

        # Size filter
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            min_size = float(self.folder_min_file_size.get() or 0)
            if file_size_mb < min_size:
                return False

            max_size_str = self.folder_max_file_size.get().strip()
            if max_size_str:
                max_size = float(max_size_str)
                if file_size_mb > max_size:
                    return False
        except (ValueError, OSError):
            return False

        return True

    def _folder_get_organized_path(self, file_path, dest_base):
        """Returns the organized destination path based on organization options."""
        filename = os.path.basename(file_path)
        dest_path = dest_base

        # Organize by type
        if self.folder_organize_by_type_var.get():
            file_ext = os.path.splitext(filename)[1].lower()
            type_mapping = {
                ".jpg": "Images",
                ".jpeg": "Images",
                ".png": "Images",
                ".gif": "Images",
                ".bmp": "Images",
                ".mp4": "Videos",
                ".avi": "Videos",
                ".mov": "Videos",
                ".wmv": "Videos",
                ".mkv": "Videos",
                ".mp3": "Audio",
                ".wav": "Audio",
                ".flac": "Audio",
                ".aac": "Audio",
                ".pdf": "Documents",
                ".doc": "Documents",
                ".docx": "Documents",
                ".txt": "Documents",
                ".zip": "Archives",
                ".rar": "Archives",
                ".7z": "Archives",
                ".tar": "Archives",
            }
            file_type = type_mapping.get(file_ext, "Other")
            dest_path = os.path.join(dest_path, file_type)

        # Organize by date
        if self.folder_organize_by_date_var.get():
            try:
                mtime = os.path.getmtime(file_path)
                date_folder = datetime.fromtimestamp(mtime).strftime("%Y/%m")
                dest_path = os.path.join(dest_path, date_folder)
            except OSError:
                dest_path = os.path.join(dest_path, "Unknown_Date")

        return os.path.join(dest_path, filename)

    def _folder_get_unique_path(self, path):
        """Get a unique path by adding a number if the file already exists."""
        if not os.path.exists(path):
            return path
        parent, name = os.path.split(path)
        is_file = "." in name and not os.path.isdir(path)
        filename, ext = os.path.splitext(name) if is_file else (name, "")
        counter = 1
        new_path = os.path.join(parent, f"{filename} ({counter}){ext}")
        while os.path.exists(new_path):
            counter += 1
            new_path = os.path.join(parent, f"{filename} ({counter}){ext}")
        return new_path

    def create_help_tab(self, tab):
        """Create the help tab with comprehensive documentation for all integrated features."""
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(tab)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(
            header_frame,
            text="Advanced Data Processor - Complete Help Guide",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(side="left", padx=10, pady=10)

        # Main content with scrollable help
        help_frame = ctk.CTkScrollableFrame(tab)
        help_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        help_frame.grid_columnconfigure(0, weight=1)

        # Comprehensive help content
        help_content = """
# 🚀 Advanced Data Processor - Complete Feature Guide

## 📋 Application Overview
This integrated application combines multiple powerful tools for data processing, analysis, and visualization:

1. **CSV Processor** - Original time series data processing
2. **Format Converter** - Multi-format file conversion with batch processing
3. **DAT File Import** - DAT file processing with DBF tag files
4. **Folder Tool** - Comprehensive folder processing and organization
5. **Help** - This comprehensive documentation

---

## 📊 CSV Processor Tab

### Purpose
Process, analyze, and visualize time series data from CSV files with advanced filtering and mathematical operations.

### Setup Sub-tab
**File Selection & Configuration**
- **Input Files**: Select multiple CSV files for batch processing
- **Output Directory**: Choose where processed files will be saved
- **Configuration Management**: Save/load processing settings
- **Export Format**: Choose output format (CSV, Excel, MAT, Parquet, HDF5, etc.)
- **Sorting Options**: Configure data sorting preferences

**Usage**:
1. Click "Select Files" to choose input CSV files
2. Click "Select Output Folder" to set destination
3. Configure processing options in other sub-tabs
4. Save configuration for future use
5. Select signals to process
6. Click "Process & Batch Export Files"

### Processing Sub-tab
**Signal Processing Operations**

#### 🔧 Signal Filtering
- **Moving Average**: Smooth data with configurable window size
- **Butterworth Filter**: Low-pass, high-pass, band-pass filtering
- **Median Filter**: Remove outliers with configurable window
- **Savitzky-Golay**: Polynomial smoothing for noisy data
- **Hampel Filter**: Robust outlier detection and removal
- **Z-Score Filter**: Statistical outlier removal

#### ⏱️ Time Resampling
- **Resample Data**: Convert to different time intervals (1s, 1min, 1h, etc.)
- **Interpolation Methods**: Linear, cubic, nearest neighbor
- **Aggregation**: Mean, sum, min, max, median

#### 📈 Signal Integration
- **Trapezoidal Integration**: Calculate cumulative values
- **Flow Calculations**: Convert rate data to total volumes
- **Custom Integration**: User-defined integration methods

#### 📉 Signal Differentiation
- **Spline Differentiation**: Smooth derivative calculation
- **Finite Difference**: Direct numerical differentiation
- **Multiple Orders**: 1st through 5th order derivatives

### Custom Variables Sub-tab
**Mathematical Formula Creation**
- **Formula Builder**: Create custom calculations using existing signals
- **Signal Reference**: Use [SignalName] syntax to reference data
- **Mathematical Functions**: sin, cos, exp, log, sqrt, etc.
- **Conditional Logic**: if/else statements for complex calculations

**Example Formulas**:
- `[Flow] * 3600` - Convert flow rate to hourly volume
- `sqrt([Pressure]^2 + [Temperature]^2)` - Calculate magnitude
- `if([Value] > 100, [Value] * 2, [Value])` - Conditional processing

---

## 🔄 Format Converter Tab

### Purpose
Convert files between multiple formats with batch processing capabilities.

### Supported Formats
**Input Formats**: CSV, TSV, Parquet, Excel (.xlsx, .xls), JSON, HDF5, Pickle, NumPy (.npy), MATLAB (.mat), Feather, Arrow, SQLite

**Output Formats**: CSV, TSV, Parquet, Excel (.xlsx), JSON, HDF5, Pickle, NumPy (.npy), MATLAB (.mat), Feather, Arrow, SQLite

### Key Features

#### 📁 File Selection
- **Individual Files**: Select specific files for conversion
- **Folder Import**: Import all supported files from a directory
- **Batch Processing**: Process multiple files simultaneously
- **File List Management**: Add, remove, and clear files

#### ⚙️ Conversion Options
- **Output Format**: Choose target file format
- **Combine Files**: Merge multiple files into single output
- **Column Selection**: Choose specific columns to include
- **Batch Processing**: Enable for large file sets
- **File Splitting**: Split large files into smaller chunks

#### 📊 Parquet Analyzer
- **Metadata Analysis**: View file structure and statistics
- **Column Information**: Data types, null counts, memory usage
- **File Properties**: Size, compression, row groups
- **Schema Details**: Complete column schema information

### Usage Workflow
1. **Select Files**: Choose input files or import from folder
2. **Configure Output**: Set output format and destination
3. **Select Columns**: Choose which columns to include (optional)
4. **Set Options**: Configure batch processing and splitting
5. **Convert**: Start the conversion process
6. **Monitor Progress**: Track conversion status and logs

---

## 📁 Folder Tool Tab

### Purpose
Comprehensive folder processing and organization with multiple operation modes.

### Operation Modes

#### 🔗 Combine & Copy
**Purpose**: Copy all files from multiple source folders into a single destination folder.

**Features**:
- **Multi-source Support**: Process files from multiple source folders
- **Automatic Renaming**: Handle naming conflicts with numbered suffixes
- **File Filtering**: Filter by extension and file size
- **Organization Options**: Organize by file type or date
- **Progress Tracking**: Real-time progress with cancellation

**Use Cases**:
- Consolidating files from multiple backup locations
- Combining photo collections from different devices
- Merging document archives

#### 📂 Flatten & Tidy
**Purpose**: Copy files from deeply nested folder structures to a flat, organized structure.

**Features**:
- **Structure Flattening**: Remove nested folder hierarchy
- **Conflict Resolution**: Handle duplicate filenames automatically
- **File Filtering**: Include only specific file types or sizes
- **Progress Monitoring**: Track operation progress

**Use Cases**:
- Organizing scattered files into a single location
- Simplifying complex folder structures
- Preparing files for backup or sharing

#### ✂️ Copy & Prune Empty
**Purpose**: Copy folder structure while skipping empty directories.

**Features**:
- **Structure Preservation**: Maintain relative folder paths
- **Empty Folder Detection**: Automatically skip empty directories
- **File Filtering**: Apply extension and size filters
- **Efficient Processing**: Only copy non-empty folders

**Use Cases**:
- Cleaning up folder structures
- Removing empty directories from backups
- Organizing file collections

#### 🗑️ Deduplicate Files
**Purpose**: Remove renamed duplicate files (e.g., "file (1).txt", "file (2).txt").

**Features**:
- **Pattern Recognition**: Detect renamed duplicates using regex
- **Newest File Retention**: Keep the most recent version
- **Safe Operation**: Preview mode available
- **In-place Processing**: Works directly on source folders

**Use Cases**:
- Cleaning up duplicate downloads
- Removing system-generated duplicates
- Organizing file collections

#### 📊 Analyze & Report
**Purpose**: Generate comprehensive analysis reports without modifying files.

**Features**:
- **File Statistics**: Count, size, and type analysis
- **Size Distribution**: Largest files identification
- **Type Breakdown**: File type distribution
- **Detailed Reports**: Comprehensive analysis in dialog

**Use Cases**:
- Understanding folder contents before processing
- Identifying large files for cleanup
- Planning storage requirements

### Advanced Features

#### 🔍 File Filtering
- **Extension Filtering**: Include only specific file extensions
- **Size Filtering**: Filter by minimum and maximum file sizes
- **Combined Filters**: Apply multiple filters simultaneously

#### 📁 Organization Options
- **By Type**: Organize files into type-based folders (Images, Documents, etc.)
- **By Date**: Organize by creation/modification date (YYYY/MM structure)
- **Combined Organization**: Use both type and date organization

#### 🛡️ Safety Features
- **Preview Mode**: Show what would be done without making changes
- **Backup Creation**: Create backups before processing
- **Cancellation**: Cancel operations at any time
- **Progress Tracking**: Real-time progress updates

### Usage Workflow
1. **Select Source Folders**: Choose folders to process
2. **Set Destination**: Choose output location (if applicable)
3. **Configure Filters**: Set file type and size filters
4. **Choose Operation**: Select the processing mode
5. **Set Options**: Configure organization and safety options
6. **Run Operation**: Start processing with progress monitoring

---

## 📈 Plotting & Analysis Tab

### Purpose
Interactive visualization and analysis of processed data.

### Key Features

#### 🎯 Smart Auto-Zoom System
- **Auto-zoom Control**: Toggle automatic zoom behavior
- **Smart Detection**: Distinguish between new signals and filter changes
- **Manual Control**: "Fit to Data" button for manual zoom
- **Zoom Preservation**: Maintain view when changing filters

#### 📊 Plotting Capabilities
- **Interactive Charts**: Zoom, pan, and explore data
- **Multiple Chart Types**: Line, scatter, and combination plots
- **Signal Selection**: Choose which signals to display
- **Color Schemes**: Multiple color schemes and custom colors
- **Legend Management**: Customize signal labels and order

#### 📈 Trendline Analysis
- **Linear Regression**: Straight line trend analysis
- **Exponential Fit**: Exponential growth/decay trends
- **Power Law**: Power function relationships
- **Polynomial**: Higher-order polynomial fits
- **R-squared Values**: Statistical fit quality indicators

#### 💾 Export Options
- **Image Export**: Save plots as PNG, JPG, PDF, SVG
- **Excel Export**: Export data and plots to Excel
- **Configuration Save**: Save plot settings for reuse

### Usage Workflow
1. **Select File**: Choose data file from dropdown
2. **Select Signals**: Choose which signals to plot
3. **Configure Display**: Set colors, styles, and layout
4. **Add Analysis**: Include trendlines if needed
5. **Export Results**: Save plots or data as needed

---

## 📋 Plots List Tab

### Purpose
Save and manage plot configurations for batch processing.

### Features
- **Configuration Save**: Save plot settings with names and descriptions
- **Batch Export**: Generate all saved plots automatically
- **Preview System**: Preview plots before saving
- **Library Management**: Organize and manage plot collection

### Usage
1. **Create Plot**: Configure plot in Plotting tab
2. **Save Configuration**: Add to plots library with name/description
3. **Batch Export**: Generate all saved plots at once
4. **Manage Library**: Edit, delete, or reorganize saved plots

---

## 📄 DAT File Import Tab

### Purpose
Process DAT files with associated DBF tag files for structured data import.

### Features
- **DAT File Selection**: Choose data files for processing
- **DBF Tag Import**: Import tag information from DBF files
- **Data Trimming**: Set time ranges for data extraction
- **Export Options**: Save processed data in various formats

### Usage
1. **Select DAT File**: Choose the data file
2. **Import Tags**: Load associated DBF tag file
3. **Configure Trimming**: Set start/end times
4. **Process & Export**: Generate output files

---

## ⚙️ Configuration Management

### Save/Load Settings
- **Configuration Save**: Save all current settings
- **Configuration Load**: Restore previous settings
- **Configuration Management**: Delete and organize saved configs
- **File Location**: Access configuration files directly

### Signal List Management
- **Save Signal Lists**: Save selected signals for reuse
- **Load Signal Lists**: Restore previous signal selections
- **Apply Saved Lists**: Quickly apply saved signal configurations

---

## 🎨 User Interface Features

### Responsive Design
- **Splitter Panels**: Adjustable panel sizes
- **Scrollable Content**: Handle large datasets efficiently
- **Modern UI**: CustomTkinter-based modern interface
- **Keyboard Shortcuts**: Efficient navigation and operation

### Progress Tracking
- **Real-time Updates**: Live progress indicators
- **Status Messages**: Clear operation feedback
- **Cancellation Support**: Stop operations at any time
- **Error Handling**: Comprehensive error reporting

---

## 🚀 Performance Features

### Optimization
- **Background Processing**: Non-blocking operations
- **Memory Management**: Efficient data handling
- **Batch Operations**: Process multiple files efficiently
- **Progress Feedback**: Real-time operation status

### File Handling
- **Large File Support**: Handle files of any size
- **Multiple Formats**: Support for 15+ file formats
- **Compression**: Built-in compression for output files
- **Error Recovery**: Robust error handling and recovery

---

## 📚 Tips & Best Practices

### Data Processing
1. **Start Small**: Test with small datasets before processing large files
2. **Use Preview Mode**: Always preview folder operations before execution
3. **Save Configurations**: Save frequently used settings
4. **Backup Data**: Create backups before major operations
5. **Monitor Progress**: Watch progress indicators for large operations

### File Organization
1. **Use Descriptive Names**: Name configurations and plots clearly
2. **Organize by Type**: Use folder tool's type organization
3. **Regular Cleanup**: Use deduplication features regularly
4. **Backup Important Data**: Always backup before major changes

### Performance
1. **Batch Processing**: Use batch modes for multiple files
2. **Filter Early**: Apply filters early in the process
3. **Monitor Memory**: Watch memory usage with large datasets
4. **Use Appropriate Formats**: Choose efficient formats for your data

---

## 🔧 Troubleshooting

### Common Issues
- **File Not Found**: Check file paths and permissions
- **Memory Errors**: Reduce batch size or use smaller datasets
- **Format Errors**: Verify file format compatibility
- **Permission Errors**: Check file and folder permissions

### Getting Help
- **Error Messages**: Read error messages carefully for clues
- **Log Files**: Check application logs for detailed information
- **Preview Mode**: Use preview features to test operations
- **Small Tests**: Test with small datasets first

---

## 📞 Support Information

This integrated application combines multiple powerful tools into a single, comprehensive data processing solution. All features are designed to work together seamlessly while maintaining the full functionality of the original standalone applications.

For technical support or feature requests, please refer to the application documentation or contact the development team.
"""

        # Create text widget for help content
        help_text = ctk.CTkTextbox(help_frame, wrap="word", font=ctk.CTkFont(size=12))
        help_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Insert help content
        help_text.insert("1.0", help_content)

        # Make text read-only
        help_text.configure(state="disabled")


class ColumnSelectionDialog(ctk.CTkToplevel):
    """Simple dialog for column selection."""

    def __init__(self, parent, columns):
        super().__init__(parent)
        self.title("Select Columns")
        self.geometry("400x500")
        self.resizable(True, True)

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        self.columns = columns
        self.result = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title = ctk.CTkLabel(
            main_frame,
            text="Select Columns to Include",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        title.pack(pady=(10, 20))

        # Buttons frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(button_frame, text="Select All", command=self.select_all).pack(
            side="left", padx=5
        )
        ctk.CTkButton(button_frame, text="Select None", command=self.select_none).pack(
            side="left", padx=5
        )

        # Scrollable frame for checkboxes
        scroll_frame = ctk.CTkScrollableFrame(main_frame, height=300)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create checkboxes for each column
        self.column_vars = {}
        for column in self.columns:
            var = ctk.BooleanVar(value=True)  # Default to selected
            self.column_vars[column] = var

            checkbox = ctk.CTkCheckBox(scroll_frame, text=column, variable=var)
            checkbox.pack(anchor="w", padx=5, pady=2)

        # Bottom buttons
        bottom_frame = ctk.CTkFrame(main_frame)
        bottom_frame.pack(fill="x", padx=10, pady=(10, 0))

        ctk.CTkButton(bottom_frame, text="OK", command=self.ok_clicked).pack(
            side="right", padx=5
        )
        ctk.CTkButton(bottom_frame, text="Cancel", command=self.cancel_clicked).pack(
            side="right", padx=5
        )

    def select_all(self):
        """Select all columns."""
        for var in self.column_vars.values():
            var.set(True)

    def select_none(self):
        """Select no columns."""
        for var in self.column_vars.values():
            var.set(False)

    def ok_clicked(self):
        """Handle OK button click."""
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one column.")
            return

        self.result = selected_columns
        self.destroy()

    def cancel_clicked(self):
        """Handle Cancel button click."""
        self.result = None
        self.destroy()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    app = IntegratedCSVProcessorApp()
    app.mainloop()

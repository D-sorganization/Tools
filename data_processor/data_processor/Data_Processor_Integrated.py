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

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, colorchooser
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
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from simpledbf import Dbf5
import re
from PIL import Image
import io
import threading
import queue
import mimetypes
import zipfile
import tempfile
import math
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
import logging
from datetime import datetime

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
    import pyarrow.parquet as pq
    import pyarrow.feather as feather
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Import the original data processor
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Data_Processor_r0 import CSVProcessorApp as OriginalCSVProcessorApp

# Import folder tool functionality
try:
    from Claude_Folders_Uno import FolderProcessorApp as OriginalFolderProcessorApp
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
        if ext in ['.csv']:
            return 'csv'
        elif ext in ['.tsv', '.txt']:
            return 'tsv'
        elif ext in ['.parquet', '.pq']:
            return 'parquet'
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
        elif ext in ['.json']:
            return 'json'
        elif ext in ['.h5', '.hdf5']:
            return 'hdf5'
        elif ext in ['.pkl', '.pickle']:
            return 'pickle'
        elif ext in ['.npy']:
            return 'numpy'
        elif ext in ['.mat']:
            return 'matlab'
        elif ext in ['.feather']:
            return 'feather'
        elif ext in ['.arrow']:
            return 'arrow'
        elif ext in ['.db', '.sqlite']:
            return 'sqlite'
        
        # Content-based detection for ambiguous extensions
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                
            # Check for CSV/TSV
            if b',' in header and b'\n' in header:
                return 'csv'
            elif b'\t' in header and b'\n' in header:
                return 'tsv'
            elif header.startswith(b'{') or header.startswith(b'['):
                return 'json'
            elif header.startswith(b'PK'):
                return 'excel'  # ZIP-based format
                
        except Exception:
            pass
            
        return None

class DataReader:
    """Handles reading data from various file formats."""
    
    @staticmethod
    def read_file(file_path: str, format_type: str, **kwargs) -> pd.DataFrame:
        """Read file based on format type."""
        try:
            if format_type == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif format_type == 'tsv':
                return pd.read_csv(file_path, sep='\t', **kwargs)
            elif format_type == 'parquet':
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for parquet files")
                return pd.read_parquet(file_path, **kwargs)
            elif format_type == 'excel':
                return pd.read_excel(file_path, **kwargs)
            elif format_type == 'json':
                return pd.read_json(file_path, **kwargs)
            elif format_type == 'hdf5':
                return pd.read_hdf(file_path, **kwargs)
            elif format_type == 'pickle':
                return pd.read_pickle(file_path)
            elif format_type == 'numpy':
                data = np.load(file_path)
                if isinstance(data, np.ndarray):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame(data.item())
            elif format_type == 'matlab':
                if not SCIPY_AVAILABLE:
                    raise ImportError("SciPy is required for MATLAB files")
                data = scipy.io.loadmat(file_path)
                # Convert MATLAB struct to DataFrame
                if len(data) == 1:
                    return pd.DataFrame(data[list(data.keys())[0]])
                else:
                    return pd.DataFrame(data)
            elif format_type == 'feather':
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for feather files")
                return pd.read_feather(file_path, **kwargs)
            elif format_type == 'arrow':
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for arrow files")
                table = pa.ipc.open_file(file_path).read_all()
                return table.to_pandas()
            elif format_type == 'sqlite':
                return pd.read_sql_query("SELECT * FROM data", f"sqlite:///{file_path}")
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise Exception(f"Error reading {file_path}: {str(e)}")

class DataWriter:
    """Handles writing data to various file formats."""
    
    @staticmethod
    def write_file(df: pd.DataFrame, file_path: str, format_type: str, **kwargs) -> None:
        """Write DataFrame to file based on format type."""
        try:
            if format_type == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif format_type == 'tsv':
                df.to_csv(file_path, sep='\t', index=False, **kwargs)
            elif format_type == 'parquet':
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for parquet files")
                df.to_parquet(file_path, **kwargs)
            elif format_type == 'excel':
                df.to_excel(file_path, index=False, **kwargs)
            elif format_type == 'json':
                df.to_json(file_path, orient='records', **kwargs)
            elif format_type == 'hdf5':
                df.to_hdf(file_path, key='data', **kwargs)
            elif format_type == 'pickle':
                df.to_pickle(file_path)
            elif format_type == 'numpy':
                np.save(file_path, df.values)
            elif format_type == 'matlab':
                if not SCIPY_AVAILABLE:
                    raise ImportError("SciPy is required for MATLAB files")
                scipy.io.savemat(file_path, {'data': df.values, 'columns': df.columns.tolist()})
            elif format_type == 'feather':
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for feather files")
                df.to_feather(file_path, **kwargs)
            elif format_type == 'arrow':
                if not PYARROW_AVAILABLE:
                    raise ImportError("PyArrow is required for arrow files")
                table = pa.Table.from_pandas(df)
                with pa.ipc.open_file(file_path, 'w') as writer:
                    writer.write(table)
            elif format_type == 'sqlite':
                import sqlite3
                conn = sqlite3.connect(file_path)
                df.to_sql('data', conn, if_exists='replace', index=False)
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
        title = ctk.CTkLabel(main_frame, text="Parquet File Metadata Analyzer", 
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=(10, 20))
        
        # Select file button
        self.select_btn = ctk.CTkButton(main_frame, text="Select Parquet File", 
                                       command=self.select_file, height=40)
        self.select_btn.pack(pady=(0, 20))
        
        # Results display
        self.results_text = ctk.CTkTextbox(main_frame, height=300)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Close button
        close_btn = ctk.CTkButton(main_frame, text="Close", command=self.destroy, height=35)
        close_btn.pack(pady=(0, 10))
        
    def select_file(self):
        """Open file dialog to select a parquet file."""
        file_path = filedialog.askopenfilename(
            title="Select Parquet File",
            filetypes=[("Parquet Files", "*.parquet *.pq"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.analyze_parquet_file(file_path)
            
    def analyze_parquet_file(self, file_path):
        """Analyze the selected parquet file."""
        try:
            if not PYARROW_AVAILABLE:
                self.results_text.insert("end", "Error: PyArrow is required for parquet analysis.\n")
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
                results += f"  Size: {self.format_file_size(row_group.total_byte_size)}\n"
                results += f"  Columns: {row_group.num_columns}\n"
                
                # Column details
                for j, col in enumerate(row_group.column_metadata):
                    results += f"    Column {j}: {col.path_in_schema[0]}\n"
                    results += f"      Values: {col.num_values:,}\n"
                    results += f"      Size: {self.format_file_size(col.total_uncompressed_size)}\n"
                    results += f"      Compressed: {self.format_file_size(col.total_compressed_size)}\n"
                    if col.statistics:
                        stats = col.statistics
                        if hasattr(stats, 'min') and hasattr(stats, 'max'):
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
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
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
        files = filedialog.askopenfilenames(
            title="Select Input Files",
            filetypes=[
                ("All Supported", "*.csv *.tsv *.txt *.parquet *.pq *.xlsx *.xls *.json *.h5 *.hdf5 *.pkl *.pickle *.npy *.mat *.feather *.arrow *.db *.sqlite"),
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
                ("All Files", "*.*")
            ]
        )
        
        if files:
            self.converter_input_files = list(files)
            self.converter_update_file_list()
            self.converter_input_label.configure(text=f"{len(files)} files selected")
            self.converter_update_convert_button()

    def converter_browse_folder(self):
        """Browse for input folder."""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            # Scan for supported files
            supported_extensions = {
                '.csv', '.tsv', '.txt', '.parquet', '.pq', '.xlsx', '.xls', 
                '.json', '.h5', '.hdf5', '.pkl', '.pickle', '.npy', '.mat', 
                '.feather', '.arrow', '.db', '.sqlite'
            }
            
            files = []
            for root, dirs, filenames in os.walk(folder):
                for filename in filenames:
                    if Path(filename).suffix.lower() in supported_extensions:
                        files.append(os.path.join(root, filename))
            
            if files:
                self.converter_input_files = files
                self.converter_update_file_list()
                self.converter_input_label.configure(text=f"{len(files)} files found in folder")
                self.converter_update_convert_button()
            else:
                messagebox.showwarning("No Files Found", "No supported files found in the selected folder.")

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
            ctk.CTkLabel(self.converter_file_list_frame, text="No files selected").pack(padx=5, pady=5)
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
            ctk.CTkButton(file_frame, text="X", width=30, 
                         command=lambda fp=file_path: self.converter_remove_file(fp)).pack(side="right", padx=5, pady=2)

    def converter_remove_file(self, file_path):
        """Remove a specific file from the list."""
        if file_path in self.converter_input_files:
            self.converter_input_files.remove(file_path)
            self.converter_update_file_list()
            self.converter_input_label.configure(text=f"{len(self.converter_input_files)} files selected")
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
            messagebox.showwarning("No Files", "Please select input files first to determine columns.")
            return
        
        try:
            first_file = self.converter_input_files[0]
            format_type = FileFormatDetector.detect_format(first_file)
            if not format_type:
                messagebox.showerror("Error", "Could not detect format for the first file.")
                return
            
            df = DataReader.read_file(first_file, format_type)
            columns = df.columns.tolist()
            
            dialog = ColumnSelectionDialog(self, columns)
            if dialog.result:
                self.converter_selected_columns = set(dialog.result)
                self.converter_use_all_columns_var.set(False)
                self.converter_columns_label.configure(text=f"{len(dialog.result)} columns selected")
                self._log_conversion_message(f"Selected {len(dialog.result)} columns: {', '.join(dialog.result[:5])}{'...' if len(dialog.result) > 5 else ''}")
            
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
            args=(output_format, combine_files, use_all_columns, batch_processing, split_files)
        )
        conversion_thread.daemon = True
        conversion_thread.start()

    def _perform_conversion(self, output_format, combine_files, use_all_columns, batch_processing, split_files):
        """Perform the actual file conversion in a background thread."""
        try:
            self.converter_status_label.configure(text="Converting files...")
            self.converter_progress.set(0)
            self.converter_convert_button.configure(state="disabled")
            
            total_files = len(self.converter_input_files)
            processed_files = 0
            
            if combine_files:
                # Combine all files into one
                self._log_conversion_message(f"Starting conversion: combining {total_files} files into {output_format.upper()}")
                
                combined_data = []
                for i, file_path in enumerate(self.converter_input_files):
                    try:
                        format_type = FileFormatDetector.detect_format(file_path)
                        if not format_type:
                            self._log_conversion_message(f"Warning: Could not detect format for {os.path.basename(file_path)}")
                            continue
                        
                        df = DataReader.read_file(file_path, format_type)
                        
                        # Apply column selection
                        if not use_all_columns and self.converter_selected_columns:
                            available_columns = [col for col in self.converter_selected_columns if col in df.columns]
                            if available_columns:
                                df = df[available_columns]
                            else:
                                self._log_conversion_message(f"Warning: No selected columns found in {os.path.basename(file_path)}")
                                continue
                        
                        combined_data.append(df)
                        self._log_conversion_message(f"Loaded {os.path.basename(file_path)}: {len(df)} rows, {len(df.columns)} columns")
                        
                        processed_files += 1
                        self.converter_progress.set(processed_files / total_files)
                        
                    except Exception as e:
                        self._log_conversion_message(f"Error reading {os.path.basename(file_path)}: {str(e)}")
                
                if combined_data:
                    try:
                        combined_df = pd.concat(combined_data, ignore_index=True)
                        output_filename = self._generate_output_filename(output_format, "combined_data")
                        output_path = os.path.join(self.converter_output_path, output_filename)
                        
                        DataWriter.write_file(combined_df, output_path, output_format)
                        self._log_conversion_message(f"Successfully created: {output_filename}")
                        self._log_conversion_message(f"Combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns")
                        
                    except Exception as e:
                        self._log_conversion_message(f"Error writing combined file: {str(e)}")
                else:
                    self._log_conversion_message("No valid data to combine")
            
            else:
                # Process files individually
                self._log_conversion_message(f"Starting conversion: processing {total_files} files individually")
                
                for i, file_path in enumerate(self.converter_input_files):
                    try:
                        format_type = FileFormatDetector.detect_format(file_path)
                        if not format_type:
                            self._log_conversion_message(f"Warning: Could not detect format for {os.path.basename(file_path)}")
                            continue
                        
                        df = DataReader.read_file(file_path, format_type)
                        
                        # Apply column selection
                        if not use_all_columns and self.converter_selected_columns:
                            available_columns = [col for col in self.converter_selected_columns if col in df.columns]
                            if available_columns:
                                df = df[available_columns]
                            else:
                                self._log_conversion_message(f"Warning: No selected columns found in {os.path.basename(file_path)}")
                                continue
                        
                        # Generate output filename
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        output_filename = self._generate_output_filename(output_format, base_name)
                        output_path = os.path.join(self.converter_output_path, output_filename)
                        
                        DataWriter.write_file(df, output_path, output_format)
                        self._log_conversion_message(f"Converted {os.path.basename(file_path)} -> {output_filename}")
                        
                        processed_files += 1
                        self.converter_progress.set(processed_files / total_files)
                        
                    except Exception as e:
                        self._log_conversion_message(f"Error converting {os.path.basename(file_path)}: {str(e)}")
            
            self.converter_status_label.configure(text=f"Conversion complete. {processed_files} files processed.")
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
            "sqlite": ".db"
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
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
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
            converter_scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            converter_scrollable_frame.grid_columnconfigure(0, weight=1)
            
            # Input section
            input_frame = ctk.CTkFrame(converter_scrollable_frame)
            input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
            input_frame.grid_columnconfigure(1, weight=1)
            
            ctk.CTkLabel(input_frame, text="Input Files:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.converter_input_label = ctk.CTkLabel(input_frame, text="No files selected")
            self.converter_input_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
            
            input_buttons_frame = ctk.CTkFrame(input_frame)
            input_buttons_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
            
            ctk.CTkButton(input_buttons_frame, text="Browse Files", 
                         command=self.converter_browse_files).pack(side="left", padx=5)
            ctk.CTkButton(input_buttons_frame, text="Browse Folder", 
                         command=self.converter_browse_folder).pack(side="left", padx=5)
            ctk.CTkButton(input_buttons_frame, text="Clear Files", 
                         command=self.converter_clear_files).pack(side="left", padx=5)
            
            # Output section
            output_frame = ctk.CTkFrame(converter_scrollable_frame)
            output_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
            output_frame.grid_columnconfigure(1, weight=1)
            
            ctk.CTkLabel(output_frame, text="Output Format:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.converter_format_var = ctk.StringVar(value="parquet")
            format_combo = ctk.CTkComboBox(output_frame, values=[
                "parquet", "csv", "tsv", "excel", "json", "hdf5", "pickle", 
                "numpy", "matlab", "feather", "arrow", "sqlite"
            ], variable=self.converter_format_var)
            format_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
            
            ctk.CTkLabel(output_frame, text="Output Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.converter_output_label = ctk.CTkLabel(output_frame, text="No output path selected")
            self.converter_output_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            
            ctk.CTkButton(output_frame, text="Browse Output", 
                         command=self.converter_browse_output).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
            
            # Options section
            options_frame = ctk.CTkFrame(converter_scrollable_frame)
            options_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
            
            self.converter_combine_var = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(options_frame, text="Combine all files into one", 
                           variable=self.converter_combine_var).pack(anchor="w", padx=5, pady=2)
            
            self.converter_use_all_columns_var = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(options_frame, text="Use all columns", 
                           variable=self.converter_use_all_columns_var).pack(anchor="w", padx=5, pady=2)
            
            self.converter_batch_var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(options_frame, text="Batch processing", 
                           variable=self.converter_batch_var).pack(anchor="w", padx=5, pady=2)
            
            self.converter_split_var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(options_frame, text="Split large files", 
                           variable=self.converter_split_var).pack(anchor="w", padx=5, pady=2)
            
            # Column selection
            column_frame = ctk.CTkFrame(converter_scrollable_frame)
            column_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
            
            ctk.CTkLabel(column_frame, text="Column Selection:").pack(anchor="w", padx=5, pady=2)
            self.converter_columns_label = ctk.CTkLabel(column_frame, text="All columns selected")
            self.converter_columns_label.pack(anchor="w", padx=5, pady=2)
            
            ctk.CTkButton(column_frame, text="Select Columns", 
                         command=self.converter_select_columns).pack(anchor="w", padx=5, pady=5)
            
            # Convert button
            self.converter_convert_button = ctk.CTkButton(converter_scrollable_frame, 
                                                         text="Convert Files", 
                                                         command=lambda: self.converter_start_conversion(),
                                                         height=40)
            self.converter_convert_button.grid(row=4, column=0, sticky="ew", padx=5, pady=10)
            
            # Progress
            self.converter_progress = ctk.CTkProgressBar(converter_scrollable_frame)
            self.converter_progress.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
            self.converter_progress.set(0)
            
            # Status
            self.converter_status_label = ctk.CTkLabel(converter_scrollable_frame, text="Ready")
            self.converter_status_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)

        def create_converter_right_content(right_panel):
            """Create the right panel content for converter"""
            right_panel.grid_rowconfigure(1, weight=1)
            right_panel.grid_columnconfigure(0, weight=1)
            
            # File list
            self.converter_file_list_frame = ctk.CTkScrollableFrame(right_panel, label_text="Selected Files", height=200)
            self.converter_file_list_frame.grid(row=0, column=0, padx=10, pady=(0, 10), sticky="ew")
            
            # Log area
            log_frame = ctk.CTkFrame(right_panel)
            log_frame.grid(row=1, column=0, padx=10, pady=0, sticky="nsew")
            log_frame.grid_columnconfigure(0, weight=1)
            log_frame.grid_rowconfigure(0, weight=1)
            
            ctk.CTkLabel(log_frame, text="Conversion Log:").pack(anchor="w", padx=5, pady=2)
            self.converter_log_text = ctk.CTkTextbox(log_frame, height=300)
            self.converter_log_text.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Buttons
            button_frame = ctk.CTkFrame(right_panel)
            button_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
            
            ctk.CTkButton(button_frame, text="Analyze Parquet", 
                         command=self.show_parquet_analyzer).pack(side="left", padx=5)
            ctk.CTkButton(button_frame, text="Clear Log", 
                         command=self.converter_clear_log).pack(side="left", padx=5)
            ctk.CTkButton(button_frame, text="Save Log", 
                         command=self.converter_save_log).pack(side="left", padx=5)

        # Create the splitter
        splitter_frame = self._create_splitter(parent_tab, create_converter_left_content, create_converter_right_content, 
                                              'converter_left_width', 400)
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def create_folder_tool_tab(self, parent_tab):
        """Create the folder tool tab with embedded folder processor functionality."""
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)
        
        if not FOLDER_TOOL_AVAILABLE:
            # Show error message if folder tool is not available
            error_frame = ctk.CTkFrame(parent_tab)
            error_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
            
            ctk.CTkLabel(error_frame, text="Folder Tool Not Available", 
                        font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)
            ctk.CTkLabel(error_frame, text="The folder tool component could not be loaded.\n"
                        "Please ensure Claude_Folders_Uno.py is available in the same directory.").pack(pady=10)
            return
        
        # Create a frame to hold the folder tool
        folder_frame = ctk.CTkFrame(parent_tab)
        folder_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        folder_frame.grid_columnconfigure(0, weight=1)
        folder_frame.grid_rowconfigure(0, weight=1)
        
        # Create a title for the folder tool section
        title_label = ctk.CTkLabel(folder_frame, text="Enhanced Folder Processor", 
                                  font=ctk.CTkFont(size=18, weight="bold"))
        title_label.grid(row=0, column=0, pady=(10, 20))
        
        # Create a button to launch the folder tool in a separate window
        launch_button = ctk.CTkButton(folder_frame, text="Launch Folder Tool", 
                                     command=self.launch_folder_tool,
                                     height=50, font=ctk.CTkFont(size=14, weight="bold"))
        launch_button.grid(row=1, column=0, pady=20)
        
        # Add description
        description_text = """
The Folder Tool provides comprehensive folder processing capabilities:

• Combine & Copy: Merge multiple folders into one
• Flatten & Tidy: Remove nested folder structures
• Copy & Prune Empty: Remove empty folders after copying
• Deduplicate Files: Remove duplicate files in-place
• Analyze & Report: Generate detailed folder analysis

Click "Launch Folder Tool" to open the full folder processing interface.
        """
        
        description_label = ctk.CTkLabel(folder_frame, text=description_text, 
                                       justify="left", wraplength=600)
        description_label.grid(row=2, column=0, pady=20, padx=20)

    def launch_folder_tool(self):
        """Launch the folder tool in a separate window."""
        if not FOLDER_TOOL_AVAILABLE:
            messagebox.showerror("Error", "Folder tool is not available.")
            return
        
        try:
            # Create a new Tkinter root window for the folder tool
            folder_root = tk.Tk()
            
            # Initialize the folder processor app
            folder_app = OriginalFolderProcessorApp(folder_root)
            
            # Set up the window to be modal relative to the main application
            folder_root.transient(self)
            folder_root.grab_set()
            
            # Center the window on screen
            folder_root.update_idletasks()
            x = (folder_root.winfo_screenwidth() // 2) - (700 // 2)
            y = (folder_root.winfo_screenheight() // 2) - (900 // 2)
            folder_root.geometry(f"700x900+{x}+{y}")
            
            # Start the folder tool
            folder_root.mainloop()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch folder tool: {str(e)}")

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
        title = ctk.CTkLabel(main_frame, text="Select Columns to Include", 
                            font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=(10, 20))
        
        # Buttons frame
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkButton(button_frame, text="Select All", 
                     command=self.select_all).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Select None", 
                     command=self.select_none).pack(side="left", padx=5)
        
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
        
        ctk.CTkButton(bottom_frame, text="OK", 
                     command=self.ok_clicked).pack(side="right", padx=5)
        ctk.CTkButton(bottom_frame, text="Cancel", 
                     command=self.cancel_clicked).pack(side="right", padx=5)
    
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

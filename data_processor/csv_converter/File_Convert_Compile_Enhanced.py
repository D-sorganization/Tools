#!/usr/bin/env python3
"""
Universal File Format Converter - ENHANCED VERSION
A PyQt6-based GUI application for converting between multiple file formats commonly used in machine learning and data science.

Supported Formats:
Input/Output: CSV, TSV, TXT, Parquet, Excel (XLSX, XLS), JSON, HDF5, Pickle, NumPy, MATLAB, Feather, Arrow, SQLite
Output Only: TFRecord, ONNX, PMML, Joblib

Enhanced Features:
- Multi-format support with automatic format detection
- Column selection with search and filtering
- Save/load column selection lists
- Efficient handling of large datasets
- Bulk conversion of files
- Progress tracking and error handling
- Memory-efficient processing
- Format-specific options and optimizations

Author: AI Assistant
Date: 2025
"""

import sys
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
import json
import pickle
import numpy as np
import h5py
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
import logging
from datetime import datetime
import threading
import queue
import mimetypes
import zipfile
import tempfile

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

# File splitting and management imports
import math
import shutil
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QCheckBox, QSpinBox, QGroupBox,
    QGridLayout, QSplitter, QFrame, QComboBox, QTabWidget,
    QListWidget, QListWidgetItem, QScrollArea, QFrame, QButtonGroup,
    QInputDialog, QDialog, QFormLayout, QDoubleSpinBox, QRadioButton,
    QSlider, QTextBrowser
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QRegularExpression
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QRegularExpressionValidator

# File format definitions
SUPPORTED_FORMATS = {
    'csv': {
        'name': 'CSV (Comma Separated Values)',
        'extensions': ['.csv'],
        'mime_types': ['text/csv'],
        'readable': True,
        'writable': True,
        'description': 'Standard comma-separated values format'
    },
    'tsv': {
        'name': 'TSV (Tab Separated Values)',
        'extensions': ['.tsv', '.txt'],
        'mime_types': ['text/tab-separated-values'],
        'readable': True,
        'writable': True,
        'description': 'Tab-separated values format'
    },
    'parquet': {
        'name': 'Parquet',
        'extensions': ['.parquet', '.pq'],
        'mime_types': ['application/octet-stream'],
        'readable': True,
        'writable': True,
        'description': 'Columnar storage format optimized for analytics'
    },
    'excel': {
        'name': 'Excel (XLSX/XLS)',
        'extensions': ['.xlsx', '.xls'],
        'mime_types': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel'],
        'readable': True,
        'writable': True,
        'description': 'Microsoft Excel format'
    },
    'json': {
        'name': 'JSON',
        'extensions': ['.json'],
        'mime_types': ['application/json'],
        'readable': True,
        'writable': True,
        'description': 'JavaScript Object Notation format'
    },
    'hdf5': {
        'name': 'HDF5',
        'extensions': ['.h5', '.hdf5'],
        'mime_types': ['application/x-hdf5'],
        'readable': True,
        'writable': True,
        'description': 'Hierarchical Data Format 5'
    },
    'pickle': {
        'name': 'Pickle (PKL)',
        'extensions': ['.pkl', '.pickle'],
        'mime_types': ['application/octet-stream'],
        'readable': True,
        'writable': True,
        'description': 'Python pickle format'
    },
    'numpy': {
        'name': 'NumPy (NPY/NPZ)',
        'extensions': ['.npy', '.npz'],
        'mime_types': ['application/octet-stream'],
        'readable': True,
        'writable': True,
        'description': 'NumPy array format'
    },
    'matlab': {
        'name': 'MATLAB (.mat)',
        'extensions': ['.mat'],
        'mime_types': ['application/x-matlab-data'],
        'readable': SCIPY_AVAILABLE,
        'writable': SCIPY_AVAILABLE,
        'description': 'MATLAB data format'
    },
    'feather': {
        'name': 'Feather',
        'extensions': ['.feather'],
        'mime_types': ['application/octet-stream'],
        'readable': True,
        'writable': True,
        'description': 'Fast columnar data format'
    },
    'arrow': {
        'name': 'Arrow',
        'extensions': ['.arrow'],
        'mime_types': ['application/octet-stream'],
        'readable': True,
        'writable': True,
        'description': 'Apache Arrow format'
    },
    'sqlite': {
        'name': 'SQLite Database',
        'extensions': ['.db', '.sqlite', '.sqlite3'],
        'mime_types': ['application/x-sqlite3'],
        'readable': True,
        'writable': True,
        'description': 'SQLite database format'
    },
    'joblib': {
        'name': 'Joblib',
        'extensions': ['.joblib'],
        'mime_types': ['application/octet-stream'],
        'readable': JOBLIB_AVAILABLE,
        'writable': JOBLIB_AVAILABLE,
        'description': 'Joblib serialization format'
    }
}

# File splitting configuration
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
    """Detect file formats based on extension and content."""
    
    @staticmethod
    def detect_format(file_path: str) -> Optional[str]:
        """Detect the format of a file."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Check by extension first
        for format_key, format_info in SUPPORTED_FORMATS.items():
            if extension in format_info['extensions']:
                return format_key
        
        # Try to detect by content for common formats
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                
            # Check for JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                return 'json'
            except:
                pass
            
            # Check for CSV/TSV
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if ',' in first_line:
                        return 'csv'
                    elif '\t' in first_line:
                        return 'tsv'
            except:
                pass
                
        except Exception:
            pass
            
        return None

class DataReader:
    """Read data from various file formats."""
    
    @staticmethod
    def read_file(file_path: str, format_type: str, **kwargs) -> pd.DataFrame:
        """Read a file and return a pandas DataFrame."""
        try:
            if format_type == 'csv':
                return pd.read_csv(file_path, **kwargs)
            elif format_type == 'tsv':
                return pd.read_csv(file_path, sep='\t', **kwargs)
            elif format_type == 'parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif format_type == 'excel':
                return pd.read_excel(file_path, **kwargs)
            elif format_type == 'json':
                return pd.read_json(file_path, **kwargs)
            elif format_type == 'hdf5':
                return pd.read_hdf(file_path, **kwargs)
            elif format_type == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    return data
                else:
                    return pd.DataFrame(data)
            elif format_type == 'numpy':
                if file_path.endswith('.npz'):
                    data = np.load(file_path)
                    # Convert to DataFrame - this is simplified
                    return pd.DataFrame({k: v for k, v in data.items()})
                else:
                    data = np.load(file_path)
                    return pd.DataFrame(data)
            elif format_type == 'matlab' and SCIPY_AVAILABLE:
                data = scipy.io.loadmat(file_path)
                # Convert to DataFrame - this is simplified
                return pd.DataFrame({k: v.flatten() if v.ndim > 1 else v for k, v in data.items() if not k.startswith('__')})
            elif format_type == 'feather':
                return feather.read_feather(file_path)
            elif format_type == 'arrow':
                return pa.ipc.open_file(file_path).read_pandas()
            elif format_type == 'sqlite':
                # For SQLite, we need to specify a table or query
                table_name = kwargs.get('table_name', None)
                if table_name:
                    return pd.read_sql(f"SELECT * FROM {table_name}", f"sqlite:///{file_path}")
                else:
                    # List available tables
                    conn = sqlite3.connect(file_path)
                    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
                    conn.close()
                    if len(tables) > 0:
                        return pd.read_sql(f"SELECT * FROM {tables.iloc[0]['name']}", f"sqlite:///{file_path}")
                    else:
                        raise ValueError("No tables found in SQLite database")
            elif format_type == 'joblib' and JOBLIB_AVAILABLE:
                data = joblib.load(file_path)
                if isinstance(data, pd.DataFrame):
                    return data
                else:
                    return pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise Exception(f"Error reading {format_type} file: {str(e)}")

class DataWriter:
    """Write data to various file formats."""
    
    @staticmethod
    def write_file(df: pd.DataFrame, file_path: str, format_type: str, **kwargs) -> None:
        """Write a pandas DataFrame to a file."""
        try:
            if format_type == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif format_type == 'tsv':
                df.to_csv(file_path, sep='\t', index=False, **kwargs)
            elif format_type == 'parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            elif format_type == 'excel':
                df.to_excel(file_path, index=False, **kwargs)
            elif format_type == 'json':
                df.to_json(file_path, orient='records', **kwargs)
            elif format_type == 'hdf5':
                df.to_hdf(file_path, key='data', mode='w', **kwargs)
            elif format_type == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(df, f)
            elif format_type == 'numpy':
                if file_path.endswith('.npz'):
                    np.savez_compressed(file_path, **{col: df[col].values for col in df.columns})
                else:
                    np.save(file_path, df.values)
            elif format_type == 'matlab' and SCIPY_AVAILABLE:
                scipy.io.savemat(file_path, {col: df[col].values for col in df.columns})
            elif format_type == 'feather':
                feather.write_feather(df, file_path)
            elif format_type == 'arrow':
                table = pa.Table.from_pandas(df)
                with pa.ipc.open_file(file_path, 'w') as writer:
                    writer.write(table)
            elif format_type == 'sqlite':
                conn = sqlite3.connect(file_path)
                table_name = kwargs.get('table_name', 'data')
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                conn.close()
            elif format_type == 'joblib' and JOBLIB_AVAILABLE:
                joblib.dump(df, file_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            raise Exception(f"Error writing {format_type} file: {str(e)}")

# Continue with the rest of the application...

class ColumnSelectionDialog(QDialog):
    """Enhanced dialog for selecting columns from files."""
    
    def __init__(self, columns: List[str], parent=None):
        super().__init__(parent)
        self.columns = columns
        self.selected_columns = set()
        self.setup_ui()
        self.load_column_lists()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Column Selection")
        self.setModal(True)
        self.resize(900, 700)
        
        layout = QVBoxLayout()
        
        # Search section
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to filter columns...")
        self.search_edit.textChanged.connect(self._filter_columns)
        search_layout.addWidget(self.search_edit)
        
        self.case_sensitive_checkbox = QCheckBox("Case sensitive")
        self.case_sensitive_checkbox.toggled.connect(self._filter_columns)
        search_layout.addWidget(self.case_sensitive_checkbox)
        
        layout.addLayout(search_layout)
        
        # Column list
        self.column_list = QListWidget()
        self.column_list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.column_list)
        
        # Selection buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_none)
        button_layout.addWidget(select_none_btn)
        
        clear_selected_btn = QPushButton("Clear Selected")
        clear_selected_btn.clicked.connect(self._clear_selected)
        button_layout.addWidget(clear_selected_btn)
        
        invert_btn = QPushButton("Invert Selection")
        invert_btn.clicked.connect(self._invert_selection)
        button_layout.addWidget(invert_btn)
        
        layout.addLayout(button_layout)
        
        # Save/Load section
        save_load_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save Selection")
        save_btn.clicked.connect(self._save_column_list)
        save_load_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Selection")
        load_btn.clicked.connect(self._load_column_list)
        save_load_layout.addWidget(load_btn)
        
        save_load_layout.addWidget(QLabel("Quick Load:"))
        self.load_combo = QComboBox()
        self.load_combo.currentTextChanged.connect(self._load_selected_list)
        save_load_layout.addWidget(self.load_combo)
        
        layout.addLayout(save_load_layout)
        
        # Status
        self.status_label = QLabel("0 columns selected")
        layout.addWidget(self.status_label)
        
        # Dialog buttons
        button_box = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._accept)
        button_box.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._reject)
        button_box.addWidget(cancel_btn)
        
        layout.addLayout(button_box)
        
        self.setLayout(layout)
        self._populate_column_list()
        self._apply_styling()
        
    def _apply_styling(self):
        """Apply custom styling."""
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #ecf0f1;
            }
            QListWidget::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            QListWidget::indicator:unchecked {
                background-color: white;
                border: 2px solid #bdc3c7;
            }
            QListWidget::indicator:checked {
                background-color: #3498db;
                border: 2px solid #3498db;
            }
            QListWidget::indicator:checked:hover {
                background-color: #2980b9;
                border: 2px solid #2980b9;
            }
            QListWidget::indicator:unchecked:hover {
                border: 2px solid #3498db;
            }
        """)
        
    def _populate_column_list(self):
        """Populate the column list with all columns."""
        self.column_list.clear()
        for column in self.columns:
            item = QListWidgetItem(column)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.column_list.addItem(item)
            
    def _filter_columns(self):
        """Filter columns based on search text."""
        search_text = self.search_edit.text()
        case_sensitive = self.case_sensitive_checkbox.isChecked()
        
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            column_name = item.text()
            
            if not search_text:
                item.setHidden(False)
            else:
                if case_sensitive:
                    matches = search_text in column_name
                else:
                    matches = search_text.lower() in column_name.lower()
                item.setHidden(not matches)
                
    def _select_all(self):
        """Select all visible columns."""
        checked_count = 0
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)
                checked_count += 1
        print(f"DEBUG: Set {checked_count} items to checked state")
        self._update_status()
        
    def _select_none(self):
        """Deselect all columns."""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
        self._update_status()
        
    def _clear_selected(self):
        """Clear selection of visible columns."""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Unchecked)
        self._update_status()
        
    def _invert_selection(self):
        """Invert selection of visible columns."""
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if not item.isHidden():
                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
        self._update_status()
        
    def _on_item_changed(self, item):
        """Handle item check state change."""
        self._update_status()
        
    def _update_status(self):
        """Update status label with selection count."""
        selected_count = 0
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_count += 1
        print(f"DEBUG: Found {selected_count} checked items out of {self.column_list.count()} total")
        self.status_label.setText(f"{selected_count} columns selected")
        
    def _save_column_list(self):
        """Save current column selection as a named list."""
        selected_columns = []
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_columns.append(item.text())
                
        if not selected_columns:
            QMessageBox.warning(self, "No Selection", "Please select at least one column to save.")
            return
            
        name, ok = QInputDialog.getText(self, "Save Column List", "Enter a name for this column list:")
        if ok and name:
            self._save_list_to_file(name, selected_columns)
            self.load_column_lists()
            
    def _save_list_to_file(self, name: str, columns: List[str]):
        """Save column list to file."""
        try:
            lists_dir = Path("column_lists")
            lists_dir.mkdir(exist_ok=True)
            
            file_path = lists_dir / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'name': name,
                    'columns': columns,
                    'created': datetime.now().isoformat()
                }, f, indent=2)
                
            QMessageBox.information(self, "Saved", f"Column list '{name}' saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save column list: {str(e)}")
            
    def _load_column_list(self):
        """Load column list from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Column List", "column_lists", "JSON Files (*.json)"
        )
        
        if file_path:
            self._load_list_from_file(file_path)
            
    def _load_list_from_file(self, file_path: str):
        """Load column list from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            columns = data.get('columns', [])
            self._apply_column_selection(columns)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load column list: {str(e)}")
            
    def load_column_lists(self):
        """Load available column lists into combo box."""
        self.load_combo.clear()
        self.load_combo.addItem("Select saved list...")
        
        try:
            lists_dir = Path("column_lists")
            if lists_dir.exists():
                for json_file in lists_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            name = data.get('name', json_file.stem)
                            self.load_combo.addItem(name)
                    except:
                        continue
        except Exception:
            pass
            
    def _load_selected_list(self, list_name: str):
        """Load selected column list from combo box."""
        if list_name == "Select saved list...":
            return
            
        try:
            lists_dir = Path("column_lists")
            file_path = lists_dir / f"{list_name}.json"
            
            if file_path.exists():
                self._load_list_from_file(str(file_path))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load column list: {str(e)}")
            
    def _apply_column_selection(self, selected_columns: List[str]):
        """Apply column selection to the list."""
        # First, uncheck all
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
            
        # Then check selected ones
        selected_set = set(selected_columns)
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.text() in selected_set:
                item.setCheckState(Qt.CheckState.Checked)
                
        self._update_status()
        
    def _accept(self):
        """Accept the selection."""
        selected_columns = []
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_columns.append(item.text())
                
        if not selected_columns:
            QMessageBox.warning(self, "No Selection", "Please select at least one column.")
            return
            
        self.selected_columns = set(selected_columns)
        self.accept()
        
    def _reject(self):
        """Reject the selection."""
        self.selected_columns = set()
        self.reject()
        
    def get_selected_columns(self) -> Set[str]:
        """Get the selected columns."""
        return self.selected_columns

class ConversionWorker(QThread):
    """Worker thread for handling file format conversion."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    conversion_completed = pyqtSignal(bool, str)
    
    def __init__(self, input_files: List[str], output_path: str, 
                 output_format: str, combine_files: bool = True, 
                 chunk_size: int = 10000, selected_columns: Set[str] = None,
                 format_options: Dict[str, Any] = None, split_config: SplitConfig = None):
        super().__init__()
        self.input_files = input_files
        self.output_path = output_path
        self.output_format = output_format
        self.combine_files = combine_files
        self.chunk_size = chunk_size
        self.selected_columns = selected_columns or set()
        self.format_options = format_options or {}
        self.split_config = split_config or SplitConfig()
        self.is_cancelled = False
        
    def run(self):
        """Execute the conversion process."""
        try:
            if self.combine_files:
                self._convert_to_single_file()
            else:
                self._convert_to_multiple_files()
                
            if not self.is_cancelled:
                self.conversion_completed.emit(True, "Conversion completed successfully!")
                
        except Exception as e:
            self.error_occurred.emit(f"Conversion error: {str(e)}")
            self.conversion_completed.emit(False, str(e))
    
    def _convert_to_single_file(self):
        """Convert all files to a single output file."""
        self.status_updated.emit("Reading files and combining data...")
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files in chunks to manage memory
        total_files = len(self.input_files)
        processed_files = 0
        
        # Read and combine all files
        combined_data = []
        
        for i, input_file in enumerate(self.input_files):
            if self.is_cancelled:
                break
                
            try:
                self.status_updated.emit(f"Processing {Path(input_file).name} ({i+1}/{total_files})")
                
                # Detect input format
                input_format = FileFormatDetector.detect_format(input_file)
                if not input_format:
                    raise ValueError(f"Could not detect format for {input_file}")
                
                # Read the file
                df = DataReader.read_file(input_file, input_format)
                
                # Apply column selection if specified
                if self.selected_columns:
                    available_columns = set(df.columns)
                    selected_available = self.selected_columns.intersection(available_columns)
                    if selected_available:
                        df = df[list(selected_available)]
                    else:
                        self.status_updated.emit(f"Warning: No selected columns found in {input_file}")
                        continue
                
                combined_data.append(df)
                processed_files += 1
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.error_occurred.emit(f"Error processing {input_file}: {str(e)}")
                continue
        
        if not combined_data:
            raise ValueError("No data to convert")
        
        # Combine all data
        self.status_updated.emit("Combining data...")
        final_df = pd.concat(combined_data, ignore_index=True)
        
        # Write the combined data
        self.status_updated.emit(f"Writing to {self.output_format} format...")
        
        # Check if file splitting is enabled
        if self.split_config.enabled:
            self.status_updated.emit("Splitting large dataset into multiple files...")
            output_files = self._split_dataframe(final_df, Path(self.output_path))
            
            if output_files:
                self.status_updated.emit(f"Created {len(output_files)} split files")
                # Update the output path to the directory containing split files
                if self.split_config.output_directory:
                    self.output_path = self.split_config.output_directory
                else:
                    self.output_path = str(Path(output_files[0]).parent)
            else:
                # Fallback to single file if splitting failed
                DataWriter.write_file(final_df, self.output_path, self.output_format, **self.format_options)
        else:
            DataWriter.write_file(final_df, self.output_path, self.output_format, **self.format_options)
                
    def _convert_to_multiple_files(self):
        """Convert files to multiple output files."""
        total_files = len(self.input_files)
        
        for i, input_file in enumerate(self.input_files):
            if self.is_cancelled:
                break
                
            try:
                self.status_updated.emit(f"Converting {Path(input_file).name} ({i+1}/{total_files})")
                
                # Detect input format
                input_format = FileFormatDetector.detect_format(input_file)
                if not input_format:
                    raise ValueError(f"Could not detect format for {input_file}")
                
                # Read the file
                df = DataReader.read_file(input_file, input_format)
                
                # Apply column selection if specified
                if self.selected_columns:
                    available_columns = set(df.columns)
                    selected_available = self.selected_columns.intersection(available_columns)
                    if selected_available:
                        df = df[list(selected_available)]
                    else:
                        self.status_updated.emit(f"Warning: No selected columns found in {input_file}")
                        continue
                
                # Generate output filename
                input_path = Path(input_file)
                output_filename = input_path.stem + "." + self.output_format
                output_file = Path(self.output_path) / output_filename
                
                # Write the file
                DataWriter.write_file(df, str(output_file), self.output_format, **self.format_options)
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.error_occurred.emit(f"Error converting {input_file}: {str(e)}")
                continue
                
    def cancel(self):
        """Cancel the conversion process."""
        self.is_cancelled = True

    def _split_dataframe(self, df: pd.DataFrame, base_path: Path) -> List[str]:
        """Split a DataFrame into multiple files based on configuration."""
        if not self.split_config.enabled:
            return []
            
        output_files = []
        total_rows = len(df)
        
        if self.split_config.method == SplitMethod.ROWS:
            output_files = self._split_by_rows(df, base_path)
        elif self.split_config.method == SplitMethod.SIZE:
            output_files = self._split_by_size(df, base_path)
        elif self.split_config.method == SplitMethod.TIME:
            output_files = self._split_by_time(df, base_path)
        elif self.split_config.method == SplitMethod.CUSTOM:
            output_files = self._split_by_custom(df, base_path)
            
        return output_files
        
    def _split_by_rows(self, df: pd.DataFrame, base_path: Path) -> List[str]:
        """Split DataFrame by row count."""
        rows_per_file = self.split_config.rows_per_file
        total_rows = len(df)
        num_files = math.ceil(total_rows / rows_per_file)
        
        output_files = []
        for i in range(num_files):
            if self.is_cancelled:
                break
                
            start_idx = i * rows_per_file
            end_idx = min((i + 1) * rows_per_file, total_rows)
            
            chunk_df = df.iloc[start_idx:end_idx]
            
            # Generate filename
            filename = self._generate_filename(base_path, i + 1, num_files)
            output_file = str(filename)
            
            # Write chunk
            self.status_updated.emit(f"Writing split file {i + 1}/{num_files}: {filename.name}")
            self._write_chunk(chunk_df, output_file, i == 0)  # Include header only for first file
            
            output_files.append(output_file)
            
            # Update progress
            progress = int((i + 1) / num_files * 100)
            self.progress_updated.emit(progress)
            
        return output_files
        
    def _split_by_size(self, df: pd.DataFrame, base_path: Path) -> List[str]:
        """Split DataFrame by target file size."""
        max_size_bytes = self.split_config.get_file_size_bytes()
        total_rows = len(df)
        
        # Estimate size per row (rough approximation)
        sample_size = min(1000, total_rows)
        sample_df = df.head(sample_size)
        
        # Create temporary file to measure size
        with tempfile.NamedTemporaryFile(suffix=f'.{self.output_format}', delete=True) as temp_file:
            self._write_chunk(sample_df, temp_file.name, True)
            temp_size = temp_file.tell()
            
        if temp_size == 0:
            # Fallback to row-based splitting
            return self._split_by_rows(df, base_path)
            
        bytes_per_row = temp_size / sample_size
        rows_per_file = max(1, int(max_size_bytes / bytes_per_row))
        
        # Use row-based splitting with calculated rows per file
        self.split_config.rows_per_file = rows_per_file
        return self._split_by_rows(df, base_path)
        
    def _split_by_time(self, df: pd.DataFrame, base_path: Path) -> List[str]:
        """Split DataFrame by time intervals."""
        time_column = self.split_config.time_column
        interval_hours = self.split_config.time_interval_hours
        
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
            
        # Convert time column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            try:
                df[time_column] = pd.to_datetime(df[time_column])
            except Exception as e:
                raise ValueError(f"Could not convert '{time_column}' to datetime: {str(e)}")
                
        # Sort by time column
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
        
        # Calculate time boundaries
        min_time = df_sorted[time_column].min()
        max_time = df_sorted[time_column].max()
        
        # Create time intervals
        intervals = []
        current_time = min_time
        while current_time <= max_time:
            next_time = current_time + pd.Timedelta(hours=interval_hours)
            intervals.append((current_time, next_time))
            current_time = next_time
            
        output_files = []
        for i, (start_time, end_time) in enumerate(intervals):
            if self.is_cancelled:
                break
                
            # Filter data for this time interval
            mask = (df_sorted[time_column] >= start_time) & (df_sorted[time_column] < end_time)
            chunk_df = df_sorted[mask]
            
            if len(chunk_df) == 0:
                continue
                
            # Generate filename with time information
            time_str = start_time.strftime("%Y%m%d_%H%M")
            filename = self._generate_filename(base_path, i + 1, len(intervals), time_str)
            output_file = str(filename)
            
            # Write chunk
            self.status_updated.emit(f"Writing time split {i + 1}/{len(intervals)}: {filename.name}")
            self._write_chunk(chunk_df, output_file, True)  # Each time split gets its own header
            
            output_files.append(output_file)
            
            # Update progress
            progress = int((i + 1) / len(intervals) * 100)
            self.progress_updated.emit(progress)
            
        return output_files
        
    def _split_by_custom(self, df: pd.DataFrame, base_path: Path) -> List[str]:
        """Split DataFrame by custom condition."""
        custom_condition = self.split_config.custom_condition
        
        if not custom_condition:
            return []
            
        # Create a safe environment for evaluating the condition
        safe_dict = {
            'row': None,
            'previous_row': None,
            'row_index': 0,
            'pd': pd,
            'np': np,
            'math': math
        }
        
        output_files = []
        current_chunk = []
        file_index = 1
        
        for index, row in df.iterrows():
            if self.is_cancelled:
                break
                
            # Update safe environment
            safe_dict['row'] = row
            safe_dict['row_index'] = index
            
            try:
                # Evaluate the custom condition
                should_split = eval(custom_condition, {"__builtins__": {}}, safe_dict)
                
                if should_split and current_chunk:
                    # Write current chunk and start new one
                    chunk_df = pd.DataFrame(current_chunk)
                    filename = self._generate_filename(base_path, file_index, 0)
                    output_file = str(filename)
                    
                    self.status_updated.emit(f"Writing custom split {file_index}: {filename.name}")
                    self._write_chunk(chunk_df, output_file, True)
                    
                    output_files.append(output_file)
                    current_chunk = []
                    file_index += 1
                    
            except Exception as e:
                self.error_occurred.emit(f"Error evaluating custom condition: {str(e)}")
                break
                
            # Add current row to chunk
            current_chunk.append(row.to_dict())
            safe_dict['previous_row'] = row
            
        # Write remaining chunk
        if current_chunk and not self.is_cancelled:
            chunk_df = pd.DataFrame(current_chunk)
            filename = self._generate_filename(base_path, file_index, 0)
            output_file = str(filename)
            
            self.status_updated.emit(f"Writing final custom split: {filename.name}")
            self._write_chunk(chunk_df, output_file, True)
            output_files.append(output_file)
            
        return output_files
        
    def _generate_filename(self, base_path: Path, part_number: int, total_parts: int = 0, 
                          time_suffix: str = "") -> Path:
        """Generate filename for split files."""
        # Get base name and extension
        base_name = base_path.stem
        extension = base_path.suffix
        
        # Create output directory if specified
        if self.split_config.output_directory:
            output_dir = Path(self.split_config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_path.parent
            
        # Generate filename using pattern
        filename = self.split_config.filename_pattern.format(
            base_name=base_name,
            part_number=part_number,
            total_parts=total_parts,
            extension=extension,
            time_suffix=time_suffix
        )
        
        return output_dir / filename
        
    def _write_chunk(self, df: pd.DataFrame, output_file: str, include_header: bool = True):
        """Write a data chunk to file with appropriate options."""
        # Prepare format options
        write_options = self.format_options.copy()
        
        # Add splitting-specific options
        if self.output_format == 'parquet':
            write_options['compression'] = self.split_config.compression
        elif self.output_format in ['csv', 'tsv']:
            write_options['index'] = False
            if not include_header and not self.split_config.include_header:
                write_options['header'] = False
                
        # Write the file
        DataWriter.write_file(df, output_file, self.output_format, **write_options)

# Continue with the main converter interface...

class UniversalFileConverter(QWidget):
    """Main widget for the universal file format converter."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.input_files = []
        self.selected_columns = set()
        self.output_format = 'parquet'
        self.combine_files = True
        self.chunk_size = 10000
        self.format_options = {}
        self.split_config = SplitConfig()
        
        self.setup_logging()
        self.setup_ui()
        self._apply_styling()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('file_converter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Universal File Format Converter")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create splitter for main sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section - Input and configuration
        top_widget = self._create_top_section()
        splitter.addWidget(top_widget)
        
        # Bottom section - Output and conversion
        bottom_widget = self._create_bottom_section()
        splitter.addWidget(bottom_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 300])
        layout.addWidget(splitter)
        
        self.setLayout(layout)
        
    def _create_top_section(self):
        """Create the top section with input configuration."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Input file selection
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout()
        
        # File selection buttons
        file_buttons_layout = QHBoxLayout()
        
        browse_folder_btn = QPushButton("Browse Folder")
        browse_folder_btn.clicked.connect(self._browse_folder)
        file_buttons_layout.addWidget(browse_folder_btn)
        
        browse_files_btn = QPushButton("Browse Files")
        browse_files_btn.clicked.connect(self._browse_files)
        file_buttons_layout.addWidget(browse_files_btn)
        
        clear_files_btn = QPushButton("Clear Files")
        clear_files_btn.clicked.connect(self._clear_files)
        file_buttons_layout.addWidget(clear_files_btn)
        
        input_layout.addLayout(file_buttons_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        input_layout.addWidget(self.file_list)
        
        # File information buttons
        file_info_layout = QHBoxLayout()
        
        self.file_info_btn = QPushButton("File Info")
        self.file_info_btn.clicked.connect(self._show_file_info)
        self.file_info_btn.setEnabled(False)
        file_info_layout.addWidget(self.file_info_btn)
        
        self.analyze_all_btn = QPushButton("Analyze All Files")
        self.analyze_all_btn.clicked.connect(self._analyze_all_files)
        self.analyze_all_btn.setEnabled(False)
        file_info_layout.addWidget(self.analyze_all_btn)
        
        input_layout.addLayout(file_info_layout)
        
        # File count
        self.file_count_label = QLabel("0 files selected")
        input_layout.addWidget(self.file_count_label)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Column selection
        column_group = QGroupBox("Column Selection")
        column_layout = QVBoxLayout()
        
        # Use all columns checkbox
        self.use_all_columns_checkbox = QCheckBox("Use all columns")
        self.use_all_columns_checkbox.setChecked(True)
        self.use_all_columns_checkbox.toggled.connect(self._on_use_all_columns_toggled)
        column_layout.addWidget(self.use_all_columns_checkbox)
        
        # Select columns button
        self.select_columns_btn = QPushButton("Select Specific Columns")
        self.select_columns_btn.clicked.connect(self._select_columns)
        self.select_columns_btn.setEnabled(False)
        column_layout.addWidget(self.select_columns_btn)
        
        # Selected columns info
        self.selected_columns_label = QLabel("All columns will be used")
        column_layout.addWidget(self.selected_columns_label)
        
        column_group.setLayout(column_layout)
        layout.addWidget(column_group)
        
        widget.setLayout(layout)
        return widget
        
    def _create_bottom_section(self):
        """Create the bottom section with output configuration."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Output configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout()
        
        # Output format selection
        self.output_format_combo = QComboBox()
        for format_key, format_info in SUPPORTED_FORMATS.items():
            if format_info['writable']:
                self.output_format_combo.addItem(format_info['name'], format_key)
        self.output_format_combo.currentTextChanged.connect(self._on_output_format_changed)
        output_layout.addRow("Output Format:", self.output_format_combo)
        
        # Combine files option
        self.combine_files_checkbox = QCheckBox("Combine all files into single output")
        self.combine_files_checkbox.setChecked(True)
        self.combine_files_checkbox.toggled.connect(self._on_combine_toggled)
        output_layout.addRow("", self.combine_files_checkbox)
        
        # Output path
        output_path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output path...")
        output_path_layout.addWidget(self.output_path_edit)
        
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self._browse_output)
        output_path_layout.addWidget(browse_output_btn)
        
        output_layout.addRow("Output Path:", output_path_layout)
        
        # Chunk size
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(1000, 1000000)
        self.chunk_size_spin.setValue(10000)
        self.chunk_size_spin.setSuffix(" rows")
        output_layout.addRow("Chunk Size:", self.chunk_size_spin)
        
        # File splitting configuration
        split_layout = QHBoxLayout()
        
        self.split_files_checkbox = QCheckBox("Split large files")
        self.split_files_checkbox.toggled.connect(self._on_split_files_toggled)
        split_layout.addWidget(self.split_files_checkbox)
        
        self.configure_splitting_btn = QPushButton("Configure Splitting")
        self.configure_splitting_btn.clicked.connect(self._configure_file_splitting)
        self.configure_splitting_btn.setEnabled(False)
        split_layout.addWidget(self.configure_splitting_btn)
        
        output_layout.addRow("File Splitting:", split_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Conversion controls
        conversion_group = QGroupBox("Conversion")
        conversion_layout = QVBoxLayout()
        
        # Convert button
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.clicked.connect(self._start_conversion)
        self.convert_btn.setEnabled(False)
        conversion_layout.addWidget(self.convert_btn)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_conversion)
        self.cancel_btn.setEnabled(False)
        conversion_layout.addWidget(self.cancel_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        conversion_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        conversion_layout.addWidget(self.status_label)
        
        conversion_group.setLayout(conversion_layout)
        layout.addWidget(conversion_group)
        
        # Log section
        log_group = QGroupBox("Conversion Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Save log button
        save_log_btn = QPushButton("Save Log")
        save_log_btn.clicked.connect(self._save_log)
        log_layout.addWidget(save_log_btn)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        widget.setLayout(layout)
        return widget
        
    def _apply_styling(self):
        """Apply custom styling."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            QComboBox {
                padding: 6px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        
    def _browse_folder(self):
        """Browse for a folder containing files to convert."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self._scan_files(folder_path)
            
    def _browse_files(self):
        """Browse for individual files to convert."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "",
            "All Supported Files (*.csv *.tsv *.txt *.parquet *.pq *.xlsx *.xls *.json *.h5 *.hdf5 *.pkl *.pickle *.npy *.npz *.mat *.feather *.arrow *.db *.sqlite *.sqlite3 *.joblib);;All Files (*)"
        )
        if file_paths:
            self.input_files.extend(file_paths)
            self._update_file_list()
            
    def _clear_files(self):
        """Clear the list of input files."""
        self.input_files.clear()
        self._update_file_list()
        
    def _scan_files(self, folder_path: str):
        """Scan a folder for supported files."""
        supported_extensions = []
        for format_info in SUPPORTED_FORMATS.values():
            supported_extensions.extend(format_info['extensions'])
            
        found_files = []
        for ext in supported_extensions:
            found_files.extend(Path(folder_path).rglob(f"*{ext}"))
            
        if found_files:
            self.input_files.extend([str(f) for f in found_files])
            self._update_file_list()
        else:
            QMessageBox.information(self, "No Files Found", 
                                  f"No supported files found in {folder_path}")
            
    def _update_file_list(self):
        """Update the file list display."""
        self.file_list.clear()
        for file_path in self.input_files:
            self.file_list.addItem(Path(file_path).name)
            
        self.file_count_label.setText(f"{len(self.input_files)} files selected")
        
        # Enable/disable file info buttons
        has_files = len(self.input_files) > 0
        self.file_info_btn.setEnabled(has_files)
        self.analyze_all_btn.setEnabled(has_files)
        
        self._update_convert_button()
        
    def _show_file_info(self):
        """Show information for the selected file."""
        current_item = self.file_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "No Selection", "Please select a file from the list.")
            return
            
        # Find the selected file path
        selected_name = current_item.text()
        selected_file = None
        for file_path in self.input_files:
            if Path(file_path).name == selected_name:
                selected_file = file_path
                break
                
        if not selected_file:
            QMessageBox.warning(self, "Error", "Selected file not found.")
            return
            
        # Analyze file
        self.status_label.setText(f"Analyzing {selected_name}...")
        QApplication.processEvents()
        
        try:
            file_info = FileSizeAnalyzer.get_file_info(selected_file)
            dialog = FileInfoDialog(file_info, self)
            dialog.exec()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze file: {str(e)}")
        finally:
            self.status_label.setText("Ready")
            
    def _analyze_all_files(self):
        """Analyze all selected files and show summary."""
        if not self.input_files:
            return
            
        # Create summary dialog
        summary_dialog = QDialog(self)
        summary_dialog.setWindowTitle("File Analysis Summary")
        summary_dialog.setModal(True)
        summary_dialog.resize(700, 600)
        
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel(f"Analysis Summary - {len(self.input_files)} Files")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create scrollable content
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        total_rows = 0
        total_columns = 0
        total_size_bytes = 0
        file_details = []
        
        # Analyze each file
        for i, file_path in enumerate(self.input_files):
            self.status_label.setText(f"Analyzing file {i+1}/{len(self.input_files)}...")
            QApplication.processEvents()
            
            try:
                file_info = FileSizeAnalyzer.get_file_info(file_path)
                file_details.append(file_info)
                
                # Accumulate totals
                total_size_bytes += file_info.get("file_size_bytes", 0)
                if "rows" in file_info:
                    total_rows += file_info["rows"]
                if "columns" in file_info:
                    total_columns = max(total_columns, file_info["columns"])  # Max columns
                    
            except Exception as e:
                file_details.append({"file_name": Path(file_path).name, "error": str(e)})
        
        # Summary section
        summary_group = QGroupBox("Summary")
        summary_layout = QFormLayout()
        
        summary_layout.addRow("Total Files:", QLabel(str(len(self.input_files))))
        summary_layout.addRow("Total Size:", QLabel(FileSizeAnalyzer.format_file_size(total_size_bytes)))
        
        if total_rows > 0:
            summary_layout.addRow("Total Rows:", QLabel(f"{total_rows:,}"))
        if total_columns > 0:
            summary_layout.addRow("Max Columns:", QLabel(f"{total_columns:,}"))
            
        summary_group.setLayout(summary_layout)
        scroll_layout.addWidget(summary_group)
        
        # Individual file details
        for file_info in file_details:
            file_group = QGroupBox(file_info.get("file_name", "Unknown"))
            file_layout = QFormLayout()
            
            if "error" in file_info:
                file_layout.addRow("Error:", QLabel(file_info["error"]))
            else:
                file_layout.addRow("Size:", QLabel(FileSizeAnalyzer.format_file_size(file_info.get("file_size_bytes", 0))))
                
                if "rows" in file_info:
                    file_layout.addRow("Rows:", QLabel(f"{file_info['rows']:,}"))
                if "columns" in file_info:
                    file_layout.addRow("Columns:", QLabel(f"{file_info['columns']:,}"))
                if "detected_format" in file_info:
                    format_name = SUPPORTED_FORMATS.get(file_info["detected_format"], {}).get("name", file_info["detected_format"])
                    file_layout.addRow("Format:", QLabel(format_name))
                    
            file_group.setLayout(file_layout)
            scroll_layout.addWidget(file_group)
        
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(summary_dialog.accept)
        layout.addWidget(close_btn)
        
        summary_dialog.setLayout(layout)
        summary_dialog.exec()
        
        self.status_label.setText("Ready")
        
    def _select_columns(self):
        """Open the column selection dialog."""
        if not self.input_files:
            QMessageBox.warning(self, "No Files", "Please select input files first.")
            return
            
        # Read the first file to get column names
        try:
            first_file = self.input_files[0]
            input_format = FileFormatDetector.detect_format(first_file)
            if not input_format:
                QMessageBox.warning(self, "Format Error", f"Could not detect format for {first_file}")
                return
                
            df = DataReader.read_file(first_file, input_format)
            columns = list(df.columns)
            
            # Open column selection dialog
            dialog = ColumnSelectionDialog(columns, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.selected_columns = dialog.get_selected_columns()
                self._update_selected_columns_display()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading file: {str(e)}")
            
    def _on_use_all_columns_toggled(self, checked: bool):
        """Handle use all columns checkbox toggle."""
        self.select_columns_btn.setEnabled(not checked)
        if checked:
            self.selected_columns = set()
            self.selected_columns_label.setText("All columns will be used")
        else:
            self.selected_columns_label.setText("No columns selected")
            
    def _update_selected_columns_display(self):
        """Update the display of selected columns."""
        if self.selected_columns:
            count = len(self.selected_columns)
            self.selected_columns_label.setText(f"{count} columns selected")
        else:
            self.selected_columns_label.setText("No columns selected")
            
    def _on_output_format_changed(self):
        """Handle output format change."""
        self.output_format = self.output_format_combo.currentData()
        
    def _on_combine_toggled(self, checked: bool):
        """Handle combine files checkbox toggle."""
        self.combine_files = checked
        
    def _on_split_files_toggled(self, checked: bool):
        """Handle split files checkbox toggle."""
        self.split_config.enabled = checked
        self.configure_splitting_btn.setEnabled(checked)
        
    def _configure_file_splitting(self):
        """Open the file splitting configuration dialog."""
        dialog = FileSplittingDialog(self)
        
        # Pre-populate with current configuration
        if self.split_config.enabled:
            dialog.enable_splitting_checkbox.setChecked(True)
            
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.split_config = dialog.get_split_config()
            
            # Update UI to reflect configuration
            self.split_files_checkbox.setChecked(self.split_config.enabled)
            self.configure_splitting_btn.setEnabled(self.split_config.enabled)
            
            # Show summary of configuration
            if self.split_config.enabled:
                method_names = {
                    SplitMethod.ROWS: "Row Count",
                    SplitMethod.SIZE: "File Size", 
                    SplitMethod.TIME: "Time Intervals",
                    SplitMethod.CUSTOM: "Custom Condition"
                }
                
                method_name = method_names.get(self.split_config.method, "Unknown")
                QMessageBox.information(
                    self, "Splitting Configured",
                    f"File splitting enabled:\n"
                    f"Method: {method_name}\n"
                    f"Output: {self.split_config.output_directory or 'Auto-created directory'}"
                )
                
    def _browse_output(self):
        """Browse for output path."""
        if self.combine_files:
            # Single file output
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save As", "",
                f"{SUPPORTED_FORMATS[self.output_format]['name']} (*{SUPPORTED_FORMATS[self.output_format]['extensions'][0]})"
            )
            if file_path:
                self.output_path_edit.setText(file_path)
        else:
            # Directory output
            folder_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if folder_path:
                self.output_path_edit.setText(folder_path)
                
        self._update_convert_button()
        
    def _update_convert_button(self):
        """Update the convert button state."""
        can_convert = (len(self.input_files) > 0 and 
                      self.output_path_edit.text().strip() != "")
        self.convert_btn.setEnabled(can_convert)
        
    def _start_conversion(self):
        """Start the conversion process."""
        if not self.input_files:
            QMessageBox.warning(self, "No Files", "Please select input files.")
            return
            
        if not self.output_path_edit.text().strip():
            QMessageBox.warning(self, "No Output", "Please select output path.")
            return
            
        # Get conversion parameters
        output_path = self.output_path_edit.text().strip()
        output_format = self.output_format_combo.currentData()
        combine_files = self.combine_files_checkbox.isChecked()
        chunk_size = self.chunk_size_spin.value()
        
        # Update UI state
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Start conversion worker
        self.conversion_worker = ConversionWorker(
            self.input_files, output_path, output_format,
            combine_files, chunk_size, self.selected_columns,
            self.format_options, self.split_config
        )
        
        self.conversion_worker.progress_updated.connect(self.progress_bar.setValue)
        self.conversion_worker.status_updated.connect(self._update_status)
        self.conversion_worker.error_occurred.connect(self._log_error)
        self.conversion_worker.conversion_completed.connect(self._conversion_finished)
        
        self.conversion_worker.start()
        
    def _cancel_conversion(self):
        """Cancel the conversion process."""
        if hasattr(self, 'conversion_worker'):
            self.conversion_worker.cancel()
            
    def _update_status(self, status: str):
        """Update the status display."""
        self.status_label.setText(status)
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
        
    def _log_error(self, error: str):
        """Log an error message."""
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error}")
        
    def _conversion_finished(self, success: bool, message: str):
        """Handle conversion completion."""
        # Update UI state
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if success:
            self.progress_bar.setValue(100)
            self._update_status("Conversion completed successfully!")
            QMessageBox.information(self, "Success", message)
        else:
            self._log_error(f"Conversion failed: {message}")
            QMessageBox.critical(self, "Error", f"Conversion failed: {message}")
            
    def _save_log(self):
        """Save the conversion log to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "", "Text Files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "Saved", "Log saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log: {str(e)}")

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal File Format Converter")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create central widget
        self.central_widget = UniversalFileConverter()
        self.setCentralWidget(self.central_widget)
        
        # Setup menu
        self._setup_menu()
        
    def _setup_menu(self):
        """Setup the application menu."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about)
        
    def _show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self, "About Universal File Format Converter",
            """
            <h3>Universal File Format Converter</h3>
            <p>A comprehensive tool for converting between various file formats commonly used in machine learning and data science.</p>
            
            <h4>Supported Formats:</h4>
            <ul>
                <li>CSV, TSV, TXT</li>
                <li>Parquet, Feather, Arrow</li>
                <li>Excel (XLSX, XLS)</li>
                <li>JSON, HDF5, Pickle</li>
                <li>NumPy (NPY, NPZ)</li>
                <li>MATLAB (.mat)</li>
                <li>SQLite databases</li>
                <li>Joblib</li>
            </ul>
            
            <h4>Advanced Features:</h4>
            <ul>
                <li>Multi-format support with automatic detection</li>
                <li>Advanced column selection and filtering</li>
                <li>Large dataset file splitting</li>
                <li>Memory-efficient chunked processing</li>
                <li>Bulk file conversion</li>
                <li>Progress tracking and detailed logging</li>
            </ul>
            
            <h4>File Splitting Options:</h4>
            <ul>
                <li>Split by row count</li>
                <li>Split by target file size</li>
                <li>Split by time intervals</li>
                <li>Custom splitting conditions</li>
            </ul>
            
            <p><b>Version:</b> Enhanced v1.1 (with File Splitting)</p>
            <p><b>Author:</b> AI Assistant</p>
            """
        )

class FileSplittingDialog(QDialog):
    """Dialog for configuring file splitting options."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.split_config = SplitConfig()
        self.setup_ui()
        self._apply_styling()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("File Splitting Configuration")
        self.setModal(True)
        self.resize(700, 600)
        
        layout = QVBoxLayout()
        
        # Enable splitting checkbox
        self.enable_splitting_checkbox = QCheckBox("Enable File Splitting")
        self.enable_splitting_checkbox.toggled.connect(self._on_enable_splitting_toggled)
        layout.addWidget(self.enable_splitting_checkbox)
        
        # Create tab widget for different splitting methods
        self.tab_widget = QTabWidget()
        self.tab_widget.setEnabled(False)
        
        # Rows tab
        self.rows_tab = self._create_rows_tab()
        self.tab_widget.addTab(self.rows_tab, "By Row Count")
        
        # Size tab
        self.size_tab = self._create_size_tab()
        self.tab_widget.addTab(self.size_tab, "By File Size")
        
        # Time tab
        self.time_tab = self._create_time_tab()
        self.tab_widget.addTab(self.time_tab, "By Time")
        
        # Custom tab
        self.custom_tab = self._create_custom_tab()
        self.tab_widget.addTab(self.custom_tab, "Custom")
        
        layout.addWidget(self.tab_widget)
        
        # Output configuration
        output_group = QGroupBox("Output Configuration")
        output_layout = QFormLayout()
        
        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Auto-create in same directory")
        output_dir_layout.addWidget(self.output_dir_edit)
        
        browse_dir_btn = QPushButton("Browse")
        browse_dir_btn.clicked.connect(self._browse_output_directory)
        output_dir_layout.addWidget(browse_dir_btn)
        
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        # Filename pattern
        self.filename_pattern_edit = QLineEdit("{base_name}_part_{part_number:04d}{extension}")
        output_layout.addRow("Filename Pattern:", self.filename_pattern_edit)
        
        # Compression (for parquet)
        self.compression_combo = QComboBox()
        self.compression_combo.addItems(["snappy", "gzip", "brotli", "lz4", "zstd"])
        output_layout.addRow("Compression:", self.compression_combo)
        
        # Include header (for CSV)
        self.include_header_checkbox = QCheckBox("Include header in each file")
        self.include_header_checkbox.setChecked(True)
        output_layout.addRow("", self.include_header_checkbox)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Preview section
        preview_group = QGroupBox("Split Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextBrowser()
        self.preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_text)
        
        refresh_preview_btn = QPushButton("Refresh Preview")
        refresh_preview_btn.clicked.connect(self._update_preview)
        preview_layout.addWidget(refresh_preview_btn)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def _create_rows_tab(self):
        """Create the rows splitting tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Row count configuration
        form_layout = QFormLayout()
        
        self.rows_per_file_spin = QSpinBox()
        self.rows_per_file_spin.setRange(1000, 10000000)
        self.rows_per_file_spin.setValue(100000)
        self.rows_per_file_spin.setSuffix(" rows")
        form_layout.addRow("Rows per file:", self.rows_per_file_spin)
        
        # Estimated files info
        self.estimated_files_label = QLabel("Estimated files: --")
        form_layout.addRow("", self.estimated_files_label)
        
        layout.addLayout(form_layout)
        
        # Help text
        help_text = QTextBrowser()
        help_text.setMaximumHeight(100)
        help_text.setPlainText(
            "Split files based on the number of rows. This is useful for:\n"
            "• Managing memory usage during processing\n"
            "• Creating files of consistent size\n"
            "• Parallel processing of data chunks"
        )
        layout.addWidget(help_text)
        
        widget.setLayout(layout)
        return widget
        
    def _create_size_tab(self):
        """Create the file size splitting tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # File size configuration
        form_layout = QFormLayout()
        
        self.max_file_size_spin = QDoubleSpinBox()
        self.max_file_size_spin.setRange(1.0, 10000.0)
        self.max_file_size_spin.setValue(100.0)
        self.max_file_size_spin.setSuffix(" MB")
        self.max_file_size_spin.setDecimals(1)
        form_layout.addRow("Maximum file size:", self.max_file_size_spin)
        
        # Size slider
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 1000)
        self.size_slider.setValue(100)
        self.size_slider.valueChanged.connect(self._on_size_slider_changed)
        form_layout.addRow("", self.size_slider)
        
        # Estimated files info
        self.estimated_files_size_label = QLabel("Estimated files: --")
        form_layout.addRow("", self.estimated_files_size_label)
        
        layout.addLayout(form_layout)
        
        # Help text
        help_text = QTextBrowser()
        help_text.setMaximumHeight(100)
        help_text.setPlainText(
            "Split files based on target file size. This is useful for:\n"
            "• Managing storage constraints\n"
            "• Optimizing transfer speeds\n"
            "• Creating files suitable for specific systems"
        )
        layout.addWidget(help_text)
        
        widget.setLayout(layout)
        return widget
        
    def _create_time_tab(self):
        """Create the time-based splitting tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Time configuration
        form_layout = QFormLayout()
        
        self.time_column_edit = QLineEdit()
        self.time_column_edit.setPlaceholderText("e.g., timestamp, date, created_at")
        form_layout.addRow("Time column name:", self.time_column_edit)
        
        self.time_interval_spin = QDoubleSpinBox()
        self.time_interval_spin.setRange(0.1, 8760.0)  # 0.1 hours to 1 year
        self.time_interval_spin.setValue(24.0)
        self.time_interval_spin.setSuffix(" hours")
        self.time_interval_spin.setDecimals(1)
        form_layout.addRow("Time interval:", self.time_interval_spin)
        
        # Time interval presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Quick presets:"))
        
        hour_btn = QPushButton("1 Hour")
        hour_btn.clicked.connect(lambda: self.time_interval_spin.setValue(1.0))
        preset_layout.addWidget(hour_btn)
        
        day_btn = QPushButton("1 Day")
        day_btn.clicked.connect(lambda: self.time_interval_spin.setValue(24.0))
        preset_layout.addWidget(day_btn)
        
        week_btn = QPushButton("1 Week")
        week_btn.clicked.connect(lambda: self.time_interval_spin.setValue(168.0))
        preset_layout.addWidget(week_btn)
        
        month_btn = QPushButton("1 Month")
        month_btn.clicked.connect(lambda: self.time_interval_spin.setValue(720.0))
        preset_layout.addWidget(month_btn)
        
        form_layout.addRow("", preset_layout)
        
        layout.addLayout(form_layout)
        
        # Help text
        help_text = QTextBrowser()
        help_text.setMaximumHeight(100)
        help_text.setPlainText(
            "Split files based on time intervals. This is useful for:\n"
            "• Time-series data analysis\n"
            "• Creating daily/weekly/monthly reports\n"
            "• Managing data retention policies"
        )
        layout.addWidget(help_text)
        
        widget.setLayout(layout)
        return widget
        
    def _create_custom_tab(self):
        """Create the custom splitting tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Custom condition
        form_layout = QFormLayout()
        
        self.custom_condition_edit = QTextEdit()
        self.custom_condition_edit.setMaximumHeight(100)
        self.custom_condition_edit.setPlaceholderText(
            "Enter Python expression that returns True when a new file should start.\n"
            "Example: row['category'] != previous_category\n"
            "Available variables: row (current row), previous_row, row_index"
        )
        form_layout.addRow("Custom condition:", self.custom_condition_edit)
        
        layout.addLayout(form_layout)
        
        # Help text
        help_text = QTextBrowser()
        help_text.setMaximumHeight(150)
        help_text.setPlainText(
            "Split files based on custom conditions. This is useful for:\n"
            "• Splitting by category changes\n"
            "• Creating files for specific data ranges\n"
            "• Complex business logic requirements\n\n"
            "Examples:\n"
            "• row['user_id'] != previous_row['user_id']\n"
            "• row['status'] == 'completed'\n"
            "• row_index % 50000 == 0"
        )
        layout.addWidget(help_text)
        
        widget.setLayout(layout)
        return widget
        
    def _apply_styling(self):
        """Apply custom styling."""
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextBrowser {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: #f8f9fa;
            }
        """)
        
    def _on_enable_splitting_toggled(self, checked: bool):
        """Handle enable splitting checkbox toggle."""
        self.tab_widget.setEnabled(checked)
        self.split_config.enabled = checked
        self._update_preview()
        
    def _on_size_slider_changed(self, value: int):
        """Handle size slider change."""
        self.max_file_size_spin.setValue(float(value))
        
    def _browse_output_directory(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            
    def _update_preview(self):
        """Update the split preview."""
        if not self.split_config.enabled:
            self.preview_text.setPlainText("File splitting is disabled.")
            return
            
        # Get current tab
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Rows tab
            rows_per_file = self.rows_per_file_spin.value()
            # Estimate based on typical file sizes
            estimated_files = f"~{rows_per_file:,} rows per file"
            self.estimated_files_label.setText(f"Estimated: {estimated_files}")
            
        elif current_tab == 1:  # Size tab
            max_size = self.max_file_size_spin.value()
            estimated_files = f"~{max_size:.1f} MB per file"
            self.estimated_files_size_label.setText(f"Estimated: {estimated_files}")
            
        elif current_tab == 2:  # Time tab
            interval = self.time_interval_spin.value()
            column = self.time_column_edit.text()
            if column:
                estimated_files = f"Split by {interval:.1f} hour intervals using '{column}' column"
            else:
                estimated_files = "Please specify a time column"
                
        else:  # Custom tab
            condition = self.custom_condition_edit.toPlainText()
            if condition.strip():
                estimated_files = f"Split based on custom condition"
            else:
                estimated_files = "Please specify a custom condition"
                
        # Update preview text
        preview_text = f"Split Configuration:\n"
        preview_text += f"Method: {self.tab_widget.tabText(current_tab)}\n"
        preview_text += f"Output: {estimated_files}\n"
        
        if self.output_dir_edit.text():
            preview_text += f"Directory: {self.output_dir_edit.text()}\n"
        else:
            preview_text += f"Directory: Auto-created\n"
            
        preview_text += f"Pattern: {self.filename_pattern_edit.text()}\n"
        preview_text += f"Compression: {self.compression_combo.currentText()}"
        
        self.preview_text.setPlainText(preview_text)
        
    def _accept(self):
        """Accept the configuration."""
        if not self.split_config.enabled:
            self.accept()
            return
            
        # Validate configuration based on current tab
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # Rows
            self.split_config.method = SplitMethod.ROWS
            self.split_config.rows_per_file = self.rows_per_file_spin.value()
            
        elif current_tab == 1:  # Size
            self.split_config.method = SplitMethod.SIZE
            self.split_config.max_file_size_mb = self.max_file_size_spin.value()
            
        elif current_tab == 2:  # Time
            self.split_config.method = SplitMethod.TIME
            self.split_config.time_column = self.time_column_edit.text().strip()
            self.split_config.time_interval_hours = self.time_interval_spin.value()
            
            if not self.split_config.time_column:
                QMessageBox.warning(self, "Missing Time Column", 
                                  "Please specify a time column name.")
                return
                
        else:  # Custom
            self.split_config.method = SplitMethod.CUSTOM
            self.split_config.custom_condition = self.custom_condition_edit.toPlainText().strip()
            
            if not self.split_config.custom_condition:
                QMessageBox.warning(self, "Missing Custom Condition", 
                                  "Please specify a custom splitting condition.")
                return
                
        # Get output configuration
        self.split_config.output_directory = self.output_dir_edit.text().strip()
        self.split_config.filename_pattern = self.filename_pattern_edit.text()
        self.split_config.compression = self.compression_combo.currentText()
        self.split_config.include_header = self.include_header_checkbox.isChecked()
        
        self.accept()
        
    def get_split_config(self) -> SplitConfig:
        """Get the split configuration."""
        return self.split_config 

class FileSizeAnalyzer:
    """Analyze file sizes and provide detailed information."""
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information."""
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File not found"}
            
        # Basic file info
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        info = {
            "file_path": str(path),
            "file_name": path.name,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_mb, 2),
            "file_size_gb": round(file_size_mb / 1024, 3),
            "extension": path.suffix.lower(),
            "exists": True
        }
        
        # Try to get format-specific info
        try:
            format_type = FileFormatDetector.detect_format(file_path)
            if format_type:
                info["detected_format"] = format_type
                
                # Get data-specific info for supported formats
                if format_type in ['csv', 'tsv', 'parquet', 'excel', 'json', 'hdf5', 'feather', 'arrow']:
                    try:
                        df = DataReader.read_file(file_path, format_type)
                        info.update({
                            "rows": len(df),
                            "columns": len(df.columns),
                            "column_names": list(df.columns),
                            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                            "data_types": df.dtypes.to_dict()
                        })
                        
                        # Estimate compression ratio for parquet
                        if format_type == 'parquet':
                            raw_size_estimate = len(df) * len(df.columns) * 8  # Rough estimate
                            compression_ratio = raw_size_estimate / file_size_bytes if file_size_bytes > 0 else 1
                            info["compression_ratio"] = round(compression_ratio, 2)
                            
                    except Exception as e:
                        info["data_error"] = str(e)
                        
        except Exception as e:
            info["format_error"] = str(e)
            
        return info
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    @staticmethod
    def estimate_parquet_size(rows: int, columns: int, avg_value_size: int = 8) -> Dict[str, Any]:
        """Estimate parquet file size based on rows and columns."""
        # Rough estimation
        raw_size_bytes = rows * columns * avg_value_size
        compressed_size_bytes = raw_size_bytes * 0.3  # Parquet typically 70% compression
        
        return {
            "raw_size_bytes": raw_size_bytes,
            "raw_size_mb": round(raw_size_bytes / (1024 * 1024), 2),
            "compressed_size_bytes": int(compressed_size_bytes),
            "compressed_size_mb": round(compressed_size_bytes / (1024 * 1024), 2),
            "compression_ratio": round(raw_size_bytes / compressed_size_bytes, 2)
        }

class FileInfoDialog(QDialog):
    """Dialog for displaying detailed file information."""
    
    def __init__(self, file_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.file_info = file_info
        self.setup_ui()
        self._apply_styling()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("File Information")
        self.setModal(True)
        self.resize(600, 500)
        
        layout = QVBoxLayout()
        
        # File name header
        file_name_label = QLabel(self.file_info.get("file_name", "Unknown File"))
        file_name_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        file_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(file_name_label)
        
        # Create scrollable content
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # Basic file information
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout()
        
        basic_layout.addRow("File Path:", QLabel(self.file_info.get("file_path", "N/A")))
        basic_layout.addRow("File Size:", QLabel(FileSizeAnalyzer.format_file_size(self.file_info.get("file_size_bytes", 0))))
        basic_layout.addRow("Extension:", QLabel(self.file_info.get("extension", "N/A")))
        
        if "detected_format" in self.file_info:
            format_name = SUPPORTED_FORMATS.get(self.file_info["detected_format"], {}).get("name", self.file_info["detected_format"])
            basic_layout.addRow("Detected Format:", QLabel(format_name))
            
        basic_group.setLayout(basic_layout)
        scroll_layout.addWidget(basic_group)
        
        # Data information (if available)
        if "rows" in self.file_info and "columns" in self.file_info:
            data_group = QGroupBox("Data Information")
            data_layout = QFormLayout()
            
            data_layout.addRow("Rows:", QLabel(f"{self.file_info['rows']:,}"))
            data_layout.addRow("Columns:", QLabel(f"{self.file_info['columns']:,}"))
            
            if "memory_usage_mb" in self.file_info:
                data_layout.addRow("Memory Usage:", QLabel(f"{self.file_info['memory_usage_mb']} MB"))
                
            if "compression_ratio" in self.file_info:
                data_layout.addRow("Compression Ratio:", QLabel(f"{self.file_info['compression_ratio']}x"))
                
            data_group.setLayout(data_layout)
            scroll_layout.addWidget(data_group)
            
            # Column information
            if "column_names" in self.file_info:
                column_group = QGroupBox("Columns")
                column_layout = QVBoxLayout()
                
                # Show first 10 columns
                columns = self.file_info["column_names"]
                if len(columns) > 10:
                    column_text = f"First 10 columns (of {len(columns)}):\n" + "\n".join(columns[:10])
                else:
                    column_text = "\n".join(columns)
                    
                column_label = QLabel(column_text)
                column_label.setWordWrap(True)
                column_layout.addWidget(column_label)
                
                column_group.setLayout(column_layout)
                scroll_layout.addWidget(column_group)
                
            # Data types
            if "data_types" in self.file_info:
                type_group = QGroupBox("Data Types")
                type_layout = QVBoxLayout()
                
                type_text = "\n".join([f"{col}: {dtype}" for col, dtype in list(self.file_info["data_types"].items())[:10]])
                if len(self.file_info["data_types"]) > 10:
                    type_text += f"\n... and {len(self.file_info['data_types']) - 10} more"
                    
                type_label = QLabel(type_text)
                type_label.setWordWrap(True)
                type_layout.addWidget(type_label)
                
                type_group.setLayout(type_layout)
                scroll_layout.addWidget(type_group)
        
        # Error information
        if "error" in self.file_info:
            error_group = QGroupBox("Error")
            error_layout = QVBoxLayout()
            error_label = QLabel(self.file_info["error"])
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            error_group.setLayout(error_layout)
            scroll_layout.addWidget(error_group)
            
        if "data_error" in self.file_info:
            error_group = QGroupBox("Data Error")
            error_layout = QVBoxLayout()
            error_label = QLabel(self.file_info["data_error"])
            error_label.setStyleSheet("color: orange;")
            error_layout.addWidget(error_label)
            error_group.setLayout(error_layout)
            scroll_layout.addWidget(error_group)
        
        # Size estimation for parquet
        if "rows" in self.file_info and "columns" in self.file_info:
            estimate_group = QGroupBox("Size Estimation")
            estimate_layout = QFormLayout()
            
            estimate = FileSizeAnalyzer.estimate_parquet_size(
                self.file_info["rows"], 
                self.file_info["columns"]
            )
            
            estimate_layout.addRow("Raw Size:", QLabel(f"{estimate['raw_size_mb']} MB"))
            estimate_layout.addRow("Compressed (Parquet):", QLabel(f"{estimate['compressed_size_mb']} MB"))
            estimate_layout.addRow("Compression Ratio:", QLabel(f"{estimate['compression_ratio']}x"))
            
            estimate_group.setLayout(estimate_layout)
            scroll_layout.addWidget(estimate_group)
        
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
    def _apply_styling(self):
        """Apply custom styling."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLabel {
                padding: 2px;
            }
        """)
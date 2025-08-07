#!/usr/bin/env python3
"""
Standalone File Size Analyzer
A simple tool to analyze file sizes and provide detailed information about data files.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import pickle
import numpy as np
import h5py
import sqlite3
import math

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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# File format definitions (simplified)
SUPPORTED_FORMATS = {
    'csv': {'name': 'CSV (Comma Separated Values)', 'extensions': ['.csv']},
    'tsv': {'name': 'TSV (Tab Separated Values)', 'extensions': ['.tsv', '.txt']},
    'parquet': {'name': 'Parquet', 'extensions': ['.parquet', '.pq']},
    'excel': {'name': 'Excel (XLSX/XLS)', 'extensions': ['.xlsx', '.xls']},
    'json': {'name': 'JSON', 'extensions': ['.json']},
    'hdf5': {'name': 'HDF5', 'extensions': ['.h5', '.hdf5']},
    'pickle': {'name': 'Pickle (PKL)', 'extensions': ['.pkl', '.pickle']},
    'numpy': {'name': 'NumPy (NPY/NPZ)', 'extensions': ['.npy', '.npz']},
    'matlab': {'name': 'MATLAB (.mat)', 'extensions': ['.mat']},
    'feather': {'name': 'Feather', 'extensions': ['.feather']},
    'arrow': {'name': 'Arrow', 'extensions': ['.arrow']},
    'sqlite': {'name': 'SQLite Database', 'extensions': ['.db', '.sqlite', '.sqlite3']},
    'joblib': {'name': 'Joblib', 'extensions': ['.joblib']}
}

class FileSizeAnalyzer:
    """Analyze file sizes and provide detailed information."""
    
    @staticmethod
    def detect_format(file_path: str) -> str:
        """Detect the format of a file."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        for format_key, format_info in SUPPORTED_FORMATS.items():
            if extension in format_info['extensions']:
                return format_key
        return 'unknown'
    
    @staticmethod
    def read_file(file_path: str, format_type: str) -> pd.DataFrame:
        """Read a file and return a pandas DataFrame."""
        try:
            if format_type == 'csv':
                return pd.read_csv(file_path)
            elif format_type == 'tsv':
                return pd.read_csv(file_path, sep='\t')
            elif format_type == 'parquet':
                return pd.read_parquet(file_path)
            elif format_type == 'excel':
                return pd.read_excel(file_path)
            elif format_type == 'json':
                return pd.read_json(file_path)
            elif format_type == 'hdf5':
                return pd.read_hdf(file_path)
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
                    return pd.DataFrame({k: v for k, v in data.items()})
                else:
                    data = np.load(file_path)
                    return pd.DataFrame(data)
            elif format_type == 'matlab' and SCIPY_AVAILABLE:
                data = scipy.io.loadmat(file_path)
                return pd.DataFrame({k: v.flatten() if v.ndim > 1 else v for k, v in data.items() if not k.startswith('__')})
            elif format_type == 'feather':
                return pd.read_feather(file_path)
            elif format_type == 'arrow':
                return pa.ipc.open_file(file_path).read_pandas()
            elif format_type == 'sqlite':
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
            format_type = FileSizeAnalyzer.detect_format(file_path)
            if format_type != 'unknown':
                info["detected_format"] = format_type
                
                # Get data-specific info for supported formats
                if format_type in ['csv', 'tsv', 'parquet', 'excel', 'json', 'hdf5', 'feather', 'arrow']:
                    try:
                        df = FileSizeAnalyzer.read_file(file_path, format_type)
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
        raw_size_bytes = rows * columns * avg_value_size
        compressed_size_bytes = raw_size_bytes * 0.3  # Parquet typically 70% compression
        
        return {
            "raw_size_bytes": raw_size_bytes,
            "raw_size_mb": round(raw_size_bytes / (1024 * 1024), 2),
            "compressed_size_bytes": int(compressed_size_bytes),
            "compressed_size_mb": round(compressed_size_bytes / (1024 * 1024), 2),
            "compression_ratio": round(raw_size_bytes / compressed_size_bytes, 2)
        }

class FileAnalyzerWindow(QMainWindow):
    """Main window for the standalone file analyzer."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Size Analyzer")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.setup_ui()
        self.apply_styling()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("File Size Analyzer")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Select a file to analyze its size, structure, and data information")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # Browse button
        browse_btn = QPushButton("Browse for File")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        # Selected file label
        self.selected_file_label = QLabel("No file selected")
        self.selected_file_label.setWordWrap(True)
        file_layout.addWidget(self.selected_file_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Analysis results
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        # Create scrollable results area
        scroll_area = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_widget.setLayout(self.results_layout)
        
        scroll_area.setWidget(self.results_widget)
        scroll_area.setWidgetResizable(True)
        results_layout.addWidget(scroll_area)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        self.central_widget.setLayout(layout)
        
    def apply_styling(self):
        """Apply custom styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
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
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QLabel {
                padding: 5px;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
            }
        """)
        
    def browse_file(self):
        """Open file dialog to select a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File to Analyze", "",
            "All Supported Files (*.csv *.tsv *.txt *.parquet *.pq *.xlsx *.xls *.json *.h5 *.hdf5 *.pkl *.pickle *.npy *.npz *.mat *.feather *.arrow *.db *.sqlite *.sqlite3 *.joblib);;All Files (*)"
        )
        
        if file_path:
            self.selected_file_label.setText(f"Selected: {Path(file_path).name}")
            self.analyze_file(file_path)
            
    def analyze_file(self, file_path: str):
        """Analyze the selected file."""
        self.status_label.setText("Analyzing file...")
        QApplication.processEvents()
        
        try:
            # Clear previous results
            for i in reversed(range(self.results_layout.count())):
                self.results_layout.itemAt(i).widget().setParent(None)
            
            # Get file information
            file_info = FileSizeAnalyzer.get_file_info(file_path)
            
            # Display results
            self.display_file_info(file_info)
            
            self.status_label.setText("Analysis complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze file: {str(e)}")
            self.status_label.setText("Analysis failed")
            
    def display_file_info(self, file_info: Dict[str, Any]):
        """Display file information in the results area."""
        # Basic information
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout()
        
        basic_layout.addRow("File Path:", QLabel(file_info.get("file_path", "N/A")))
        basic_layout.addRow("File Size:", QLabel(FileSizeAnalyzer.format_file_size(file_info.get("file_size_bytes", 0))))
        basic_layout.addRow("Extension:", QLabel(file_info.get("extension", "N/A")))
        
        if "detected_format" in file_info:
            format_name = SUPPORTED_FORMATS.get(file_info["detected_format"], {}).get("name", file_info["detected_format"])
            basic_layout.addRow("Detected Format:", QLabel(format_name))
            
        basic_group.setLayout(basic_layout)
        self.results_layout.addWidget(basic_group)
        
        # Data information (if available)
        if "rows" in file_info and "columns" in file_info:
            data_group = QGroupBox("Data Information")
            data_layout = QFormLayout()
            
            data_layout.addRow("Rows:", QLabel(f"{file_info['rows']:,}"))
            data_layout.addRow("Columns:", QLabel(f"{file_info['columns']:,}"))
            
            if "memory_usage_mb" in file_info:
                data_layout.addRow("Memory Usage:", QLabel(f"{file_info['memory_usage_mb']} MB"))
                
            if "compression_ratio" in file_info:
                data_layout.addRow("Compression Ratio:", QLabel(f"{file_info['compression_ratio']}x"))
                
            data_group.setLayout(data_layout)
            self.results_layout.addWidget(data_group)
            
            # Column information
            if "column_names" in file_info:
                column_group = QGroupBox("Columns")
                column_layout = QVBoxLayout()
                
                # Show first 20 columns
                columns = file_info["column_names"]
                if len(columns) > 20:
                    column_text = f"First 20 columns (of {len(columns)}):\n" + "\n".join(columns[:20])
                else:
                    column_text = "\n".join(columns)
                    
                column_label = QLabel(column_text)
                column_label.setWordWrap(True)
                column_layout.addWidget(column_label)
                
                column_group.setLayout(column_layout)
                self.results_layout.addWidget(column_group)
                
            # Data types
            if "data_types" in file_info:
                type_group = QGroupBox("Data Types")
                type_layout = QVBoxLayout()
                
                type_text = "\n".join([f"{col}: {dtype}" for col, dtype in list(file_info["data_types"].items())[:15]])
                if len(file_info["data_types"]) > 15:
                    type_text += f"\n... and {len(file_info['data_types']) - 15} more"
                    
                type_label = QLabel(type_text)
                type_label.setWordWrap(True)
                type_layout.addWidget(type_label)
                
                type_group.setLayout(type_layout)
                self.results_layout.addWidget(type_group)
        
        # Size estimation for parquet
        if "rows" in file_info and "columns" in file_info:
            estimate_group = QGroupBox("Size Estimation")
            estimate_layout = QFormLayout()
            
            estimate = FileSizeAnalyzer.estimate_parquet_size(
                file_info["rows"], 
                file_info["columns"]
            )
            
            estimate_layout.addRow("Raw Size:", QLabel(f"{estimate['raw_size_mb']} MB"))
            estimate_layout.addRow("Compressed (Parquet):", QLabel(f"{estimate['compressed_size_mb']} MB"))
            estimate_layout.addRow("Compression Ratio:", QLabel(f"{estimate['compression_ratio']}x"))
            
            estimate_group.setLayout(estimate_layout)
            self.results_layout.addWidget(estimate_group)
        
        # Error information
        if "error" in file_info:
            error_group = QGroupBox("Error")
            error_layout = QVBoxLayout()
            error_label = QLabel(file_info["error"])
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            error_group.setLayout(error_layout)
            self.results_layout.addWidget(error_group)
            
        if "data_error" in file_info:
            error_group = QGroupBox("Data Error")
            error_layout = QVBoxLayout()
            error_label = QLabel(file_info["data_error"])
            error_label.setStyleSheet("color: orange;")
            error_layout.addWidget(error_label)
            error_group.setLayout(error_layout)
            self.results_layout.addWidget(error_group)

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("File Size Analyzer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Data Processing Tools")
    
    # Create and show main window
    window = FileAnalyzerWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
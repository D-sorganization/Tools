#!/usr/bin/env python3
"""
Parquet File Size Analyzer
A simple tool to analyze parquet file metadata without loading the entire file.
"""

import sys
import os
from pathlib import Path
import pyarrow.parquet as pq
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLabel
from PyQt6.QtCore import Qt


class ParquetAnalyzer(QMainWindow):
    """Simple GUI for analyzing parquet files."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parquet File Analyzer")
        self.setGeometry(100, 100, 600, 400)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Parquet File Metadata Analyzer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Select file button
        self.select_btn = QPushButton("Select Parquet File")
        self.select_btn.clicked.connect(self.select_file)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        layout.addWidget(self.select_btn)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.results_text)
        
        central_widget.setLayout(layout)
        
    def select_file(self):
        """Open file dialog to select a parquet file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Parquet File", 
            "", 
            "Parquet Files (*.parquet *.pq);;All Files (*)"
        )
        
        if file_path:
            self.analyze_parquet_file(file_path)
            
    def analyze_parquet_file(self, file_path):
        """Analyze the selected parquet file."""
        try:
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
            
            # Schema information
            results += f"=== Schema ===\n"
            for i, field in enumerate(parquet_file.schema):
                results += f"{i+1:2d}. {field.name}: {field.physical_type}\n"
            
            # Row group details
            if parquet_file.metadata.num_row_groups > 0:
                results += f"\n=== Row Group Details ===\n"
                for i in range(min(3, parquet_file.metadata.num_row_groups)):
                    rg = parquet_file.metadata.row_group(i)
                    results += f"Row Group {i+1}:\n"
                    results += f"  Rows: {rg.num_rows:,}\n"
                    results += f"  Size: {self.format_file_size(rg.total_byte_size)}\n"
                    results += "\n"
                
                if parquet_file.metadata.num_row_groups > 3:
                    results += f"... and {parquet_file.metadata.num_row_groups - 3} more row groups\n\n"
            
            # File statistics
            if parquet_file.metadata.num_rows > 0:
                avg_row_size = file_size / parquet_file.metadata.num_rows
                results += f"=== Statistics ===\n"
                results += f"Average row size: {avg_row_size:.2f} bytes\n"
                results += f"Total data size: {self.format_file_size(parquet_file.metadata.serialized_size)}\n"
            
            self.results_text.setPlainText(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze file: {str(e)}")
            self.results_text.setPlainText(f"Error: {str(e)}")
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Parquet File Analyzer")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = ParquetAnalyzer()
    window.show()
    
    # Start the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
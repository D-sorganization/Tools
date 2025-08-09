#!/usr/bin/env python3
"""
CSV to Parquet Bulk Converter - STABLE VERSION
A PyQt6-based GUI application for converting multiple CSV files to Parquet format.

Features:
- Bulk conversion of CSV files to Parquet
- Support for large file sets (10,000+ files)
- Option to combine into single file or multiple files
- Progress tracking and error handling
- Memory-efficient processing for large datasets
- Can be integrated as a subtab in other PyQt6 applications

Author: AI Assistant
Date: 2025
"""

import sys
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import threading
import queue

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QCheckBox, QSpinBox, QGroupBox,
    QGridLayout, QSplitter, QFrame, QComboBox, QTabWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor


class ConversionWorker(QThread):
    """Worker thread for handling CSV to Parquet conversion."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    conversion_completed = pyqtSignal(bool, str)
    
    def __init__(self, csv_files: List[str], output_path: str, 
                 combine_files: bool = True, chunk_size: int = 10000, auto_convert: bool = True):
        super().__init__()
        self.csv_files = csv_files
        self.output_path = output_path
        self.combine_files = combine_files
        self.chunk_size = chunk_size
        self.auto_convert = auto_convert
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
        """Convert all CSV files to a single Parquet file."""
        self.status_updated.emit("Reading CSV files and combining data...")
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files in chunks to manage memory
        total_files = len(self.csv_files)
        processed_files = 0
        
        # Use PyArrow for efficient writing
        writer = None
        schema = None
        
        for i, csv_file in enumerate(self.csv_files):
            if self.is_cancelled:
                break
                
            try:
                self.status_updated.emit(f"Processing {Path(csv_file).name} ({i+1}/{total_files})")
                
                # Read CSV in chunks to handle large files
                chunk_iter = pd.read_csv(csv_file, chunksize=self.chunk_size)
                
                for chunk in chunk_iter:
                    if self.is_cancelled:
                        break
                    
                    # Infer schema from first chunk if not set
                    if schema is None:
                        # Use more flexible schema inference
                        if self.auto_convert:
                            # Convert mixed types to string to avoid conversion errors
                            chunk_converted = chunk.astype(str)
                            table = pa.Table.from_pandas(chunk_converted)
                        else:
                            table = pa.Table.from_pandas(chunk)
                        schema = table.schema
                        writer = pq.ParquetWriter(self.output_path, schema)
                    else:
                        # Convert to PyArrow table with existing schema, but be more flexible
                        try:
                            if self.auto_convert:
                                # Convert mixed types to string to avoid conversion errors
                                chunk_converted = chunk.astype(str)
                                table = pa.Table.from_pandas(chunk_converted, schema=schema)
                            else:
                                table = pa.Table.from_pandas(chunk, schema=schema)
                        except Exception as schema_error:
                            # If schema conversion fails, try with inferred schema for this chunk
                            self.error_occurred.emit(f"Schema conversion warning for {Path(csv_file).name}: {str(schema_error)}")
                            if self.auto_convert:
                                chunk_converted = chunk.astype(str)
                                table = pa.Table.from_pandas(chunk_converted)
                            else:
                                table = pa.Table.from_pandas(chunk)
                    
                    writer.write_table(table)
                
                processed_files += 1
                progress = int((processed_files / total_files) * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.error_occurred.emit(f"Error processing {csv_file}: {str(e)}")
                continue
        
        if writer:
            writer.close()
            
        self.status_updated.emit(f"Successfully converted {processed_files} files to {self.output_path}")
    
    def _convert_to_multiple_files(self):
        """Convert each CSV file to a separate Parquet file."""
        total_files = len(self.csv_files)
        processed_files = 0
        
        for i, csv_file in enumerate(self.csv_files):
            if self.is_cancelled:
                break
                
            try:
                self.status_updated.emit(f"Converting {Path(csv_file).name} ({i+1}/{total_files})")
                
                # Generate output filename
                csv_path = Path(csv_file)
                output_file = Path(self.output_path) / f"{csv_path.stem}.parquet"
                
                # Read and convert
                df = pd.read_csv(csv_file)
                if self.auto_convert:
                    # Convert mixed types to string to avoid conversion errors
                    df = df.astype(str)
                df.to_parquet(output_file, index=False)
                
                processed_files += 1
                progress = int((processed_files / total_files) * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                self.error_occurred.emit(f"Error converting {csv_file}: {str(e)}")
                continue
        
        self.status_updated.emit(f"Successfully converted {processed_files} files")
    
    def cancel(self):
        """Cancel the conversion process."""
        self.is_cancelled = True


class CSVToParquetConverter(QWidget):
    """Main GUI widget for CSV to Parquet conversion."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.csv_files = []
        self.worker_thread = None
        self.setup_ui()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('csv_to_parquet.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("CSV to Parquet Bulk Converter")
        self.setMinimumSize(800, 600)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section - File selection and options
        top_widget = self._create_top_section()
        splitter.addWidget(top_widget)
        
        # Bottom section - Progress and log
        bottom_widget = self._create_bottom_section()
        splitter.addWidget(bottom_widget)
        
        # Set splitter proportions
        splitter.setSizes([300, 300])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # Apply styling
        self._apply_styling()
    
    def _create_top_section(self):
        """Create the top section with file selection and options."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout()
        
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Select folder containing CSV files...")
        self.folder_path_edit.setReadOnly(True)
        
        browse_btn = QPushButton("Browse Folder")
        browse_btn.clicked.connect(self._browse_folder)
        
        self.file_count_label = QLabel("No files selected")
        self.file_count_label.setStyleSheet("color: gray;")
        
        file_layout.addWidget(QLabel("CSV Files Folder:"), 0, 0)
        file_layout.addWidget(self.folder_path_edit, 0, 1)
        file_layout.addWidget(browse_btn, 0, 2)
        file_layout.addWidget(self.file_count_label, 1, 0, 1, 3)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Output options group
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output file or folder...")
        self.output_path_edit.setReadOnly(True)
        
        output_browse_btn = QPushButton("Browse Output")
        output_browse_btn.clicked.connect(self._browse_output)
        
        self.combine_checkbox = QCheckBox("Combine all files into single Parquet file")
        self.combine_checkbox.setChecked(True)
        self.combine_checkbox.toggled.connect(self._on_combine_toggled)
        
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(1000, 100000)
        self.chunk_size_spin.setValue(10000)
        self.chunk_size_spin.setSuffix(" rows")
        
        self.auto_convert_checkbox = QCheckBox("Auto-convert data types (handle mixed types)")
        self.auto_convert_checkbox.setChecked(True)
        self.auto_convert_checkbox.setToolTip("Automatically handle mixed data types in columns")
        
        output_layout.addWidget(QLabel("Output Path:"), 0, 0)
        output_layout.addWidget(self.output_path_edit, 0, 1)
        output_layout.addWidget(output_browse_btn, 0, 2)
        output_layout.addWidget(self.combine_checkbox, 1, 0, 1, 3)
        output_layout.addWidget(QLabel("Chunk Size (for large files):"), 2, 0)
        output_layout.addWidget(self.chunk_size_spin, 2, 1)
        output_layout.addWidget(self.auto_convert_checkbox, 3, 0, 1, 3)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.clicked.connect(self._start_conversion)
        self.convert_btn.setEnabled(False)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_conversion)
        self.cancel_btn.setEnabled(False)
        
        button_layout.addWidget(self.convert_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def _create_bottom_section(self):
        """Create the bottom section with progress and log."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.status_label = QLabel("Ready to convert")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Log section
        log_group = QGroupBox("Conversion Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        
        log_buttons_layout = QHBoxLayout()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        
        save_log_btn = QPushButton("Save Log")
        save_log_btn.clicked.connect(self._save_log)
        
        log_buttons_layout.addWidget(clear_log_btn)
        log_buttons_layout.addWidget(save_log_btn)
        log_buttons_layout.addStretch()
        
        log_layout.addWidget(self.log_text)
        log_layout.addLayout(log_buttons_layout)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        widget.setLayout(layout)
        return widget
    
    def _apply_styling(self):
        """Apply custom styling to the interface."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
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
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton#cancel {
                background-color: #f44336;
            }
            QPushButton#cancel:hover {
                background-color: #da190b;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #f9f9f9;
            }
        """)
        
        # Apply cancel button styling
        self.cancel_btn.setObjectName("cancel")
    
    def _browse_folder(self):
        """Browse for folder containing CSV files."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with CSV Files", ""
        )
        
        if folder_path:
            self.folder_path_edit.setText(folder_path)
            self._scan_csv_files(folder_path)
    
    def _scan_csv_files(self, folder_path: str):
        """Scan folder for CSV files."""
        try:
            csv_files = []
            folder = Path(folder_path)
            
            # Find all CSV files recursively
            for csv_file in folder.rglob("*.csv"):
                csv_files.append(str(csv_file))
            
            self.csv_files = csv_files
            
            if csv_files:
                self.file_count_label.setText(f"Found {len(csv_files)} CSV files")
                self.file_count_label.setStyleSheet("color: green; font-weight: bold;")
                self.log_text.append(f"Found {len(csv_files)} CSV files in {folder_path}")
            else:
                self.file_count_label.setText("No CSV files found")
                self.file_count_label.setStyleSheet("color: red;")
                self.log_text.append("No CSV files found in the selected folder")
            
            self._update_convert_button()
            
        except Exception as e:
            self.log_text.append(f"Error scanning folder: {str(e)}")
            self.file_count_label.setText("Error scanning folder")
            self.file_count_label.setStyleSheet("color: red;")
    
    def _browse_output(self):
        """Browse for output file or folder."""
        if self.combine_checkbox.isChecked():
            # Single file output
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Parquet File", "", "Parquet Files (*.parquet)"
            )
            if file_path:
                self.output_path_edit.setText(file_path)
        else:
            # Folder output
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Output Folder", ""
            )
            if folder_path:
                self.output_path_edit.setText(folder_path)
        
        self._update_convert_button()
    
    def _on_combine_toggled(self, checked: bool):
        """Handle combine checkbox toggle."""
        if checked:
            self.output_path_edit.setPlaceholderText("Select output file...")
        else:
            self.output_path_edit.setPlaceholderText("Select output folder...")
        
        # Clear current output path when switching modes
        self.output_path_edit.clear()
        self._update_convert_button()
    
    def _update_convert_button(self):
        """Update convert button state based on current settings."""
        has_files = len(self.csv_files) > 0
        has_output = bool(self.output_path_edit.text().strip())
        self.convert_btn.setEnabled(has_files and has_output)
    
    def _start_conversion(self):
        """Start the conversion process."""
        if not self.csv_files:
            QMessageBox.warning(self, "No Files", "Please select CSV files first.")
            return
        
        if not self.output_path_edit.text().strip():
            QMessageBox.warning(self, "No Output", "Please select output path.")
            return
        
        # Confirm conversion
        msg = f"Convert {len(self.csv_files)} CSV files to Parquet?"
        if self.combine_checkbox.isChecked():
            msg += "\nAll files will be combined into a single Parquet file."
        else:
            msg += "\nEach file will be converted to a separate Parquet file."
        
        reply = QMessageBox.question(
            self, "Confirm Conversion", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._execute_conversion()
    
    def _execute_conversion(self):
        """Execute the conversion in a worker thread."""
        # Update UI state
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Clear log
        self.log_text.clear()
        self.log_text.append(f"Starting conversion at {datetime.now().strftime('%H:%M:%S')}")
        self.log_text.append(f"Input files: {len(self.csv_files)}")
        self.log_text.append(f"Output: {self.output_path_edit.text()}")
        self.log_text.append(f"Combine files: {self.combine_checkbox.isChecked()}")
        self.log_text.append("-" * 50)
        
        # Create and start worker thread
        self.worker_thread = ConversionWorker(
            csv_files=self.csv_files,
            output_path=self.output_path_edit.text(),
            combine_files=self.combine_checkbox.isChecked(),
            chunk_size=self.chunk_size_spin.value(),
            auto_convert=self.auto_convert_checkbox.isChecked()
        )
        
        # Connect signals
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self._update_status)
        self.worker_thread.error_occurred.connect(self._log_error)
        self.worker_thread.conversion_completed.connect(self._conversion_finished)
        
        # Start conversion
        self.worker_thread.start()
    
    def _cancel_conversion(self):
        """Cancel the current conversion."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.status_label.setText("Cancelling conversion...")
            self.log_text.append("Cancelling conversion...")
    
    def _update_status(self, status: str):
        """Update status label and log."""
        self.status_label.setText(status)
        self.log_text.append(status)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _log_error(self, error: str):
        """Log error message."""
        self.log_text.append(f"ERROR: {error}")
        self.logger.error(error)
    
    def _conversion_finished(self, success: bool, message: str):
        """Handle conversion completion."""
        # Update UI state
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("Conversion completed successfully!")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.log_text.append(f"SUCCESS: {message}")
            
            QMessageBox.information(
                self, "Conversion Complete", 
                f"Conversion completed successfully!\n\n{message}"
            )
        else:
            self.status_label.setText("Conversion failed!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.log_text.append(f"FAILED: {message}")
            
            QMessageBox.critical(
                self, "Conversion Failed", 
                f"Conversion failed!\n\nError: {message}"
            )
    
    def _save_log(self):
        """Save the current log to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log File", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "Log Saved", f"Log saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV to Parquet Bulk Converter")
        self.setMinimumSize(900, 700)
        
        # Create central widget
        self.converter_widget = CSVToParquetConverter()
        self.setCentralWidget(self.converter_widget)
        
        # Setup menu bar
        self._setup_menu()
    
    def _setup_menu(self):
        """Setup the menu bar."""
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
        """Show about dialog."""
        QMessageBox.about(
            self, "About CSV to Parquet Converter",
            """<h3>CSV to Parquet Bulk Converter</h3>
            <p>A PyQt6-based application for converting multiple CSV files to Parquet format.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Bulk conversion of CSV files to Parquet</li>
                <li>Support for large file sets (10,000+ files)</li>
                <li>Option to combine into single file or multiple files</li>
                <li>Memory-efficient processing for large datasets</li>
                <li>Progress tracking and error handling</li>
            </ul>
            <p><b>Dependencies:</b></p>
            <ul>
                <li>PyQt6</li>
                <li>pandas</li>
                <li>pyarrow</li>
            </ul>"""
        )


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("CSV to Parquet Converter")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Data Processor")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CSV to Parquet Bulk Converter - ENHANCED VERSION
A PyQt6-based GUI application for converting multiple CSV files to Parquet format.

Enhanced Features:
- Column selection with search and filtering
- Save/load column selection lists
- Efficient handling of large column sets (2000+ columns)
- Bulk conversion of CSV files to Parquet
- Support for large file sets (10,000+ files)
- Option to combine into single file or multiple files
- Progress tracking and error handling
- Memory-efficient processing for large datasets
- Can be integrated as a subtab in other PyQt6 applications

Author: AI Assistant
Date: 2025
"""

import json
import logging
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PyQt6.QtCore import QRegularExpression, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QIcon, QPalette,
                         QRegularExpressionValidator)
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QDialog, QFileDialog, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QSpinBox, QSplitter,
                             QTabWidget, QTextEdit, QVBoxLayout, QWidget)


class ColumnSelectionDialog(QDialog):
    """Dialog for selecting columns from CSV files."""

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
        self.resize(800, 600)

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
        self.column_list.itemChanged.connect(
            self._on_item_changed
        )  # Connect the signal
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

        save_btn = QPushButton("Save Column List")
        save_btn.clicked.connect(self._save_column_list)
        save_load_layout.addWidget(save_btn)

        save_load_layout.addWidget(QLabel("Load:"))
        self.load_combo = QComboBox()
        self.load_combo.currentTextChanged.connect(self._load_selected_list)
        save_load_layout.addWidget(self.load_combo)

        layout.addLayout(save_load_layout)

        # Status
        self.status_label = QLabel("0 columns selected")
        layout.addWidget(self.status_label)

        # Dialog buttons
        dialog_buttons = QHBoxLayout()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._accept)
        dialog_buttons.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._reject)
        dialog_buttons.addWidget(cancel_btn)

        layout.addLayout(dialog_buttons)

        self.setLayout(layout)
        self._apply_styling()
        self._populate_column_list()

    def _apply_styling(self):
        """Apply custom styling."""
        self.setStyleSheet(
            """
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
        """
        )

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
        print(
            f"DEBUG: Found {selected_count} checked items out of {self.column_list.count()} total"
        )
        self.status_label.setText(f"{selected_count} columns selected")

    def _save_column_list(self):
        """Save current column selection as a named list."""
        selected_columns = []
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_columns.append(item.text())

        if not selected_columns:
            QMessageBox.warning(
                self, "No Selection", "Please select at least one column to save."
            )
            return

        name, ok = QInputDialog.getText(
            self, "Save Column List", "Enter a name for this column list:"
        )
        if ok and name:
            self._save_list_to_file(name, selected_columns)
            self.load_column_lists()

    def _save_list_to_file(self, name: str, columns: List[str]):
        """Save column list to file."""
        try:
            lists_dir = Path("column_lists")
            lists_dir.mkdir(exist_ok=True)

            file_path = lists_dir / f"{name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": name,
                        "columns": columns,
                        "created": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )

            QMessageBox.information(
                self, "Saved", f"Column list '{name}' saved successfully!"
            )

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
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            columns = data.get("columns", [])
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
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            name = data.get("name", json_file.stem)
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
            QMessageBox.warning(
                self, "No Selection", "Please select at least one column."
            )
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
    """Worker thread for handling CSV to Parquet conversion."""

    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    conversion_completed = pyqtSignal(bool, str)

    def __init__(
        self,
        csv_files: List[str],
        output_path: str,
        combine_files: bool = True,
        chunk_size: int = 10000,
        auto_convert: bool = True,
        selected_columns: Set[str] = None,
    ):
        super().__init__()
        self.csv_files = csv_files
        self.output_path = output_path
        self.combine_files = combine_files
        self.chunk_size = chunk_size
        self.auto_convert = auto_convert
        self.selected_columns = selected_columns or set()
        self.is_cancelled = False

    def run(self):
        """Execute the conversion process."""
        try:
            if self.combine_files:
                self._convert_to_single_file()
            else:
                self._convert_to_multiple_files()

            if not self.is_cancelled:
                self.conversion_completed.emit(
                    True, "Conversion completed successfully!"
                )

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
                self.status_updated.emit(
                    f"Processing {Path(csv_file).name} ({i+1}/{total_files})"
                )

                # Read CSV in chunks to handle large files
                chunk_iter = pd.read_csv(csv_file, chunksize=self.chunk_size)

                for chunk in chunk_iter:
                    if self.is_cancelled:
                        break

                    # Filter columns if specified
                    if self.selected_columns:
                        available_columns = set(chunk.columns)
                        columns_to_keep = list(
                            available_columns.intersection(self.selected_columns)
                        )
                        if columns_to_keep:
                            chunk = chunk[columns_to_keep]
                        else:
                            self.error_occurred.emit(
                                f"No selected columns found in {Path(csv_file).name}"
                            )
                            continue

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
                                table = pa.Table.from_pandas(
                                    chunk_converted, schema=schema
                                )
                            else:
                                table = pa.Table.from_pandas(chunk, schema=schema)
                        except Exception as schema_error:
                            # If schema conversion fails, try with inferred schema for this chunk
                            self.error_occurred.emit(
                                f"Schema conversion warning for {Path(csv_file).name}: {str(schema_error)}"
                            )
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

        self.status_updated.emit(
            f"Successfully converted {processed_files} files to {self.output_path}"
        )

    def _convert_to_multiple_files(self):
        """Convert each CSV file to a separate Parquet file."""
        total_files = len(self.csv_files)
        processed_files = 0

        for i, csv_file in enumerate(self.csv_files):
            if self.is_cancelled:
                break

            try:
                self.status_updated.emit(
                    f"Converting {Path(csv_file).name} ({i+1}/{total_files})"
                )

                # Generate output filename
                csv_path = Path(csv_file)
                output_file = Path(self.output_path) / f"{csv_path.stem}.parquet"

                # Read and convert
                df = pd.read_csv(csv_file)

                # Filter columns if specified
                if self.selected_columns:
                    available_columns = set(df.columns)
                    columns_to_keep = list(
                        available_columns.intersection(self.selected_columns)
                    )
                    if columns_to_keep:
                        df = df[columns_to_keep]
                    else:
                        self.error_occurred.emit(
                            f"No selected columns found in {csv_file}"
                        )
                        continue

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
        self.selected_columns = set()
        self.setup_ui()
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("csv_to_parquet.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("CSV to Parquet Bulk Converter - Enhanced")
        self.setMinimumSize(1000, 700)

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
        splitter.setSizes([400, 300])

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
        self.folder_path_edit.setPlaceholderText(
            "Select folder containing CSV files..."
        )
        self.folder_path_edit.setReadOnly(True)

        browse_folder_btn = QPushButton("Browse Folder")
        browse_folder_btn.clicked.connect(self._browse_folder)

        browse_files_btn = QPushButton("Select Files")
        browse_files_btn.clicked.connect(self._browse_files)

        self.select_all_files_checkbox = QCheckBox("Select all files in folder")
        self.select_all_files_checkbox.setChecked(True)
        self.select_all_files_checkbox.toggled.connect(
            self._on_select_all_files_toggled
        )

        self.file_count_label = QLabel("No files selected")
        self.file_count_label.setStyleSheet("color: gray;")

        file_layout.addWidget(QLabel("CSV Files:"), 0, 0)
        file_layout.addWidget(self.folder_path_edit, 0, 1)
        file_layout.addWidget(browse_folder_btn, 0, 2)
        file_layout.addWidget(browse_files_btn, 0, 3)
        file_layout.addWidget(self.select_all_files_checkbox, 1, 0, 1, 4)
        file_layout.addWidget(self.file_count_label, 2, 0, 1, 4)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Column selection group
        column_group = QGroupBox("Column Selection")
        column_layout = QGridLayout()

        self.column_selection_btn = QPushButton("Select Columns")
        self.column_selection_btn.clicked.connect(self._select_columns)
        self.column_selection_btn.setEnabled(False)

        self.column_count_label = QLabel("No columns selected")
        self.column_count_label.setStyleSheet("color: gray;")

        self.use_all_columns_checkbox = QCheckBox(
            "Use all columns (skip column selection)"
        )
        self.use_all_columns_checkbox.setChecked(True)
        self.use_all_columns_checkbox.toggled.connect(self._on_use_all_columns_toggled)

        column_layout.addWidget(QLabel("Column Selection:"), 0, 0)
        column_layout.addWidget(self.column_selection_btn, 0, 1)
        column_layout.addWidget(self.use_all_columns_checkbox, 0, 2)
        column_layout.addWidget(self.column_count_label, 1, 0, 1, 3)

        column_group.setLayout(column_layout)
        layout.addWidget(column_group)

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

        self.auto_convert_checkbox = QCheckBox(
            "Auto-convert data types (handle mixed types)"
        )
        self.auto_convert_checkbox.setChecked(True)
        self.auto_convert_checkbox.setToolTip(
            "Automatically handle mixed data types in columns"
        )

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
        self.setStyleSheet(
            """
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
        """
        )

        # Apply cancel button styling
        self.cancel_btn.setObjectName("cancel")

    def _browse_folder(self):
        """Browse for folder containing CSV files."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with CSV Files", ""
        )

        if folder_path:
            self.folder_path_edit.setText(folder_path)
            if self.select_all_files_checkbox.isChecked():
                self._scan_csv_files(folder_path)

    def _browse_files(self):
        """Browse for individual CSV files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV Files", "", "CSV Files (*.csv)"
        )

        if file_paths:
            # Set the folder path to the directory of the first file
            folder_path = str(Path(file_paths[0]).parent)
            self.folder_path_edit.setText(folder_path)
            self.csv_files = file_paths

            if file_paths:
                self.file_count_label.setText(f"Selected {len(file_paths)} CSV files")
                self.file_count_label.setStyleSheet("color: green; font-weight: bold;")
                self.log_text.append(f"Selected {len(file_paths)} CSV files")

                # Enable column selection if not using all columns
                if not self.use_all_columns_checkbox.isChecked():
                    self.column_selection_btn.setEnabled(True)
            else:
                self.file_count_label.setText("No CSV files selected")
                self.file_count_label.setStyleSheet("color: red;")
                self.column_selection_btn.setEnabled(False)

            self._update_convert_button()

    def _on_select_all_files_toggled(self, checked: bool):
        """Handle select all files checkbox toggle."""
        if checked and self.folder_path_edit.text():
            # Re-scan the folder to get all files
            self._scan_csv_files(self.folder_path_edit.text())
        elif not checked:
            # Clear the file list but keep the folder path
            self.csv_files = []
            self.file_count_label.setText("No files selected")
            self.file_count_label.setStyleSheet("color: gray;")
            self.column_selection_btn.setEnabled(False)
            self._update_convert_button()

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
                self.log_text.append(
                    f"Found {len(csv_files)} CSV files in {folder_path}"
                )

                # Enable column selection if not using all columns
                if not self.use_all_columns_checkbox.isChecked():
                    self.column_selection_btn.setEnabled(True)
            else:
                self.file_count_label.setText("No CSV files found")
                self.file_count_label.setStyleSheet("color: red;")
                self.log_text.append("No CSV files found in the selected folder")
                self.column_selection_btn.setEnabled(False)

            self._update_convert_button()

        except Exception as e:
            self.log_text.append(f"Error scanning folder: {str(e)}")
            self.file_count_label.setText("Error scanning folder")
            self.file_count_label.setStyleSheet("color: red;")

    def _select_columns(self):
        """Open column selection dialog."""
        if not self.csv_files:
            QMessageBox.warning(self, "No Files", "Please select CSV files first.")
            return

        try:
            # Read columns from first file
            first_file = self.csv_files[0]
            df_sample = pd.read_csv(first_file, nrows=1)  # Read just headers
            columns = list(df_sample.columns)

            # Open column selection dialog
            dialog = ColumnSelectionDialog(columns, self)
            dialog.exec()

            # Get selected columns
            self.selected_columns = dialog.get_selected_columns()

            if self.selected_columns:
                self.column_count_label.setText(
                    f"{len(self.selected_columns)} columns selected"
                )
                self.column_count_label.setStyleSheet(
                    "color: green; font-weight: bold;"
                )
                self.log_text.append(
                    f"Selected {len(self.selected_columns)} columns for conversion"
                )
            else:
                self.column_count_label.setText("No columns selected")
                self.column_count_label.setStyleSheet("color: red;")

            self._update_convert_button()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read columns: {str(e)}")

    def _on_use_all_columns_toggled(self, checked: bool):
        """Handle use all columns checkbox toggle."""
        if checked:
            self.column_selection_btn.setEnabled(False)
            self.selected_columns = set()
            self.column_count_label.setText("Using all columns")
            self.column_count_label.setStyleSheet("color: blue; font-weight: bold;")
        else:
            if self.csv_files:
                self.column_selection_btn.setEnabled(True)
            self.column_count_label.setText("No columns selected")
            self.column_count_label.setStyleSheet("color: red;")

        self._update_convert_button()

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

        # Check column selection
        has_columns = True
        if not self.use_all_columns_checkbox.isChecked():
            has_columns = len(self.selected_columns) > 0

        self.convert_btn.setEnabled(has_files and has_output and has_columns)

    def _start_conversion(self):
        """Start the conversion process."""
        if not self.csv_files:
            QMessageBox.warning(self, "No Files", "Please select CSV files first.")
            return

        if not self.output_path_edit.text().strip():
            QMessageBox.warning(self, "No Output", "Please select output path.")
            return

        # Check column selection
        if not self.use_all_columns_checkbox.isChecked() and not self.selected_columns:
            QMessageBox.warning(self, "No Columns", "Please select columns to include.")
            return

        # Confirm conversion
        msg = f"Convert {len(self.csv_files)} CSV files to Parquet?"
        if self.combine_checkbox.isChecked():
            msg += "\nAll files will be combined into a single Parquet file."
        else:
            msg += "\nEach file will be converted to a separate Parquet file."

        if self.use_all_columns_checkbox.isChecked():
            msg += "\nUsing all columns."
        else:
            msg += f"\nUsing {len(self.selected_columns)} selected columns."

        reply = QMessageBox.question(
            self,
            "Confirm Conversion",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
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
        self.log_text.append(
            f"Starting conversion at {datetime.now().strftime('%H:%M:%S')}"
        )
        self.log_text.append(f"Input files: {len(self.csv_files)}")
        self.log_text.append(f"Output: {self.output_path_edit.text()}")
        self.log_text.append(f"Combine files: {self.combine_checkbox.isChecked()}")

        if self.use_all_columns_checkbox.isChecked():
            self.log_text.append("Using all columns")
        else:
            self.log_text.append(f"Using {len(self.selected_columns)} selected columns")

        self.log_text.append("-" * 50)

        # Create and start worker thread
        self.worker_thread = ConversionWorker(
            csv_files=self.csv_files,
            output_path=self.output_path_edit.text(),
            combine_files=self.combine_checkbox.isChecked(),
            chunk_size=self.chunk_size_spin.value(),
            auto_convert=self.auto_convert_checkbox.isChecked(),
            selected_columns=(
                self.selected_columns
                if not self.use_all_columns_checkbox.isChecked()
                else None
            ),
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
                self,
                "Conversion Complete",
                f"Conversion completed successfully!\n\n{message}",
            )
        else:
            self.status_label.setText("Conversion failed!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.log_text.append(f"FAILED: {message}")

            QMessageBox.critical(
                self, "Conversion Failed", f"Conversion failed!\n\nError: {message}"
            )

    def _save_log(self):
        """Save the current log to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Log File", "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "Log Saved", f"Log saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV to Parquet Bulk Converter - Enhanced")
        self.setMinimumSize(1100, 800)

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
            self,
            "About CSV to Parquet Converter - Enhanced",
            """<h3>CSV to Parquet Bulk Converter - Enhanced</h3>
            <p>A PyQt6-based application for converting multiple CSV files to Parquet format with advanced column selection.</p>
            <p><b>Enhanced Features:</b></p>
            <ul>
                <li>Column selection with search and filtering</li>
                <li>Save/load column selection lists</li>
                <li>Efficient handling of large column sets (2000+ columns)</li>
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
            </ul>""",
        )


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("CSV to Parquet Converter - Enhanced")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Data Processor")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

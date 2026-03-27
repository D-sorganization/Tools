from numba import jit
# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Format Converter Tab and Logic for Data Processor."""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
import pandas as pd
from upstream_drift_tools.data_processing.io import (
    DataReader,
    DataWriter,
    FileFormatDetector,
)

from ..models.split_config import SplitConfig
from .column_selection_dialog import ColumnSelectionDialog
from .parquet_analyzer import ParquetAnalyzerDialog

logger = logging.getLogger(__name__)


class FormatConverterMixin:
    """Mixin containing UI and logic for the Format Converter tab."""

    # These would be initialized in the main app's __init__
    converter_input_files: list[str | Path]
    converter_output_path: str
    converter_selected_columns: set[str]
    converter_split_config: SplitConfig

    # UI elements (would be assigned during create_format_converter_tab)
    converter_input_label: ctk.CTkLabel
    converter_output_label: ctk.CTkLabel
    converter_columns_label: ctk.CTkLabel
    converter_format_var: ctk.StringVar
    converter_combine_var: ctk.BooleanVar
    converter_use_all_columns_var: ctk.BooleanVar
    converter_batch_var: ctk.BooleanVar
    converter_split_var: ctk.BooleanVar
    converter_convert_button: ctk.CTkButton
    converter_progress: ctk.CTkProgressBar
    converter_status_label: ctk.CTkLabel
    converter_log_text: ctk.CTkTextbox
    converter_file_list_frame: ctk.CTkScrollableFrame

    def converter_select_columns(self) -> None:
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

            dialog = ColumnSelectionDialog(self, columns)  # type: ignore
            if dialog.result:
                self.converter_selected_columns = set(dialog.result)
                self.converter_use_all_columns_var.set(False)
                self.converter_columns_label.configure(
                    text=f"{len(dialog.result)} columns selected"
                )
                self._log_conversion_message(
                    f"Selected {len(dialog.result)} columns: "
                    f"{', '.join(dialog.result[:5])}"
                    f"{'...' if len(dialog.result) > 5 else ''}"
                )

        except (RuntimeError, AttributeError) as e:
            messagebox.showerror("Error", f"Error reading file: {str(e)}")

    def create_format_converter_tab(self, parent_tab: ctk.CTkFrame) -> None:
        """Create the format converter tab UI."""
        if not (parent_tab is not None):
            raise ValueError("parent_tab must be provided")
        parent_tab.grid_columnconfigure(0, weight=1)
        parent_tab.grid_rowconfigure(0, weight=1)

        # Create the splitter (assumes self._create_splitter exists on the mixing class)
        splitter_frame = self._create_splitter(  # type: ignore
            parent_tab,
            self._create_converter_left_panel,
            self._create_converter_right_panel,
            "converter_left_width",
            400,
        )
        splitter_frame.grid(row=0, column=0, sticky="nsew")

    def _create_converter_left_panel(self, left_panel: ctk.CTkFrame) -> None:
        """Create the left panel content for the format converter tab."""
        if not (left_panel is not None):
            raise ValueError("left_panel must be provided")
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(0, weight=1)

        converter_scrollable_frame = ctk.CTkScrollableFrame(left_panel)
        converter_scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        converter_scrollable_frame.grid_columnconfigure(0, weight=1)

        self._create_input_section(converter_scrollable_frame)
        self._create_output_section(converter_scrollable_frame)
        self._create_options_section(converter_scrollable_frame)
        self._create_column_section(converter_scrollable_frame)

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
        self.converter_status_label.grid(row=6, column=0, sticky="w", padx=5, pady=5)

    def _create_input_section(self, parent: ctk.CTkFrame) -> None:
        """Create the input files section of the converter tab."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        input_frame = ctk.CTkFrame(parent)
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(input_frame, text="Select Input Files/Folder:").grid(
            row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w"
        )
        self.converter_input_label = ctk.CTkLabel(input_frame, text="No files selected")
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

    def _create_output_section(self, parent: ctk.CTkFrame) -> None:
        """Create the output format/path section of the converter tab."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        output_frame = ctk.CTkFrame(parent)
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
        self.converter_output_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ctk.CTkButton(
            output_frame, text="Browse Output", command=self.converter_browse_output
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def _create_options_section(self, parent: ctk.CTkFrame) -> None:
        """Create the options checkboxes section of the converter tab."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        options_frame = ctk.CTkFrame(parent)
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

    def _create_column_section(self, parent: ctk.CTkFrame) -> None:
        """Create the column selection section of the converter tab."""
        if not (parent is not None):
            raise ValueError("parent must be provided")
        column_frame = ctk.CTkFrame(parent)
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

    def _create_converter_right_panel(self, right_panel: ctk.CTkFrame) -> None:
        """Create the right panel content for the format converter tab."""
        if not (right_panel is not None):
            raise ValueError("right_panel must be provided")
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

        ctk.CTkLabel(log_frame, text="Conversion Log:").pack(anchor="w", padx=5, pady=2)
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

    def converter_browse_files(self) -> None:
        """Browse for input files."""
        try:
            files = filedialog.askopenfilenames(
                title="Select Input Files",
                filetypes=[
                    (
                        "All Supported",
                        "*.csv *.tsv *.txt *.parquet *.pq"
                        "*.xlsx *.xls *.json *.h5 *.hdf5"
                        "*.pkl *.pickle *.npy *.mat *.feather *.arrow *.db *.sqlite",
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
        except (KeyError, ValueError, TypeError) as e:
            messagebox.showerror("Error", f"Failed to browse files: {str(e)}")

    def converter_browse_folder(self) -> None:
        """Browse for input folder."""
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
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
            for root, _dirs, filenames in os.walk(folder):
                for filename in filenames:
                    if Path(filename).suffix.lower() in supported_extensions:
                        files.append(Path(root) / filename)

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

    def converter_clear_files(self) -> None:
        """Clear all selected files."""
        self.converter_input_files = []
        self.converter_update_file_list()
        self.converter_input_label.configure(text="No files selected")
        self.converter_update_convert_button()

    def converter_update_file_list(self) -> None:
        """Update the file list display."""
        for widget in self.converter_file_list_frame.winfo_children():
            widget.destroy()

        if not self.converter_input_files:
            ctk.CTkLabel(self.converter_file_list_frame, text="No files selected").pack(
                padx=5, pady=5
            )
            return

        for file_path in self.converter_input_files:
            file_frame = ctk.CTkFrame(self.converter_file_list_frame)
            file_frame.pack(fill="x", padx=5, pady=2)

            filename = Path(file_path).name
            if len(filename) > 40:
                filename = filename[:37] + "..."

            ctk.CTkLabel(file_frame, text=filename).pack(side="left", padx=5, pady=2)

            ctk.CTkButton(
                file_frame,
                text="X",
                width=30,
                command=lambda fp=file_path: self.converter_remove_file(fp),
            ).pack(side="right", padx=5, pady=2)

    def converter_remove_file(self, file_path: str | Path) -> None:
        """Remove a specific file from the list."""
        if file_path in self.converter_input_files:
            self.converter_input_files.remove(file_path)
            self.converter_update_file_list()
            self.converter_input_label.configure(
                text=f"{len(self.converter_input_files)} files selected"
            )
            self.converter_update_convert_button()

    def converter_browse_output(self) -> None:
        """Browse for output directory."""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.converter_output_path = folder
            self.converter_output_label.configure(text=folder)
            self.converter_update_convert_button()

    def converter_update_convert_button(self) -> None:
        """Update the convert button state."""
        if self.converter_input_files and self.converter_output_path:
            self.converter_convert_button.configure(state="normal")
        else:
            self.converter_convert_button.configure(state="disabled")

    def converter_start_conversion(self) -> None:
        """Start the file conversion process."""
        if not self.converter_input_files:
            messagebox.showwarning("No Files", "Please select input files first.")
            return

        if not self.converter_output_path:
            messagebox.showwarning("No Output", "Please select an output directory.")
            return

        output_format = self.converter_format_var.get()
        combine_files = self.converter_combine_var.get()
        use_all_columns = self.converter_use_all_columns_var.get()
        batch_processing = self.converter_batch_var.get()
        split_files = self.converter_split_var.get()

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
        output_format: str,
        combine_files: bool,
        use_all_columns: bool,
        batch_processing: bool,
        split_files: bool,
    ) -> None:
        """Perform the actual file conversion in a background thread.

        Contracts (DbC):
        - Precondition: output_format must not be empty.
        - Precondition: converter_input_files must not be empty.
        - Precondition: converter_output_path must be a valid directory string.
        """
        if not (output_format):
            raise ValueError("output_format cannot be empty")
        if not (self.converter_input_files):
            raise ValueError("No input files selected")
        if not (self.converter_output_path):
            raise ValueError("No output path selected")

        try:
            self.converter_status_label.configure(text="Converting files...")
            self.converter_progress.set(0)
            self.converter_convert_button.configure(state="disabled")

            total_files = len(self.converter_input_files)

            if combine_files:
                processed_files = self._convert_combined(
                    output_format, use_all_columns, total_files
                )
            else:
                processed_files = self._convert_individually(
                    output_format, use_all_columns, total_files
                )

            self.converter_status_label.configure(
                text=f"Conversion complete. {processed_files} files processed."
            )
            self.converter_progress.set(1.0)

        except (PermissionError, OSError) as e:
            self._log_conversion_message(f"Conversion error: {str(e)}")
            self.converter_status_label.configure(text="Conversion failed")
        finally:
            self.converter_convert_button.configure(state="normal")

    def _read_and_filter_file(
        self, file_path: str, use_all_columns: bool
    ) -> pd.DataFrame | None:
        """Read a file and optionally filter to selected columns.

        Returns None if the file cannot be read or has no matching columns.
        """
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        format_type = FileFormatDetector.detect_format(file_path)
        if not format_type:
            fname = Path(file_path).name
            self._log_conversion_message(
                f"Warning: Could not detect format for {fname}"
            )
            return None

        df = DataReader.read_file(file_path, format_type)

        if not use_all_columns and self.converter_selected_columns:
            available_columns = [
                col for col in self.converter_selected_columns if col in df.columns
            ]
            if available_columns:
                df = df[available_columns]
            else:
                fname = Path(file_path).name
                self._log_conversion_message(
                    f"Warning: No selected columns found in {fname}"
                )
                return None

        return df

    def _convert_combined(
        self, output_format: str, use_all_columns: bool, total_files: int
    ) -> int:
        """Combine all input files into a single output file."""
        if not (output_format is not None):
            raise ValueError("output_format must be provided")
        self._log_conversion_message(
            f"Starting conversion: combining {total_files} files into "
            f"{output_format.upper()}"
        )

        combined_data: list[pd.DataFrame] = []
        processed_files = 0

        for file_path in self.converter_input_files:
            try:
                df = self._read_and_filter_file(file_path, use_all_columns)
                if df is None:
                    continue

                combined_data.append(df)
                fname = Path(file_path).name
                self._log_conversion_message(
                    f"Loaded {fname}: {len(df)} rows, {len(df.columns)} columns"
                )
                processed_files += 1
                self.converter_progress.set(processed_files / total_files)
            except (PermissionError, OSError) as e:
                self._log_conversion_message(
                    f"Error reading {Path(file_path).name}: {str(e)}"
                )

        if combined_data:
            try:
                combined_df = pd.concat(combined_data, ignore_index=True)
                output_filename = self._generate_output_filename(
                    output_format, "combined_data"
                )
                output_path = Path(self.converter_output_path) / output_filename
                DataWriter.write_file(combined_df, output_path, output_format)
                self._log_conversion_message(f"Successfully created: {output_filename}")
                self._log_conversion_message(
                    f"Combined data: {len(combined_df)} rows, "
                    f"{len(combined_df.columns)} columns"
                )
            except (PermissionError, OSError) as e:
                self._log_conversion_message(f"Error writing combined file: {str(e)}")
        else:
            self._log_conversion_message("No valid data to combine")

        return processed_files

    @jit(nopython=True, fastmath=True)
    def _convert_individually(
        self, output_format: str, use_all_columns: bool, total_files: int
    ) -> int:
        """Convert each input file to the output format separately."""
        if not (output_format is not None):
            raise ValueError("output_format must be provided")
        self._log_conversion_message(
            f"Starting conversion: processing {total_files} files individually"
        )

        processed_files = 0
        for file_path in self.converter_input_files:
            try:
                df = self._read_and_filter_file(file_path, use_all_columns)
                if df is None:
                    continue

                base_name = Path(file_path).stem
                output_filename = self._generate_output_filename(
                    output_format, base_name
                )
                output_path = Path(self.converter_output_path) / output_filename

                DataWriter.write_file(df, output_path, output_format)
                self._log_conversion_message(
                    f"Converted {Path(file_path).name} -> {output_filename}"
                )
                processed_files += 1
                self.converter_progress.set(processed_files / total_files)
            except (PermissionError, OSError) as e:
                self._log_conversion_message(
                    f"Error converting {Path(file_path).name}: {str(e)}"
                )

        return processed_files

    def _generate_output_filename(
        self, output_format: str, base_name: str | None = None
    ) -> str:
        """Generate output filename with proper extension."""
        if not (output_format is not None):
            raise ValueError("output_format must be provided")
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

    def _log_conversion_message(self, message: str) -> None:
        """Add a message to the conversion log."""
        if not (message is not None):
            raise ValueError("message must be provided")
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        # self.after would be available on the class mixing in this one (ctk.CTkFrame)
        self.after(0, lambda: self.converter_log_text.insert("end", log_message))  # type: ignore
        self.after(0, lambda: self.converter_log_text.see("end"))  # type: ignore

    def converter_clear_log(self) -> None:
        """Clear the conversion log."""
        self.converter_log_text.delete("1.0", "end")

    def converter_save_log(self) -> None:
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
                    from utils.file_utils import safe_write_text

                    safe_write_text(file_path, log_content)
                    messagebox.showinfo("Success", f"Log saved to {file_path}")
                except OSError as e:
                    messagebox.showerror("Error", f"Failed to save log: {str(e)}")

    def show_parquet_analyzer(self) -> None:
        """Show the parquet analyzer dialog."""
        dialog = ParquetAnalyzerDialog(self)  # type: ignore
        dialog.grab_set()

    # Stub for method expected on self
    def after(self, ms: int, func: Any) -> None: ...

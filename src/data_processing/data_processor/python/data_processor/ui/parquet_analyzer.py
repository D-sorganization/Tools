"""Dialog for analyzing parquet file metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

try:
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ParquetAnalyzerDialog(ctk.CTkToplevel):
    """Dialog for analyzing parquet file metadata."""

    def __init__(self, parent: ctk.CTk | ctk.CTkToplevel | None = None) -> None:
        super().__init__(parent)
        self.title("Parquet File Analyzer")
        self.geometry("600x500")
        self.resizable(True, True)

        # Make dialog modal
        if parent:
            self.transient(parent)
        self.grab_set()

        self.setup_ui()

    def setup_ui(self) -> None:
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

    def select_file(self) -> None:
        """Open file dialog to select a parquet file."""
        file_path = filedialog.askopenfilename(
            title="Select Parquet File",
            filetypes=[("Parquet Files", "*.parquet *.pq"), ("All Files", "*.*")],
        )

        if file_path:
            self.analyze_parquet_file(file_path)

    def analyze_parquet_file(self, file_path: str) -> None:
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
            result_lines = [
                "=== Parquet File Analysis ===",
                f"File: {Path(file_path).name}",
                f"Path: {file_path}",
                f"Size: {self.format_file_size(file_size)}",
                "",
                "=== Metadata ===",
                f"Rows: {parquet_file.metadata.num_rows:,}",
                f"Columns: {parquet_file.metadata.num_columns}",
                f"Row Groups: {parquet_file.metadata.num_row_groups}",
                "",
                "=== Schema ===",
            ]

            schema = parquet_file.schema_arrow
            result_lines.extend([f"{field.name}: {field.type}" for field in schema])

            result_lines.append("")
            result_lines.append("=== Row Group Details ===")

            for i in range(parquet_file.metadata.num_row_groups):
                row_group = parquet_file.metadata.row_group(i)
                result_lines.extend(
                    [
                        f"Row Group {i}:",
                        f"  Rows: {row_group.num_rows:,}",
                        f"  Size: {self.format_file_size(row_group.total_byte_size)}",
                        f"  Columns: {row_group.num_columns}",
                    ]
                )

                # Column details
                for j in range(row_group.num_columns):
                    col = row_group.column(j)
                    result_lines.extend(
                        [
                            f"    Column {j}: {col.path_in_schema}",
                            f"      Values: {col.num_values:,}",
                            (
                                f"      Size: "
                                f"{self.format_file_size(col.total_uncompressed_size)}"
                            ),
                            (
                                f"      Compressed: "
                                f"{self.format_file_size(col.total_compressed_size)}"
                            ),
                        ]
                    )

                    if col.statistics:
                        stats = col.statistics
                        if stats.has_min_max:
                            result_lines.extend(
                                [f"      Min: {stats.min}", f"      Max: {stats.max}"]
                            )
                    result_lines.append("")

            results = "\n".join(result_lines)

            # Clear and insert results
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", results)

        except (PermissionError, OSError) as e:
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Error analyzing file: {str(e)}")

    def format_file_size(self, size_bytes: float) -> str:
        """Format file size in human readable format."""
        if not (size_bytes is not None):
            raise ValueError("size_bytes must be provided")
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

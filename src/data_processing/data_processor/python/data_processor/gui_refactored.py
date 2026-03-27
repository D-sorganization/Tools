# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

# mypy: ignore-errors
"""GUI Application using refactored core modules.

This is a refactored version of the Data Processor GUI that uses the
core business logic modules instead of implementing everything inline.

Benefits:
- Thin GUI layer focused on presentation
- Business logic testable independently
- Easier to maintain and extend
- Reuses battle-tested core modules
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tkinter import TclError, filedialog, messagebox
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

# Import core business logic
from .core.data_loader import DataLoader
from .core.signal_processor import SignalProcessor

# Import utilities
from .logging_config import get_logger
from .models.processing_config import (
    DifferentiationConfig,
    FilterConfig,
    IntegrationConfig,
)

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class DataProcessorGUI(ctk.CTk):  # type: ignore
    """Main GUI application using refactored core modules."""

    def __init__(self) -> None:
        """Initialize the Data Processor GUI."""
        super().__init__()

        # Initialize core modules
        self.data_loader = DataLoader(use_high_performance=True)
        self.signal_processor = SignalProcessor()

        # Application state
        self.selected_files: list[str] = []
        self.current_data: pd.DataFrame | None = None
        self.available_signals: list[str] = []
        self.selected_signals: list[str] = []

        # Filter configuration
        self.filter_config = FilterConfig()

        # Window configuration
        self.title("Data Processor - Refactored GUI")
        self.geometry("1200x800")

        # Create UI
        self.create_ui()
        self.setup_shortcuts()

        logger.info("Data Processor GUI initialized")

    @contextmanager
    def busy_cursor(self) -> Generator[None, None, None]:
        """Context manager to show busy cursor during long operations."""
        try:
            previous_cursor = self.cget("cursor")
        except TclError:
            previous_cursor = "arrow"

        try:
            self._set_cursor_recursive(self, "watch")
            self.update_idletasks()
            yield
        finally:
            self._set_cursor_recursive(self, previous_cursor)
            self.update_idletasks()

    def _set_cursor_recursive(self, widget: Any, cursor: str) -> None:
        """Set cursor recursively for widget and all children."""
        if not (cursor is not None):
            raise ValueError("cursor must be provided")
        try:
            widget.configure(cursor=cursor)
        except (TclError, AttributeError):
            # Ignore errors if widget doesn't support cursor or is destroyed
            pass

        try:
            for child in widget.winfo_children():
                self._set_cursor_recursive(child, cursor)
        except (TclError, AttributeError):
            pass

    def create_ui(self) -> None:
        """Create the user interface."""
        # Main container with padding
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Data Processor - Refactored Architecture",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title_label.pack(pady=(0, 20))

        # Create tab view
        self.tab_view = ctk.CTkTabview(main_frame)
        self.tab_view.pack(fill="both", expand=True)

        # Create tabs
        self.tab_view.add("File Selection")
        self.tab_view.add("Signal Processing")
        self.tab_view.add("Advanced Operations")
        self.tab_view.add("Export")

        # Populate tabs
        self.create_file_selection_tab()
        self.create_signal_processing_tab()
        self.create_advanced_operations_tab()
        self.create_export_tab()

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready",
            font=ctk.CTkFont(size=12),
        )
        self.status_label.pack(side="bottom", pady=(10, 0))

    def create_file_selection_tab(self) -> None:
        """Create the file selection tab."""
        tab = self.tab_view.tab("File Selection")

        # File selection section
        file_frame = ctk.CTkFrame(tab)
        file_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            file_frame,
            text="Select CSV Files",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(10, 10))

        # Buttons
        button_frame = ctk.CTkFrame(file_frame)
        button_frame.pack(pady=10)

        ctk.CTkButton(
            button_frame,
            text="Select Files (Ctrl+O)",
            command=self.select_files,
            width=180,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Clear Selection",
            command=self.clear_files,
            width=150,
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Load Data (Ctrl+L)",
            command=self.load_data,
            width=180,
        ).pack(side="left", padx=5)

        # File list
        self.file_listbox = ctk.CTkTextbox(file_frame, height=200)
        self.file_listbox.pack(fill="both", expand=True, padx=10, pady=10)

        # Signal detection section
        signal_frame = ctk.CTkFrame(tab)
        signal_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            signal_frame,
            text="Detected Signals",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(10, 10))

        ctk.CTkButton(
            signal_frame,
            text="Detect Signals",
            command=self.detect_signals,
            width=200,
        ).pack(pady=5)

        self.signal_listbox = ctk.CTkTextbox(signal_frame, height=200)
        self.signal_listbox.pack(fill="both", expand=True, padx=10, pady=10)

    def create_signal_processing_tab(self) -> None:
        """Create the signal processing tab."""
        tab = self.tab_view.tab("Signal Processing")

        # Filter selection
        filter_frame = ctk.CTkFrame(tab)
        filter_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            filter_frame,
            text="Filter Type",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=5)

        self.filter_type_var = ctk.StringVar(value="Moving Average")
        filter_types = [
            "Moving Average",
            "Butterworth Low-pass",
            "Butterworth High-pass",
            "Median Filter",
            "Gaussian Filter",
            "Hampel Filter",
            "Z-Score Filter",
            "Savitzky-Golay",
            "FFT Low-pass",
            "FFT High-pass",
        ]

        self.filter_menu = ctk.CTkOptionMenu(
            filter_frame,
            variable=self.filter_type_var,
            values=filter_types,
            command=self.on_filter_type_changed,
        )
        self.filter_menu.pack(pady=5)

        # Filter parameters
        self.param_frame = ctk.CTkFrame(tab)
        self.param_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.create_filter_parameters()

        # Apply button
        ctk.CTkButton(
            tab,
            text="Apply Filter",
            command=self.apply_filter,
            width=200,
            height=40,
        ).pack(pady=20)

    def create_filter_parameters(self) -> None:
        """Create filter parameter controls."""
        # Clear existing parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        filter_type = self.filter_type_var.get()

        if "Moving Average" in filter_type:
            ctk.CTkLabel(self.param_frame, text="Window Size:").pack(pady=5)
            self.ma_window_entry = ctk.CTkEntry(self.param_frame)
            self.ma_window_entry.insert(0, "10")
            self.ma_window_entry.pack(pady=5)

        elif "Butterworth" in filter_type:
            ctk.CTkLabel(self.param_frame, text="Filter Order:").pack(pady=5)
            self.bw_order_entry = ctk.CTkEntry(self.param_frame)
            self.bw_order_entry.insert(0, "3")
            self.bw_order_entry.pack(pady=5)

            ctk.CTkLabel(self.param_frame, text="Cutoff Frequency:").pack(pady=5)
            self.bw_cutoff_entry = ctk.CTkEntry(self.param_frame)
            self.bw_cutoff_entry.insert(0, "0.1")
            self.bw_cutoff_entry.pack(pady=5)

        elif "Median" in filter_type:
            ctk.CTkLabel(self.param_frame, text="Kernel Size:").pack(pady=5)
            self.median_kernel_entry = ctk.CTkEntry(self.param_frame)
            self.median_kernel_entry.insert(0, "5")
            self.median_kernel_entry.pack(pady=5)

        elif "Gaussian" in filter_type:
            ctk.CTkLabel(self.param_frame, text="Sigma:").pack(pady=5)
            self.gaussian_sigma_entry = ctk.CTkEntry(self.param_frame)
            self.gaussian_sigma_entry.insert(0, "1.0")
            self.gaussian_sigma_entry.pack(pady=5)

    def create_advanced_operations_tab(self) -> None:
        """Create the advanced operations tab."""
        tab = self.tab_view.tab("Advanced Operations")

        # Integration section
        int_frame = ctk.CTkFrame(tab)
        int_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            int_frame,
            text="Integration",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=5)

        ctk.CTkButton(
            int_frame,
            text="Integrate Selected Signals",
            command=self.integrate_signals,
        ).pack(pady=5)

        # Differentiation section
        diff_frame = ctk.CTkFrame(tab)
        diff_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            diff_frame,
            text="Differentiation",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=5)

        ctk.CTkButton(
            diff_frame,
            text="Differentiate Selected Signals",
            command=self.differentiate_signals,
        ).pack(pady=5)

        # Custom formula section
        formula_frame = ctk.CTkFrame(tab)
        formula_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            formula_frame,
            text="Custom Formula",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=5)

        ctk.CTkLabel(formula_frame, text="Signal Name:").pack(pady=5)
        self.formula_name_entry = ctk.CTkEntry(formula_frame, width=300)
        self.formula_name_entry.pack(pady=5)

        ctk.CTkLabel(formula_frame, text="Formula (e.g., signal1 + signal2):").pack(
            pady=5,
        )
        self.formula_entry = ctk.CTkEntry(formula_frame, width=300)
        self.formula_entry.pack(pady=5)

        ctk.CTkButton(
            formula_frame,
            text="Apply Custom Formula",
            command=self.apply_custom_formula,
        ).pack(pady=10)

    def create_export_tab(self) -> None:
        """Create the export tab."""
        tab = self.tab_view.tab("Export")

        # Export options
        export_frame = ctk.CTkFrame(tab)
        export_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            export_frame,
            text="Export Processed Data",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=20)

        # Format selection
        ctk.CTkLabel(export_frame, text="Export Format:").pack(pady=5)

        self.export_format_var = ctk.StringVar(value="csv")
        formats = ["csv", "excel", "parquet", "hdf5", "feather"]

        ctk.CTkOptionMenu(
            export_frame,
            variable=self.export_format_var,
            values=formats,
        ).pack(pady=5)

        # Export button
        ctk.CTkButton(
            export_frame,
            text="Export Data (Ctrl+S)",
            command=self.export_data,
            width=200,
            height=40,
        ).pack(pady=20)

        # Statistics section
        stats_frame = ctk.CTkFrame(tab)
        stats_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            stats_frame,
            text="Data Statistics",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(pady=5)

        self.stats_textbox = ctk.CTkTextbox(stats_frame, height=200)
        self.stats_textbox.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkButton(
            stats_frame,
            text="Calculate Statistics",
            command=self.calculate_statistics,
        ).pack(pady=10)

    def setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        self.bind("<Control-o>", lambda event: self.select_files())
        self.bind("<Control-l>", lambda event: self.load_data())
        self.bind("<Control-s>", lambda event: self.export_data())
        self.bind("<Control-q>", lambda event: self.quit())
        logger.info("Keyboard shortcuts initialized")

    # Event handlers

    def on_filter_type_changed(self, choice: str) -> None:
        """Handle filter type change."""
        self.create_filter_parameters()

    def select_files(self) -> None:
        """Select CSV files."""
        files = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )

        if files:
            self.selected_files = list(files)
            self.update_file_list()
            self.update_status(f"Selected {len(files)} files")
            logger.info(f"Selected {len(files)} files")

    def clear_files(self) -> None:
        """Clear file selection."""
        if self.selected_files and not messagebox.askyesno(
            "Confirm Clear",
            "Are you sure you want to clear the file selection?",
        ):
            return

        self.selected_files = []
        self.update_file_list()
        self.update_status("File selection cleared")

    def update_file_list(self) -> None:
        """Update the file list display."""
        self.file_listbox.delete("1.0", "end")
        for file_path in self.selected_files:
            self.file_listbox.insert("end", f"{Path(file_path).name}\n")

    def load_data(self) -> None:
        """Load data from selected files using core DataLoader."""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select files first")
            return

        try:
            self.update_status("Loading data...")
            with self.busy_cursor():
                # Use core data loader module
                if len(self.selected_files) == 1:
                    self.current_data = self.data_loader.load_csv_file(
                        self.selected_files[0],
                    )
                else:
                    # Load multiple files and combine
                    dataframes = self.data_loader.load_multiple_files(
                        self.selected_files
                    )
                    self.current_data = self.data_loader.combine_dataframes(dataframes)

                if self.current_data is not None:
                    # Detect and convert time column
                    time_col = self.data_loader.detect_time_column(self.current_data)
                    if time_col:
                        self.current_data = self.data_loader.convert_time_column(
                            self.current_data,
                            time_col,
                        )

                    # Get numeric signals
                    self.available_signals = self.data_loader.get_numeric_signals(
                        self.current_data,
                    )

            if self.current_data is not None:
                # Check needed outside busy_cursor if current_data is None
                self.update_status(
                    f"Loaded {len(self.current_data)} rows, "
                    f"{len(self.available_signals)} signals",
                )

                messagebox.showinfo(
                    "Success",
                    f"Loaded {len(self.current_data)} rows\n"
                    f"{len(self.available_signals)} signals detected",
                )

                logger.info(f"Loaded data: {len(self.current_data)} rows")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load data:\n{e}")

    def detect_signals(self) -> None:
        """Detect signals from selected files using core DataLoader."""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select files first")
            return

        try:
            self.update_status("Detecting signals...")
            signals: list[str] | set[str] = []
            with self.busy_cursor():
                # Use core data loader for signal detection
                signals = self.data_loader.detect_signals(self.selected_files)

            self.signal_listbox.delete("1.0", "end")
            for signal in sorted(signals):
                self.signal_listbox.insert("end", f"{signal}\n")

            self.update_status(f"Detected {len(signals)} unique signals")

            logger.info(f"Detected {len(signals)} signals")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error detecting signals: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to detect signals:\n{e}")

    def apply_filter(self) -> None:
        """Apply filter to current data using core SignalProcessor."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        try:
            self.update_status("Applying filter...")

            with self.busy_cursor():
                # Build filter configuration from UI
                filter_type = self.filter_type_var.get()

                if "Moving Average" in filter_type:
                    window = int(self.ma_window_entry.get())
                    self.filter_config = FilterConfig(
                        filter_type=filter_type,
                        parameters={"ma_window": window},
                    )

                elif "Butterworth" in filter_type:
                    order = int(self.bw_order_entry.get())
                    cutoff = float(self.bw_cutoff_entry.get())
                    self.filter_config = FilterConfig(
                        filter_type=filter_type,
                        parameters={"bw_order": order, "bw_cutoff": cutoff},
                    )

                elif "Median" in filter_type:
                    kernel = int(self.median_kernel_entry.get())
                    self.filter_config = FilterConfig(
                        filter_type=filter_type,
                        parameters={"median_kernel": kernel},
                    )

                elif "Gaussian" in filter_type:
                    sigma = float(self.gaussian_sigma_entry.get())
                    self.filter_config = FilterConfig(
                        filter_type=filter_type,
                        parameters={"gaussian_sigma": sigma},
                    )

                # Apply filter using core signal processor
                self.current_data = self.signal_processor.apply_filter(
                    self.current_data,
                    self.filter_config,
                )

            self.update_status(f"Applied {filter_type} filter")
            messagebox.showinfo(
                "Success",
                f"{filter_type} filter applied successfully",
            )
            logger.info(f"Applied filter: {filter_type}")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.error(f"Error applying filter: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to apply filter:\n{e}")

    def integrate_signals(self) -> None:
        """Integrate signals using core SignalProcessor."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        try:
            self.update_status("Integrating signals...")
            with self.busy_cursor():
                # Use core signal processor for integration
                int_config = IntegrationConfig(
                    signals=self.available_signals,
                    method="cumulative",
                )

                self.current_data = self.signal_processor.integrate_signals(
                    self.current_data,
                    int_config,
                )

            self.update_status("Integration applied")
            messagebox.showinfo("Success", "Signals integrated successfully")

            logger.info("Applied integration")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error integrating signals: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to integrate:\n{e}")

    def differentiate_signals(self) -> None:
        """Differentiate signals using core SignalProcessor."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        try:
            self.update_status("Differentiating signals...")
            with self.busy_cursor():
                # Use core signal processor for differentiation
                diff_config = DifferentiationConfig(
                    signals=self.available_signals,
                    order=1,
                    method="central",
                )

                self.current_data = self.signal_processor.differentiate_signals(
                    self.current_data,
                    diff_config,
                )

            self.update_status("Differentiation applied")
            messagebox.showinfo("Success", "Signals differentiated successfully")

            logger.info("Applied differentiation")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error differentiating signals: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to differentiate:\n{e}")

    def apply_custom_formula(self) -> None:
        """Apply custom formula using core SignalProcessor."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        formula_name = self.formula_name_entry.get().strip()
        formula = self.formula_entry.get().strip()

        if not formula_name or not formula:
            messagebox.showwarning(
                "Invalid Input",
                "Please provide both name and formula",
            )
            return

        try:
            self.update_status(f"Applying formula: {formula_name}...")

            success = False
            with self.busy_cursor():
                # Use core signal processor for custom formula
                self.current_data, success = self.signal_processor.apply_custom_formula(
                    self.current_data,
                    formula_name,
                    formula,
                )

            if success:
                self.update_status(f"Created signal: {formula_name}")
                messagebox.showinfo(
                    "Success",
                    f"Signal '{formula_name}' created successfully",
                )
                logger.info(f"Applied custom formula: {formula_name} = {formula}")
            else:
                self.update_status("Failed to apply formula")
                messagebox.showerror("Error", "Failed to apply formula")

        except (ValueError, TypeError, KeyError, ArithmeticError) as e:
            logger.error(f"Error applying formula: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to apply formula:\n{e}")

    def calculate_statistics(self) -> None:
        """Calculate statistics using core SignalProcessor."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load data first")
            return

        try:
            self.update_status("Calculating statistics...")
            with self.busy_cursor():
                self.stats_textbox.delete("1.0", "end")
                self.stats_textbox.insert("end", "=== Signal Statistics ===\n\n")

                for signal in self.available_signals[:10]:  # Limit to first 10
                    # Pass only the specific signal column as a DataFrame
                    stats = self.signal_processor.detect_signal_statistics(
                        self.current_data[[signal]]
                    )

                    self.stats_textbox.insert("end", f"{signal}:\n")
                    self.stats_textbox.insert("end", f"  Mean: {stats['mean']:.2f}\n")
                    self.stats_textbox.insert("end", f"  Std: {stats['std']:.2f}\n")
                    self.stats_textbox.insert("end", f"  Min: {stats['min']:.2f}\n")
                    self.stats_textbox.insert("end", f"  Max: {stats['max']:.2f}\n")
                    self.stats_textbox.insert(
                        "end",
                        f"  Median: {stats['median']:.2f}\n\n",
                    )

            self.update_status("Statistics calculated")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error calculating statistics: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to calculate statistics:\n{e}")

    def export_data(self) -> None:
        """Export data using core DataLoader."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load and process data first")
            return

        format_type = self.export_format_var.get()

        # Get save filename
        file_types = {
            "csv": [("CSV Files", "*.csv")],
            "excel": [("Excel Files", "*.xlsx")],
            "parquet": [("Parquet Files", "*.parquet")],
            "hdf5": [("HDF5 Files", "*.h5")],
            "feather": [("Feather Files", "*.feather")],
        }

        filename = filedialog.asksaveasfilename(
            title="Save Processed Data",
            filetypes=file_types.get(format_type, [("All Files", "*.*")]),
        )

        if not filename:
            return

        try:
            self.update_status("Exporting data...")
            success = False

            success = False
            with self.busy_cursor():
                # Use core data loader for export
                success = self.data_loader.save_dataframe(
                    self.current_data,
                    filename,
                    format_type=format_type,
                )

            if success:
                self.update_status(f"Data exported to {Path(filename).name}")
                messagebox.showinfo(
                    "Success",
                    f"Data exported successfully to:\n{filename}",
                )
                logger.info(f"Exported data to {filename}")
            else:
                self.update_status("Export failed")
                messagebox.showerror("Error", "Failed to export data")

        except (PermissionError, OSError) as e:
            logger.error(f"Error exporting data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to export data:\n{e}")

    def update_status(self, message: str) -> None:
        """Update status bar."""
        if not (message is not None):
            raise ValueError("message must be provided")
        self.status_label.configure(text=message)
        logger.debug(f"Status: {message}")


def main() -> None:
    """Run the refactored GUI application."""
    app = DataProcessorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

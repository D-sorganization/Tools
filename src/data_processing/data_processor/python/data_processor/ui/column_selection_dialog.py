from typing import Any

"""Column Selection Dialog for Data Processor."""

from __future__ import annotations

from tkinter import messagebox
from typing import Any

import customtkinter as ctk


class ColumnSelectionDialog(ctk.CTkToplevel):
    """Simple dialog for column selection."""

    def __init__(self, parent, columns):
        if not (parent is not None):
            raise ValueError("parent must be provided")
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

    def setup_ui(self) -> Any:
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

    def select_all(self) -> None:
        """Select all columns."""
        for var in self.column_vars.values():
            var.set(True)

    def select_none(self) -> None:
        """Select no columns."""
        for var in self.column_vars.values():
            var.set(False)

    def ok_clicked(self) -> None:
        """Handle OK button click."""
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if not selected_columns:
            messagebox.showwarning("Warning", "Please select at least one column.")
            return

        self.result = selected_columns
        self.destroy()

    def cancel_clicked(self) -> None:
        """Handle Cancel button click."""
        self.result = None
        self.destroy()

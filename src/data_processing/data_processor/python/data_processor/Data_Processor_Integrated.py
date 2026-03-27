from typing import Any

# mypy: ignore-errors
"""Integrated version of the Data Processor GUI.

This module combines the various refactored components into a single application.
"""

import logging
from typing import Any

import customtkinter as ctk

# Import models
from .models.split_config import SplitConfig
from .ui.folder_tool_tab import FolderToolMixin

# Import mixins
from .ui.format_converter_tab import FormatConverterMixin
from .ui.help_tab import HelpTabMixin

# Import base class
try:
    from Data_Processor_r0 import (
        CSVProcessorApp as OriginalCSVProcessorApp,
    )
except ImportError:
    # Use refactored GUI as base if legacy base is unavailable
    try:
        from .gui_refactored import DataProcessorGUI as OriginalCSVProcessorApp
    except ImportError:
        # Final fallback to a basic ctk.CTk if everything else fails
        OriginalCSVProcessorApp = ctk.CTk

# Folder tool availability flag
FOLDER_TOOL_AVAILABLE = True

logger = logging.getLogger(__name__)


class IntegratedCSVProcessorApp(
    OriginalCSVProcessorApp, FormatConverterMixin, FolderToolMixin, HelpTabMixin
):
    """Extended application class with integrated compiler converter functionality."""

    def __init__(self, *args, **kwargs):
        # Initialize converter variables BEFORE calling parent class
        self.converter_input_files = []
        self.converter_output_path = ""
        self.converter_format_var = ctk.StringVar(value="parquet")
        self.converter_combine_var = ctk.BooleanVar(value=True)
        self.converter_use_all_columns_var = ctk.BooleanVar(value=True)
        self.converter_batch_var = ctk.BooleanVar(value=False)
        self.converter_split_var = ctk.BooleanVar(value=False)
        self.converter_selected_columns = set()
        self.converter_split_config = SplitConfig()

        # Initialize folder tool variables
        self.folder_source_folders = []
        self.folder_destination = ""
        self.folder_cancel_flag = False

        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Folder tool Tkinter variables
        self.folder_operation_mode = ctk.StringVar(value="combine")
        self.folder_filter_extensions = ctk.StringVar(value="")
        self.folder_min_file_size = ctk.StringVar(value="0")
        self.folder_max_file_size = ctk.StringVar(value="1000")
        self.folder_organize_by_type_var = ctk.BooleanVar(value=False)
        self.folder_organize_by_date_var = ctk.BooleanVar(value=False)
        self.folder_preview_mode_var = ctk.BooleanVar(value=False)
        self.folder_status_var = ctk.StringVar(value="Ready")

        self.title("Advanced CSV Time Series Processor & Analyzer - Integrated")

        # Refine tab ordering
        # Original tabs are created in OriginalCSVProcessorApp.create_ui()
        if hasattr(self, "main_tab_view"):
            # Remove and re-add Help to keep it at the end
            if "Help" in self.main_tab_view._tab_dict:
                self.main_tab_view.delete("Help")

            # Move/Remove DAT File Import to reorder
            dat_import_exists = "DAT File Import" in self.main_tab_view._tab_dict
            if dat_import_exists:
                self.main_tab_view.delete("DAT File Import")

            # Add Format Converter tab
            self.main_tab_view.add("Format Converter")
            self.create_format_converter_tab(self.main_tab_view.tab("Format Converter"))

            # Add DAT File Import tab back
            if dat_import_exists:
                self.main_tab_view.add("DAT File Import")
                if hasattr(self, "create_dat_import_tab"):
                    self.create_dat_import_tab(
                        self.main_tab_view.tab("DAT File Import")
                    )

            # Add Folder Tool tab
            if FOLDER_TOOL_AVAILABLE:
                self.main_tab_view.add("Folder Tool")
                self.create_folder_tool_tab(self.main_tab_view.tab("Folder Tool"))

            # Add Help tab
            self.main_tab_view.add("Help")
            self.create_help_tab(self.main_tab_view.tab("Help"))

        logger.info("Integrated CSV Processor App initialized")

    def _create_splitter(
        self, parent, left_func, right_func, width_attr, default_width
    ) -> Any:
        """Helper to create a split layout with two panels.

        This implementation replaces the missing original method.
        """
        if not (parent is not None):
            raise ValueError("parent must be provided")
        frame = ctk.CTkFrame(parent)
        frame.grid_columnconfigure(0, weight=0)  # Left panel doesn't expand
        frame.grid_columnconfigure(1, weight=1)  # Right panel expands
        frame.grid_rowconfigure(0, weight=1)

        left_panel = ctk.CTkFrame(frame, width=default_width)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_panel.grid_propagate(False)  # Keep fixed width

        right_panel = ctk.CTkFrame(frame)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        left_func(left_panel)
        right_func(right_panel)

        return frame


if __name__ == "__main__":
    app = IntegratedCSVProcessorApp()
    app.mainloop()

#!/usr/bin/env python3
"""
Fix type annotation issues in Data_Processor_r0.py
"""

import re
from pathlib import Path


def fix_type_annotations(file_path: str) -> bool:
    """Fix type annotation issues"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    changes_made = False

    # Fix the main type annotation issues
    replacements = [
        # Fix the plot_signal_vars type annotation - it should be dict not BooleanVar
        (
            "self.plot_signal_vars: dict[str, tk.BooleanVar] = {}",
            "self.plot_signal_vars: dict[str, tk.BooleanVar] = {}",
        ),
        # Fix function signatures that are missing type annotations
        (
            "def _filter_signals(self, event):",
            "def _filter_signals(self, event: tk.Event[tk.Misc]) -> None:",
        ),
        ("def _clear_search(self):", "def _clear_search(self) -> None:"),
        ("def _on_bulk_mode_change(self):", "def _on_bulk_mode_change(self) -> None:"),
        (
            "def _on_dataset_naming_change(self):",
            "def _on_dataset_naming_change(self) -> None:",
        ),
        (
            "def _on_window_configure(self, event):",
            "def _on_window_configure(self, event: tk.Event[tk.Misc]) -> None:",
        ),
        # Fix Event type parameters
        (
            "def _on_canvas_configure(self, event):",
            "def _on_canvas_configure(self, event: tk.Event[tk.Misc]) -> None:",
        ),
        (
            "def _on_mousewheel(self, event):",
            "def _on_mousewheel(self, event: tk.Event[tk.Misc]) -> None:",
        ),
        (
            "def _on_plot_canvas_configure(self, event):",
            "def _on_plot_canvas_configure(self, event: tk.Event[tk.Misc]) -> None:",
        ),
        (
            "def _on_plot_mousewheel(self, event):",
            "def _on_plot_mousewheel(self, event: tk.Event[tk.Misc]) -> None:",
        ),
        (
            "def _on_plots_list_select(self, event):",
            "def _on_plots_list_select(self, event: tk.Event[tk.Misc]) -> None:",
        ),
        # Fix missing return type annotations
        ("def get_deriv(w: pd.Series):", "def get_deriv(w: pd.Series) -> float:"),
    ]

    for old, new in replacements:
        if old in content and new not in content:
            content = content.replace(old, new)
            changes_made = True

    # Fix BooleanVar indexing issues - these should use .get() method
    # Pattern: self.signal_vars[signal] -> self.signal_vars[signal].get()
    boolean_var_patterns = [
        (
            r"self\.signal_vars\[([^\]]+)\]\.set\(True\)",
            r"self.signal_vars[\1].set(True)",
        ),  # Keep .set() calls
        (
            r"self\.signal_vars\[([^\]]+)\]\.set\(False\)",
            r"self.signal_vars[\1].set(False)",
        ),  # Keep .set() calls
        (
            r"if self\.signal_vars\[([^\]]+)\]:",
            r"if self.signal_vars[\1].get():",
        ),  # Fix boolean checks
        (
            r"and self\.signal_vars\[([^\]]+)\]",
            r"and self.signal_vars[\1].get()",
        ),  # Fix boolean checks in and
        (
            r"self\.plot_signal_vars\[([^\]]+)\]\.set\(True\)",
            r"self.plot_signal_vars[\1].set(True)",
        ),  # Keep .set() calls
        (
            r"self\.plot_signal_vars\[([^\]]+)\]\.set\(False\)",
            r"self.plot_signal_vars[\1].set(False)",
        ),  # Keep .set() calls
        (
            r"if self\.plot_signal_vars\[([^\]]+)\]:",
            r"if self.plot_signal_vars[\1].get():",
        ),  # Fix boolean checks
        (
            r"and self\.plot_signal_vars\[([^\]]+)\]",
            r"and self.plot_signal_vars[\1].get()",
        ),  # Fix boolean checks in and
    ]

    for pattern, replacement in boolean_var_patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            changes_made = True

    # Fix some specific problematic lines
    specific_fixes = [
        # Fix the duplicate attribute definitions
        (
            "        self.plot_signal_vars: dict[str, tk.BooleanVar] = {}\n        # Initialize plot signal variables (will be populated when file is selected in plotting tab)\n        self.plot_signal_vars = {}",
            "        # Initialize plot signal variables (will be populated when file is selected in plotting tab)\n        self.plot_signal_vars: dict[str, tk.BooleanVar] = {}",
        ),
        # Fix the custom_legend_entries duplicate
        (
            "        self.custom_legend_entries: dict[str, str] = {}\n        # Custom legend entries for plots\n        self.custom_legend_entries = {}",
            "        # Custom legend entries for plots\n        self.custom_legend_entries: dict[str, str] = {}",
        ),
        # Fix the plots_list duplicate
        (
            '        self.plots_list: list[dict[str, Any]] = []\n        # Signal List Management variables\n        self.saved_signal_list: list[str] = []\n        self.saved_signal_list_name = ""\n\n        # Integration and Differentiation variables\n        self.integrator_signal_vars: dict[str, tk.BooleanVar] = {}\n        self.deriv_signal_vars: dict[str, tk.BooleanVar] = {}\n        self.derivative_vars = {}\n        for _ in range(\n            1, MAX_DERIVATIVE_ORDER + 1\n        ):  # Support up to 5th order derivatives\n            self.derivative_vars[i] = tk.BooleanVar(value=False)\n\n        # Plot view state management\n        self.saved_plot_view = None\n\n        # Custom legend entries for plots\n        self.custom_legend_entries: dict[str, str] = {}\n\n        # Custom colors for plots\n        self.custom_colors = [\n            "#1f77b4",\n            "#ff7f0e",\n            "#2ca02c",\n            "#d62728",\n            "#9467bd",\n            "#8c564b",\n            "#e377c2",\n            "#7f7f7f",\n            "#bcbd22",\n            "#17becf",\n        ]\n\n        # Create Main UI\n        self.main_tab_view = ctk.CTkTabview(self)\n        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")\n\n        self.main_tab_view.add("Processing")\n        self.main_tab_view.add("Plotting & Analysis")\n        self.main_tab_view.add("Plots List")\n        self.main_tab_view.add("DAT File Import")\n        self.main_tab_view.add("Help")\n\n        self.create_setup_and_process_tab(self.main_tab_view.tab("Processing"))\n        self.create_plotting_tab(self.main_tab_view.tab("Plotting & Analysis"))\n        self.create_plots_list_tab(self.main_tab_view.tab("Plots List"))\n        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))\n        self.create_help_tab(self.main_tab_view.tab("Help"))\n\n        self.create_status_bar()\n        self.status_label.configure(\n            text="Ready. Select input files or import a DAT file.",\n        )\n\n        # Load saved plots and other settings\n        self._load_plots_from_file()\n        self.plots_list = []',
            '        self.plots_list: list[dict[str, Any]] = []\n        # Signal List Management variables\n        self.saved_signal_list: list[str] = []\n        self.saved_signal_list_name = ""\n\n        # Integration and Differentiation variables\n        self.integrator_signal_vars: dict[str, tk.BooleanVar] = {}\n        self.deriv_signal_vars: dict[str, tk.BooleanVar] = {}\n        self.derivative_vars = {}\n        for _ in range(\n            1, MAX_DERIVATIVE_ORDER + 1\n        ):  # Support up to 5th order derivatives\n            self.derivative_vars[i] = tk.BooleanVar(value=False)\n\n        # Plot view state management\n        self.saved_plot_view = None\n\n        # Custom legend entries for plots\n        self.custom_legend_entries: dict[str, str] = {}\n\n        # Custom colors for plots\n        self.custom_colors = [\n            "#1f77b4",\n            "#ff7f0e",\n            "#2ca02c",\n            "#d62728",\n            "#9467bd",\n            "#8c564b",\n            "#e377c2",\n            "#7f7f7f",\n            "#bcbd22",\n            "#17becf",\n        ]\n\n        # Create Main UI\n        self.main_tab_view = ctk.CTkTabview(self)\n        self.main_tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")\n\n        self.main_tab_view.add("Processing")\n        self.main_tab_view.add("Plotting & Analysis")\n        self.main_tab_view.add("Plots List")\n        self.main_tab_view.add("DAT File Import")\n        self.main_tab_view.add("Help")\n\n        self.create_setup_and_process_tab(self.main_tab_view.tab("Processing"))\n        self.create_plotting_tab(self.main_tab_view.tab("Plotting & Analysis"))\n        self.create_plots_list_tab(self.main_tab_view.tab("Plots List"))\n        self.create_dat_import_tab(self.main_tab_view.tab("DAT File Import"))\n        self.create_help_tab(self.main_tab_view.tab("Help"))\n\n        self.create_status_bar()\n        self.status_label.configure(\n            text="Ready. Select input files or import a DAT file.",\n        )\n\n        # Load saved plots and other settings\n        self._load_plots_from_file()',
        ),
    ]

    for old, new in specific_fixes:
        if old in content:
            content = content.replace(old, new)
            changes_made = True

    if changes_made:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed type annotations in {file_path}")
        return True
    return False


def main() -> None:
    """Fix type annotation issues in Data_Processor_r0.py"""
    file_path = (
        "data_processing/data_processor/python/data_processor/Data_Processor_r0.py"
    )
    path = Path(file_path)
    if path.exists():
        print("Fixing type annotations...")
        fix_type_annotations(str(path))
    else:
        print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()

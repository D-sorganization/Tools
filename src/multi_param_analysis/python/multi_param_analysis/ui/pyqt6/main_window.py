#!/usr/bin/env python3
"""Multi-Parameter Analysis PyQt6 Main Window.

A PyQt6 GUI for running multi-parameter sensitivity analysis.
"""

from __future__ import annotations

import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# LOD: Extract deep Qt constant chains (>2 levels) to module-level aliases.
_SCROLLBAR_ALWAYS_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
_ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
_SIZE_EXPANDING = QSizePolicy.Policy.Expanding
_SIZE_FIXED = QSizePolicy.Policy.Fixed

# Catppuccin Mocha color palette
CATPPUCCIN_MOCHA = {
    "rosewater": "#f5e0dc",
    "flamingo": "#f2cdcd",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "overlay2": "#9399b2",
    "overlay1": "#7f849c",
    "overlay0": "#6c7086",
    "surface2": "#585b70",
    "surface1": "#45475a",
    "surface0": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {CATPPUCCIN_MOCHA["base"]};
}}

QWidget {{
    background-color: {CATPPUCCIN_MOCHA["base"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    font-family: "Segoe UI", "Arial", sans-serif;
}}

QScrollArea {{
    border: none;
    background-color: {CATPPUCCIN_MOCHA["base"]};
}}

QTabWidget::pane {{
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
    background-color: {CATPPUCCIN_MOCHA["mantle"]};
    border-radius: 4px;
}}

QTabBar::tab {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["subtext1"]};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
    color: {CATPPUCCIN_MOCHA["blue"]};
}}

QGroupBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {CATPPUCCIN_MOCHA["mauve"]};
}}

QLabel {{
    color: {CATPPUCCIN_MOCHA["text"]};
    background-color: transparent;
}}

QDoubleSpinBox, QSpinBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
}}

QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {CATPPUCCIN_MOCHA["blue"]};
}}

QComboBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
    min-width: 150px;
}}

QComboBox:hover {{
    border: 1px solid {CATPPUCCIN_MOCHA["blue"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    selection-background-color: {CATPPUCCIN_MOCHA["surface2"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface1"]};
}}

QCheckBox {{
    color: {CATPPUCCIN_MOCHA["text"]};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
}}

QCheckBox::indicator:checked {{
    background-color: {CATPPUCCIN_MOCHA["blue"]};
    border-color: {CATPPUCCIN_MOCHA["blue"]};
}}

QTextEdit {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 8px;
    font-family: "Consolas", "Courier New", monospace;
}}

QPushButton {{
    background-color: {CATPPUCCIN_MOCHA["blue"]};
    color: {CATPPUCCIN_MOCHA["crust"]};
    border: none;
    border-radius: 4px;
    padding: 10px 24px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: {CATPPUCCIN_MOCHA["sapphire"]};
}}

QPushButton:pressed {{
    background-color: {CATPPUCCIN_MOCHA["lavender"]};
}}

QPushButton#runBtn {{
    background-color: {CATPPUCCIN_MOCHA["green"]};
}}

QPushButton#runBtn:hover {{
    background-color: {CATPPUCCIN_MOCHA["teal"]};
}}
"""


class MultiParamAnalysisWindow(QMainWindow):
    """Main window for Multi-Parameter Analysis application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._results: dict | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Multi-Parameter Analysis")
        self.setMinimumSize(700, 800)
        self.setStyleSheet(STYLESHEET)

        # Central widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(_SCROLLBAR_ALWAYS_OFF)
        self.setCentralWidget(scroll_area)

        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title_label = QLabel("Multi-Parameter Analysis")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(_ALIGN_CENTER)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab_widget.addTab(self._create_parameters_tab(), "Parameters")
        self.tab_widget.addTab(self._create_options_tab(), "Options")
        self.tab_widget.addTab(self._create_results_tab(), "Results")

        # Run button
        run_btn = QPushButton("Run Analysis")
        run_btn.setObjectName("runBtn")
        run_btn_clicked = run_btn.clicked
        run_btn_clicked.connect(self._run_analysis)
        main_layout.addWidget(run_btn)

        main_layout.addStretch()

    def _create_parameters_tab(self) -> QWidget:
        """Create the parameters configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Parameter 1
        param1_group = QGroupBox("Parameter 1 (X-Axis)")
        param1_layout = QGridLayout(param1_group)
        param1_layout.setSpacing(10)

        param1_layout.addWidget(QLabel("Variable:"), 0, 0)
        self.param1_combo = QComboBox()
        self.param1_combo.addItems(
            [
                "Temperature",
                "O2/Feed Ratio",
                "Steam/Feed Ratio",
                "Pressure",
                "Feed Rate",
            ]
        )
        param1_layout.addWidget(self.param1_combo, 0, 1)

        param1_layout.addWidget(QLabel("Minimum:"), 1, 0)
        self.param1_min = QDoubleSpinBox()
        self.param1_min.setRange(-1e6, 1e6)
        self.param1_min.setDecimals(2)
        self.param1_min.setValue(600)
        self.param1_min.setSizePolicy(_SIZE_EXPANDING, _SIZE_FIXED)
        param1_layout.addWidget(self.param1_min, 1, 1)

        param1_layout.addWidget(QLabel("Maximum:"), 2, 0)
        self.param1_max = QDoubleSpinBox()
        self.param1_max.setRange(-1e6, 1e6)
        self.param1_max.setDecimals(2)
        self.param1_max.setValue(1200)
        param1_layout.addWidget(self.param1_max, 2, 1)

        param1_layout.addWidget(QLabel("Steps:"), 3, 0)
        self.param1_steps = QSpinBox()
        self.param1_steps.setRange(2, 100)
        self.param1_steps.setValue(10)
        param1_layout.addWidget(self.param1_steps, 3, 1)

        layout.addWidget(param1_group)

        # Parameter 2
        param2_group = QGroupBox("Parameter 2 (Y-Axis)")
        param2_layout = QGridLayout(param2_group)
        param2_layout.setSpacing(10)

        param2_layout.addWidget(QLabel("Variable:"), 0, 0)
        self.param2_combo = QComboBox()
        self.param2_combo.addItems(
            [
                "O2/Feed Ratio",
                "Temperature",
                "Steam/Feed Ratio",
                "Pressure",
                "Feed Rate",
            ]
        )
        param2_layout.addWidget(self.param2_combo, 0, 1)

        param2_layout.addWidget(QLabel("Minimum:"), 1, 0)
        self.param2_min = QDoubleSpinBox()
        self.param2_min.setRange(-1e6, 1e6)
        self.param2_min.setDecimals(3)
        self.param2_min.setValue(0.1)
        self.param2_min.setSizePolicy(_SIZE_EXPANDING, _SIZE_FIXED)
        param2_layout.addWidget(self.param2_min, 1, 1)

        param2_layout.addWidget(QLabel("Maximum:"), 2, 0)
        self.param2_max = QDoubleSpinBox()
        self.param2_max.setRange(-1e6, 1e6)
        self.param2_max.setDecimals(3)
        self.param2_max.setValue(0.5)
        param2_layout.addWidget(self.param2_max, 2, 1)

        param2_layout.addWidget(QLabel("Steps:"), 3, 0)
        self.param2_steps = QSpinBox()
        self.param2_steps.setRange(2, 100)
        self.param2_steps.setValue(10)
        param2_layout.addWidget(self.param2_steps, 3, 1)

        layout.addWidget(param2_group)

        # Output variable
        output_group = QGroupBox("Output Variable")
        output_layout = QGridLayout(output_group)

        output_layout.addWidget(QLabel("Variable:"), 0, 0)
        self.output_combo = QComboBox()
        self.output_combo.addItems(
            [
                "Efficiency",
                "Syngas HHV",
                "H2 Yield",
                "CO Yield",
                "Carbon Conversion",
                "Cold Gas Efficiency",
            ]
        )
        output_layout.addWidget(self.output_combo, 0, 1)

        layout.addWidget(output_group)

        layout.addStretch()
        return tab

    def _create_options_tab(self) -> QWidget:
        """Create the analysis options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Execution options
        exec_group = QGroupBox("Execution Options")
        exec_layout = QGridLayout(exec_group)
        exec_layout.setSpacing(10)

        self.parallel_checkbox = QCheckBox("Use parallel processing")
        self.parallel_checkbox.setChecked(True)
        exec_layout.addWidget(self.parallel_checkbox, 0, 0, 1, 2)

        exec_layout.addWidget(QLabel("Max Workers:"), 1, 0)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(4)
        exec_layout.addWidget(self.workers_spin, 1, 1)

        layout.addWidget(exec_group)

        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QGridLayout(analysis_group)
        analysis_layout.setSpacing(10)

        self.sensitivity_checkbox = QCheckBox("Calculate sensitivity indices")
        self.sensitivity_checkbox.setChecked(True)
        analysis_layout.addWidget(self.sensitivity_checkbox, 0, 0, 1, 2)

        self.normalize_checkbox = QCheckBox("Normalize results")
        analysis_layout.addWidget(self.normalize_checkbox, 1, 0, 1, 2)

        layout.addWidget(analysis_group)

        # Demo function
        demo_group = QGroupBox("Demo Function")
        demo_layout = QGridLayout(demo_group)

        demo_layout.addWidget(QLabel("Function:"), 0, 0)
        self.demo_func_combo = QComboBox()
        self.demo_func_combo.addItems(
            [
                "Rosenbrock",
                "Rastrigin",
                "Sphere",
                "Himmelblau",
                "Beale",
            ]
        )
        demo_layout.addWidget(self.demo_func_combo, 0, 1)

        layout.addWidget(demo_group)

        layout.addStretch()
        return tab

    def _create_results_tab(self) -> QWidget:
        """Create the results display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setSpacing(8)

        labels = [
            ("Grid Points:", "grid_points"),
            ("Min Value:", "min_value"),
            ("Max Value:", "max_value"),
            ("Mean Value:", "mean_value"),
            ("Std Deviation:", "std_value"),
            ("Optimal X:", "opt_x"),
            ("Optimal Y:", "opt_y"),
        ]

        self.stat_labels: dict[str, QLabel] = {}
        for row, (label_text, key) in enumerate(labels):
            stats_layout.addWidget(QLabel(label_text), row, 0)
            value_label = QLabel("-")
            value_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['sapphire']};")
            self.stat_labels[key] = value_label
            stats_layout.addWidget(value_label, row, 1)

        layout.addWidget(stats_group)

        # Sensitivity
        sens_group = QGroupBox("Sensitivity Analysis")
        sens_layout = QVBoxLayout(sens_group)
        self.sensitivity_text = QTextEdit()
        self.sensitivity_text.setReadOnly(True)
        self.sensitivity_text.setMaximumHeight(120)
        self.sensitivity_text.setPlaceholderText(
            "Sensitivity indices will appear here..."
        )
        sens_layout.addWidget(self.sensitivity_text)
        layout.addWidget(sens_group)

        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText(
            "Run analysis to see results preview...\n\n"
            "Results will show a grid of output values."
        )
        preview_layout.addWidget(self.preview_text)
        layout.addWidget(preview_group)

        return tab

    def _run_analysis(self) -> None:
        """Run the multi-parameter analysis."""
        # Get parameters
        param1_name = self.param1_combo.currentText()
        param2_name = self.param2_combo.currentText()
        output_name = self.output_combo.currentText()

        param1_values = np.linspace(
            self.param1_min.value(),
            self.param1_max.value(),
            self.param1_steps.value(),
        )
        param2_values = np.linspace(
            self.param2_min.value(),
            self.param2_max.value(),
            self.param2_steps.value(),
        )

        # Run demo analysis
        self._run_demo_analysis(
            param1_name, param2_name, output_name, param1_values, param2_values
        )

    def _run_demo_analysis(
        self,
        param1_name: str,
        param2_name: str,
        output_name: str,
        param1_values: np.ndarray,
        param2_values: np.ndarray,
    ) -> None:
        """Run demo analysis with test function.

        Preconditions:
            param1_name: must be a non-empty str
            param2_name: must be a non-empty str
            output_name: must be a non-empty str
            param1_values: must be a numpy ndarray
            param2_values: must be a numpy ndarray
        """
        if not isinstance(param1_name, str):
            raise TypeError(
                f"param1_name must be a str, got {type(param1_name).__name__}"
            )
        if not isinstance(param2_name, str):
            raise TypeError(
                f"param2_name must be a str, got {type(param2_name).__name__}"
            )
        if not isinstance(output_name, str):
            raise TypeError(
                f"output_name must be a str, got {type(output_name).__name__}"
            )
        if not isinstance(param1_values, np.ndarray):
            raise TypeError(
                f"param1_values must be a numpy ndarray, got {type(param1_values).__name__}"
            )
        if not isinstance(param2_values, np.ndarray):
            raise TypeError(
                f"param2_values must be a numpy ndarray, got {type(param2_values).__name__}"
            )

        demo_func = self.demo_func_combo.currentText()

        # Create meshgrid
        X, Y = np.meshgrid(param1_values, param2_values)

        # Normalize to [-5, 5] range for demo functions
        x_norm = 10 * (X - X.min()) / (X.max() - X.min()) - 5
        y_norm = 10 * (Y - Y.min()) / (Y.max() - Y.min()) - 5

        # Evaluate demo function
        if demo_func == "Rosenbrock":
            Z = (1 - x_norm) ** 2 + 100 * (y_norm - x_norm**2) ** 2
        elif demo_func == "Rastrigin":
            Z = (
                20
                + x_norm**2
                + y_norm**2
                - 10 * (np.cos(2 * np.pi * x_norm) + np.cos(2 * np.pi * y_norm))
            )
        elif demo_func == "Sphere":
            Z = x_norm**2 + y_norm**2
        elif demo_func == "Himmelblau":
            Z = (x_norm**2 + y_norm - 11) ** 2 + (x_norm + y_norm**2 - 7) ** 2
        else:  # Beale
            Z = (
                (1.5 - x_norm + x_norm * y_norm) ** 2
                + (2.25 - x_norm + x_norm * y_norm**2) ** 2
                + (2.625 - x_norm + x_norm * y_norm**3) ** 2
            )

        # Store results
        self._results = {
            "param1_values": param1_values,
            "param2_values": param2_values,
            "output_values": Z,
            "param1_name": param1_name,
            "param2_name": param2_name,
            "output_name": output_name,
        }

        # Update statistics
        grid_points = len(param1_values) * len(param2_values)
        self.stat_labels["grid_points"].setText(str(grid_points))
        self.stat_labels["min_value"].setText(f"{Z.min():.4f}")
        self.stat_labels["max_value"].setText(f"{Z.max():.4f}")
        self.stat_labels["mean_value"].setText(f"{Z.mean():.4f}")
        self.stat_labels["std_value"].setText(f"{Z.std():.4f}")

        # Find optimal
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        self.stat_labels["opt_x"].setText(f"{param1_values[min_idx[1]]:.4f}")
        self.stat_labels["opt_y"].setText(f"{param2_values[min_idx[0]]:.4f}")

        # Update sensitivity
        if self.sensitivity_checkbox.isChecked():
            self._calculate_sensitivity(param1_values, param2_values, Z)

        # Update preview
        self._update_preview(param1_values, param2_values, Z)

        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)

    def _calculate_sensitivity(
        self,
        param1_values: np.ndarray,
        param2_values: np.ndarray,
        Z: np.ndarray,
    ) -> None:
        """Calculate sensitivity indices.

        Preconditions:
            param1_values: must be a numpy ndarray
            param2_values: must be a numpy ndarray
            Z: must be a numpy ndarray
        """
        if not isinstance(param1_values, np.ndarray):
            raise TypeError(
                f"param1_values must be a numpy ndarray, got {type(param1_values).__name__}"
            )
        if not isinstance(param2_values, np.ndarray):
            raise TypeError(
                f"param2_values must be a numpy ndarray, got {type(param2_values).__name__}"
            )
        if not isinstance(Z, np.ndarray):
            raise TypeError(f"Z must be a numpy ndarray, got {type(Z).__name__}")

        # Simple variance-based sensitivity
        total_var = Z.var()

        if total_var > 0:
            # Main effect of param1 (average over param2)
            param1_means = Z.mean(axis=0)
            param1_var = param1_means.var()
            s1 = param1_var / total_var

            # Main effect of param2 (average over param1)
            param2_means = Z.mean(axis=1)
            param2_var = param2_means.var()
            s2 = param2_var / total_var

            # Interaction effect
            interaction = 1 - s1 - s2
        else:
            s1 = s2 = interaction = 0.0

        sens_text = []
        sens_text.append("Variance-Based Sensitivity Indices")
        sens_text.append("=" * 40)
        sens_text.append(
            f"\nFirst-order index (S1) for {self.param1_combo.currentText()}:"
        )
        sens_text.append(f"  S1 = {s1:.4f} ({s1 * 100:.1f}% of variance)")
        sens_text.append(
            f"\nFirst-order index (S2) for {self.param2_combo.currentText()}:"
        )
        sens_text.append(f"  S2 = {s2:.4f} ({s2 * 100:.1f}% of variance)")
        sens_text.append("\nInteraction effect:")
        sens_text.append(
            f"  S12 = {interaction:.4f} ({interaction * 100:.1f}% of variance)"
        )

        self.sensitivity_text.setPlainText("\n".join(sens_text))

    def _update_preview(
        self,
        param1_values: np.ndarray,
        param2_values: np.ndarray,
        Z: np.ndarray,
    ) -> None:
        """Update the data preview.

        Preconditions:
            param1_values: must be a numpy ndarray
            param2_values: must be a numpy ndarray
            Z: must be a numpy ndarray
        """
        if not isinstance(param1_values, np.ndarray):
            raise TypeError(
                f"param1_values must be a numpy ndarray, got {type(param1_values).__name__}"
            )
        if not isinstance(param2_values, np.ndarray):
            raise TypeError(
                f"param2_values must be a numpy ndarray, got {type(param2_values).__name__}"
            )
        if not isinstance(Z, np.ndarray):
            raise TypeError(f"Z must be a numpy ndarray, got {type(Z).__name__}")

        lines = []
        lines.append("Multi-Parameter Analysis Results")
        lines.append("=" * 50)
        lines.append(f"\nFunction: {self.demo_func_combo.currentText()}")
        lines.append(
            f"Grid: {len(param1_values)} x {len(param2_values)} = {Z.size} points"
        )
        lines.append(
            f"\n{self.param1_combo.currentText()}: {param1_values[0]:.2f} to {param1_values[-1]:.2f}"
        )
        lines.append(
            f"{self.param2_combo.currentText()}: {param2_values[0]:.3f} to {param2_values[-1]:.3f}"
        )

        lines.append("\nOutput Grid (sample):")
        lines.append("-" * 50)

        # Show corner values
        n1, n2 = min(5, len(param1_values)), min(5, len(param2_values))

        # Header
        header = "       "
        for j in range(n2):
            header += f" {param2_values[j]:8.3f}"
        lines.append(header)

        # Data rows
        for i in range(n1):
            row = f"{param1_values[i]:6.1f} "
            for j in range(n2):
                row += f" {Z[j, i]:8.2f}"
            lines.append(row)

        if len(param1_values) > n1 or len(param2_values) > n2:
            lines.append("... (truncated)")

        self.preview_text.setPlainText("\n".join(lines))
        self.preview_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['text']};")


def main() -> int:
    """Run the Multi-Parameter Analysis application."""
    app = QApplication(sys.argv)
    window = MultiParamAnalysisWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

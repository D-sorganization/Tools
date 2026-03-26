#!/usr/bin/env python3
"""Humanoid Character Builder PyQt6 Main Window.

A PyQt6 GUI for building parametric humanoid characters with
anthropometric calculations and URDF export.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

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

QDoubleSpinBox {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    padding: 6px 10px;
}}

QDoubleSpinBox:focus {{
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

QSlider::groove:horizontal {{
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    height: 8px;
    background: {CATPPUCCIN_MOCHA["surface0"]};
    border-radius: 4px;
}}

QSlider::handle:horizontal {{
    background: {CATPPUCCIN_MOCHA["blue"]};
    border: 1px solid {CATPPUCCIN_MOCHA["blue"]};
    width: 16px;
    margin: -4px 0;
    border-radius: 8px;
}}

QSlider::sub-page:horizontal {{
    background: {CATPPUCCIN_MOCHA["sapphire"]};
    border-radius: 4px;
}}

QTableWidget {{
    background-color: {CATPPUCCIN_MOCHA["surface0"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    border: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
    border-radius: 4px;
    gridline-color: {CATPPUCCIN_MOCHA["surface1"]};
}}

QTableWidget::item {{
    padding: 4px;
}}

QTableWidget::item:selected {{
    background-color: {CATPPUCCIN_MOCHA["surface2"]};
}}

QHeaderView::section {{
    background-color: {CATPPUCCIN_MOCHA["surface1"]};
    color: {CATPPUCCIN_MOCHA["text"]};
    padding: 6px;
    border: none;
    border-bottom: 1px solid {CATPPUCCIN_MOCHA["surface2"]};
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

QPushButton#buildBtn {{
    background-color: {CATPPUCCIN_MOCHA["green"]};
}}

QPushButton#buildBtn:hover {{
    background-color: {CATPPUCCIN_MOCHA["teal"]};
}}

QPushButton#exportBtn {{
    background-color: {CATPPUCCIN_MOCHA["peach"]};
}}

QPushButton#exportBtn:hover {{
    background-color: {CATPPUCCIN_MOCHA["yellow"]};
}}
"""


class BuildType(Enum):
    """Body build types."""

    ECTOMORPH = "Ectomorph (Lean)"
    MESOMORPH = "Mesomorph (Athletic)"
    ENDOMORPH = "Endomorph (Heavy)"
    AVERAGE = "Average"


class GenderModel(Enum):
    """Gender model for anthropometric calculations."""

    MALE = "Male"
    FEMALE = "Female"
    NEUTRAL = "Neutral"


@dataclass
class SegmentData:
    """Data for a body segment."""

    name: str
    mass_kg: float
    length_m: float
    width_m: float
    depth_m: float


# de Leva (1996) anthropometric data (simplified)
SEGMENT_MASS_RATIOS = {
    "head": 0.0694,
    "neck": 0.0240,
    "thorax": 0.2160,
    "lumbar": 0.1390,
    "pelvis": 0.1117,
    "upper_arm": 0.0271,
    "forearm": 0.0162,
    "hand": 0.0061,
    "thigh": 0.1416,
    "shin": 0.0433,
    "foot": 0.0137,
}

SEGMENT_LENGTH_RATIOS = {
    "head": 0.1395,
    "neck": 0.052,
    "thorax": 0.170,
    "lumbar": 0.108,
    "pelvis": 0.078,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.108,
    "thigh": 0.245,
    "shin": 0.246,
    "foot": 0.152,
}


class HumanoidBuilderWindow(QMainWindow):
    """Main window for Humanoid Character Builder application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._segments: list[SegmentData] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Humanoid Character Builder")
        self.setMinimumSize(800, 900)
        self.setStyleSheet(STYLESHEET)

        # Central widget with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(scroll_area)

        central_widget = QWidget()
        scroll_area.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title
        title_label = QLabel("Humanoid Character Builder")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab_widget.addTab(self._create_body_params_tab(), "Body Parameters")
        self.tab_widget.addTab(self._create_proportions_tab(), "Proportions")
        self.tab_widget.addTab(self._create_results_tab(), "Results")
        self.tab_widget.addTab(self._create_export_tab(), "Export")

        # Build button
        build_btn = QPushButton("Build Character")
        build_btn.setObjectName("buildBtn")
        build_btn.clicked.connect(self._build_character)
        main_layout.addWidget(build_btn)

        main_layout.addStretch()

    def _create_body_params_tab(self) -> QWidget:
        """Create the body parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Primary parameters
        primary_group = QGroupBox("Primary Parameters")
        primary_layout = QGridLayout(primary_group)
        primary_layout.setSpacing(10)

        # Height
        primary_layout.addWidget(QLabel("Height (m):"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.5, 3.0)
        self.height_spin.setDecimals(2)
        self.height_spin.setValue(1.75)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        primary_layout.addWidget(self.height_spin, 0, 1)

        # Mass
        primary_layout.addWidget(QLabel("Mass (kg):"), 1, 0)
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(10.0, 300.0)
        self.mass_spin.setDecimals(1)
        self.mass_spin.setValue(75.0)
        self.mass_spin.setSingleStep(0.5)
        primary_layout.addWidget(self.mass_spin, 1, 1)

        # Build type
        primary_layout.addWidget(QLabel("Build Type:"), 2, 0)
        self.build_combo = QComboBox()
        for build in BuildType:
            self.build_combo.addItem(build.value)
        self.build_combo.setCurrentIndex(3)  # Average
        primary_layout.addWidget(self.build_combo, 2, 1)

        # Gender model
        primary_layout.addWidget(QLabel("Gender Model:"), 3, 0)
        self.gender_combo = QComboBox()
        for gender in GenderModel:
            self.gender_combo.addItem(gender.value)
        self.gender_combo.setCurrentIndex(2)  # Neutral
        primary_layout.addWidget(self.gender_combo, 3, 1)

        layout.addWidget(primary_group)

        # Build factors
        factors_group = QGroupBox("Build Factors")
        factors_layout = QGridLayout(factors_group)
        factors_layout.setSpacing(10)

        # Muscularity
        factors_layout.addWidget(QLabel("Muscularity:"), 0, 0)
        self.muscularity_slider = QSlider(Qt.Orientation.Horizontal)
        self.muscularity_slider.setRange(0, 100)
        self.muscularity_slider.setValue(50)
        factors_layout.addWidget(self.muscularity_slider, 0, 1)
        self.muscularity_label = QLabel("0.50")
        self.muscularity_label.setMinimumWidth(40)
        factors_layout.addWidget(self.muscularity_label, 0, 2)
        self.muscularity_slider.valueChanged.connect(
            lambda v: self.muscularity_label.setText(f"{v / 100:.2f}")
        )

        # Body fat
        factors_layout.addWidget(QLabel("Body Fat:"), 1, 0)
        self.bodyfat_slider = QSlider(Qt.Orientation.Horizontal)
        self.bodyfat_slider.setRange(0, 100)
        self.bodyfat_slider.setValue(20)
        factors_layout.addWidget(self.bodyfat_slider, 1, 1)
        self.bodyfat_label = QLabel("0.20")
        self.bodyfat_label.setMinimumWidth(40)
        factors_layout.addWidget(self.bodyfat_label, 1, 2)
        self.bodyfat_slider.valueChanged.connect(
            lambda v: self.bodyfat_label.setText(f"{v / 100:.2f}")
        )

        layout.addWidget(factors_group)

        # BMI display
        bmi_group = QGroupBox("Body Metrics")
        bmi_layout = QGridLayout(bmi_group)

        bmi_layout.addWidget(QLabel("BMI:"), 0, 0)
        self.bmi_label = QLabel("-")
        self.bmi_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['sapphire']};")
        bmi_layout.addWidget(self.bmi_label, 0, 1)

        bmi_layout.addWidget(QLabel("Category:"), 1, 0)
        self.bmi_category_label = QLabel("-")
        self.bmi_category_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['sapphire']};")
        bmi_layout.addWidget(self.bmi_category_label, 1, 1)

        layout.addWidget(bmi_group)

        # Connect signals for BMI updates
        self.height_spin.valueChanged.connect(self._update_bmi)
        self.mass_spin.valueChanged.connect(self._update_bmi)
        self._update_bmi()

        layout.addStretch()
        return tab

    def _create_proportions_tab(self) -> QWidget:
        """Create the proportions adjustment tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Proportions group
        props_group = QGroupBox("Body Proportions (1.0 = Default)")
        props_layout = QGridLayout(props_group)
        props_layout.setSpacing(10)

        self.proportion_sliders: dict[str, tuple[QSlider, QLabel]] = {}

        proportions = [
            ("Shoulder Width", "shoulder_width"),
            ("Hip Width", "hip_width"),
            ("Arm Length", "arm_length"),
            ("Leg Length", "leg_length"),
            ("Torso Length", "torso_length"),
            ("Head Size", "head_scale"),
            ("Neck Length", "neck_length"),
            ("Hand Size", "hand_scale"),
            ("Foot Size", "foot_scale"),
        ]

        for row, (label_text, key) in enumerate(proportions):
            props_layout.addWidget(QLabel(f"{label_text}:"), row, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(50, 150)  # 0.5 to 1.5
            slider.setValue(100)  # 1.0
            props_layout.addWidget(slider, row, 1)

            value_label = QLabel("1.00")
            value_label.setMinimumWidth(40)
            props_layout.addWidget(value_label, row, 2)

            slider.valueChanged.connect(
                lambda v, lbl=value_label: lbl.setText(f"{v / 100:.2f}")
            )

            self.proportion_sliders[key] = (slider, value_label)

        layout.addWidget(props_group)

        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_proportions)
        layout.addWidget(reset_btn)

        layout.addStretch()
        return tab

    def _create_results_tab(self) -> QWidget:
        """Create the results display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Summary
        summary_group = QGroupBox("Build Summary")
        summary_layout = QGridLayout(summary_group)
        summary_layout.setSpacing(8)

        labels = [
            ("Total Height:", "total_height"),
            ("Total Mass:", "total_mass"),
            ("Segment Count:", "segment_count"),
            ("Total Computed Mass:", "computed_mass"),
        ]

        self.summary_labels: dict[str, QLabel] = {}
        for row, (label_text, key) in enumerate(labels):
            summary_layout.addWidget(QLabel(label_text), row, 0)
            value_label = QLabel("-")
            value_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['sapphire']};")
            self.summary_labels[key] = value_label
            summary_layout.addWidget(value_label, row, 1)

        layout.addWidget(summary_group)

        # Segments table
        segments_group = QGroupBox("Segment Details")
        segments_layout = QVBoxLayout(segments_group)

        self.segments_table = QTableWidget()
        self.segments_table.setColumnCount(5)
        self.segments_table.setHorizontalHeaderLabels(
            ["Segment", "Mass (kg)", "Length (m)", "Width (m)", "Depth (m)"]
        )
        header = self.segments_table.horizontalHeader()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for i in range(1, 5):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.segments_table.setMinimumHeight(300)

        segments_layout.addWidget(self.segments_table)
        layout.addWidget(segments_group)

        return tab

    def _create_export_tab(self) -> QWidget:
        """Create the export options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Export options
        options_group = QGroupBox("Export Options")
        options_layout = QGridLayout(options_group)
        options_layout.setSpacing(10)

        options_layout.addWidget(QLabel("Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["URDF Package", "URDF Only", "JSON Config"])
        options_layout.addWidget(self.format_combo, 0, 1)

        options_layout.addWidget(QLabel("Mesh Format:"), 1, 0)
        self.mesh_format_combo = QComboBox()
        self.mesh_format_combo.addItems(["STL", "OBJ", "DAE"])
        options_layout.addWidget(self.mesh_format_combo, 1, 1)

        options_layout.addWidget(QLabel("Character Name:"), 2, 0)
        self.name_edit = QTextEdit()
        self.name_edit.setPlainText("humanoid")
        self.name_edit.setMaximumHeight(30)
        options_layout.addWidget(self.name_edit, 2, 1)

        layout.addWidget(options_group)

        # Preview
        preview_group = QGroupBox("Configuration Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText(
            "Build character to see configuration preview..."
        )
        preview_layout.addWidget(self.preview_text)

        layout.addWidget(preview_group)

        # Export buttons
        btn_layout = QHBoxLayout()

        preview_btn = QPushButton("Update Preview")
        preview_btn.clicked.connect(self._update_preview)
        btn_layout.addWidget(preview_btn)

        export_btn = QPushButton("Export")
        export_btn.setObjectName("exportBtn")
        export_btn.clicked.connect(self._export_character)
        btn_layout.addWidget(export_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

        return tab

    def _update_bmi(self) -> None:
        """Update BMI display."""
        height = self.height_spin.value()
        mass = self.mass_spin.value()

        bmi = mass / (height * height)
        self.bmi_label.setText(f"{bmi:.1f}")

        if bmi < 18.5:
            category = "Underweight"
            color = CATPPUCCIN_MOCHA["yellow"]
        elif bmi < 25:
            category = "Normal"
            color = CATPPUCCIN_MOCHA["green"]
        elif bmi < 30:
            category = "Overweight"
            color = CATPPUCCIN_MOCHA["peach"]
        else:
            category = "Obese"
            color = CATPPUCCIN_MOCHA["red"]

        self.bmi_category_label.setText(category)
        self.bmi_category_label.setStyleSheet(f"color: {color};")

    def _reset_proportions(self) -> None:
        """Reset all proportions to default."""
        for slider, label in self.proportion_sliders.values():
            slider.setValue(100)
            label.setText("1.00")

    def _get_gender_factor(self) -> float:
        """Get gender factor for anthropometric calculations."""
        gender_text = self.gender_combo.currentText()
        if gender_text == "Male":
            return 1.0
        elif gender_text == "Female":
            return 0.0
        else:
            return 0.5

    def _build_character(self) -> None:
        """Build the character with current parameters."""
        height = self.height_spin.value()
        mass = self.mass_spin.value()

        # Get proportions
        proportions = {}
        for key, (slider, _) in self.proportion_sliders.items():
            proportions[key] = slider.value() / 100.0

        # Calculate segment data
        self._segments = []
        total_mass = 0.0

        segments_info = [
            ("Head", "head"),
            ("Neck", "neck"),
            ("Thorax", "thorax"),
            ("Lumbar", "lumbar"),
            ("Pelvis", "pelvis"),
            ("L Upper Arm", "upper_arm"),
            ("R Upper Arm", "upper_arm"),
            ("L Forearm", "forearm"),
            ("R Forearm", "forearm"),
            ("L Hand", "hand"),
            ("R Hand", "hand"),
            ("L Thigh", "thigh"),
            ("R Thigh", "thigh"),
            ("L Shin", "shin"),
            ("R Shin", "shin"),
            ("L Foot", "foot"),
            ("R Foot", "foot"),
        ]

        for display_name, key in segments_info:
            mass_ratio = SEGMENT_MASS_RATIOS.get(key, 0.01)
            length_ratio = SEGMENT_LENGTH_RATIOS.get(key, 0.05)

            seg_mass = mass * mass_ratio
            seg_length = height * length_ratio

            # Apply proportion adjustments
            if "arm" in key.lower():
                seg_length *= proportions.get("arm_length", 1.0)
            elif "thigh" in key or "shin" in key:
                seg_length *= proportions.get("leg_length", 1.0)
            elif "head" in key:
                seg_length *= proportions.get("head_scale", 1.0)
            elif "hand" in key:
                seg_length *= proportions.get("hand_scale", 1.0)
            elif "foot" in key:
                seg_length *= proportions.get("foot_scale", 1.0)
            elif "neck" in key:
                seg_length *= proportions.get("neck_length", 1.0)

            # Estimate width and depth
            width = seg_length * 0.3
            depth = seg_length * 0.25

            self._segments.append(
                SegmentData(
                    name=display_name,
                    mass_kg=seg_mass,
                    length_m=seg_length,
                    width_m=width,
                    depth_m=depth,
                )
            )
            total_mass += seg_mass

        # Update summary
        self.summary_labels["total_height"].setText(f"{height:.2f} m")
        self.summary_labels["total_mass"].setText(f"{mass:.1f} kg")
        self.summary_labels["segment_count"].setText(str(len(self._segments)))
        self.summary_labels["computed_mass"].setText(f"{total_mass:.1f} kg")

        # Update table
        self.segments_table.setRowCount(len(self._segments))
        for row, seg in enumerate(self._segments):
            self.segments_table.setItem(row, 0, QTableWidgetItem(seg.name))
            self.segments_table.setItem(row, 1, QTableWidgetItem(f"{seg.mass_kg:.3f}"))
            self.segments_table.setItem(row, 2, QTableWidgetItem(f"{seg.length_m:.4f}"))
            self.segments_table.setItem(row, 3, QTableWidgetItem(f"{seg.width_m:.4f}"))
            self.segments_table.setItem(row, 4, QTableWidgetItem(f"{seg.depth_m:.4f}"))

        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)

        # Update preview
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the configuration preview."""
        if not self._segments:
            self.preview_text.setPlainText("Build character first to see preview.")
            return

        lines = []
        lines.append("Humanoid Character Configuration")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Name: {self.name_edit.toPlainText()}")
        lines.append(f"Height: {self.height_spin.value():.2f} m")
        lines.append(f"Mass: {self.mass_spin.value():.1f} kg")
        lines.append(f"Build Type: {self.build_combo.currentText()}")
        lines.append(f"Gender Model: {self.gender_combo.currentText()}")
        lines.append("")
        lines.append("Proportions:")
        for key, (slider, _) in self.proportion_sliders.items():
            lines.append(f"  {key}: {slider.value() / 100:.2f}")
        lines.append("")
        lines.append(f"Export Format: {self.format_combo.currentText()}")
        lines.append(f"Mesh Format: {self.mesh_format_combo.currentText()}")
        lines.append("")
        lines.append(f"Total Segments: {len(self._segments)}")

        self.preview_text.setPlainText("\n".join(lines))

    def _export_character(self) -> None:
        """Export the character configuration."""
        if not self._segments:
            self.preview_text.setPlainText(
                "Error: Build character first before export."
            )
            return

        # In a full implementation, this would save to file
        self.preview_text.setPlainText(
            "Export functionality would save to:\n\n"
            f"  {self.name_edit.toPlainText()}/\n"
            f"    humanoid.urdf\n"
            f"    meshes/\n"
            f"    config/body_params.yaml\n\n"
            "(File dialog would appear in full implementation)"
        )


def main() -> int:
    """Run the Humanoid Character Builder application."""
    app = QApplication(sys.argv)
    window = HumanoidBuilderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

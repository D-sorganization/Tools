#!/usr/bin/env python3
"""Parametric URDF Builder PyQt6 Main Window.

A thin UI shell that delegates all business logic to the extracted modules:
  - urdf_generator: URDF XML generation & validation
  - preview_generator: Human-readable model previews
  - anthropometric_model: Shared constants & config
  - theme: Reusable palette & stylesheet
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from urdf_builder_gui.theme import CATPPUCCIN_MOCHA, build_stylesheet

logger = logging.getLogger(__name__)

STYLESHEET = build_stylesheet()


class URDFBuilderWindow(QMainWindow):
    """Main window for Parametric URDF Builder application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Parametric URDF Builder")
        self.setMinimumSize(700, 800)
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
        title_label = QLabel("Parametric URDF Builder")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['blue']};")
        main_layout.addWidget(title_label)

        # Tab widget for different input modes
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.tab_widget.addTab(self._create_body_params_tab(), "Body Parameters")
        self.tab_widget.addTab(self._create_proportions_tab(), "Proportions")
        self.tab_widget.addTab(self._create_options_tab(), "Options")

        # Action buttons
        main_layout.addWidget(self._create_action_group())

        # Results
        main_layout.addWidget(self._create_results_group())

        main_layout.addStretch()

    def _create_body_params_tab(self) -> QWidget:
        """Create the body parameters tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Basic parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QGridLayout(basic_group)
        basic_layout.setSpacing(10)

        # Robot name
        basic_layout.addWidget(QLabel("Robot Name:"), 0, 0)
        self.name_input = QComboBox()
        self.name_input.setEditable(True)
        self.name_input.addItems(["humanoid", "robot", "character", "avatar"])
        self.name_input.setCurrentText("humanoid")
        basic_layout.addWidget(self.name_input, 0, 1)

        # Height
        basic_layout.addWidget(QLabel("Height (m):"), 1, 0)
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.5, 3.0)
        self.height_input.setDecimals(2)
        self.height_input.setValue(1.75)
        self.height_input.setSingleStep(0.01)
        self.height_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        basic_layout.addWidget(self.height_input, 1, 1)

        # Mass
        basic_layout.addWidget(QLabel("Mass (kg):"), 2, 0)
        self.mass_input = QDoubleSpinBox()
        self.mass_input.setRange(20.0, 200.0)
        self.mass_input.setDecimals(1)
        self.mass_input.setValue(70.0)
        self.mass_input.setSingleStep(0.5)
        basic_layout.addWidget(self.mass_input, 2, 1)

        # Gender factor
        basic_layout.addWidget(QLabel("Gender Factor:"), 3, 0)
        gender_container = QWidget()
        gender_layout = QGridLayout(gender_container)
        gender_layout.setContentsMargins(0, 0, 0, 0)

        self.gender_slider = QSlider(Qt.Orientation.Horizontal)
        self.gender_slider.setRange(0, 100)
        self.gender_slider.setValue(50)
        self.gender_slider.valueChanged.connect(self._on_gender_changed)
        gender_layout.addWidget(self.gender_slider, 0, 0, 1, 3)

        gender_layout.addWidget(QLabel("Female"), 1, 0)
        self.gender_label = QLabel("Neutral")
        self.gender_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gender_layout.addWidget(self.gender_label, 1, 1)
        gender_layout.addWidget(QLabel("Male"), 1, 2)

        basic_layout.addWidget(gender_container, 3, 1)

        layout.addWidget(basic_group)

        # Model template
        template_group = QGroupBox("Model Template")
        template_layout = QGridLayout(template_group)

        template_layout.addWidget(QLabel("Template:"), 0, 0)
        self.template_combo = QComboBox()
        self.template_combo.addItems(
            [
                "Full Humanoid",
                "Upper Body Only",
                "Lower Body Only",
                "Torso + Arms",
                "Torso + Legs",
                "Custom",
            ]
        )
        template_layout.addWidget(self.template_combo, 0, 1)

        layout.addWidget(template_group)

        layout.addStretch()
        return tab

    def _create_proportions_tab(self) -> QWidget:
        """Create the proportions adjustment tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Proportion adjustments
        props_group = QGroupBox("Body Proportions")
        props_layout = QGridLayout(props_group)
        props_layout.setSpacing(10)

        self.proportion_sliders: dict[str, QSlider] = {}

        proportions = [
            ("shoulder_width", "Shoulder Width", 100),
            ("hip_width", "Hip Width", 100),
            ("arm_length", "Arm Length", 100),
            ("leg_length", "Leg Length", 100),
            ("torso_length", "Torso Length", 100),
            ("head_size", "Head Size", 100),
        ]

        for row, (key, label, default) in enumerate(proportions):
            props_layout.addWidget(QLabel(f"{label}:"), row, 0)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(50, 150)
            slider.setValue(default)
            self.proportion_sliders[key] = slider
            props_layout.addWidget(slider, row, 1)

            value_label = QLabel(f"{default}%")
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v}%"))
            props_layout.addWidget(value_label, row, 2)

        layout.addWidget(props_group)

        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_proportions)
        layout.addWidget(reset_btn)

        layout.addStretch()
        return tab

    def _create_options_tab(self) -> QWidget:
        """Create the advanced options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Geometry options
        geom_group = QGroupBox("Geometry Options")
        geom_layout = QGridLayout(geom_group)

        geom_layout.addWidget(QLabel("Default Geometry:"), 0, 0)
        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["Capsule", "Cylinder", "Box", "Sphere"])
        geom_layout.addWidget(self.geometry_combo, 0, 1)

        geom_layout.addWidget(QLabel("Collision Geometry:"), 1, 0)
        self.collision_combo = QComboBox()
        self.collision_combo.addItems(["Same as Visual", "Simplified", "None"])
        geom_layout.addWidget(self.collision_combo, 1, 1)

        layout.addWidget(geom_group)

        # Joint options
        joint_group = QGroupBox("Joint Options")
        joint_layout = QGridLayout(joint_group)

        joint_layout.addWidget(QLabel("Default Damping:"), 0, 0)
        self.damping_input = QDoubleSpinBox()
        self.damping_input.setRange(0.0, 100.0)
        self.damping_input.setDecimals(2)
        self.damping_input.setValue(0.5)
        joint_layout.addWidget(self.damping_input, 0, 1)

        joint_layout.addWidget(QLabel("Default Friction:"), 1, 0)
        self.friction_input = QDoubleSpinBox()
        self.friction_input.setRange(0.0, 100.0)
        self.friction_input.setDecimals(2)
        self.friction_input.setValue(0.0)
        joint_layout.addWidget(self.friction_input, 1, 1)

        layout.addWidget(joint_group)

        # Inertia options
        inertia_group = QGroupBox("Inertia Calculation")
        inertia_layout = QGridLayout(inertia_group)

        inertia_layout.addWidget(QLabel("Calculation Mode:"), 0, 0)
        self.inertia_mode_combo = QComboBox()
        self.inertia_mode_combo.addItems(["Primitive", "Mesh-based", "Scaled"])
        inertia_layout.addWidget(self.inertia_mode_combo, 0, 1)

        inertia_layout.addWidget(QLabel("Default Density (kg/m³):"), 1, 0)
        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(500.0, 2000.0)
        self.density_input.setDecimals(0)
        self.density_input.setValue(1050.0)
        inertia_layout.addWidget(self.density_input, 1, 1)

        layout.addWidget(inertia_group)

        layout.addStretch()
        return tab

    def _create_action_group(self) -> QGroupBox:
        """Create the action buttons group."""
        group = QGroupBox("Actions")
        layout = QGridLayout(group)
        layout.setSpacing(10)

        # Generate button
        generate_btn = QPushButton("Generate URDF")
        generate_btn.clicked.connect(self._generate_urdf)
        layout.addWidget(generate_btn, 0, 0)

        # Preview button
        preview_btn = QPushButton("Preview Structure")
        preview_btn.clicked.connect(self._preview_structure)
        layout.addWidget(preview_btn, 0, 1)

        # Export button
        export_btn = QPushButton("Export URDF File")
        export_btn.setObjectName("exportBtn")
        export_btn.clicked.connect(self._export_urdf)
        layout.addWidget(export_btn, 1, 0, 1, 2)

        return group

    def _create_results_group(self) -> QGroupBox:
        """Create the results display group."""
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(250)
        self.results_text.setPlaceholderText(
            "Generated URDF will appear here...\n\n" "Click 'Generate URDF' to create a model."
        )
        layout.addWidget(self.results_text)

        return group

    # ── Event handlers ──────────────────────────────────────────────────

    def _on_gender_changed(self, value: int) -> None:
        """Update gender label based on slider value."""
        if value < 33:
            self.gender_label.setText("Female")
        elif value > 66:
            self.gender_label.setText("Male")
        else:
            self.gender_label.setText("Neutral")

    def _reset_proportions(self) -> None:
        """Reset all proportion sliders to defaults."""
        for slider in self.proportion_sliders.values():
            slider.setValue(100)

    # ── LoD gateway ─────────────────────────────────────────────────────

    def _get_proportions(self) -> dict[str, float]:
        """Get current proportion factors."""
        return {key: slider.value() / 100.0 for key, slider in self.proportion_sliders.items()}

    def _get_config(self) -> Any:
        """Collect all GUI settings into a single URDFConfig object.

        This is the Law of Demeter gateway: the generator never
        reaches into GUI widgets directly.
        """
        from urdf_builder_gui.anthropometric_model import URDFConfig

        return URDFConfig(
            robot_name=self.name_input.currentText(),
            height_m=self.height_input.value(),
            mass_kg=self.mass_input.value(),
            gender_factor=self.gender_slider.value() / 100.0,
            template=self.template_combo.currentText(),
            geometry_type=self.geometry_combo.currentText().lower(),
            collision_geometry=self.collision_combo.currentText(),
            inertia_mode=self.inertia_mode_combo.currentText(),
            damping=self.damping_input.value(),
            friction=self.friction_input.value(),
            density=self.density_input.value(),
            proportions=self._get_proportions(),
        )

    # ── Actions (thin shells) ───────────────────────────────────────────

    def _generate_urdf(self) -> None:
        """Generate URDF from current parameters."""
        try:
            from urdf_builder_gui.urdf_generator import generate_urdf_xml

            config = self._get_config()
            urdf_xml = generate_urdf_xml(config)
            self.results_text.setPlainText(urdf_xml)
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")
            logger.info("URDF generated successfully for '%s'", config.robot_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("URDF generation failed: %s", exc)
            self.results_text.setPlainText(f"Generation failed: {exc}")
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")

    def _preview_structure(self) -> None:
        """Preview model structure via the extracted generator."""
        try:
            from urdf_builder_gui.preview_generator import generate_preview_text

            config = self._get_config()
            preview = generate_preview_text(config)
            self.results_text.setPlainText(preview)
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['text']};")
        except Exception as exc:  # noqa: BLE001
            logger.error("Preview generation failed: %s", exc)
            self.results_text.setPlainText(f"Preview failed: {exc}")
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")

    def _export_urdf(self) -> None:
        """Export URDF to file with structural validation."""
        content = self.results_text.toPlainText()

        # Must have generated URDF first
        if not content or not content.strip().startswith("<?xml"):
            self.results_text.setPlainText("Please generate URDF first before exporting.")
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['yellow']};")
            return

        # Validate before writing
        from urdf_builder_gui.urdf_generator import validate_urdf_structure

        is_valid, errors = validate_urdf_structure(content)
        if not is_valid:
            msg = "URDF validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            self.results_text.setPlainText(msg)
            self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")
            logger.warning("Export blocked: URDF validation failed: %s", errors)
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save URDF File",
            f"{self.name_input.currentText()}.urdf",
            "URDF Files (*.urdf);;XML Files (*.xml);;All Files (*)",
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.results_text.append(f"\n\nExported to: {file_path}")
                self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['green']};")
                logger.info("URDF exported to %s", file_path)
            except OSError as e:
                logger.error("Export failed: %s", e)
                self.results_text.append(f"\n\nExport failed: {e}")
                self.results_text.setStyleSheet(f"color: {CATPPUCCIN_MOCHA['red']};")


def main() -> int:
    """Run the Parametric URDF Builder application."""
    app = QApplication(sys.argv)
    window = URDFBuilderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

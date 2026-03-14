"""ControlsMixin -- visual controls panel for ElectrodeAdvisorWidget."""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...configs.ui_defaults import (
    DEFAULT_INTERACTION_MODE,
    ELECTRODE_EXTENSION_SLIDER,
    TRANSPARENCY_SLIDERS,
    ZOOM_SLIDER,
)

logger = logging.getLogger(__name__)


class ControlsMixin:
    """Mixin providing the visual controls panel on the right side."""

    # -- Attributes provided by the host class (declared for mypy) --
    _export_3d_plot: Any
    _export_charts: Any
    _on_auto_scale_changed: Any
    _on_color_scheme_changed: Any
    _on_electrode_extension_changed: Any
    _on_input_changed: Any
    _on_interaction_mode_changed: Any
    _on_metal_conductivity_changed: Any
    _on_zoom_slider_changed: Any
    _reset_3d_view: Any
    _set_view_preset: Any

    def _create_visual_controls_panel(self) -> None:
        """Create the visual controls panel for the right side."""
        self.visual_controls_panel = QFrame()
        self.visual_controls_panel.setFrameStyle(
            QFrame.Shape.StyledPanel,
        )

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        control_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

        control_scroll_widget = QWidget()
        control_layout = QVBoxLayout(control_scroll_widget)

        # Build each control section
        control_layout.addWidget(self._build_color_group())
        control_layout.addWidget(self._build_view_presets())
        control_layout.addWidget(self._build_zoom_group())
        control_layout.addWidget(self._build_interaction_group())
        control_layout.addWidget(self._build_transparency_group())
        control_layout.addWidget(self._build_visual_settings())
        control_layout.addWidget(self._build_export_group())
        control_layout.addWidget(self._build_display_components())
        control_layout.addWidget(self._build_conductive_model())
        control_layout.addWidget(self._build_current_display())
        control_layout.addStretch()

        control_scroll.setWidget(control_scroll_widget)

        panel_layout = QVBoxLayout(self.visual_controls_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(control_scroll)

    # -- Section builders ------------------------------------------------

    def _build_color_group(self) -> QGroupBox:
        """Build color controls (color-by mode, scheme, scale)."""
        group = QGroupBox("Color Controls")
        layout = QFormLayout(group)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(
            [
                "Default colors",
                "Current intensity",
                "Power dissipation",
            ]
        )
        self.color_mode_combo.setCurrentText("Current intensity")
        self.color_mode_combo.currentTextChanged.connect(
            self._on_input_changed,
        )
        layout.addRow("Color by:", self.color_mode_combo)

        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(
            [
                "Default",
                "Heat Map",
                "Cool Tones",
                "High Contrast",
                "Viridis",
                "Plasma",
                "Copper",
            ]
        )
        self.color_scheme_combo.currentTextChanged.connect(
            self._on_color_scheme_changed,
        )
        layout.addRow("Color Scheme:", self.color_scheme_combo)

        self.auto_scale_checkbox = QCheckBox("Auto-scale colors")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.stateChanged.connect(
            self._on_input_changed,
        )
        layout.addRow(self.auto_scale_checkbox)

        self.min_scale_input = QDoubleSpinBox()
        self.min_scale_input.setRange(0, 1000)
        self.min_scale_input.setValue(0)
        self.min_scale_input.setDecimals(1)
        self.min_scale_input.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons,
        )
        self.min_scale_input.setEnabled(False)
        self.min_scale_input.valueChanged.connect(
            self._on_input_changed,
        )
        layout.addRow("Min scale value:", self.min_scale_input)

        self.max_scale_input = QDoubleSpinBox()
        self.max_scale_input.setRange(0, 10000)
        self.max_scale_input.setValue(1000)
        self.max_scale_input.setDecimals(1)
        self.max_scale_input.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons,
        )
        self.max_scale_input.setEnabled(False)
        self.max_scale_input.valueChanged.connect(
            self._on_input_changed,
        )
        layout.addRow("Max scale value:", self.max_scale_input)

        self.auto_scale_checkbox.stateChanged.connect(
            self._on_auto_scale_changed,
        )
        return group

    def _build_view_presets(self) -> QGroupBox:
        """Build view preset buttons (default, top, side, front)."""
        group = QGroupBox("View Presets")
        layout = QGridLayout(group)

        presets = [
            ("Default", "default", 0, 0),
            ("Top Down", "top", 0, 1),
            ("Side View", "side", 1, 0),
            ("Front View", "front", 1, 1),
        ]
        for label, key, row, col in presets:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _checked=False, k=key: self._set_view_preset(k),
            )
            layout.addWidget(btn, row, col)
            setattr(self, f"view_{key}_btn", btn)

        return group

    def _build_zoom_group(self) -> QGroupBox:
        """Build zoom slider control."""
        group = QGroupBox("Zoom Control")
        layout = QVBoxLayout(group)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(
            ZOOM_SLIDER.min_value,
            ZOOM_SLIDER.max_value,
        )
        self.zoom_slider.setValue(ZOOM_SLIDER.default_value)
        self.zoom_slider.valueChanged.connect(
            self._on_zoom_slider_changed,
        )
        layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.zoom_label)
        return group

    def _build_interaction_group(self) -> QGroupBox:
        """Build 3D interaction mode controls."""
        group = QGroupBox("3D Interaction")
        layout = QVBoxLayout(group)

        self.rotation_mode_radio = QRadioButton("Rotation Mode")
        self.rotation_mode_radio.setChecked(True)
        self.rotation_mode_radio.toggled.connect(
            self._on_interaction_mode_changed,
        )

        self.pan_mode_radio = QRadioButton("Pan/Drag Mode")
        self.pan_mode_radio.toggled.connect(
            self._on_interaction_mode_changed,
        )

        layout.addWidget(self.rotation_mode_radio)
        layout.addWidget(self.pan_mode_radio)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self._reset_3d_view)
        layout.addWidget(self.reset_view_btn)

        self.interaction_mode = DEFAULT_INTERACTION_MODE
        return group

    def _build_transparency_group(self) -> QGroupBox:
        """Build per-component transparency sliders."""
        group = QGroupBox("Transparency")
        layout = QFormLayout(group)

        slider_specs = [
            ("electrodes", "Electrodes:", "electrode_alpha_slider"),
            ("glass", "Glass:", "glass_alpha_slider"),
            ("metal", "Metal:", "metal_alpha_slider"),
            ("paths", "Paths:", "path_alpha_slider"),
            ("refractory", "Refractory:", "refractory_alpha_slider"),
            ("metal_shell", "Shell:", "metal_shell_alpha_slider"),
        ]
        for cfg_key, label, attr in slider_specs:
            cfg = TRANSPARENCY_SLIDERS[cfg_key]
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(cfg.min_value, cfg.max_value)
            slider.setValue(cfg.default_value)
            slider.valueChanged.connect(self._on_input_changed)
            layout.addRow(label, slider)
            setattr(self, attr, slider)

        return group

    def _build_visual_settings(self) -> QGroupBox:
        """Build visual adjustment controls (electrode extension)."""
        group = QGroupBox("Visual Settings")
        layout = QFormLayout(group)

        cfg = ELECTRODE_EXTENSION_SLIDER
        self.electrode_extension_slider = QSlider(
            Qt.Orientation.Horizontal,
        )
        self.electrode_extension_slider.setRange(
            cfg.min_value,
            cfg.max_value,
        )
        self.electrode_extension_slider.setValue(cfg.default_value)
        self.electrode_extension_slider.setSingleStep(cfg.single_step)
        self.electrode_extension_slider.setTickInterval(
            cfg.tick_interval,
        )
        self.electrode_extension_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow,
        )
        self.electrode_extension_value_label = QLabel(
            f"{self.electrode_extension_slider.value()} in",
        )
        self.electrode_extension_slider.valueChanged.connect(
            self._on_electrode_extension_changed,
        )

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(self.electrode_extension_slider)
        row_layout.addWidget(self.electrode_extension_value_label)
        layout.addRow("Electrode Extension (in):", row)
        return group

    def _build_export_group(self) -> QGroupBox:
        """Build export buttons (3D plot and charts)."""
        group = QGroupBox("Export")
        layout = QVBoxLayout(group)

        self.export_plot_btn = QPushButton("Export 3D Plot")
        self.export_plot_btn.clicked.connect(self._export_3d_plot)
        layout.addWidget(self.export_plot_btn)

        self.export_charts_btn = QPushButton("Export Charts")
        self.export_charts_btn.clicked.connect(self._export_charts)
        layout.addWidget(self.export_charts_btn)
        return group

    def _build_display_components(self) -> QGroupBox:
        """Build component visibility toggle checkboxes."""
        group = QGroupBox("Display Components")
        layout = QGridLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        toggles = [
            ("show_refractory_checkbox", "Refractory", False, 0, 0),
            ("show_glass_checkbox", "Glass", True, 1, 0),
            ("show_electrodes_checkbox", "Electrodes", True, 2, 0),
            ("show_metal_shell_checkbox", "Shell", False, 0, 1),
            ("show_metal_checkbox", "Metal", True, 1, 1),
            ("show_paths_checkbox", "Conductive Paths", True, 2, 1),
            ("show_axis_labels_checkbox", "Axis Labels", False, 3, 0),
            ("show_electrode_labels_checkbox", "Depth", True, 3, 1),
        ]
        for attr, label, default, row, col in toggles:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.stateChanged.connect(self._on_input_changed)
            layout.addWidget(cb, row, col)
            setattr(self, attr, cb)

        return group

    def _build_conductive_model(self) -> QGroupBox:
        """Build conductive model toggle."""
        group = QGroupBox("Conductive Model")
        layout = QFormLayout(group)

        self.metal_conductive_checkbox = QCheckBox(
            "Enable Metal Layer Conduction",
        )
        self.metal_conductive_checkbox.setChecked(True)
        self.metal_conductive_checkbox.stateChanged.connect(
            self._on_metal_conductivity_changed,
        )
        layout.addRow(self.metal_conductive_checkbox)

        info = QLabel(
            "When disabled, current flows only through glass paths",
        )
        info.setStyleSheet(
            "QLabel { color: #666666; font-size: 9pt; }",
        )
        layout.addRow(info)
        return group

    def _build_current_display(self) -> QGroupBox:
        """Build current and resistance display toggles."""
        group = QGroupBox("Current Display")
        layout = QVBoxLayout(group)
        layout.setSpacing(2)
        layout.setContentsMargins(6, 6, 6, 6)

        _info_style = "QLabel { color: #666666; font-size: 8pt; }"

        self.show_current_values_checkbox = QCheckBox(
            "Show Current Values on Paths",
        )
        self.show_current_values_checkbox.setChecked(False)
        self.show_current_values_checkbox.stateChanged.connect(
            self._on_input_changed,
        )
        layout.addWidget(self.show_current_values_checkbox)

        current_info = QLabel(
            "Displays current (A) on each conductive path",
        )
        current_info.setStyleSheet(_info_style)
        layout.addWidget(current_info)

        self.show_resistance_values_checkbox = QCheckBox(
            "Show Resistance Values on Paths",
        )
        self.show_resistance_values_checkbox.setChecked(False)
        self.show_resistance_values_checkbox.stateChanged.connect(
            self._on_input_changed,
        )
        layout.addWidget(self.show_resistance_values_checkbox)

        resistance_info = QLabel(
            "Displays resistance (\u03a9) on each conductive path",
        )
        resistance_info.setStyleSheet(_info_style)
        layout.addWidget(resistance_info)
        return group

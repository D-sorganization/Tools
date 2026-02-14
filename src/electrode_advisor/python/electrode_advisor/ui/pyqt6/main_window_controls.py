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
        """Create the visual controls panel for the right side"""
        self.visual_controls_panel = QFrame()
        self.visual_controls_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        # self.visual_controls_panel.setMaximumWidth(400)  # REMOVED: Allow flexible sizing

        # Create scroll area for the controls
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create scrollable widget
        control_scroll_widget = QWidget()
        control_layout = QVBoxLayout(control_scroll_widget)

        # COLOR CONTROLS AT TOP
        # Color scheme selection
        color_group = QGroupBox("Color Controls")
        color_layout = QFormLayout(color_group)

        # Path coloring mode selection
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(
            [
                "Default colors",
                "Current intensity",
                "Power dissipation",
                "Temperature gradient",
            ]
        )
        self.color_mode_combo.setCurrentText("Current intensity")
        self.color_mode_combo.currentTextChanged.connect(self._on_input_changed)
        color_layout.addRow("Color by:", self.color_mode_combo)

        # Color scheme selection
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
            self._on_color_scheme_changed
        )
        color_layout.addRow("Color Scheme:", self.color_scheme_combo)

        # Color scale range controls
        self.auto_scale_checkbox = QCheckBox("Auto-scale colors")
        self.auto_scale_checkbox.setChecked(True)
        self.auto_scale_checkbox.stateChanged.connect(self._on_input_changed)
        color_layout.addRow(self.auto_scale_checkbox)

        self.min_scale_input = QDoubleSpinBox()
        self.min_scale_input.setRange(0, 1000)
        self.min_scale_input.setValue(0)
        self.min_scale_input.setDecimals(1)
        self.min_scale_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.min_scale_input.setEnabled(False)
        self.min_scale_input.valueChanged.connect(self._on_input_changed)
        color_layout.addRow("Min scale value:", self.min_scale_input)

        self.max_scale_input = QDoubleSpinBox()
        self.max_scale_input.setRange(0, 10000)
        self.max_scale_input.setValue(1000)
        self.max_scale_input.setDecimals(1)
        self.max_scale_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.max_scale_input.setEnabled(False)
        self.max_scale_input.valueChanged.connect(self._on_input_changed)
        color_layout.addRow("Max scale value:", self.max_scale_input)

        # Connect auto-scale to enable/disable manual scale inputs
        self.auto_scale_checkbox.stateChanged.connect(self._on_auto_scale_changed)

        control_layout.addWidget(color_group)

        # View presets
        preset_group = QGroupBox("View Presets")
        preset_layout = QGridLayout(preset_group)

        self.view_default_btn = QPushButton("Default")
        self.view_default_btn.clicked.connect(lambda: self._set_view_preset("default"))
        preset_layout.addWidget(self.view_default_btn, 0, 0)

        self.view_top_btn = QPushButton("Top Down")
        self.view_top_btn.clicked.connect(lambda: self._set_view_preset("top"))
        preset_layout.addWidget(self.view_top_btn, 0, 1)

        self.view_side_btn = QPushButton("Side View")
        self.view_side_btn.clicked.connect(lambda: self._set_view_preset("side"))
        preset_layout.addWidget(self.view_side_btn, 1, 0)

        self.view_front_btn = QPushButton("Front View")
        self.view_front_btn.clicked.connect(lambda: self._set_view_preset("front"))
        preset_layout.addWidget(self.view_front_btn, 1, 1)

        control_layout.addWidget(preset_group)

        # Zoom slider
        zoom_group = QGroupBox("Zoom Control")
        zoom_layout = QVBoxLayout(zoom_group)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(ZOOM_SLIDER.min_value, ZOOM_SLIDER.max_value)
        self.zoom_slider.setValue(ZOOM_SLIDER.default_value)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        zoom_layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)

        control_layout.addWidget(zoom_group)

        # 3D Interaction Mode Controls
        interaction_group = QGroupBox("3D Interaction")
        interaction_layout = QVBoxLayout(interaction_group)

        # Create radio buttons for interaction modes
        self.rotation_mode_radio = QRadioButton("Rotation Mode")
        self.rotation_mode_radio.setChecked(True)  # Default mode
        self.rotation_mode_radio.toggled.connect(self._on_interaction_mode_changed)

        self.pan_mode_radio = QRadioButton("Pan/Drag Mode")
        self.pan_mode_radio.toggled.connect(self._on_interaction_mode_changed)

        interaction_layout.addWidget(self.rotation_mode_radio)
        interaction_layout.addWidget(self.pan_mode_radio)

        # Reset view button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self._reset_3d_view)
        interaction_layout.addWidget(self.reset_view_btn)

        control_layout.addWidget(interaction_group)

        # Initialize interaction mode tracking
        self.interaction_mode = DEFAULT_INTERACTION_MODE

        # Component visibility toggles - created below in proper section

        # Transparency controls
        trans_group = QGroupBox("Transparency")
        # trans_group.setMinimumHeight(180)  # REMOVED: Allow flexible sizing
        trans_layout = QFormLayout(trans_group)

        electrode_cfg = TRANSPARENCY_SLIDERS["electrodes"]
        self.electrode_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.electrode_alpha_slider.setRange(
            electrode_cfg.min_value, electrode_cfg.max_value
        )
        self.electrode_alpha_slider.setValue(electrode_cfg.default_value)
        self.electrode_alpha_slider.valueChanged.connect(self._on_input_changed)
        trans_layout.addRow("Electrodes:", self.electrode_alpha_slider)

        glass_cfg = TRANSPARENCY_SLIDERS["glass"]
        self.glass_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.glass_alpha_slider.setRange(glass_cfg.min_value, glass_cfg.max_value)
        self.glass_alpha_slider.setValue(glass_cfg.default_value)
        self.glass_alpha_slider.valueChanged.connect(self._on_input_changed)
        trans_layout.addRow("Glass:", self.glass_alpha_slider)

        metal_cfg = TRANSPARENCY_SLIDERS["metal"]
        self.metal_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.metal_alpha_slider.setRange(metal_cfg.min_value, metal_cfg.max_value)
        self.metal_alpha_slider.setValue(metal_cfg.default_value)
        self.metal_alpha_slider.valueChanged.connect(self._on_input_changed)
        trans_layout.addRow("Metal:", self.metal_alpha_slider)

        paths_cfg = TRANSPARENCY_SLIDERS["paths"]
        self.path_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.path_alpha_slider.setRange(paths_cfg.min_value, paths_cfg.max_value)
        self.path_alpha_slider.setValue(paths_cfg.default_value)
        self.path_alpha_slider.valueChanged.connect(self._on_input_changed)
        trans_layout.addRow("Paths:", self.path_alpha_slider)

        refractory_cfg = TRANSPARENCY_SLIDERS["refractory"]
        self.refractory_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.refractory_alpha_slider.setRange(
            refractory_cfg.min_value, refractory_cfg.max_value
        )
        self.refractory_alpha_slider.setValue(refractory_cfg.default_value)
        self.refractory_alpha_slider.valueChanged.connect(self._on_input_changed)
        trans_layout.addRow("Refractory:", self.refractory_alpha_slider)

        shell_cfg = TRANSPARENCY_SLIDERS["metal_shell"]
        self.metal_shell_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.metal_shell_alpha_slider.setRange(shell_cfg.min_value, shell_cfg.max_value)
        self.metal_shell_alpha_slider.setValue(shell_cfg.default_value)
        self.metal_shell_alpha_slider.valueChanged.connect(self._on_input_changed)
        trans_layout.addRow("Shell:", self.metal_shell_alpha_slider)

        control_layout.addWidget(trans_group)

        # Visual adjustment controls
        visual_group = QGroupBox("Visual Settings")
        visual_layout = QFormLayout(visual_group)

        self.electrode_extension_slider = QSlider(Qt.Orientation.Horizontal)
        self.electrode_extension_slider.setRange(
            ELECTRODE_EXTENSION_SLIDER.min_value, ELECTRODE_EXTENSION_SLIDER.max_value
        )
        self.electrode_extension_slider.setValue(
            ELECTRODE_EXTENSION_SLIDER.default_value
        )
        self.electrode_extension_slider.setSingleStep(
            ELECTRODE_EXTENSION_SLIDER.single_step
        )
        self.electrode_extension_slider.setTickInterval(
            ELECTRODE_EXTENSION_SLIDER.tick_interval
        )
        self.electrode_extension_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.electrode_extension_value_label = QLabel(
            f"{self.electrode_extension_slider.value()} in"
        )
        self.electrode_extension_slider.valueChanged.connect(
            self._on_electrode_extension_changed
        )
        slider_row_widget = QWidget()
        slider_row_layout = QHBoxLayout(slider_row_widget)
        slider_row_layout.setContentsMargins(0, 0, 0, 0)
        slider_row_layout.setSpacing(8)
        slider_row_layout.addWidget(self.electrode_extension_slider)
        slider_row_layout.addWidget(self.electrode_extension_value_label)
        visual_layout.addRow("Electrode Extension (in):", slider_row_widget)
        control_layout.addWidget(visual_group)

        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_plot_btn = QPushButton("Export 3D Plot")
        self.export_plot_btn.clicked.connect(self._export_3d_plot)
        export_layout.addWidget(self.export_plot_btn)

        self.export_charts_btn = QPushButton("Export Charts")
        self.export_charts_btn.clicked.connect(self._export_charts)
        export_layout.addWidget(self.export_charts_btn)

        control_layout.addWidget(export_group)

        control_layout.addStretch()

        # Set up the scroll area
        control_scroll.setWidget(control_scroll_widget)

        # Add scroll area to the panel
        panel_layout = QVBoxLayout(self.visual_controls_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(control_scroll)

        # Component visibility toggles - Updated labels and compact spacing
        display_group = QGroupBox("Display Components")
        # display_group.setMinimumHeight(120)  # REMOVED: Allow flexible sizing
        display_layout = QGridLayout(display_group)
        display_layout.setSpacing(2)  # Further decreased spacing between entries
        display_layout.setContentsMargins(6, 6, 6, 6)  # Even tighter margins

        # Column 1 (left)
        self.show_refractory_checkbox = QCheckBox("Refractory")
        self.show_refractory_checkbox.setChecked(False)  # Default unchecked
        self.show_refractory_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_refractory_checkbox, 0, 0)

        self.show_glass_checkbox = QCheckBox("Glass")
        self.show_glass_checkbox.setChecked(True)
        self.show_glass_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_glass_checkbox, 1, 0)

        self.show_electrodes_checkbox = QCheckBox("Electrodes")
        self.show_electrodes_checkbox.setChecked(True)
        self.show_electrodes_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_electrodes_checkbox, 2, 0)

        # Column 2 (right) - Updated labels, removed vessel shell
        self.show_metal_shell_checkbox = QCheckBox(
            "Shell"
        )  # Changed from "Metal Shell"
        self.show_metal_shell_checkbox.setChecked(False)  # Default unchecked
        self.show_metal_shell_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_metal_shell_checkbox, 0, 1)

        self.show_metal_checkbox = QCheckBox("Metal")  # Changed from "Metal Layer"
        self.show_metal_checkbox.setChecked(True)
        self.show_metal_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_metal_checkbox, 1, 1)

        self.show_paths_checkbox = QCheckBox("Conductive Paths")
        self.show_paths_checkbox.setChecked(True)
        self.show_paths_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_paths_checkbox, 2, 1)

        # Current Display Controls - separate group for better organization
        current_display_group = QGroupBox("Current Display")
        current_display_layout = QVBoxLayout(current_display_group)
        current_display_layout.setSpacing(2)
        current_display_layout.setContentsMargins(6, 6, 6, 6)

        self.show_current_values_checkbox = QCheckBox("Show Current Values on Paths")
        self.show_current_values_checkbox.setChecked(
            False
        )  # Default off for cleaner view
        self.show_current_values_checkbox.stateChanged.connect(self._on_input_changed)
        current_display_layout.addWidget(self.show_current_values_checkbox)

        # Add explanatory label
        current_info_label = QLabel("Displays current (A) on each conductive path")
        current_info_label.setStyleSheet("QLabel { color: #666666; font-size: 8pt; }")
        current_display_layout.addWidget(current_info_label)

        # Resistance display checkbox
        self.show_resistance_values_checkbox = QCheckBox(
            "Show Resistance Values on Paths"
        )
        self.show_resistance_values_checkbox.setChecked(
            False
        )  # Default off for cleaner view
        self.show_resistance_values_checkbox.stateChanged.connect(
            self._on_input_changed
        )
        current_display_layout.addWidget(self.show_resistance_values_checkbox)

        # Add explanatory label for resistance
        resistance_info_label = QLabel(
            "Displays resistance (Ω) on each conductive path"
        )
        resistance_info_label.setStyleSheet(
            "QLabel { color: #666666; font-size: 8pt; }"
        )
        current_display_layout.addWidget(resistance_info_label)

        self.show_axis_labels_checkbox = QCheckBox("Axis Labels")
        self.show_axis_labels_checkbox.setChecked(False)
        self.show_axis_labels_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_axis_labels_checkbox, 3, 0)

        # Electrode Depth Labels - Updated label
        self.show_electrode_labels_checkbox = QCheckBox(
            "Depth"
        )  # Changed from "Electrode Depth"
        self.show_electrode_labels_checkbox.setChecked(True)
        self.show_electrode_labels_checkbox.stateChanged.connect(self._on_input_changed)
        display_layout.addWidget(self.show_electrode_labels_checkbox, 3, 1)

        control_layout.addWidget(display_group)

        # Conductive Model Controls
        model_group = QGroupBox("Conductive Model")
        model_layout = QFormLayout(model_group)

        self.metal_conductive_checkbox = QCheckBox("Enable Metal Layer Conduction")
        self.metal_conductive_checkbox.setChecked(True)  # Default enabled
        self.metal_conductive_checkbox.stateChanged.connect(
            self._on_metal_conductivity_changed
        )
        model_layout.addRow(self.metal_conductive_checkbox)

        # Add explanatory label
        info_label = QLabel("When disabled, current flows only through glass paths")
        info_label.setStyleSheet("QLabel { color: #666666; font-size: 9pt; }")
        model_layout.addRow(info_label)

        control_layout.addWidget(model_group)

        # Add the current display group
        control_layout.addWidget(current_display_group)

        control_layout.addStretch()

        # Set up the scroll area
        control_scroll.setWidget(control_scroll_widget)

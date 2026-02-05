"""TRC Vessel Designer Main Window - PyQt6 GUI.

This is a new PyQt6 GUI that uses the shared TRC geometry engine
from Tools/src/shared/python/upstream_drift_tools/calculators/mechanical/.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Import the shared engine from Tools
from shared.python.upstream_drift_tools.calculators.mechanical.trc_geometry import (
    LayerConfig,
    TRCGeometryEngine,
    VesselDimensions,
    VesselGeometryResult,
)

logger = logging.getLogger(__name__)


# Default refractory presets
REFRACTORY_PRESETS: dict[str, list[dict[str, Any]]] = {
    "standard": [
        {
            "name": "High-Alumina Working Lining",
            "thickness": 6.0,
            "density": 150.0,
            "color": "#94a3b8",
        },
        {
            "name": "Insulating Firebrick",
            "thickness": 4.5,
            "density": 60.0,
            "color": "#cbd5e1",
        },
        {
            "name": "Microporous Insulation",
            "thickness": 1.0,
            "density": 20.0,
            "color": "#e2e8f0",
        },
    ],
    "high_temp": [
        {
            "name": "Chrome-Alumina Working",
            "thickness": 8.0,
            "density": 200.0,
            "color": "#475569",
        },
        {
            "name": "High-Alumina Backup",
            "thickness": 4.0,
            "density": 150.0,
            "color": "#64748b",
        },
        {
            "name": "Insulating Firebrick",
            "thickness": 4.5,
            "density": 60.0,
            "color": "#94a3b8",
        },
        {
            "name": "Calcium Silicate Board",
            "thickness": 2.0,
            "density": 30.0,
            "color": "#e2e8f0",
        },
    ],
    "economy": [
        {
            "name": "Castable Refractory",
            "thickness": 8.0,
            "density": 130.0,
            "color": "#78716c",
        },
        {
            "name": "Ceramic Fiber Blanket",
            "thickness": 2.0,
            "density": 12.0,
            "color": "#d6d3d1",
        },
    ],
}


class TRCVesselDesignerWidget(QWidget):
    """Main widget for TRC Vessel Designer using the shared geometry engine."""

    # Signals
    design_updated = pyqtSignal(dict)
    calculation_complete = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the TRC Vessel Designer widget."""
        super().__init__(parent)

        # Initialize the shared geometry engine
        self.engine = TRCGeometryEngine()

        # Default dimensions
        self.dimensions = VesselDimensions(
            cylinder_height=72.0,  # 6 feet
            cylinder_diameter=24.0,  # 2 feet
            cone_height=24.0,  # 2 feet
            cone_bottom_diameter=6.0,  # 6 inches
            cone_interior_hole=4.0,  # 4 inches
            top_refractory_thickness=6.0,
        )

        # Default layers
        self.layers = self._create_layers_from_preset("standard")

        # Results storage
        self.results: VesselGeometryResult | None = None

        # Operating conditions
        self.operating_temp = 1400.0  # C
        self.operating_pressure = 101.325  # kPa
        self.volumetric_flow = 2000.0  # m3/hr

        # Initialize UI
        self._init_ui()
        self._apply_styling()

        # Defer initial calculation
        QTimer.singleShot(100, self.calculate_geometry)

    def _create_layers_from_preset(self, preset_name: str) -> list[LayerConfig]:
        """Create layer configurations from a preset."""
        preset = REFRACTORY_PRESETS.get(preset_name, REFRACTORY_PRESETS["standard"])
        return [
            LayerConfig(
                name=layer["name"],
                thickness=layer["thickness"],
                density=layer["density"],
                color=layer["color"],
            )
            for layer in preset
        ]

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(6)
        main_splitter.setChildrenCollapsible(False)

        # Left panel - inputs
        left_panel = self._create_input_panel()
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        main_splitter.addWidget(left_scroll)

        # Center panel - results and visualization
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        # Status panel
        self.status_panel = self._create_status_panel()
        center_layout.addWidget(self.status_panel)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Results tab
        results_panel = self._create_results_panel()
        self.results_tabs.addTab(results_panel, "Results")

        # Layer details tab
        layer_panel = self._create_layer_details_panel()
        self.results_tabs.addTab(layer_panel, "Layer Details")

        center_layout.addWidget(self.results_tabs)
        main_splitter.addWidget(center_widget)

        # Set splitter proportions
        main_splitter.setSizes([400, 600])

        main_layout.addWidget(main_splitter)

    def _create_input_panel(self) -> QWidget:
        """Create the input parameter panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Vessel dimensions group
        dim_group = QGroupBox("Vessel Dimensions")
        dim_layout = QFormLayout(dim_group)

        self.cylinder_height_input = QDoubleSpinBox()
        self.cylinder_height_input.setRange(12, 300)
        self.cylinder_height_input.setValue(72.0)
        self.cylinder_height_input.setDecimals(1)
        self.cylinder_height_input.setSuffix(" in")
        self.cylinder_height_input.valueChanged.connect(self._on_input_changed)
        dim_layout.addRow("Cylinder Height:", self.cylinder_height_input)

        self.cylinder_diameter_input = QDoubleSpinBox()
        self.cylinder_diameter_input.setRange(6, 120)
        self.cylinder_diameter_input.setValue(24.0)
        self.cylinder_diameter_input.setDecimals(1)
        self.cylinder_diameter_input.setSuffix(" in")
        self.cylinder_diameter_input.valueChanged.connect(self._on_input_changed)
        dim_layout.addRow("Cylinder Diameter:", self.cylinder_diameter_input)

        self.cone_height_input = QDoubleSpinBox()
        self.cone_height_input.setRange(0, 100)
        self.cone_height_input.setValue(24.0)
        self.cone_height_input.setDecimals(1)
        self.cone_height_input.setSuffix(" in")
        self.cone_height_input.valueChanged.connect(self._on_input_changed)
        dim_layout.addRow("Cone Height:", self.cone_height_input)

        self.cone_bottom_input = QDoubleSpinBox()
        self.cone_bottom_input.setRange(1, 24)
        self.cone_bottom_input.setValue(6.0)
        self.cone_bottom_input.setDecimals(1)
        self.cone_bottom_input.setSuffix(" in")
        self.cone_bottom_input.valueChanged.connect(self._on_input_changed)
        dim_layout.addRow("Cone Bottom Diameter:", self.cone_bottom_input)

        layout.addWidget(dim_group)

        # Refractory group
        ref_group = QGroupBox("Refractory Configuration")
        ref_layout = QFormLayout(ref_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            ["Standard (3-layer)", "High Temperature (4-layer)", "Economy (2-layer)"]
        )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        ref_layout.addRow("Preset:", self.preset_combo)

        self.total_thickness_label = QLabel("11.5 in")
        self.total_thickness_label.setStyleSheet(
            "font-weight: bold; background: #f0f0f0; padding: 4px; border-radius: 4px;"
        )
        ref_layout.addRow("Total Thickness:", self.total_thickness_label)

        layout.addWidget(ref_group)

        # Operating conditions group
        op_group = QGroupBox("Operating Conditions")
        op_layout = QFormLayout(op_group)

        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(500, 2000)
        self.temp_input.setValue(1400.0)
        self.temp_input.setDecimals(0)
        self.temp_input.setSuffix(" C")
        self.temp_input.valueChanged.connect(self._on_input_changed)
        op_layout.addRow("Operating Temperature:", self.temp_input)

        self.pressure_input = QDoubleSpinBox()
        self.pressure_input.setRange(50, 500)
        self.pressure_input.setValue(101.325)
        self.pressure_input.setDecimals(1)
        self.pressure_input.setSuffix(" kPa")
        self.pressure_input.valueChanged.connect(self._on_input_changed)
        op_layout.addRow("Operating Pressure:", self.pressure_input)

        self.flow_input = QDoubleSpinBox()
        self.flow_input.setRange(100, 50000)
        self.flow_input.setValue(2000.0)
        self.flow_input.setDecimals(0)
        self.flow_input.setSuffix(" m3/hr")
        self.flow_input.valueChanged.connect(self._on_input_changed)
        op_layout.addRow("Gas Flow Rate:", self.flow_input)

        layout.addWidget(op_group)

        layout.addStretch()
        return panel

    def _create_status_panel(self) -> QGroupBox:
        """Create the status panel."""
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        status_group.setMaximumHeight(60)

        self.status_label = QLabel("System Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)

        return status_group

    def _create_results_panel(self) -> QWidget:
        """Create the results display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Volume summary
        vol_group = QGroupBox("Volume Summary")
        vol_layout = QGridLayout(vol_group)

        self.net_volume_label = QLabel("-- ft3")
        self.net_volume_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2563eb;"
        )
        vol_layout.addWidget(QLabel("Net Internal Volume:"), 0, 0)
        vol_layout.addWidget(self.net_volume_label, 0, 1)

        self.gross_volume_label = QLabel("-- ft3")
        vol_layout.addWidget(QLabel("Gross Volume:"), 1, 0)
        vol_layout.addWidget(self.gross_volume_label, 1, 1)

        self.refractory_volume_label = QLabel("-- ft3")
        vol_layout.addWidget(QLabel("Refractory Volume:"), 2, 0)
        vol_layout.addWidget(self.refractory_volume_label, 2, 1)

        layout.addWidget(vol_group)

        # Mass summary
        mass_group = QGroupBox("Mass Summary")
        mass_layout = QGridLayout(mass_group)

        self.total_mass_label = QLabel("-- lb")
        self.total_mass_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        mass_layout.addWidget(QLabel("Total Refractory Mass:"), 0, 0)
        mass_layout.addWidget(self.total_mass_label, 0, 1)

        self.surface_area_label = QLabel("-- ft2")
        mass_layout.addWidget(QLabel("Outside Surface Area:"), 1, 0)
        mass_layout.addWidget(self.surface_area_label, 1, 1)

        layout.addWidget(mass_group)

        # Residence time
        res_group = QGroupBox("Residence Time")
        res_layout = QGridLayout(res_group)

        self.residence_time_label = QLabel("-- s")
        self.residence_time_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #16a34a;"
        )
        res_layout.addWidget(QLabel("Residence Time:"), 0, 0)
        res_layout.addWidget(self.residence_time_label, 0, 1)

        self.void_diameter_label = QLabel("-- in")
        res_layout.addWidget(QLabel("Void Diameter:"), 1, 0)
        res_layout.addWidget(self.void_diameter_label, 1, 1)

        layout.addWidget(res_group)

        layout.addStretch()
        return panel

    def _create_layer_details_panel(self) -> QWidget:
        """Create the layer details panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.layer_details_group = QGroupBox("Refractory Layers")
        self.layer_details_layout = QVBoxLayout(self.layer_details_group)

        layout.addWidget(self.layer_details_group)
        layout.addStretch()

        return panel

    def _apply_styling(self) -> None:
        """Apply styling to the widget."""
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 248, 255))
        self.setPalette(palette)

        self.results_tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background: white;
            }
            QTabBar::tab {
                background: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #cccccc;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: white;
            }
            """
        )

    def _on_input_changed(self) -> None:
        """Handle input parameter changes."""
        self.calculate_geometry()

    def _on_preset_changed(self, index: int) -> None:
        """Handle preset selection changes."""
        presets = ["standard", "high_temp", "economy"]
        if 0 <= index < len(presets):
            self.layers = self._create_layers_from_preset(presets[index])
            self._update_thickness_display()
            self.calculate_geometry()

    def _update_thickness_display(self) -> None:
        """Update the total thickness display."""
        total = sum(layer.thickness for layer in self.layers)
        self.total_thickness_label.setText(f"{total:.1f} in")

    def calculate_geometry(self) -> None:
        """Calculate vessel geometry using the shared engine."""
        try:
            self._update_status("Calculating...", "info")

            # Update dimensions from inputs
            self.dimensions = VesselDimensions(
                cylinder_height=self.cylinder_height_input.value(),
                cylinder_diameter=self.cylinder_diameter_input.value(),
                cone_height=self.cone_height_input.value(),
                cone_bottom_diameter=self.cone_bottom_input.value(),
                cone_interior_hole=4.0,
                top_refractory_thickness=(
                    self.layers[0].thickness if self.layers else 6.0
                ),
            )

            # Calculate geometry using shared engine
            self.results = self.engine.calculate_geometry(self.dimensions, self.layers)

            # Update display
            self._update_results_display()
            self._update_layer_details()
            self._update_status("Calculation complete", "ok")

            # Emit signal
            self.calculation_complete.emit(self._get_results_dict())

        except Exception as e:
            logger.exception("Calculation failed")
            self._update_status(f"Error: {e}", "error")

    def _update_results_display(self) -> None:
        """Update the results display."""
        if not self.results:
            return

        # Volume results
        self.net_volume_label.setText(f"{self.results.interior_volume_ft3:.2f} ft3")
        self.gross_volume_label.setText(f"{self.results.total_volume_ft3:.2f} ft3")

        refractory_vol = (
            self.results.total_volume_ft3 - self.results.interior_volume_ft3
        )
        self.refractory_volume_label.setText(f"{refractory_vol:.2f} ft3")

        # Mass results
        self.total_mass_label.setText(f"{self.results.total_mass_lb:.1f} lb")
        self.surface_area_label.setText(
            f"{self.results.outside_surface_area_ft2:.1f} ft2"
        )

        # Void dimensions
        self.void_diameter_label.setText(f"{self.results.void_diameter_inches:.1f} in")

        # Calculate residence time
        # Convert interior volume from ft3 to m3 (1 ft3 = 0.0283168 m3)
        interior_volume_m3 = self.results.interior_volume_ft3 * 0.0283168
        flow_m3_per_sec = self.flow_input.value() / 3600

        if flow_m3_per_sec > 0:
            residence_time = interior_volume_m3 / flow_m3_per_sec
            self.residence_time_label.setText(f"{residence_time:.1f} s")
        else:
            self.residence_time_label.setText("-- s")

    def _update_layer_details(self) -> None:
        """Update the layer details panel."""
        # Clear existing widgets
        while self.layer_details_layout.count():
            item = self.layer_details_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        if not self.results:
            return

        # Add layer details
        for layer_result in self.results.layers:
            layer_widget = QWidget()
            layer_layout = QGridLayout(layer_widget)
            layer_widget.setStyleSheet(
                "background: #f8f9fa; border-radius: 4px; padding: 4px;"
            )

            layer_layout.addWidget(QLabel(f"<b>{layer_result.name}</b>"), 0, 0, 1, 2)
            layer_layout.addWidget(QLabel("Volume:"), 1, 0)
            layer_layout.addWidget(QLabel(f"{layer_result.volume_ft3:.3f} ft3"), 1, 1)
            layer_layout.addWidget(QLabel("Mass:"), 2, 0)
            layer_layout.addWidget(QLabel(f"{layer_result.mass_lb:.1f} lb"), 2, 1)

            self.layer_details_layout.addWidget(layer_widget)

        self.layer_details_layout.addStretch()

    def _update_status(self, message: str, status_type: str = "ok") -> None:
        """Update the status display."""
        self.status_label.setText(message)

        colors = {
            "ok": "#c8ffc8",
            "info": "#e0e0e0",
            "warn": "#ffffb4",
            "error": "#ff9696",
        }
        color = colors.get(status_type, colors["ok"])
        self.status_panel.setStyleSheet(f"background-color: {color}")

    def _get_results_dict(self) -> dict[str, Any]:
        """Get results as a dictionary."""
        if not self.results:
            return {}

        return {
            "total_volume_ft3": self.results.total_volume_ft3,
            "interior_volume_ft3": self.results.interior_volume_ft3,
            "total_mass_lb": self.results.total_mass_lb,
            "surface_area_ft2": self.results.outside_surface_area_ft2,
            "void_diameter_in": self.results.void_diameter_inches,
            "layers": [
                {
                    "name": lr.name,
                    "volume_ft3": lr.volume_ft3,
                    "mass_lb": lr.mass_lb,
                }
                for lr in self.results.layers
            ],
        }

    def get_current_state(self) -> dict[str, Any]:
        """Get current state for saving."""
        return {
            "dimensions": {
                "cylinder_height": self.cylinder_height_input.value(),
                "cylinder_diameter": self.cylinder_diameter_input.value(),
                "cone_height": self.cone_height_input.value(),
                "cone_bottom_diameter": self.cone_bottom_input.value(),
            },
            "preset_index": self.preset_combo.currentIndex(),
            "operating_conditions": {
                "temperature": self.temp_input.value(),
                "pressure": self.pressure_input.value(),
                "flow_rate": self.flow_input.value(),
            },
            "results": self._get_results_dict(),
        }

    def set_current_state(self, state: dict[str, Any]) -> None:
        """Restore state from saved data."""
        if "dimensions" in state:
            dims = state["dimensions"]
            self.cylinder_height_input.setValue(dims.get("cylinder_height", 72.0))
            self.cylinder_diameter_input.setValue(dims.get("cylinder_diameter", 24.0))
            self.cone_height_input.setValue(dims.get("cone_height", 24.0))
            self.cone_bottom_input.setValue(dims.get("cone_bottom_diameter", 6.0))

        if "preset_index" in state:
            self.preset_combo.setCurrentIndex(state["preset_index"])

        if "operating_conditions" in state:
            ops = state["operating_conditions"]
            self.temp_input.setValue(ops.get("temperature", 1400.0))
            self.pressure_input.setValue(ops.get("pressure", 101.325))
            self.flow_input.setValue(ops.get("flow_rate", 2000.0))

        self.calculate_geometry()

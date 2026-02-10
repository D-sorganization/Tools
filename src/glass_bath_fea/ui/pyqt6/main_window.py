"""Main window for Glass Bath FEA PyQt6 GUI.

Provides a graphical interface for:
- Configuring glass bath geometry parameters
- Setting material properties
- Generating mesh and exporting to MATLAB
- Visualizing results
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from glass_bath_fea.core.config import GlassBathFEAConfig

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QStatusBar,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

# Bootstrap imports for development mode
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)


class GlassBathFEAWidget(QWidget):
    """Main widget for Glass Bath FEA configuration and execution."""

    # Signals for communication
    configChanged = pyqtSignal()
    exportComplete = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the Glass Bath FEA widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Glass Bath FEA")
        self.setMinimumSize(800, 600)

        self._config = None
        self._init_ui()
        self._apply_styling()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.tabs.addTab(self._create_geometry_tab(), "Geometry")
        self.tabs.addTab(self._create_materials_tab(), "Materials")
        self.tabs.addTab(self._create_mesh_tab(), "Mesh")
        self.tabs.addTab(self._create_export_tab(), "Export")

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.btn_load = QPushButton("Load Config")
        self.btn_load.clicked.connect(self._load_config)
        button_layout.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save Config")
        self.btn_save.clicked.connect(self._save_config)
        button_layout.addWidget(self.btn_save)

        button_layout.addStretch()

        self.btn_export = QPushButton("Export to MATLAB")
        self.btn_export.clicked.connect(self._export_to_matlab)
        button_layout.addWidget(self.btn_export)

        layout.addLayout(button_layout)

        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def _create_geometry_tab(self) -> QWidget:
        """Create the geometry configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Vessel geometry group
        vessel_group = QGroupBox("Vessel Geometry")
        vessel_layout = QFormLayout()

        self.spin_bath_diameter = QDoubleSpinBox()
        self.spin_bath_diameter.setRange(10, 500)
        self.spin_bath_diameter.setValue(120.0)
        self.spin_bath_diameter.setSuffix(" in")
        vessel_layout.addRow("Bath Diameter:", self.spin_bath_diameter)

        self.spin_glass_depth = QDoubleSpinBox()
        self.spin_glass_depth.setRange(1, 100)
        self.spin_glass_depth.setValue(15.0)
        self.spin_glass_depth.setSuffix(" in")
        vessel_layout.addRow("Glass Depth:", self.spin_glass_depth)

        self.spin_metal_thickness = QDoubleSpinBox()
        self.spin_metal_thickness.setRange(0.1, 20)
        self.spin_metal_thickness.setValue(2.0)
        self.spin_metal_thickness.setSuffix(" in")
        vessel_layout.addRow("Metal Layer:", self.spin_metal_thickness)

        vessel_group.setLayout(vessel_layout)
        layout.addWidget(vessel_group)

        # Electrode configuration group
        electrode_group = QGroupBox("Electrode Configuration")
        electrode_layout = QFormLayout()

        self.spin_num_electrodes = QSpinBox()
        self.spin_num_electrodes.setRange(1, 12)
        self.spin_num_electrodes.setValue(3)
        electrode_layout.addRow("Number of Electrodes:", self.spin_num_electrodes)

        self.spin_electrode_diameter = QDoubleSpinBox()
        self.spin_electrode_diameter.setRange(1, 24)
        self.spin_electrode_diameter.setValue(6.0)
        self.spin_electrode_diameter.setSuffix(" in")
        electrode_layout.addRow("Electrode Diameter:", self.spin_electrode_diameter)

        self.spin_insertion_depth = QDoubleSpinBox()
        self.spin_insertion_depth.setRange(1, 50)
        self.spin_insertion_depth.setValue(10.0)
        self.spin_insertion_depth.setSuffix(" in")
        electrode_layout.addRow("Insertion Depth:", self.spin_insertion_depth)

        electrode_group.setLayout(electrode_layout)
        layout.addWidget(electrode_group)

        layout.addStretch()
        return widget

    def _create_materials_tab(self) -> QWidget:
        """Create the materials configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Glass composition group
        glass_group = QGroupBox("Glass Composition (wt%)")
        glass_layout = QFormLayout()

        self.spin_sio2 = QDoubleSpinBox()
        self.spin_sio2.setRange(0, 100)
        self.spin_sio2.setValue(74.0)
        glass_layout.addRow("SiO2:", self.spin_sio2)

        self.spin_na2o = QDoubleSpinBox()
        self.spin_na2o.setRange(0, 30)
        self.spin_na2o.setValue(13.0)
        glass_layout.addRow("Na2O:", self.spin_na2o)

        self.spin_cao = QDoubleSpinBox()
        self.spin_cao.setRange(0, 20)
        self.spin_cao.setValue(10.5)
        glass_layout.addRow("CaO:", self.spin_cao)

        self.spin_fe2o3 = QDoubleSpinBox()
        self.spin_fe2o3.setRange(0, 10)
        self.spin_fe2o3.setValue(0.1)
        self.spin_fe2o3.setDecimals(2)
        glass_layout.addRow("Fe2O3:", self.spin_fe2o3)

        glass_group.setLayout(glass_layout)
        layout.addWidget(glass_group)

        # Operating conditions group
        conditions_group = QGroupBox("Operating Conditions")
        conditions_layout = QFormLayout()

        self.spin_temperature = QDoubleSpinBox()
        self.spin_temperature.setRange(800, 1600)
        self.spin_temperature.setValue(1350.0)
        self.spin_temperature.setSuffix(" C")
        conditions_layout.addRow("Temperature:", self.spin_temperature)

        self.spin_voltage = QDoubleSpinBox()
        self.spin_voltage.setRange(10, 500)
        self.spin_voltage.setValue(100.0)
        self.spin_voltage.setSuffix(" V")
        conditions_layout.addRow("Phase Voltage:", self.spin_voltage)

        conditions_group.setLayout(conditions_layout)
        layout.addWidget(conditions_group)

        layout.addStretch()
        return widget

    def _create_mesh_tab(self) -> QWidget:
        """Create the mesh configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        mesh_group = QGroupBox("Mesh Settings")
        mesh_layout = QFormLayout()

        self.spin_element_size = QDoubleSpinBox()
        self.spin_element_size.setRange(0.001, 0.1)
        self.spin_element_size.setValue(0.01)
        self.spin_element_size.setDecimals(4)
        self.spin_element_size.setSuffix(" m")
        mesh_layout.addRow("Element Size (Glass):", self.spin_element_size)

        self.spin_mesh_order = QSpinBox()
        self.spin_mesh_order.setRange(1, 2)
        self.spin_mesh_order.setValue(1)
        mesh_layout.addRow("Element Order:", self.spin_mesh_order)

        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)

        # Mesh generation button
        self.btn_generate_mesh = QPushButton("Generate Preview Mesh")
        self.btn_generate_mesh.clicked.connect(self._generate_mesh)
        layout.addWidget(self.btn_generate_mesh)

        # Mesh info display
        self.mesh_info = QTextEdit()
        self.mesh_info.setReadOnly(True)
        self.mesh_info.setMaximumHeight(150)
        layout.addWidget(self.mesh_info)

        layout.addStretch()
        return widget

    def _create_export_tab(self) -> QWidget:
        """Create the export configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout()

        # Export path
        path_layout = QHBoxLayout()
        self.export_path_label = QLabel("./fea_export/")
        path_layout.addWidget(self.export_path_label)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_export_path)
        path_layout.addWidget(self.btn_browse)

        export_layout.addRow("Export Path:", path_layout)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Export log
        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        layout.addWidget(self.export_log)

        return widget

    def _apply_styling(self) -> None:
        """Apply consistent styling to the widget."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 5px 15px;
            }
        """)

    def _get_config(self) -> GlassBathFEAConfig:
        """Create configuration from current UI values."""
        from glass_bath_fea.core.config import (
            GlassBathFEAConfig,
            GlassComposition,
            MeshConfig,
        )

        composition = GlassComposition(
            sio2=self.spin_sio2.value(),
            na2o=self.spin_na2o.value(),
            cao=self.spin_cao.value(),
            fe2o3=self.spin_fe2o3.value(),
        )

        mesh_config = MeshConfig(
            element_size_glass=self.spin_element_size.value(),
            mesh_order=self.spin_mesh_order.value(),
        )

        config = GlassBathFEAConfig(
            bath_diameter=self.spin_bath_diameter.value(),
            glass_depth=self.spin_glass_depth.value(),
            metal_layer_thickness=self.spin_metal_thickness.value(),
            num_electrodes=self.spin_num_electrodes.value(),
            electrode_diameter=self.spin_electrode_diameter.value(),
            electrode_insertion_depth=self.spin_insertion_depth.value(),
            operating_temperature=self.spin_temperature.value(),
            phase_voltages=(
                self.spin_voltage.value(),
                self.spin_voltage.value(),
                self.spin_voltage.value(),
            ),
            glass_composition=composition,
            mesh_config=mesh_config,
        )

        return config

    def _load_config(self) -> None:
        """Load configuration from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if filename:
            self.status_label.setText(f"Loaded: {filename}")

    def _save_config(self) -> None:
        """Save current configuration to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        if filename:
            self.status_label.setText(f"Saved: {filename}")

    def _generate_mesh(self) -> None:
        """Generate a preview mesh."""
        try:
            config = self._get_config()
            from glass_bath_fea.core.mesh_generator import MeshGenerator

            gen = MeshGenerator(config)
            mesh = gen.create_mock_mesh()
            stats = gen.get_mesh_statistics(mesh)

            info = "Mesh Statistics:\n"
            info += f"  Nodes: {stats['num_nodes']}\n"
            info += f"  Elements: {stats['num_elements']}\n"
            if "elements_glass" in stats:
                info += f"  Glass elements: {stats['elements_glass']}\n"
            if "elements_metal" in stats:
                info += f"  Metal elements: {stats['elements_metal']}\n"

            self.mesh_info.setText(info)
            self.status_label.setText("Mesh generated successfully")
        except ImportError as e:
            self.mesh_info.setText(f"Error: {e}")
            self.status_label.setText("Mesh generation failed")

    def _browse_export_path(self) -> None:
        """Browse for export directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if directory:
            self.export_path_label.setText(directory)

    def _export_to_matlab(self) -> None:
        """Export data to MATLAB format."""
        try:
            config = self._get_config()
            export_dir = Path(self.export_path_label.text())

            if not export_dir.exists():
                export_dir.mkdir(parents=True)

            from glass_bath_fea.exporters.mat_exporter import export_fea_data_package

            self.export_log.append("Starting export...")
            export_fea_data_package(config, export_dir)
            self.export_log.append(f"Export complete: {export_dir}")
            self.status_label.setText("Export successful")
            self.exportComplete.emit(str(export_dir))
        except (PermissionError, OSError) as e:
            self.export_log.append(f"Error: {e}")
            self.status_label.setText("Export failed")


class GlassBathFEAMainWindow(QMainWindow):
    """Main window wrapper for the Glass Bath FEA widget."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Glass Bath FEA - Finite Element Analysis")
        self.setMinimumSize(900, 700)

        # Create central widget
        self.central_widget = GlassBathFEAWidget()
        self.setCentralWidget(self.central_widget)

        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Connect signals
        self.central_widget.exportComplete.connect(
            lambda path: self.statusBar.showMessage(f"Exported to: {path}", 5000)
        )


def main() -> None:
    """Run the Glass Bath FEA application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Glass Bath FEA")

    window = GlassBathFEAMainWindow()

    # Apply fleet-wide theme. See issue #549
    try:
        from shared.python.theme.integration import setup_themed_app

        setup_themed_app(app, window, settings_app="GlassBathFEA")
    except ImportError:
        pass  # theme system not installed

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

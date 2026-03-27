"""PyQt6 GUI for the vessel drafter tool."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vessel_drafter.analysis.vessel_drafter_metrics import (
    build_material_metrics_report,
)
from vessel_drafter.gui.material_summary_table import MaterialSummaryTable
from vessel_drafter.gui.vessel_drafter_port_panel import (
    PortFieldSpec,
    PortTableSection,
    PortValueDialog,
    make_double_spin,
)
from vessel_drafter.gui.vessel_drafter_rendering import (
    render_cross_section,
    render_plan,
)
from vessel_drafter.gui.vessel_drafter_three_d_canvas import (
    VesselDrafterThreeDCanvas,
)
from vessel_drafter.gui.zoomable_graphics_view import ZoomableGraphicsView
from vessel_drafter.models.vessel_drafter import (
    DEFAULT_VESSEL_DRAFTER_LAYOUT,
    VesselDrafterLayout,
    VesselLidPort,
    VesselSidePort,
)
from vessel_drafter.preview.vessel_drafter_preview import (
    build_cross_section_preview,
    build_plan_preview,
)
from vessel_drafter.preview.vessel_drafter_scene import build_vessel_3d_scene
from vessel_drafter.preview.vessel_drafter_view_options import (
    Vessel3DViewOptions,
)


class VesselDrafterWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vessel Drafter")
        self.resize(1400, 880)
        self._suppress_preview_updates = False
        self._three_d_preview_dirty = True

        self.inner_diameter_spin = make_double_spin(50.0, 1.0, 500.0)
        self.glass_depth_spin = make_double_spin(14.0, 1.0, 250.0)
        self.plenum_height_spin = make_double_spin(14.0, 1.0, 250.0)
        self.head_depth_spin = make_double_spin(12.5, 1.0, 250.0)
        self.hot_face_spin = make_double_spin(6.0, 0.1, 50.0)
        self.ifb_spin = make_double_spin(4.5, 0.1, 50.0)
        self.duraboard_spin = make_double_spin(1.0, 0.1, 20.0)
        self.steel_spin = make_double_spin(0.5, 0.1, 10.0)
        self.electrode_count_spin = QSpinBox()
        self.electrode_count_spin.setRange(1, 12)
        self.electrode_count_spin.setValue(3)
        self.electrode_diameter_spin = make_double_spin(2.0, 0.1, 20.0)
        self.electrode_insertion_spin = make_double_spin(14.0, 0.1, 100.0)
        self.electrode_extension_spin = make_double_spin(36.0, 0.1, 150.0)

        self.side_port_panel = PortTableSection(
            "Side Ports",
            ("Clock Angle", "Diameter", "Height Above Glass"),
        )
        self.lid_port_panel = PortTableSection(
            "Lid Ports",
            ("Clock Angle", "Diameter", "Distance From Center"),
        )

        self.cross_section_scene = QGraphicsScene(self)
        self.cross_section_view = ZoomableGraphicsView(self)
        self.cross_section_view.setScene(self.cross_section_scene)
        self.plan_scene = QGraphicsScene(self)
        self.plan_view = ZoomableGraphicsView(self)
        self.plan_view.setScene(self.plan_scene)
        self.preview_tabs = QTabWidget(self)
        self.three_d_canvas = VesselDrafterThreeDCanvas()
        self.material_summary_table = MaterialSummaryTable(self)
        self.layer_visibility_checkboxes = self._build_layer_visibility_checkboxes()
        self.section_cut_checkbox = QCheckBox("Split on vertical plane")
        self.section_cut_angle_spin = self._build_section_cut_angle_spin()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        self._build_ui()
        self._connect_signals()
        self.write_layout(DEFAULT_VESSEL_DRAFTER_LAYOUT)
        self.update_preview()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        controls_form = QFormLayout()
        controls_form.addRow("Inner diameter (in)", self.inner_diameter_spin)
        controls_form.addRow("Glass depth (in)", self.glass_depth_spin)
        controls_form.addRow("Plenum height (in)", self.plenum_height_spin)
        controls_form.addRow("Head depth (in)", self.head_depth_spin)
        controls_form.addRow("Hot face (in)", self.hot_face_spin)
        controls_form.addRow("IFB (in)", self.ifb_spin)
        controls_form.addRow("Duraboard (in)", self.duraboard_spin)
        controls_form.addRow("Steel (in)", self.steel_spin)
        controls_form.addRow("Electrode count", self.electrode_count_spin)
        controls_form.addRow("Electrode diameter (in)", self.electrode_diameter_spin)
        controls_form.addRow("Electrode insertion (in)", self.electrode_insertion_spin)
        controls_form.addRow("Electrode extension (in)", self.electrode_extension_spin)

        refresh_button = QPushButton("Refresh Preview")
        refresh_button.clicked.connect(self.update_preview)
        export_button = QPushButton("Export...")
        export_button.clicked.connect(self._handle_export)

        controls_root = QWidget()
        controls_layout = QVBoxLayout(controls_root)
        controls_layout.addLayout(controls_form)
        controls_layout.addWidget(self.side_port_panel)
        controls_layout.addWidget(self.lid_port_panel)
        controls_layout.addWidget(refresh_button)
        controls_layout.addWidget(export_button)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_root)
        controls_scroll.setMinimumWidth(340)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self._build_preview_tabs(), 1)

        main_layout = QHBoxLayout(root)
        main_layout.addWidget(controls_scroll, 0)
        main_layout.addLayout(preview_layout, 1)

    def _connect_signals(self) -> None:
        widgets = (
            self.inner_diameter_spin,
            self.glass_depth_spin,
            self.plenum_height_spin,
            self.head_depth_spin,
            self.hot_face_spin,
            self.ifb_spin,
            self.duraboard_spin,
            self.steel_spin,
            self.electrode_count_spin,
            self.electrode_diameter_spin,
            self.electrode_insertion_spin,
            self.electrode_extension_spin,
        )
        for widget in widgets:
            widget.valueChanged.connect(self.update_preview)

        self.side_port_panel.add_button.clicked.connect(self._prompt_add_side_port)
        self.lid_port_panel.add_button.clicked.connect(self._prompt_add_lid_port)
        self.side_port_panel.remove_button.clicked.connect(
            self._remove_selected_side_ports
        )
        self.lid_port_panel.remove_button.clicked.connect(
            self._remove_selected_lid_ports
        )
        self.side_port_panel.table.itemChanged.connect(self.update_preview)
        self.lid_port_panel.table.itemChanged.connect(self.update_preview)
        for checkbox in self.layer_visibility_checkboxes.values():
            checkbox.toggled.connect(self.refresh_three_d_preview)
        self.section_cut_checkbox.toggled.connect(self._handle_section_cut_toggled)
        self.section_cut_angle_spin.valueChanged.connect(
            self._handle_section_cut_angle_changed
        )
        self.preview_tabs.currentChanged.connect(self._handle_preview_tab_changed)

    def write_layout(self, layout: VesselDrafterLayout) -> None:
        if not (layout is not None):
            raise ValueError("layout must be provided")
        self._suppress_preview_updates = True
        self.inner_diameter_spin.setValue(layout.inner_diameter_in)
        self.glass_depth_spin.setValue(layout.glass_depth_in)
        self.plenum_height_spin.setValue(layout.plenum_height_in)
        self.head_depth_spin.setValue(layout.head_depth_in)
        self.hot_face_spin.setValue(layout.hot_face_thickness_in)
        self.ifb_spin.setValue(layout.ifb_thickness_in)
        self.duraboard_spin.setValue(layout.duraboard_thickness_in)
        self.steel_spin.setValue(layout.steel_thickness_in)
        self.electrode_count_spin.setValue(layout.electrode_count)
        self.electrode_diameter_spin.setValue(layout.electrode_diameter_in)
        self.electrode_insertion_spin.setValue(
            layout.electrode_insertion_into_inner_circle_in
        )
        self.electrode_extension_spin.setValue(
            layout.electrode_extension_past_inner_circle_in
        )
        self.side_port_panel.set_rows(
            tuple(
                (
                    port.normalized_clock_angle_degrees,
                    port.diameter_in,
                    port.height_above_glass_surface_in,
                )
                for port in layout.side_ports
            )
        )
        self.lid_port_panel.set_rows(
            tuple(
                (
                    port.normalized_clock_angle_degrees,
                    port.diameter_in,
                    port.radial_distance_from_center_in,
                )
                for port in layout.lid_ports
            )
        )
        self._suppress_preview_updates = False

    def add_side_port(self, port: VesselSidePort) -> None:
        self.side_port_panel.append_row(
            (
                port.normalized_clock_angle_degrees,
                port.diameter_in,
                port.height_above_glass_surface_in,
            )
        )
        self.update_preview()

    def add_lid_port(self, port: VesselLidPort) -> None:
        self.lid_port_panel.append_row(
            (
                port.normalized_clock_angle_degrees,
                port.diameter_in,
                port.radial_distance_from_center_in,
            )
        )
        self.update_preview()

    def read_layout(self) -> VesselDrafterLayout:
        return VesselDrafterLayout(
            inner_diameter_in=self.inner_diameter_spin.value(),
            glass_depth_in=self.glass_depth_spin.value(),
            plenum_height_in=self.plenum_height_spin.value(),
            head_depth_in=self.head_depth_spin.value(),
            hot_face_thickness_in=self.hot_face_spin.value(),
            ifb_thickness_in=self.ifb_spin.value(),
            duraboard_thickness_in=self.duraboard_spin.value(),
            steel_thickness_in=self.steel_spin.value(),
            electrode_count=self.electrode_count_spin.value(),
            electrode_diameter_in=self.electrode_diameter_spin.value(),
            electrode_insertion_into_inner_circle_in=self.electrode_insertion_spin.value(),
            electrode_extension_past_inner_circle_in=self.electrode_extension_spin.value(),
            side_ports=tuple(
                VesselSidePort(
                    clock_angle_degrees=angle,
                    diameter_in=diameter,
                    height_above_glass_surface_in=height,
                )
                for angle, diameter, height in self.side_port_panel.rows()
            ),
            lid_ports=tuple(
                VesselLidPort(
                    clock_angle_degrees=angle,
                    diameter_in=diameter,
                    radial_distance_from_center_in=radius,
                )
                for angle, diameter, radius in self.lid_port_panel.rows()
            ),
        )

    def _read_layout(self) -> VesselDrafterLayout:
        return self.read_layout()

    def update_preview(self) -> None:
        if self._suppress_preview_updates:
            return
        try:
            layout = self.read_layout()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return

        render_cross_section(
            self.cross_section_scene,
            build_cross_section_preview(layout),
        )
        render_plan(self.plan_scene, build_plan_preview(layout))
        self._three_d_preview_dirty = True
        self._refresh_three_d_preview_if_visible(layout)
        metrics = build_material_metrics_report(layout)
        self.material_summary_table.set_report(metrics)
        self.cross_section_view.sync_to_scene()
        self.plan_view.sync_to_scene()
        self.status_label.setText(
            f"Outer diameter: {layout.outer_diameter_in:.2f} in | "
            f"Full height: {layout.full_height_in:.2f} in | "
            f"Ports: {len(layout.side_ports)} side, {len(layout.lid_ports)} lid | "
            f"Refractory: {metrics.refractory_total_volume_ft3:.2f} ft^3, "
            f"{metrics.refractory_total_surface_area_ft2:.2f} ft^2, "
            f"{metrics.refractory_total_mass_lb:.1f} lb"
        )

    def refresh_three_d_preview(self) -> None:
        if self._suppress_preview_updates:
            return
        try:
            layout = self.read_layout()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._three_d_preview_dirty = True
        self._refresh_three_d_preview_if_visible(layout)

    def _handle_export(self) -> None:
        layout = self._read_layout()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Vessel",
            "vessel_drafter",
            "STEP Files (*.step *.stp);;STL Files (*.stl);;BREP Files (*.brep);;GLTF Files (*.gltf *.glb);;All Files (*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")
        fmt_map = {
            "step": "step",
            "stp": "step",
            "stl": "stl",
            "brep": "brep",
            "gltf": "gltf",
            "glb": "gltf",
        }
        fmt = fmt_map.get(ext, "step")
        try:
            from vessel_drafter.exporters.vessel_export import export_vessel

            export_vessel(
                layout, output_dir=path.parent, stem=path.stem, formats=(fmt,)
            )
            QMessageBox.information(self, "Export Complete", f"Exported to {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _prompt_add_side_port(self) -> None:
        dialog = PortValueDialog(
            "Add Side Port",
            (
                PortFieldSpec("Clock angle (deg)", 0.0, 0.0, 360.0),
                PortFieldSpec("Diameter (in)", 3.0, 0.1, 100.0),
                PortFieldSpec("Height above glass (in)", 4.0, 0.0, 250.0),
            ),
            self,
        )
        if dialog.exec():
            angle, diameter, height = dialog.values()
            self.add_side_port(
                VesselSidePort(
                    clock_angle_degrees=angle,
                    diameter_in=diameter,
                    height_above_glass_surface_in=height,
                )
            )

    def _prompt_add_lid_port(self) -> None:
        dialog = PortValueDialog(
            "Add Lid Port",
            (
                PortFieldSpec("Clock angle (deg)", 0.0, 0.0, 360.0),
                PortFieldSpec("Diameter (in)", 4.0, 0.1, 100.0),
                PortFieldSpec("Distance from center (in)", 8.0, 0.0, 500.0),
            ),
            self,
        )
        if dialog.exec():
            angle, diameter, radius = dialog.values()
            self.add_lid_port(
                VesselLidPort(
                    clock_angle_degrees=angle,
                    diameter_in=diameter,
                    radial_distance_from_center_in=radius,
                )
            )

    def _remove_selected_side_ports(self) -> None:
        self.side_port_panel.remove_selected_rows()
        self.update_preview()

    def _remove_selected_lid_ports(self) -> None:
        self.lid_port_panel.remove_selected_rows()
        self.update_preview()

    def _build_preview_panel(
        self,
        title: str,
        view: ZoomableGraphicsView,
    ) -> QWidget:
        if not (title is not None):
            raise ValueError("title must be provided")
        title_label = QLabel(title)
        zoom_in_button = QPushButton("+")
        zoom_in_button.clicked.connect(view.zoom_in)
        zoom_out_button = QPushButton("-")
        zoom_out_button.clicked.connect(view.zoom_out)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(view.reset_zoom)

        header_layout = QHBoxLayout()
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(zoom_out_button)
        header_layout.addWidget(zoom_in_button)
        header_layout.addWidget(reset_button)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.addLayout(header_layout)
        panel_layout.addWidget(view, 1)
        return panel

    def _build_preview_tabs(self) -> QTabWidget:
        previews_tab = QWidget()
        previews_layout = QVBoxLayout(previews_tab)
        previews_layout.addWidget(
            self._build_preview_panel("Cross-Section Preview", self.cross_section_view),
            1,
        )
        previews_layout.addWidget(
            self._build_preview_panel("Top View Preview", self.plan_view),
            1,
        )

        three_d_tab = QWidget()
        three_d_layout = QHBoxLayout(three_d_tab)
        three_d_layout.addWidget(self.three_d_canvas, 1)
        three_d_layout.addWidget(self._build_three_d_sidebar(), 0)

        self.preview_tabs.addTab(previews_tab, "2D Previews")
        self.preview_tabs.addTab(three_d_tab, "3D Preview")
        return self.preview_tabs

    def _build_three_d_sidebar(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        instructions = QLabel(
            "Drag to rotate the model. Use the checkboxes to hide layers."
        )
        instructions.setWordWrap(True)
        reset_view_button = QPushButton("Reset 3D View")
        reset_view_button.clicked.connect(self.three_d_canvas.reset_view)

        layout.addWidget(instructions)
        layout.addWidget(reset_view_button)
        layout.addWidget(self._build_layer_visibility_panel())
        layout.addWidget(self._build_section_cut_panel())
        layout.addWidget(QLabel("Material Summary"))
        layout.addWidget(self.material_summary_table)
        layout.addStretch(1)
        panel.setMinimumWidth(360)
        return panel

    def _build_layer_visibility_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Visible Layers"))
        for checkbox in self.layer_visibility_checkboxes.values():
            layout.addWidget(checkbox)
        layout.addStretch(1)
        return panel

    def _build_section_cut_panel(self) -> QWidget:
        panel = QWidget()
        layout = QFormLayout(panel)
        layout.addRow(self.section_cut_checkbox)
        layout.addRow("Split angle (deg)", self.section_cut_angle_spin)
        return panel

    def _build_layer_visibility_checkboxes(self) -> dict[str, QCheckBox]:
        materials = DEFAULT_VESSEL_DRAFTER_LAYOUT.material_properties_by_name
        labels = (
            "glass_bath",
            "hot_face_refractory",
            "ifb",
            "duraboard",
            "steel_shell",
            "electrodes",
        )
        checkboxes: dict[str, QCheckBox] = {}
        for label in labels:
            checkbox = QCheckBox(materials[label].display_name)
            checkbox.setChecked(True)
            checkboxes[label] = checkbox
        return checkboxes

    def _build_section_cut_angle_spin(self) -> QDoubleSpinBox:
        spin = make_double_spin(0.0, 0.0, 360.0)
        spin.setSingleStep(15.0)
        spin.setEnabled(False)
        return cast(QDoubleSpinBox, spin)

    def _update_three_d_preview(self, layout: VesselDrafterLayout) -> None:
        if not (layout is not None):
            raise ValueError("layout must be provided")
        view_options = self._read_three_d_view_options()
        self.three_d_canvas.draw_scene(
            build_vessel_3d_scene(
                layout,
                visible_labels=self._visible_layer_labels(),
                view_options=view_options,
            ),
            view_options,
        )
        self._three_d_preview_dirty = False

    def _visible_layer_labels(self) -> set[str]:
        return {
            label
            for label, checkbox in self.layer_visibility_checkboxes.items()
            if checkbox.isChecked()
        }

    def _read_three_d_view_options(self) -> Vessel3DViewOptions:
        return Vessel3DViewOptions(
            split_enabled=self.section_cut_checkbox.isChecked(),
            split_angle_degrees=self.section_cut_angle_spin.value(),
        )

    def _handle_preview_tab_changed(self, index: int) -> None:
        if index != self.preview_tabs.indexOf(self.preview_tabs.widget(1)):
            return
        self.refresh_three_d_preview()

    def _refresh_three_d_preview_if_visible(self, layout: VesselDrafterLayout) -> None:
        if self._three_d_preview_dirty and self.preview_tabs.currentIndex() == 1:
            self._update_three_d_preview(layout)

    def _handle_section_cut_toggled(self, checked: bool) -> None:
        if not (checked is not None):
            raise ValueError("checked must be provided")
        self.section_cut_angle_spin.setEnabled(checked)
        self.three_d_canvas.queue_default_view(self._read_three_d_view_options())
        self.refresh_three_d_preview()

    def _handle_section_cut_angle_changed(self, _: float) -> None:
        if self.section_cut_checkbox.isChecked():
            self.three_d_canvas.queue_default_view(self._read_three_d_view_options())
        self.refresh_three_d_preview()


def launch() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = VesselDrafterWindow()
    window.show()
    return int(app.exec())

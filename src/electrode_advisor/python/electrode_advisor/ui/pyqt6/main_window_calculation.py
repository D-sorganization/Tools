"""Calculation and event-handling mixin for the ElectrodeAdvisorWidget.

Contains _calculate_system, _update_status, _validate_glass_height,
_on_metal_conductivity_changed, _on_input_changed, _on_zoom_slider_changed,
_setup_timers, _periodic_update, and _run_optimization.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtWidgets import QDoubleSpinBox, QMessageBox

logger = logging.getLogger(__name__)


class CalculationMixin:
    """Mixin providing calculation and event handling for ElectrodeAdvisorWidget."""

    def _setup_timers(self) -> None:
        """Setup update timers"""
        from ...configs.ui_defaults import PERIODIC_UPDATE_MS

        self.calc_timer = QTimer()
        self.calc_timer.timeout.connect(self._periodic_update)
        self.calc_timer.start(PERIODIC_UPDATE_MS)

    @pyqtSlot()
    def _on_input_changed(self) -> None:
        """Handle input parameter changes"""
        if getattr(self, "_initialization_complete", False):
            self._calculate_system()

    @pyqtSlot()
    def _on_zoom_slider_changed(self) -> None:
        """Handle zoom slider changes"""
        zoom_value = self.zoom_slider.value()  # type: ignore[attr-defined]
        self.zoom_label.setText(f"Zoom: {zoom_value}%")  # type: ignore[attr-defined]

        if self.electrode_ax is not None:  # type: ignore[attr-defined]
            xlim = self.electrode_ax.get_xlim()  # type: ignore[attr-defined]
            ylim = self.electrode_ax.get_ylim()  # type: ignore[attr-defined]

            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2

            base_range = (
                self.bath_diameter_input.value() / 2  # type: ignore[attr-defined]
                + self.electrode_extension_slider.value()  # type: ignore[attr-defined]
            )

            zoom_factor = zoom_value / 100.0
            new_range = base_range / zoom_factor * 1.1

            self.electrode_ax.set_xlim(  # type: ignore[attr-defined]
                x_center - new_range / 2, x_center + new_range / 2
            )
            self.electrode_ax.set_ylim(  # type: ignore[attr-defined]
                y_center - new_range / 2, y_center + new_range / 2
            )
            if self.electrode_canvas is not None:  # type: ignore[attr-defined]
                self.electrode_canvas.draw()  # type: ignore[attr-defined]
            if hasattr(self.electrode_ax, "set_zlim"):  # type: ignore[attr-defined]
                z_range = (
                    (
                        self.glass_layer_height_input.value()  # type: ignore[attr-defined]
                        + self.metal_layer_height_input.value()  # type: ignore[attr-defined]
                    )
                    / zoom_factor
                    * 1.2
                )
                self.electrode_ax.set_zlim(0, z_range)  # type: ignore[attr-defined]

            if self.electrode_canvas is not None:  # type: ignore[attr-defined]
                self.electrode_canvas.draw()  # type: ignore[attr-defined]

    @pyqtSlot()
    def _validate_glass_height(self) -> None:
        """Validate that glass height is above electrode tips"""
        try:
            glass_height = self.glass_layer_height_input.value()  # type: ignore[attr-defined]
            metal_height = self.metal_layer_height_input.value()  # type: ignore[attr-defined]

            max_electrode_depth: float = 0.0
            for i in range(3):
                if i in self.depth_inputs:  # type: ignore[attr-defined]
                    depth = self.depth_inputs[i].value()  # type: ignore[attr-defined]
                    max_electrode_depth = max(max_electrode_depth, depth)

            min_glass_height = max_electrode_depth + metal_height + 1.0

            if glass_height < min_glass_height:
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "Glass Height Warning",
                    f"Glass height ({glass_height:.1f} in) should be above electrode tips.\n"
                    f"Minimum recommended height: {min_glass_height:.1f} in\n"
                    f"(Electrode depth: {max_electrode_depth:.1f} in + "
                    f"Metal height: {metal_height:.1f} in + Safety margin: 1.0 in)",
                )
                self.glass_layer_height_input.setValue(min_glass_height)  # type: ignore[attr-defined]

            self._calculate_system()

        except (RuntimeError, AttributeError) as e:
            logger.exception("Error in glass height validation: %s", e)
            self._calculate_system()

    @pyqtSlot()
    def _on_metal_conductivity_changed(self) -> None:
        """Handle metal layer conductivity toggle"""
        try:
            is_enabled = self.metal_conductive_checkbox.isChecked()  # type: ignore[attr-defined]
            logger.debug(
                "[DEBUG] Metal layer conductivity: %s",
                "Enabled" if is_enabled else "Disabled",
            )

            if hasattr(self, "show_metal_checkbox"):
                if not is_enabled:
                    self.show_metal_checkbox.setChecked(False)  # type: ignore[attr-defined]
                    self.show_metal_checkbox.setEnabled(False)  # type: ignore[attr-defined]
                    self.show_metal_checkbox.setStyleSheet(  # type: ignore[attr-defined]
                        "QCheckBox { color: #888888; }"
                    )
                    self.show_metal_checkbox.setToolTip(  # type: ignore[attr-defined]
                        "Metal layer visualization disabled when conduction is off"
                    )
                else:
                    self.show_metal_checkbox.setEnabled(True)  # type: ignore[attr-defined]
                    self.show_metal_checkbox.setChecked(True)  # type: ignore[attr-defined]
                    self.show_metal_checkbox.setStyleSheet("")  # type: ignore[attr-defined]
                    self.show_metal_checkbox.setToolTip("")  # type: ignore[attr-defined]

            self._calculate_system()
            self._update_results_tables()  # type: ignore[attr-defined]
            self._update_analysis_display()  # type: ignore[attr-defined]

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error handling metal conductivity change: %s", e)

    def _calculate_system(self) -> None:
        """Calculate System method.

        Returns:
            None
        """
        try:
            logger.debug("[DEBUG] _calculate_system called")
            self.config.vertical_spreading_factor = (  # type: ignore[attr-defined]
                self.vertical_spreading_input.value()  # type: ignore[attr-defined]
            )
            self.config.horizontal_spreading_factor = (  # type: ignore[attr-defined]
                self.horizontal_spreading_input.value()  # type: ignore[attr-defined]
            )

            depths = np.array(
                [
                    self.depth_inputs[0].value(),  # type: ignore[attr-defined]
                    self.depth_inputs[1].value(),  # type: ignore[attr-defined]
                    self.depth_inputs[2].value(),  # type: ignore[attr-defined]
                ]
            )

            bath_diameter = self.bath_diameter_input.value()  # type: ignore[attr-defined]
            electrode_diameter = float(self.electrode_diameter_combo.currentText())  # type: ignore[attr-defined]
            metal_layer_height = self.metal_layer_height_input.value()  # type: ignore[attr-defined]
            bath_temperature = self.bath_temp_input.value()  # type: ignore[attr-defined]

            voltages = np.array(
                [
                    cast(QDoubleSpinBox, self.phase_inputs["1-2"]["voltage"]).value(),  # type: ignore[attr-defined]
                    cast(QDoubleSpinBox, self.phase_inputs["2-3"]["voltage"]).value(),  # type: ignore[attr-defined]
                    cast(QDoubleSpinBox, self.phase_inputs["3-1"]["voltage"]).value(),  # type: ignore[attr-defined]
                ]
            )

            k_factors = {
                "K_tt": self.k_tt_input.value() * self.config.k_scaling_factor,  # type: ignore[attr-defined]
                "K_vert": self.k_vert_input.value() * self.config.k_scaling_factor,  # type: ignore[attr-defined]
            }

            conductive_height = self.conductive_layer_height_input.value()  # type: ignore[attr-defined]
            self.calculation_results = self.electrical_model.calculate_system_state(  # type: ignore[attr-defined]
                depths=depths,
                bath_diameter=bath_diameter,
                tip_diameter=electrode_diameter,
                metal_depth=metal_layer_height,
                k_factors=k_factors,
                bath_temperature=bath_temperature,
                voltages=voltages,
                conductive_height=conductive_height,
            )
            logger.debug("[DEBUG] calculation_results: %s", self.calculation_results)  # type: ignore[attr-defined]

            self._update_3d_visualization()  # type: ignore[attr-defined]
            self._update_temperature_profile()  # type: ignore[attr-defined]
            self._update_current_distribution()  # type: ignore[attr-defined]
            self._update_power_distribution()  # type: ignore[attr-defined]
            self._update_results_tables()  # type: ignore[attr-defined]
            self._update_analysis_display()  # type: ignore[attr-defined]
            self._update_status("Calculation completed successfully", "ok")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            error_msg = f"Calculation error: {e!s}"
            logger.exception(error_msg)
            self._update_status(error_msg, "error")

    def _update_status(self, message: str, status_type: str = "ok") -> None:
        """Update status display"""
        self.status_label.setText(message)  # type: ignore[attr-defined]

        if self.config.colors is None:  # type: ignore[attr-defined]
            return

        color_map = {
            "ok": self.config.colors["status_ok"],  # type: ignore[attr-defined]
            "warn": self.config.colors["status_warn"],  # type: ignore[attr-defined]
            "error": self.config.colors["status_err"],  # type: ignore[attr-defined]
        }

        color = color_map.get(status_type, self.config.colors["status_ok"])  # type: ignore[attr-defined]
        color_str = color.name() if hasattr(color, "name") else str(color)
        self.status_panel.setStyleSheet(f"background-color: {color_str}")  # type: ignore[attr-defined]

    @pyqtSlot()
    def _run_optimization(self) -> None:
        """Run electrode position optimization"""
        QMessageBox.information(
            self,  # type: ignore[arg-type]
            "Optimization",
            "Optimization feature will be implemented with full algorithm integration",
        )
        self.optimization_complete.emit({"status": "pending"})  # type: ignore[attr-defined]

    def _periodic_update(self) -> None:
        """Periodic update for real-time monitoring"""

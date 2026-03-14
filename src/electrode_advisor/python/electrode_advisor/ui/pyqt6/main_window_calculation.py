"""Calculation and event-handling mixin for the ElectrodeAdvisorWidget.

Contains _calculate_system, _update_status, _validate_glass_height,
_on_metal_conductivity_changed, _on_input_changed, _on_zoom_slider_changed,
_setup_timers, and _run_optimization.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSlot
from PyQt6.QtWidgets import QDoubleSpinBox, QMessageBox

logger = logging.getLogger(__name__)


class CalculationMixin:
    """Mixin providing calculation and event handling for ElectrodeAdvisorWidget."""

    def _setup_timers(self) -> None:
        """Setup update timers (periodic update removed — was a no-op)."""
        self.calc_timer = QTimer()

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

    def _compute_effective_conductivity(self) -> bool:
        """Return whether metal layer conduction is currently enabled.

        Pure query — no side effects. Extracted from _on_metal_conductivity_changed
        to separate concern of reading state from concern of updating display (#1370).
        """
        return bool(self.metal_conductive_checkbox.isChecked())  # type: ignore[attr-defined]

    def _update_metal_conductivity_display(self, is_enabled: bool) -> None:
        """Update show_metal_checkbox appearance based on conductivity state (#1370)."""
        if not hasattr(self, "show_metal_checkbox"):
            return
        if not is_enabled:
            self.show_metal_checkbox.setChecked(False)  # type: ignore[attr-defined]
            self.show_metal_checkbox.setEnabled(False)  # type: ignore[attr-defined]
            self.show_metal_checkbox.setStyleSheet("QCheckBox { color: #888888; }")  # type: ignore[attr-defined]
            self.show_metal_checkbox.setToolTip(  # type: ignore[attr-defined]
                "Metal layer visualization disabled when conduction is off"
            )
        else:
            self.show_metal_checkbox.setEnabled(True)  # type: ignore[attr-defined]
            self.show_metal_checkbox.setChecked(True)  # type: ignore[attr-defined]
            self.show_metal_checkbox.setStyleSheet("")  # type: ignore[attr-defined]
            self.show_metal_checkbox.setToolTip("")  # type: ignore[attr-defined]

    @pyqtSlot()
    def _on_metal_conductivity_changed(self) -> None:
        """Handle metal layer conductivity toggle"""
        try:
            is_enabled = self._compute_effective_conductivity()
            logger.debug(
                "[DEBUG] Metal layer conductivity: %s",
                "Enabled" if is_enabled else "Disabled",
            )
            self._update_metal_conductivity_display(is_enabled)
            self._calculate_system()
            self._update_results_tables()  # type: ignore[attr-defined]
            self._update_analysis_display()  # type: ignore[attr-defined]

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error handling metal conductivity change: %s", e)

    def _read_calculation_params(self) -> dict[str, Any]:
        """Read UI inputs needed for electrical-model calculations.

        Extracted from _calculate_system and _compute_balanced_depths to
        eliminate duplicated parameter reading (fixes #1414).

        Returns
        -------
        dict
            Keys: depths, bath_diameter, electrode_diameter,
            metal_layer_height, bath_temperature, voltages,
            k_factors, conductive_height.
        """
        depths = np.array(
            [
                self.depth_inputs[0].value(),  # type: ignore[attr-defined]
                self.depth_inputs[1].value(),  # type: ignore[attr-defined]
                self.depth_inputs[2].value(),  # type: ignore[attr-defined]
            ]
        )
        bath_diameter = self.bath_diameter_input.value()  # type: ignore[attr-defined]
        electrode_diameter = float(
            self.electrode_diameter_combo.currentText()  # type: ignore[attr-defined]
        )
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
        return {
            "depths": depths,
            "bath_diameter": bath_diameter,
            "electrode_diameter": electrode_diameter,
            "metal_layer_height": metal_layer_height,
            "bath_temperature": bath_temperature,
            "voltages": voltages,
            "k_factors": k_factors,
            "conductive_height": conductive_height,
        }

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

            params = self._read_calculation_params()
            depths = params["depths"]
            bath_diameter = params["bath_diameter"]
            bath_temperature = params["bath_temperature"]

            # DbC preconditions (#1365)
            assert bath_diameter > 0, f"bath_diameter must be > 0, got {bath_diameter}"
            assert all(d >= 0 for d in depths), f"depths must be >= 0, got {depths}"
            in_range = 800 <= bath_temperature <= 1600
            assert in_range, f"temperature {bath_temperature} not in [800,1600]"

            self.calculation_results = self.electrical_model.calculate_system_state(  # type: ignore[attr-defined]
                depths=depths,
                bath_diameter=bath_diameter,
                tip_diameter=params["electrode_diameter"],
                metal_depth=params["metal_layer_height"],
                k_factors=params["k_factors"],
                bath_temperature=bath_temperature,
                voltages=params["voltages"],
                conductive_height=params["conductive_height"],
            )
            logger.debug("[DEBUG] calculation_results: %s", self.calculation_results)  # type: ignore[attr-defined]

            # DbC postcondition (#1380): model must return a non-empty dict
            assert isinstance(self.calculation_results, dict), (  # type: ignore[attr-defined]
                "calculate_system_state must return a dict"
            )
            assert len(self.calculation_results) > 0, (  # type: ignore[attr-defined]
                "calculate_system_state must return a non-empty result"
            )

            self._update_3d_visualization()  # type: ignore[attr-defined]
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

    def _compute_balanced_depths(
        self,
        target_resistance: float,
        phase_index: int,
        current_depths: np.ndarray,
        lo: float = 1.0,
        hi: float = 40.0,
        tol: float = 1e-3,
        max_iter: int = 50,
    ) -> float:
        """Use bisection to find the depth for one electrode that hits target_resistance.

        Pure computation — no side effects on UI state.

        Parameters
        ----------
        target_resistance:
            Desired total resistance (Ohm) for the phase pair.
        phase_index:
            Index (0, 1, 2) of the electrode whose depth is varied.
        current_depths:
            Numpy array of the three current electrode depths (inches).
        lo, hi:
            Bisection search bounds for depth (inches).
        tol:
            Convergence tolerance on depth (inches).
        max_iter:
            Maximum bisection iterations.

        Returns
        -------
        float
            Depth (inches) that produces a resistance closest to target_resistance,
            or the midpoint if bisection does not converge within max_iter.
        """
        params = self._read_calculation_params()

        def resistance_at_depth(depth: float) -> float:
            trial_depths = current_depths.copy()
            trial_depths[phase_index] = depth
            result = self.electrical_model.calculate_system_state(  # type: ignore[attr-defined]
                depths=trial_depths,
                bath_diameter=params["bath_diameter"],
                tip_diameter=params["electrode_diameter"],
                metal_depth=params["metal_layer_height"],
                k_factors=params["k_factors"],
                bath_temperature=params["bath_temperature"],
                voltages=params["voltages"],
                conductive_height=params["conductive_height"],
            )
            phase_keys = ["1-2", "2-3", "3-1"]
            phase_key = phase_keys[phase_index]
            paths = result.get("current_paths", {})
            return float(paths.get(phase_key, {}).get("total", float("inf")))

        r_lo = resistance_at_depth(lo)
        r_hi = resistance_at_depth(hi)

        # Resistance decreases with depth (deeper electrode → lower resistance).
        # Ensure the target lies within [r_hi, r_lo].
        if target_resistance <= r_hi or target_resistance >= r_lo:
            mid = (lo + hi) / 2.0
            logger.debug(
                "[DEBUG] bisect: target %.4f out of range [%.4f, %.4f]; returning mid %.4f",
                target_resistance,
                r_hi,
                r_lo,
                mid,
            )
            return mid

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            r_mid = resistance_at_depth(mid)
            if abs(r_mid - target_resistance) < tol or (hi - lo) / 2.0 < tol:
                return mid
            # Deeper → lower R, so if r_mid > target we need to go deeper
            if r_mid > target_resistance:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    @pyqtSlot()
    def _run_optimization(self) -> None:
        """Run electrode position optimization using bisection to equalize resistances.

        Issue #1362: replaces the placeholder stub with a real bisection-based
        electrode advancement algorithm.  For each electrode, finds the insertion
        depth that brings its per-phase resistance to the mean of the current
        resistances, yielding a more balanced power distribution.
        """
        try:
            if not self.calculation_results:  # type: ignore[attr-defined]
                QMessageBox.warning(
                    self,  # type: ignore[arg-type]
                    "Optimization",
                    "Run a calculation first before optimizing.",
                )
                return

            current_paths = self.calculation_results.get("current_paths", {})  # type: ignore[attr-defined]
            phase_keys = ["1-2", "2-3", "3-1"]
            resistances = [
                current_paths.get(pk, {}).get("total", 0.0) for pk in phase_keys
            ]
            target_resistance = float(np.mean(resistances))

            current_depths = np.array(
                [
                    self.depth_inputs[0].value(),  # type: ignore[attr-defined]
                    self.depth_inputs[1].value(),  # type: ignore[attr-defined]
                    self.depth_inputs[2].value(),  # type: ignore[attr-defined]
                ]
            )

            new_depths = current_depths.copy()
            for i in range(3):
                new_depths[i] = self._compute_balanced_depths(
                    target_resistance=target_resistance,
                    phase_index=i,
                    current_depths=new_depths,
                )

            for i in range(3):
                self.depth_inputs[i].setValue(new_depths[i])  # type: ignore[attr-defined]

            self._calculate_system()
            self.optimization_complete.emit(
                {"status": "complete", "depths": new_depths.tolist()}
            )  # type: ignore[attr-defined]
            QMessageBox.information(
                self,  # type: ignore[arg-type]
                "Optimization Complete",
                f"Balanced electrode depths:\n"
                f"  E1: {new_depths[0]:.2f} in\n"
                f"  E2: {new_depths[1]:.2f} in\n"
                f"  E3: {new_depths[2]:.2f} in\n"
                f"Target resistance: {target_resistance:.4f} Ω",
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Optimization failed: %s", e)
            QMessageBox.critical(
                self,  # type: ignore[arg-type]
                "Optimization Error",
                f"Optimization failed: {e!s}",
            )

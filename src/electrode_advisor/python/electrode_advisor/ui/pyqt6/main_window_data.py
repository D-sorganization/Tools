"""DataMixin -- system calculation, status updates, results tables, and chart updates."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from PyQt6.QtWidgets import QDoubleSpinBox, QLineEdit, QTableWidgetItem

logger = logging.getLogger(__name__)


class DataMixin:
    """Mixin providing calculation, status, and results display methods."""

    # -- Attributes provided by the host class (declared for mypy) --
    bath_diameter_input: Any
    bath_temp_input: Any
    conductive_layer_height_input: Any
    config: Any
    current_ax: Any
    current_canvas: Any
    depth_inputs: Any
    electrical_model: Any
    electrode_diameter_combo: Any
    horizontal_spreading_input: Any
    k_tt_input: Any
    k_vert_input: Any
    metal_conductive_checkbox: Any
    metal_layer_height_input: Any
    path_labels: Any
    phase_inputs: Any
    power_ax: Any
    power_canvas: Any
    power_factor_input: Any
    resistance_table: Any
    status_label: Any
    status_panel: Any
    total_power_display: Any
    vertical_spreading_input: Any
    _update_3d_visualization: Any

    def _calculate_system(self) -> None:
        """Calculate System method.

        Returns:
            None
        """
        try:
            logger.debug("[DEBUG] _calculate_system called")
            # Update configuration with current spreading factors
            self.config.vertical_spreading_factor = (
                self.vertical_spreading_input.value()
            )
            self.config.horizontal_spreading_factor = (
                self.horizontal_spreading_input.value()
            )

            # Read all input values
            depths = np.array(
                [
                    self.depth_inputs[0].value(),
                    self.depth_inputs[1].value(),
                    self.depth_inputs[2].value(),
                ]
            )

            bath_diameter = self.bath_diameter_input.value()
            electrode_diameter = float(self.electrode_diameter_combo.currentText())
            metal_layer_height = self.metal_layer_height_input.value()
            bath_temperature = self.bath_temp_input.value()

            # Voltages from phase inputs
            voltages = np.array(
                [
                    cast(QDoubleSpinBox, self.phase_inputs["1-2"]["voltage"]).value(),
                    cast(QDoubleSpinBox, self.phase_inputs["2-3"]["voltage"]).value(),
                    cast(QDoubleSpinBox, self.phase_inputs["3-1"]["voltage"]).value(),
                ]
            )

            # K factors
            k_factors = {
                "K_tt": self.k_tt_input.value() * self.config.k_scaling_factor,
                "K_vert": self.k_vert_input.value() * self.config.k_scaling_factor,
            }

            # Calculate system state using the electrical model
            conductive_height = self.conductive_layer_height_input.value()
            self.calculation_results = self.electrical_model.calculate_system_state(
                depths=depths,
                bath_diameter=bath_diameter,
                tip_diameter=electrode_diameter,
                metal_depth=metal_layer_height,
                k_factors=k_factors,
                bath_temperature=bath_temperature,
                voltages=voltages,
                conductive_height=conductive_height,
            )
            logger.debug("[DEBUG] calculation_results: %s", self.calculation_results)

            # Update displays
            self._update_3d_visualization()
            # #1377: _update_temperature_profile removed — was a no-op stub
            self._update_current_distribution()
            self._update_power_distribution()
            self._update_results_tables()
            self._update_analysis_display()
            self._update_status("Calculation completed successfully", "ok")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            # Handle calculation errors gracefully
            error_msg = f"Calculation error: {e!s}"
            logger.exception(error_msg)
            self._update_status(error_msg, "error")

    def _update_status(self, message: str, status_type: str = "ok") -> None:
        """Update status display"""
        self.status_label.setText(message)

        if self.config.colors is None:
            return

        color_map = {
            "ok": self.config.colors["status_ok"],
            "warn": self.config.colors["status_warn"],
            "error": self.config.colors["status_err"],
        }

        color = color_map.get(status_type, self.config.colors["status_ok"])
        # Colors may be hex strings (shared engine) or QColor objects
        color_str = color.name() if hasattr(color, "name") else str(color)
        self.status_panel.setStyleSheet(f"background-color: {color_str}")

    def _update_results_tables(self) -> None:
        """Update the results tables with new path information"""
        try:
            if not self.calculation_results:
                return

            # Check if metal conductivity is enabled
            metal_conductive = self.metal_conductive_checkbox.isChecked()

            # Update resistance table
            phases = ["1-2", "2-3", "3-1"]
            current_paths = self.calculation_results.get("current_paths", {})

            for i, phase in enumerate(phases):
                if phase in current_paths:
                    path_data = current_paths[phase]

                    # Phase name
                    self.resistance_table.setItem(i, 0, QTableWidgetItem(phase))

                    # Direct glass resistance
                    direct_res = path_data.get("direct_glass", 0)
                    self.resistance_table.setItem(
                        i, 1, QTableWidgetItem(f"{direct_res:.3f}")
                    )

                    # Via metal resistance (show as N/A if metal conduction disabled)
                    if metal_conductive:
                        metal_res = path_data.get("via_metal", 0)
                        self.resistance_table.setItem(
                            i, 2, QTableWidgetItem(f"{metal_res:.3f}")
                        )
                    else:
                        self.resistance_table.setItem(i, 2, QTableWidgetItem("N/A"))

                    # Total resistance
                    total_res = path_data.get("total", 0)
                    self.resistance_table.setItem(
                        i, 3, QTableWidgetItem(f"{total_res:.3f}")
                    )
                    if (
                        phase in self.phase_inputs
                        and "resistance" in self.phase_inputs[phase]
                    ):
                        cast(QLineEdit, self.phase_inputs[phase]["resistance"]).setText(
                            f"{total_res:.3f}"
                        )

                    # Current split - adjust for metal conduction state
                    if metal_conductive:
                        direct_frac = path_data.get("direct_fraction", 0) * 100
                        metal_frac = path_data.get("metal_fraction", 0) * 100
                        split_text = (
                            f"Direct: {direct_frac:.1f}% / Metal: {metal_frac:.1f}%"
                        )
                    else:
                        # When metal conduction is off, all current goes through glass
                        split_text = "Direct: 100.0% / Metal: 0.0%"

                    self.resistance_table.setItem(i, 4, QTableWidgetItem(split_text))

            # Update power display in phase inputs - now uses consistent power factor calculation
            # Note: Power displays are now updated only by _update_power_distribution()
            # for consistency
            # This section is commented out to avoid conflicting updates
            # actual_currents = self.calculation_results.get('actual_currents', {})
            # for phase in phases:
            #     if phase in self.phase_inputs and phase in actual_currents:
            #         current = actual_currents[phase]
            #         voltage = self.phase_inputs[phase]['voltage'].value()
            #         power_factor = self.power_factor_input.value()
            #         power = current * voltage * power_factor / 1000  # kW with power factor
            #         self.phase_inputs[phase]['power'].setText(f"{power:.1f}")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error updating results tables: %s", e)

    def _update_analysis_display(self) -> None:
        """Update the analysis display with current distribution info"""
        try:
            if not self.calculation_results:
                return

            # Check if metal conductivity is enabled
            metal_conductive = self.metal_conductive_checkbox.isChecked()

            current_dist = self.calculation_results.get("current_distribution", {})

            # Average the metrics across all phases
            if current_dist:
                if metal_conductive:
                    # Normal operation with metal paths
                    avg_direct = (
                        np.mean(
                            [v["direct_glass_fraction"] for v in current_dist.values()]
                        )
                        * 100
                    )
                    avg_metal = (
                        np.mean(
                            [v["via_metal_fraction"] for v in current_dist.values()]
                        )
                        * 100
                    )
                    avg_ratio = np.mean(
                        [v["resistance_ratio"] for v in current_dist.values()]
                    )

                    # Thermal efficiency estimate (simplified)
                    thermal_eff = (avg_direct / 100) * 0.85 + (avg_metal / 100) * 0.95
                else:
                    # Metal conduction disabled - all current through glass
                    avg_direct = 100.0
                    avg_metal = 0.0
                    avg_ratio = 1.0  # No ratio when only one path type

                    # Higher thermal efficiency when all current goes through glass
                    # (no metal losses)
                    thermal_eff = 0.85

                self.path_labels["Direct Glass Fraction"].setText(f"{avg_direct:.1f}%")
                self.path_labels["Via Metal Fraction"].setText(f"{avg_metal:.1f}%")
                self.path_labels["Path Resistance Ratio"].setText(f"{avg_ratio:.2f}")
                self.path_labels["Thermal Efficiency"].setText(f"{thermal_eff:.1%}")

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error updating analysis display: %s", e)

    def _update_temperature_profile(self) -> None:
        """Temperature profile removed - this functionality is no longer needed"""

    def _update_current_distribution(self) -> None:
        """Update current distribution plot with phase and line currents"""
        try:
            if not self.current_ax:
                return

            self.current_ax.clear()

            # Get current values
            phase_currents = [
                self.phase_inputs["1-2"]["current"].value(),
                self.phase_inputs["2-3"]["current"].value(),
                self.phase_inputs["3-1"]["current"].value(),
            ]

            phases = ["1-2", "2-3", "3-1"]
            colors = ["#FF4444", "#44FF44", "#4444FF"]

            # Create side-by-side bars for phase and line currents
            x = np.arange(len(phases))
            width = 0.35

            # Phase currents
            bars1 = self.current_ax.bar(
                x - width / 2,
                phase_currents,
                width,
                label="Phase",
                color=colors,
                alpha=0.8,
            )

            # Line currents: delta config → I_line = √3 × I_phase (#1357)
            sqrt3 = float(np.sqrt(3))
            line_currents = [current * sqrt3 for current in phase_currents]
            bars2 = self.current_ax.bar(
                x + width / 2,
                line_currents,
                width,
                label="Line",
                color=colors,
                alpha=0.5,
            )

            # Increase y-axis limits to prevent cutoff
            max_current = max(phase_currents + line_currents) if phase_currents else 100
            self.current_ax.set_ylim(0, max_current * 1.3)  # 30% headroom

            # Add value labels on bars
            for bar, current in zip(bars1, phase_currents, strict=False):
                height = bar.get_height()
                self.current_ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max_current * 0.02,
                    f"{current:.1f}A",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            for bar, current in zip(bars2, line_currents, strict=False):
                height = bar.get_height()
                self.current_ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max_current * 0.02,
                    f"{current:.1f}A",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if self.current_ax:
                self.current_ax.set_xlabel("Phase")
                self.current_ax.set_ylabel("Current (A)")
                self.current_ax.set_title("Current Distribution")
                self.current_ax.set_xticks(x)
                self.current_ax.set_xticklabels(phases)

                # Position legend below the plot
                self.current_ax.legend(
                    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2
                )

                self.current_ax.grid(True, alpha=0.3)

            if self.current_canvas is not None:
                self.current_canvas.draw()

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error updating current distribution: %s", e)

    def _update_power_distribution(self) -> None:
        """Update power distribution plot with corrected 3-phase delta calculations"""
        try:
            if not self.power_ax:
                return

            self.power_ax.clear()

            # Calculate power for each phase in a 3-phase delta system
            powers = []
            phases = ["Phase 1-2", "Phase 2-3", "Phase 3-1"]
            colors = ["#FF8C00", "#32CD32", "#1E90FF"]

            phase_keys = ["1-2", "2-3", "3-1"]

            # For each phase in delta configuration
            for phase_key in phase_keys:
                current = cast(
                    QDoubleSpinBox, self.phase_inputs[phase_key]["current"]
                ).value()
                voltage = cast(
                    QDoubleSpinBox, self.phase_inputs[phase_key]["voltage"]
                ).value()

                # For resistive loads (molten glass), power factor = 1.0 for each phase
                # Individual phase real power: P = V * I * cos(φ) where cos(φ) = 1 for resistive
                power = current * voltage / 1000  # Convert to kW
                powers.append(power)

            # Calculate total three-phase power
            # For balanced 3-phase delta with resistive loads:
            total_resistive_power = sum(powers)

            # Apply system power factor for any reactive components (transformers, etc.)
            power_factor = self.power_factor_input.value()

            # The system power factor accounts for reactive components in the circuit
            # but doesn't affect the resistive power in the glass
            # Display both values for clarity
            total_apparent_power = (
                total_resistive_power / power_factor
                if power_factor > 0
                else total_resistive_power
            )

            # Update displays
            self.total_power_display.setText(f"{total_resistive_power:.1f}")

            # Update individual phase power displays (always resistive power)
            for i, phase_key in enumerate(phase_keys):
                cast(QLineEdit, self.phase_inputs[phase_key]["power"]).setText(
                    f"{powers[i]:.1f}"
                )

            # Create bar chart (#1379: numpy imported at module level)

            x_positions = np.arange(len(phases))
            bars = self.power_ax.bar(x_positions, powers, color=colors, alpha=0.7)
            self.power_ax.set_xticks(x_positions)
            self.power_ax.set_xticklabels(phases)

            # Increase y-axis limits to prevent cutoff
            max_power = max(powers) if powers else 50
            self.power_ax.set_ylim(0, max_power * 1.3)

            # Add value labels on bars
            for i, (bar, power) in enumerate(zip(bars, powers, strict=False)):
                height = bar.get_height()
                self.power_ax.text(
                    i,
                    height + max_power * 0.02,
                    f"{power:.1f}kW",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Add system power factor info in subtitle
            pf_text = f"System PF: {power_factor:.2f} | Total Resistive: {total_resistive_power:.1f}kW"
            if abs(total_apparent_power - total_resistive_power) > 0.1:
                pf_text += f" | Apparent: {total_apparent_power:.1f}kW"

            if self.power_ax:
                self.power_ax.set_xlabel("Phase")
                self.power_ax.set_ylabel("Power (kW)")
                self.power_ax.set_title(f"Power Distribution\n{pf_text}")
                self.power_ax.grid(True, alpha=0.3)

            if self.power_canvas is not None:
                self.power_canvas.draw()

        except ImportError as e:
            logger.exception("Error updating power distribution: %s", e)
            import traceback

            traceback.print_exc()

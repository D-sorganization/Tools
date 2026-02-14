"""ResultsAndChartsMixin -- results tables and chart update logic.

Handles updating results tables, analysis displays, current/power distribution
charts, color mapping for conductive paths, and electrode sphere drawing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
from PyQt6.QtWidgets import QDoubleSpinBox, QLineEdit, QTableWidgetItem

if TYPE_CHECKING:
    from typing import cast
else:
    from typing import cast

logger = logging.getLogger(__name__)


class ResultsAndChartsMixin:
    """Mixin providing results table and chart update methods.

    Expected to be mixed into a QWidget subclass that defines:
    - ``calculation_results``
    - ``self.config``
    - Various chart axes and canvas attributes
    """

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

    def _get_current_based_color(self, path_type: str, phase_index: int = 0) -> str:
        """Get color based on selected coloring mode with proper scaling"""
        # Get coloring mode
        color_mode = self.color_mode_combo.currentText()

        if color_mode == "Default colors":
            if self.config.color_schemes is not None:
                # Return default colors from scheme
                return str(
                    self.config.color_schemes["default"].get(path_type, "lightblue")
                )
            return "lightblue"

        # Get calculation results
        if not hasattr(self, "calculation_results") or not self.calculation_results:
            return "lightblue"

        # Get the value to map to color
        if color_mode == "Current intensity":
            value = self._get_path_current(path_type, phase_index)
        elif color_mode == "Power dissipation":
            value = self._get_path_power(path_type, phase_index)
        elif color_mode == "Temperature gradient":
            value = self._get_path_temperature(path_type, phase_index)
        else:
            return "lightblue"

        # Get color scale bounds
        if self.auto_scale_checkbox.isChecked():
            # Calculate min/max from all paths
            min_val, max_val = self._calculate_color_scale_bounds(color_mode)
        else:
            min_val = cast(QDoubleSpinBox, self.min_scale_input).value()
            max_val = cast(QDoubleSpinBox, self.max_scale_input).value()

        # Normalize value to 0-1 range
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        else:
            normalized = 0.5

        # Get color from appropriate colormap
        return self._value_to_color(normalized, color_mode)

    def _get_path_current(self, path_type: str, phase_index: int) -> float:
        """Get current value for specific path"""
        actual_currents = self.calculation_results.get("actual_currents", {})
        current_paths = self.calculation_results.get("current_paths", {})

        # Check if metal conduction is enabled
        metal_conductive = self.metal_conductive_checkbox.isChecked()

        phase_keys = ["1-2", "2-3", "3-1"]
        if phase_index < len(phase_keys):
            phase_key = phase_keys[phase_index]
            total_current = actual_currents.get(phase_key, 0.0)
            path_data = current_paths.get(phase_key, {})

            if "direct" in path_type:
                # When metal conduction is off, all current goes through glass
                fraction = (
                    1.0 if not metal_conductive else path_data.get("direct_fraction", 0)
                )
                return float(total_current * fraction)
            if "metal" in path_type:
                # When metal conduction is off, no current through metal
                if not metal_conductive:
                    return 0.0
                return float(total_current * path_data.get("metal_fraction", 0))
        return 0.0

    def _get_path_resistance(self, path_type: str, phase_index: int) -> float:
        """Get resistance value for specific path"""
        current_paths = self.calculation_results.get("current_paths", {})

        # Check if metal conduction is enabled
        metal_conductive = self.metal_conductive_checkbox.isChecked()

        phase_keys = ["1-2", "2-3", "3-1"]
        if phase_index < len(phase_keys):
            phase_key = phase_keys[phase_index]
            path_data = current_paths.get(phase_key, {})

            if "direct" in path_type:
                return float(path_data.get("direct_glass", 0.0))
            if "metal" in path_type:
                # When metal conduction is off, resistance is effectively infinite
                if not metal_conductive:
                    return float("inf")
                return float(path_data.get("via_metal", 0.0))
        return 0.0

    def _get_path_power(self, path_type: str, phase_index: int) -> float:
        """Get power dissipation for specific path"""
        current = self._get_path_current(path_type, phase_index)
        current_paths = self.calculation_results.get("current_paths", {})

        phase_keys = ["1-2", "2-3", "3-1"]
        if phase_index < len(phase_keys):
            phase_key = phase_keys[phase_index]
            path_data = current_paths.get(phase_key, {})

            if "direct" in path_type:
                resistance = path_data.get("direct_glass", 1.0)
            elif "metal" in path_type:
                resistance = path_data.get("via_metal", 1.0)
            else:
                resistance = 1.0

            return float(current**2 * resistance)
        return 0.0

    def _get_path_temperature(self, path_type: str, phase_index: int) -> float:
        """Get estimated temperature for path based on power dissipation"""
        base_temp = self.bath_temp_input.value()
        power = self._get_path_power(path_type, phase_index)

        # Simple temperature rise model (would be more complex in reality)
        temp_rise = power * 0.001  # Simplified: 1°C per kW
        return base_temp + temp_rise

    def _calculate_color_scale_bounds(self, color_mode: str) -> tuple[float, float]:
        """Calculate min/max values for color scaling"""
        values = []

        for phase_idx in range(3):
            for path_type in ["direct_glass", "via_metal"]:
                if color_mode == "Current intensity":
                    values.append(self._get_path_current(path_type, phase_idx))
                elif color_mode == "Power dissipation":
                    values.append(self._get_path_power(path_type, phase_idx))
                elif color_mode == "Temperature gradient":
                    values.append(self._get_path_temperature(path_type, phase_idx))

        if values:
            return min(values), max(values)
        return 0.0, 1.0

    def _value_to_color(self, normalized_value: float, color_mode: str) -> str:
        """Convert normalized value (0-1) to color based on mode"""

        # Select colormap based on mode
        if color_mode == "Current intensity":
            cmap = colormaps.get_cmap("coolwarm")  # Blue to red
        elif color_mode == "Power dissipation":
            cmap = colormaps.get_cmap("hot")  # Black to red to yellow to white
        elif color_mode == "Temperature gradient":
            cmap = colormaps.get_cmap("plasma")  # Purple to pink to yellow
        else:
            cmap = colormaps.get_cmap("viridis")  # Default

        # Get RGBA color
        rgba = cmap(normalized_value)

        # Convert to hex
        return mcolors.to_hex(rgba)

    # Drawing geometry methods (_draw_3d_metal_layer, _draw_3d_glass_layer,
    # _draw_3d_electrodes, _draw_horizontal_cylinder, _draw_3d_refractory_layer,
    # _draw_3d_metal_shell) are provided by DrawingMixin via inheritance.

    def _draw_electrode_sphere(
        self,
        x_center: float,
        y_center: float,
        z_center: float,
        radius: float,
        color: Any,
        alpha: float,
    ) -> None:
        """Draw a spherical tip at the electrode end"""
        if self.electrode_ax is None:
            return
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)

        # Sphere coordinates
        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center

        # Draw sphere
        self.electrode_ax.plot_surface(
            x_sphere, y_sphere, z_sphere, color=color, alpha=alpha, linewidth=0
        )

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

            # Line currents (for demonstration, using phase current * 0.8)
            line_currents = [current * 0.8 for current in phase_currents]
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

            # Create bar chart
            import numpy as np

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

"""VisualizationMixin -- 3D visualization update and color mapping helpers."""

from __future__ import annotations

import logging
from typing import Any, cast

import matplotlib.colors as mcolors
from matplotlib import colormaps
from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox

from ...configs.color_schemes import get_color_scheme

logger = logging.getLogger(__name__)


class VisualizationMixin:
    """Mixin providing the main 3D visualization update and color helpers."""

    # -- Attributes provided by the host class (declared for mypy) --
    auto_scale_checkbox: Any
    bath_diameter_input: Any
    bath_temp_input: Any
    calculation_results: Any
    color_mode_combo: Any
    config: Any
    current_color_scheme: Any
    depth_inputs: Any
    electrode_alpha_slider: Any
    electrode_ax: Any
    electrode_canvas: Any
    electrode_diameter_combo: Any
    electrode_extension_slider: Any
    glass_alpha_slider: Any
    glass_layer_height_input: Any
    max_scale_input: Any
    metal_alpha_slider: Any
    metal_conductive_checkbox: Any
    metal_layer_height_input: Any
    metal_shell_alpha_slider: Any
    min_scale_input: Any
    path_alpha_slider: Any
    refractory_alpha_slider: Any
    refractory_thickness_input: Any
    zoom_slider: Any
    _draw_3d_conductive_paths_new: Any
    _draw_3d_electrodes: Any
    _draw_3d_glass_layer: Any
    _draw_3d_metal_layer: Any
    _draw_3d_metal_shell: Any
    _draw_3d_refractory_layer: Any

    def _update_3d_visualization(self) -> None:
        """Update the 3D electrode visualization with new path geometry."""
        try:
            if not self._is_visualization_ready():
                return

            if self.electrode_ax is not None:
                self.electrode_ax.clear()

            params = self._read_geometry_params()
            self._draw_visible_components(params)
            self._configure_axis_labels()
            self._configure_axis_limits(params)

            logger.debug("[DEBUG] Drawing canvas...")
            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error updating 3D visualization: %s", e)

    def _is_visualization_ready(self) -> bool:
        """Check whether the visualization subsystem is initialised and ready."""
        if not getattr(self, "matplotlib_initialized", False):
            logger.debug("[DEBUG] Matplotlib not initialized, skipping")
            return False
        if not getattr(self, "_initialization_complete", False):
            logger.debug("[DEBUG] Initialization not complete, skipping")
            return False
        try:
            if (
                not hasattr(self, "show_refractory_checkbox")
                or not self.show_refractory_checkbox
            ):
                logger.debug("[DEBUG] Widgets not available, skipping")
                return False
            _ = self.show_refractory_checkbox.isChecked()
        except RuntimeError as e:
            logger.exception("[DEBUG] Widget access error: %s, skipping", e)
            return False
        return True

    def _read_geometry_params(self) -> dict:
        """Read current geometry parameters from the UI widgets."""
        bath_diameter = self.bath_diameter_input.value()
        electrode_diameter = float(self.electrode_diameter_combo.currentText())
        return {
            "bath_radius": bath_diameter / 2.0,
            "electrode_radius": electrode_diameter / 2.0,
            "metal_height": self.metal_layer_height_input.value(),
            "glass_height": self.glass_layer_height_input.value(),
            "refractory_thickness": self.refractory_thickness_input.value(),
            "depths": [inp.value() for inp in self.depth_inputs[:3]],
        }

    @staticmethod
    def _safe_checkbox_check(owner: object, checkbox_name: str) -> bool:
        """Return checkbox state, or False if the widget is unavailable."""
        try:
            checkbox = getattr(owner, checkbox_name, None)
            if isinstance(checkbox, QCheckBox):
                return checkbox.isChecked()
            return False
        except (RuntimeError, AttributeError):
            return False

    def _draw_visible_components(self, p: dict) -> None:
        """Draw 3-D components whose visibility checkbox is ticked."""
        total_height = p["glass_height"] + p["metal_height"]

        if self._safe_checkbox_check(self, "show_refractory_checkbox"):
            self._draw_3d_refractory_layer(
                p["bath_radius"], total_height, p["refractory_thickness"]
            )
        if self._safe_checkbox_check(self, "show_metal_shell_checkbox"):
            self._draw_3d_metal_shell(
                p["bath_radius"], total_height, p["refractory_thickness"]
            )

        metal_cb = getattr(self, "metal_conductive_checkbox", None)
        metal_on = metal_cb.isChecked() if metal_cb else True
        if self._safe_checkbox_check(self, "show_metal_checkbox") and metal_on:
            self._draw_3d_metal_layer(p["bath_radius"], p["metal_height"])

        if self._safe_checkbox_check(self, "show_glass_checkbox"):
            self._draw_3d_glass_layer(
                p["bath_radius"], p["metal_height"], p["glass_height"]
            )
        if self._safe_checkbox_check(self, "show_electrodes_checkbox"):
            self._draw_3d_electrodes(
                p["depths"],
                p["electrode_radius"],
                p["bath_radius"],
                p["metal_height"],
                p["glass_height"],
            )
        if self._safe_checkbox_check(self, "show_paths_checkbox"):
            self._draw_3d_conductive_paths_new(
                p["depths"],
                p["electrode_radius"],
                p["bath_radius"],
                p["metal_height"],
                p["glass_height"],
            )

    def _configure_axis_labels(self) -> None:
        """Show or hide axis labels and ticks based on the checkbox."""
        if self._safe_checkbox_check(self, "show_axis_labels_checkbox"):
            if self.electrode_ax:
                self.electrode_ax.set_xlabel("X (inches)")
                self.electrode_ax.set_ylabel("Y (inches)")
                if hasattr(self.electrode_ax, "set_zlabel"):
                    self.electrode_ax.set_zlabel("Height (inches)")
                self.electrode_ax.tick_params(
                    axis="x", which="both", bottom=True, top=False, labelbottom=True
                )
                self.electrode_ax.tick_params(
                    axis="y", which="both", left=True, right=False, labelleft=True
                )
                if hasattr(self.electrode_ax, "zaxis"):
                    self.electrode_ax.zaxis.set_tick_params(labelleft=True)
        else:
            if self.electrode_ax:
                self.electrode_ax.set_xlabel("")
                self.electrode_ax.set_ylabel("")
                if hasattr(self.electrode_ax, "set_zlabel"):
                    self.electrode_ax.set_zlabel("")
        if hasattr(self, "electrode_ax") and self.electrode_ax:
            self.electrode_ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            self.electrode_ax.tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )
            if hasattr(self.electrode_ax, "zaxis"):
                self.electrode_ax.zaxis.set_tick_params(labelleft=False)

        if self.electrode_ax is not None:
            self.electrode_ax.set_title("")

    def _configure_axis_limits(self, p: dict) -> None:
        """Set axis limits, aspect ratio, and camera angle."""
        total_height = p["glass_height"] + p["metal_height"]
        extension_length = float(self.electrode_extension_slider.value())
        max_range = max(p["bath_radius"] + extension_length, total_height)

        zoom_factor = self.zoom_slider.value() / 100.0
        scaled_range = max_range / zoom_factor * 1.1

        if self.electrode_ax:
            self.electrode_ax.set_xlim(-scaled_range, scaled_range)
            self.electrode_ax.set_ylim(-scaled_range, scaled_range)
            if hasattr(self.electrode_ax, "set_zlim"):
                self.electrode_ax.set_zlim(0, total_height / zoom_factor * 1.2)
            if hasattr(self.electrode_ax, "set_box_aspect"):
                self.electrode_ax.set_box_aspect([1, 1, total_height / (2 * max_range)])
            if hasattr(self.electrode_ax, "view_init"):
                self.electrode_ax.view_init(elev=20, azim=45)

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
        return float(base_temp + temp_rise)

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

    def _get_color_scheme_colors(self) -> list[str]:
        """Get colors based on current color scheme."""
        return get_color_scheme(self.current_color_scheme)

    def _get_transparency_values(self) -> dict[str, float]:
        """Get current transparency values from sliders"""
        return {
            "electrodes": self.electrode_alpha_slider.value() / 100.0,
            "glass": self.glass_alpha_slider.value() / 100.0,
            "metal": self.metal_alpha_slider.value() / 100.0,
            "paths": self.path_alpha_slider.value() / 100.0,
            "refractory": self.refractory_alpha_slider.value() / 100.0,
            "metal_shell": self.metal_shell_alpha_slider.value() / 100.0,
        }

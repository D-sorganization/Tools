"""PathsMixin -- real geometry drawing and conductive path rendering.

Delegates duplicated drawing logic to :mod:`~electrode_advisor.utils.shared_drawing`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...utils.shared_drawing import (
    annotate_path_value,
    annotate_resistance_value,
    build_trapezoidal_prism,
    compute_wall_position,
    draw_trapezoidal_path,
)
from ...utils.visualization import ElectrodeVisualization

logger = logging.getLogger(__name__)


class PathsMixin:
    """Mixin providing 3D geometry and conductive path drawing."""

    # -- Attributes provided by the host class (declared for mypy) --
    bath_diameter_input: Any
    calculation_results: Any
    conductive_layer_height_input: Any
    config: Any
    electrode_alpha_slider: Any
    electrode_ax: Any
    electrode_canvas: Any
    electrode_diameter_combo: Any
    electrode_extension_slider: Any
    glass_alpha_slider: Any
    glass_layer_height_input: Any
    metal_alpha_slider: Any
    metal_conductive_checkbox: Any
    metal_layer_height_input: Any
    metal_shell_alpha_slider: Any
    path_alpha_slider: Any
    refractory_alpha_slider: Any
    refractory_thickness_input: Any
    show_axis_labels_checkbox: Any
    show_electrode_labels_checkbox: Any
    show_electrodes_checkbox: Any
    show_glass_checkbox: Any
    show_metal_checkbox: Any
    show_metal_shell_checkbox: Any
    show_paths_checkbox: Any
    show_refractory_checkbox: Any
    _draw_correct_via_metal_path: Any
    _get_current_based_color: Any
    _get_path_resistance: Any

    def _draw_3d_real_geometry(self) -> None:
        """Draw only the real, physically correct geometry in the 3D plot."""
        ax = self.electrode_ax
        if ax is None:
            return
        ax.clear()

        results = self.calculation_results
        if not results or "electrode_positions" not in results:
            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()
            return

        # Get geometry
        positions = results["electrode_positions"]
        bath_diameter = self.bath_diameter_input.value()
        tip_diameter = float(self.electrode_diameter_combo.currentText())
        glass_depth = self.glass_layer_height_input.value()
        metal_depth = self.metal_layer_height_input.value()
        refractory_thickness = self.refractory_thickness_input.value()

        # Initialize visualization utility
        if not hasattr(self, "visualizer"):
            self.visualizer = ElectrodeVisualization()

        self._draw_real_geometry_layers(
            ax, bath_diameter, glass_depth, metal_depth, refractory_thickness
        )
        self._draw_real_geometry_electrodes(ax, positions, tip_diameter)
        self._draw_real_geometry_paths(
            ax, results, positions, bath_diameter, glass_depth, metal_depth
        )
        self._configure_real_geometry_axes(
            ax, bath_diameter, refractory_thickness, glass_depth, metal_depth
        )

        if self.electrode_canvas is not None:
            self.electrode_canvas.draw()

    def _draw_real_geometry_layers(
        self,
        ax: Any,
        bath_diameter: float,
        glass_depth: float,
        metal_depth: float,
        refractory_thickness: float,
    ) -> None:
        """Draw refractory, glass, metal, and shell layers."""
        if self.show_refractory_checkbox.isChecked():
            self.visualizer.draw_cylinder(
                ax,
                radius=(bath_diameter / 2 + refractory_thickness),
                height=glass_depth + metal_depth,
                z0=0,
                color="#bfa46f",
                alpha=self.refractory_alpha_slider.value() / 100,
            )
        if self.show_glass_checkbox.isChecked():
            self.visualizer.draw_cylinder(
                ax,
                radius=bath_diameter / 2,
                height=glass_depth,
                z0=metal_depth,
                color="#ff8c00",
                alpha=self.glass_alpha_slider.value() / 100,
            )
        if self.show_metal_checkbox.isChecked():
            self.visualizer.draw_cylinder(
                ax,
                radius=bath_diameter / 2,
                height=metal_depth,
                z0=0,
                color="#888888",
                alpha=self.metal_alpha_slider.value() / 100,
            )
        if self.show_metal_shell_checkbox.isChecked():
            self.visualizer.draw_cylinder(
                ax,
                radius=(bath_diameter / 2 + refractory_thickness + 1),
                height=glass_depth + metal_depth,
                z0=0,
                color="#444444",
                alpha=self.metal_shell_alpha_slider.value() / 100,
                linewidth=1,
                wireframe=True,
            )

    def _draw_real_geometry_electrodes(
        self, ax: Any, positions: list[dict], tip_diameter: float
    ) -> None:
        """Draw electrode cylinders and optional labels."""
        if not self.show_electrodes_checkbox.isChecked():
            return
        for pos in positions:
            base = pos["base"]
            tip = pos["tip"]
            self.visualizer.draw_cylinder_between(
                ax,
                base,
                tip,
                radius=tip_diameter / 2,
                color="#888888",
                alpha=self.electrode_alpha_slider.value() / 100,
            )
            if self.show_electrode_labels_checkbox.isChecked():
                ax.text(
                    tip[0],
                    tip[1],
                    tip[2],
                    f"{pos['depth']:.1f}",
                    color="k",
                    fontsize=10,
                )

    def _draw_real_geometry_paths(
        self,
        ax: Any,
        results: dict,
        positions: list[dict],
        bath_diameter: float,
        glass_depth: float,
        metal_depth: float,
    ) -> None:
        """Draw conductive paths (real geometry only)."""
        if not (self.show_paths_checkbox.isChecked() and "current_paths" in results):
            return
        for phase in results["current_paths"]:
            i, j = int(phase[0]) - 1, int(phase[2]) - 1
            self.visualizer.draw_trapezoidal_prism(
                ax,
                positions[i],
                positions[j],
                bath_diameter / 2,
                glass_depth,
                color="#4169E1",
                alpha=self.path_alpha_slider.value() / 100,
            )
            self.visualizer.draw_via_metal_path(
                ax,
                positions[i],
                positions[j],
                bath_diameter / 2,
                metal_depth,
                glass_depth,
                color="#DC143C",
                alpha=self.path_alpha_slider.value() / 100,
            )

    def _configure_real_geometry_axes(
        self,
        ax: Any,
        bath_diameter: float,
        refractory_thickness: float,
        glass_depth: float,
        metal_depth: float,
    ) -> None:
        """Set axis labels, limits, and camera angle."""
        show_labels = self.show_axis_labels_checkbox.isChecked()
        ax.set_xlabel("X (in)" if show_labels else "")
        ax.set_ylabel("Y (in)" if show_labels else "")
        try:
            ax.set_zlabel("Z (in)" if show_labels else "")
        except (AttributeError, ValueError) as e:
            logger.debug("set_zlabel not available: %s", e)

        lim = bath_diameter / 2 + refractory_thickness + 2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        try:
            ax.set_zlim(0, glass_depth + metal_depth)
        except (AttributeError, ValueError) as e:
            logger.debug("set_zlim not available: %s", e)
        try:
            ax.view_init(elev=25, azim=45)
        except (AttributeError, ValueError) as e:
            logger.debug("view_init not available: %s", e)

    def _draw_3d_conductive_paths_new(
        self,
        depths: list[float],
        electrode_radius: float,
        bath_radius: float,
        metal_height: float,
        glass_height: float,
    ) -> None:
        """Draw the new 6-path conductive model with correct geometry."""
        if self.electrode_ax is None:
            return
        metal_conductive = self.metal_conductive_checkbox.isChecked()
        conductive_height = self.conductive_layer_height_input.value()
        path_alpha = self.path_alpha_slider.value() / 100.0

        electrode_positions = self._compute_electrode_positions(
            depths, bath_radius, metal_height, glass_height
        )

        for i in range(3):
            j = (i + 1) % 3
            phase_key = f"{i + 1}-{j + 1}"

            current_paths, actual_currents = self._get_phase_data()
            phase_current = actual_currents.get(phase_key, 0.0)
            phase_data = current_paths.get(phase_key, {})

            direct_fraction = phase_data.get(
                "direct_fraction", 1.0 if not metal_conductive else 0.5
            )
            metal_fraction = phase_data.get(
                "metal_fraction", 0.0 if not metal_conductive else 0.5
            )

            direct_current = phase_current * direct_fraction
            metal_current = phase_current * metal_fraction if metal_conductive else 0.0

            direct_resistance = self._get_path_resistance("direct_glass", i)
            metal_resistance = self._get_path_resistance("via_metal", i)

            direct_color = self._get_current_based_color("direct_glass", i)
            self._draw_correct_trapezoidal_path(
                electrode_positions[i],
                electrode_positions[j],
                conductive_height,
                bath_radius,
                color=direct_color,
                alpha=path_alpha * 0.8,
                label=f"Direct Glass {i + 1}-{j + 1}",
                current_value=direct_current,
                resistance_value=direct_resistance,
            )

            if metal_conductive:
                metal_color = self._get_current_based_color("via_metal", i)
                self._draw_correct_via_metal_path(
                    electrode_positions[i],
                    electrode_positions[j],
                    metal_height,
                    electrode_radius,
                    bath_radius,
                    color=metal_color,
                    alpha=path_alpha * 0.6,
                    label=f"Via Metal {i + 1}-{j + 1}",
                    current_value=metal_current,
                    resistance_value=metal_resistance,
                )

    def _compute_electrode_positions(
        self,
        depths: list[float],
        bath_radius: float,
        metal_height: float,
        glass_height: float,
    ) -> list[dict[str, Any]]:
        """Compute the 3 electrode positions at 120-degree intervals."""
        angles = [0, 120, 240]
        refractory_thickness = self.refractory_thickness_input.value()
        electrode_extension = self.electrode_extension_slider.value()
        total_electrode_length = (
            bath_radius + refractory_thickness + electrode_extension
        )
        electrode_positions: list[dict[str, Any]] = []

        for depth, angle in zip(depths, angles, strict=False):
            angle_rad = np.radians(angle)
            electrode_z = metal_height + glass_height - depth  # #1358/#1375
            x_tip = (bath_radius - depth) * np.cos(angle_rad)
            y_tip = (bath_radius - depth) * np.sin(angle_rad)
            x_base = total_electrode_length * np.cos(angle_rad)
            y_base = total_electrode_length * np.sin(angle_rad)
            electrode_positions.append(
                {
                    "tip": np.array([x_tip, y_tip, electrode_z]),
                    "base": np.array([x_base, y_base, electrode_z]),
                    "angle": angle_rad,
                    "depth": depth,
                }
            )
        return electrode_positions

    def _get_phase_data(self) -> tuple[dict, dict]:
        """Return (current_paths, actual_currents) from calculation results."""
        if hasattr(self, "calculation_results") and self.calculation_results:
            current_paths = self.calculation_results.get("current_paths", {})
            actual_currents = self.calculation_results.get("actual_currents", {})
        else:
            current_paths = {}
            actual_currents = {}
        return current_paths, actual_currents

    def _draw_correct_trapezoidal_path(
        self,
        electrode1_pos: dict[str, Any],
        electrode2_pos: dict[str, Any],
        conductive_height: float,
        bath_radius: float,
        color: str = "blue",
        alpha: float = 0.4,
        label: str = "",
        current_value: float = 0.0,
        resistance_value: float = 0.0,
    ) -> None:
        """Draw the correct trapezoidal prism path within the glass bath area.

        Delegates to :func:`~shared_drawing.draw_trapezoidal_path`.
        """
        if self.electrode_ax is None:
            return
        draw_trapezoidal_path(
            owner=self,
            ax=self.electrode_ax,
            electrode1_pos=electrode1_pos,
            electrode2_pos=electrode2_pos,
            conductive_height=conductive_height,
            bath_radius=bath_radius,
            vertical_spreading_factor=self.config.vertical_spreading_factor,
            color=color,
            alpha=alpha,
            current_value=current_value,
            resistance_value=resistance_value,
        )

    @staticmethod
    def _compute_wall_position(
        electrode_pos: dict[str, Any],
        bath_radius: float,
    ) -> np.ndarray:
        """Compute glass bath wall intersection for an electrode."""
        return compute_wall_position(electrode_pos, bath_radius)

    @staticmethod
    def _build_trapezoidal_prism(
        wall1: np.ndarray,
        tip1: np.ndarray,
        tip2: np.ndarray,
        wall2: np.ndarray,
        electrode_z: float,
        effective_height: float,
    ) -> list[list[list[float]]]:
        """Build 6-face trapezoidal prism vertices from wall/tip positions."""
        return build_trapezoidal_prism(
            wall1, tip1, tip2, wall2, electrode_z, effective_height
        )

    def _annotate_path_value(
        self,
        ax: Any,
        mid_x: float,
        mid_y: float,
        mid_z: float,
        value: float,
        checkbox_name: str,
        fmt: str,
        bg_color: str,
        text_color: str,
    ) -> None:
        """Annotate a path with a formatted value label."""
        annotate_path_value(
            self,
            ax,
            mid_x,
            mid_y,
            mid_z,
            value,
            checkbox_name,
            fmt,
            bg_color,
            text_color,
        )

    def _annotate_resistance_value(
        self,
        ax: Any,
        mid_x: float,
        mid_y: float,
        electrode_z: float,
        resistance_value: float,
        current_value: float,
        bg_color: str,
        text_color: str,
    ) -> None:
        """Annotate a path with resistance value."""
        annotate_resistance_value(
            self,
            ax,
            mid_x,
            mid_y,
            electrode_z,
            resistance_value,
            current_value,
            bg_color,
            text_color,
        )

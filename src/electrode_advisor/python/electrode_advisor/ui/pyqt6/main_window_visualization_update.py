"""VisualizationUpdateMixin -- dynamic 3D visualization updates.

Handles updating the 3D electrode visualization with calculation results,
drawing conductive paths (trapezoidal prisms, via-metal segments), and
rendering the real geometry view.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtWidgets import QCheckBox

from ...utils.constants import ELECTRODE_ANGLES_DEG, ELECTRODE_COUNT
from ...utils.shared_drawing import (
    draw_trapezoidal_path,
    draw_via_metal_path,
)
from ...utils.visualization import ElectrodeVisualization

logger = logging.getLogger(__name__)


class VisualizationUpdateMixin:
    """Mixin providing dynamic 3D visualization update methods.

    Expected to be mixed into a QWidget subclass that also inherits DrawingMixin
    and defines relevant electrode/calculation attributes.
    """

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

        positions = results["electrode_positions"]
        bath_diameter = self.bath_diameter_input.value()
        tip_diameter = float(self.electrode_diameter_combo.currentText())
        glass_depth = self.glass_layer_height_input.value()
        metal_depth = self.metal_layer_height_input.value()
        refractory_thickness = self.refractory_thickness_input.value()

        if not hasattr(self, "visualizer"):
            self.visualizer = ElectrodeVisualization()

        self._draw_real_geom_layers(
            ax, bath_diameter, glass_depth, metal_depth, refractory_thickness
        )
        self._draw_real_geom_electrodes(ax, positions, tip_diameter)
        self._draw_real_geom_paths(
            ax, results, positions, bath_diameter, glass_depth, metal_depth
        )
        self._configure_real_geom_axes(
            ax, bath_diameter, refractory_thickness, glass_depth, metal_depth
        )

        if self.electrode_canvas is not None:
            self.electrode_canvas.draw()

    def _draw_real_geom_layers(
        self,
        ax: Any,
        bath_diameter: float,
        glass_depth: float,
        metal_depth: float,
        refractory_thickness: float,
    ) -> None:
        """Draw refractory, glass, metal, and shell layers for real geometry."""
        assert bath_diameter is not None, "bath_diameter must be provided"
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

    def _draw_real_geom_electrodes(
        self,
        ax: Any,
        positions: list[dict],
        tip_diameter: float,
    ) -> None:
        """Draw electrode cylinders and optional labels for real geometry."""
        assert positions is not None, "positions must be provided"
        if not self.show_electrodes_checkbox.isChecked():
            return
        for pos in positions:
            base, tip = pos["base"], pos["tip"]
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

    def _draw_real_geom_paths(
        self,
        ax: Any,
        results: dict,
        positions: list[dict],
        bath_diameter: float,
        glass_depth: float,
        metal_depth: float,
    ) -> None:
        """Draw conductive paths for real geometry view."""
        assert results is not None, "results must be provided"
        if not (self.show_paths_checkbox.isChecked() and "current_paths" in results):
            return
        for phase in results["current_paths"]:
            parts = phase.split("-")
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
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

    def _configure_real_geom_axes(
        self,
        ax: Any,
        bath_diameter: float,
        refractory_thickness: float,
        glass_depth: float,
        metal_depth: float,
    ) -> None:
        """Set axis labels, limits, and camera angle for real geometry."""
        assert bath_diameter is not None, "bath_diameter must be provided"
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

    def _update_3d_visualization(self) -> None:
        """Update the 3D electrode visualization with new path geometry."""
        try:
            if not self._is_viz_update_ready():
                return

            if self.electrode_ax is not None:
                self.electrode_ax.clear()

            params = self._read_viz_geometry_params()
            self._draw_viz_visible_components(params)
            self._configure_viz_axis_labels()
            self._configure_viz_axis_limits(params)

            logger.debug("[DEBUG] Drawing canvas...")
            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error updating 3D visualization: %s", e)

    def _is_viz_update_ready(self) -> bool:
        """Check whether the visualization subsystem is ready for an update."""
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

    def _read_viz_geometry_params(self) -> dict:
        """Read current geometry parameters from the UI widgets."""
        bath_diameter = self.bath_diameter_input.value()
        electrode_diameter = float(self.electrode_diameter_combo.currentText())
        return {
            "bath_radius": bath_diameter / 2.0,
            "electrode_radius": electrode_diameter / 2.0,
            "metal_height": self.metal_layer_height_input.value(),
            "glass_height": self.glass_layer_height_input.value(),
            "refractory_thickness": self.refractory_thickness_input.value(),
            "depths": [self.depth_inputs[i].value() for i in range(ELECTRODE_COUNT)],
        }

    @staticmethod
    def _safe_viz_checkbox(owner: object, checkbox_name: str) -> bool:
        """Return checkbox state, or False if the widget is unavailable."""
        try:
            checkbox = getattr(owner, checkbox_name, None)
            if isinstance(checkbox, QCheckBox):
                return checkbox.isChecked()
            return False
        except (RuntimeError, AttributeError):
            return False

    def _draw_viz_visible_components(self, p: dict) -> None:
        """Draw 3-D components whose visibility checkbox is ticked."""
        assert p is not None, "p must be provided"
        total_height = p["glass_height"] + p["metal_height"]

        if self._safe_viz_checkbox(self, "show_refractory_checkbox"):
            self._draw_3d_refractory_layer(
                p["bath_radius"], total_height, p["refractory_thickness"]
            )
        if self._safe_viz_checkbox(self, "show_metal_shell_checkbox"):
            self._draw_3d_metal_shell(
                p["bath_radius"], total_height, p["refractory_thickness"]
            )

        metal_cb = getattr(self, "metal_conductive_checkbox", None)
        metal_on = metal_cb.isChecked() if metal_cb else True
        if self._safe_viz_checkbox(self, "show_metal_checkbox") and metal_on:
            self._draw_3d_metal_layer(p["bath_radius"], p["metal_height"])

        if self._safe_viz_checkbox(self, "show_glass_checkbox"):
            self._draw_3d_glass_layer(
                p["bath_radius"], p["metal_height"], p["glass_height"]
            )
        if self._safe_viz_checkbox(self, "show_electrodes_checkbox"):
            self._draw_3d_electrodes(
                p["depths"],
                p["electrode_radius"],
                p["bath_radius"],
                p["metal_height"],
                p["glass_height"],
            )
        if self._safe_viz_checkbox(self, "show_paths_checkbox"):
            self._draw_3d_conductive_paths_new(
                p["depths"],
                p["electrode_radius"],
                p["bath_radius"],
                p["metal_height"],
                p["glass_height"],
            )

    def _configure_viz_axis_labels(self) -> None:
        """Show or hide axis labels and ticks based on the checkbox."""
        if self._safe_viz_checkbox(self, "show_axis_labels_checkbox"):
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
                self.electrode_ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                )
                self.electrode_ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                )
                if hasattr(self.electrode_ax, "zaxis"):
                    self.electrode_ax.zaxis.set_tick_params(labelleft=False)

        if self.electrode_ax is not None:
            self.electrode_ax.set_title("")

    def _configure_viz_axis_limits(self, p: dict) -> None:
        """Set axis limits, aspect ratio, and camera angle."""
        assert p is not None, "p must be provided"
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

    def _draw_3d_conductive_paths_new(
        self,
        depths: list[float],
        electrode_radius: float,
        bath_radius: float,
        metal_height: float,
        glass_height: float,
    ) -> None:
        """Draw the new 6-path conductive model with correct geometry."""
        assert depths is not None, "depths must be provided"
        if self.electrode_ax is None:
            return
        metal_conductive = self.metal_conductive_checkbox.isChecked()
        conductive_height = self.conductive_layer_height_input.value()
        path_alpha = self.path_alpha_slider.value() / 100.0

        electrode_positions = self._compute_electrode_positions_for_paths(
            depths, bath_radius, metal_height, glass_height
        )

        for i in range(ELECTRODE_COUNT):
            j = (i + 1) % ELECTRODE_COUNT
            phase_key = f"{i + 1}-{j + 1}"

            current_paths, actual_currents = self._get_phase_current_data()
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

            direct_color = self._get_current_based_color("direct_glass", i)
            self._draw_correct_trapezoidal_path(
                electrode_positions[i],
                electrode_positions[j],
                conductive_height,
                bath_radius,
                color=direct_color,
                alpha=path_alpha * 0.8,
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
                    current_value=metal_current,
                    resistance_value=self._get_path_resistance("via_metal", i),
                )

    def _compute_electrode_positions_for_paths(
        self,
        depths: list[float],
        bath_radius: float,
        metal_height: float,
        glass_height: float,
    ) -> list[dict[str, Any]]:
        """Compute electrode positions at 120-degree intervals for path drawing."""
        assert depths is not None, "depths must be provided"
        angles = ELECTRODE_ANGLES_DEG
        refractory_thickness = self.refractory_thickness_input.value()
        electrode_extension = self.electrode_extension_slider.value()
        total_length = bath_radius + refractory_thickness + electrode_extension
        positions: list[dict[str, Any]] = []

        for depth, angle in zip(depths, angles, strict=False):
            angle_rad = np.radians(angle)
            electrode_z = metal_height + glass_height - depth
            x_tip = (bath_radius - depth) * np.cos(angle_rad)
            y_tip = (bath_radius - depth) * np.sin(angle_rad)
            x_base = total_length * np.cos(angle_rad)
            y_base = total_length * np.sin(angle_rad)
            positions.append(
                {
                    "tip": np.array([x_tip, y_tip, electrode_z]),
                    "base": np.array([x_base, y_base, electrode_z]),
                    "angle": angle_rad,
                    "depth": depth,
                }
            )
        return positions

    def _get_phase_current_data(self) -> tuple[dict, dict]:
        """Return (current_paths, actual_currents) from calculation results."""
        if hasattr(self, "calculation_results") and self.calculation_results:
            return (
                self.calculation_results.get("current_paths", {}),
                self.calculation_results.get("actual_currents", {}),
            )
        return {}, {}

    def _draw_correct_trapezoidal_path(
        self,
        electrode1_pos: dict[str, Any],
        electrode2_pos: dict[str, Any],
        conductive_height: float,
        bath_radius: float,
        color: str = "blue",
        alpha: float = 0.4,
        current_value: float = 0.0,
        resistance_value: float = 0.0,
    ) -> None:
        """Draw the correct trapezoidal prism path within the glass bath area.

        Delegates to :func:`~shared_drawing.draw_trapezoidal_path`.
        """
        assert electrode1_pos is not None, "electrode1_pos must be provided"
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

    def _draw_correct_via_metal_path(
        self,
        electrode1_pos: dict[str, Any],
        electrode2_pos: dict[str, Any],
        metal_height: float,
        electrode_radius: float,
        bath_radius: float,
        color: str = "red",
        alpha: float = 0.3,
        current_value: float = 0.0,
        resistance_value: float = 0.0,
    ) -> None:
        """Draw the correct 3-segment via-metal path with vertical extrusions.

        Delegates to :func:`~shared_drawing.draw_via_metal_path`.
        """
        assert electrode1_pos is not None, "electrode1_pos must be provided"
        if self.electrode_ax is None:
            return
        draw_via_metal_path(
            owner=self,
            ax=self.electrode_ax,
            electrode1_pos=electrode1_pos,
            electrode2_pos=electrode2_pos,
            metal_height=metal_height,
            electrode_radius=electrode_radius,
            bath_radius=bath_radius,
            color=color,
            alpha=alpha,
            current_value=current_value,
            resistance_value=resistance_value,
            horizontal_spreading_factor=self.config.horizontal_spreading_factor,
        )

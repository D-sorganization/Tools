"""VisualizationUpdateMixin -- dynamic 3D visualization updates.

Handles updating the 3D electrode visualization with calculation results,
drawing conductive paths (trapezoidal prisms, via-metal segments), and
rendering the real geometry view.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt6.QtWidgets import QCheckBox

from ...utils.visualization import ElectrodeVisualization

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VisualizationUpdateMixin:
    """Mixin providing dynamic 3D visualization update methods.

    Expected to be mixed into a QWidget subclass that also inherits DrawingMixin
    and defines relevant electrode/calculation attributes.
    """

    def _draw_3d_real_geometry(self) -> None:
        """Draw only the real, physically correct geometry in the 3D plot"""
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

        # Draw refractory (cylinder)
        if self.show_refractory_checkbox.isChecked():
            self.visualizer.draw_cylinder(
                ax,
                radius=(bath_diameter / 2 + refractory_thickness),
                height=glass_depth + metal_depth,
                z0=0,
                color="#bfa46f",
                alpha=self.refractory_alpha_slider.value() / 100,
            )

        # Draw glass (cylinder)
        if self.show_glass_checkbox.isChecked():
            self.visualizer.draw_cylinder(
                ax,
                radius=bath_diameter / 2,
                height=glass_depth,
                z0=metal_depth,
                color="#ff8c00",
                alpha=self.glass_alpha_slider.value() / 100,
            )

        # Draw metal (cylinder)
        if self.show_metal_checkbox.isChecked():
            self.visualizer.draw_cylinder(
                ax,
                radius=bath_diameter / 2,
                height=metal_depth,
                z0=0,
                color="#888888",
                alpha=self.metal_alpha_slider.value() / 100,
            )

        # Draw shell (thin cylinder)
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

        # Draw electrodes (cylinders)
        if self.show_electrodes_checkbox.isChecked():
            for pos in positions:
                # Electrode from top to tip
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
                    # ax.text expects x, y, z, s
                    ax.text(
                        tip[0],
                        tip[1],
                        tip[2],
                        f"{pos['depth']:.1f}",
                        color="k",
                        fontsize=10,
                    )

        # Draw conductive paths (real geometry only)
        if self.show_paths_checkbox.isChecked() and "current_paths" in results:
            for phase in results["current_paths"]:
                i, j = int(phase[0]) - 1, int(phase[2]) - 1
                positions[i]["tip"]
                positions[j]["tip"]
                # Draw direct glass path (trapezoidal prism)
                self.visualizer.draw_trapezoidal_prism(
                    ax,
                    positions[i],
                    positions[j],
                    bath_diameter / 2,
                    glass_depth,
                    color="#4169E1",
                    alpha=self.path_alpha_slider.value() / 100,
                )
                # Draw via-metal path (composite segments)
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

        # Axis labels (matplotlib 3d axes use set_xlabel, set_ylabel, set_zlabel,
        # but some environments may not support set_zlabel directly)
        ax.set_xlabel("X (in)" if self.show_axis_labels_checkbox.isChecked() else "")
        ax.set_ylabel("Y (in)" if self.show_axis_labels_checkbox.isChecked() else "")
        # set_zlabel is not always present, so use hasattr
        # set_zlabel is not always present, so use try/except
        try:
            ax.set_zlabel(
                "Z (in)" if self.show_axis_labels_checkbox.isChecked() else ""
            )
        except (AttributeError, ValueError) as zlabel_error:
            logger.debug("set_zlabel not available: %s", zlabel_error)

        # Set aspect and limits
        lim = bath_diameter / 2 + refractory_thickness + 2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # set_zlim is not always present, so use hasattr
        try:
            ax.set_zlim(0, glass_depth + metal_depth)
        except (AttributeError, ValueError) as zlim_error:
            logger.debug("set_zlim not available: %s", zlim_error)
        try:
            ax.view_init(elev=25, azim=45)
        except (AttributeError, ValueError) as view_error:
            logger.debug("view_init not available: %s", view_error)
        if self.electrode_canvas is not None:
            self.electrode_canvas.draw()

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
            "depths": [inp.value() for inp in self.depth_inputs[:3]],
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

    def _configure_viz_axis_limits(self, p: dict) -> None:
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
                self.electrode_ax.set_box_aspect(
                    [1, 1, total_height / (2 * max_range)]
                )
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
        """Draw the new 6-path conductive model with correct geometry"""
        if self.electrode_ax is None:
            return
        # Check if metal conductivity is enabled
        metal_conductive = self.metal_conductive_checkbox.isChecked()

        # Get conductive layer height parameter
        conductive_height = self.conductive_layer_height_input.value()

        # Get path alpha from slider
        path_alpha = self.path_alpha_slider.value() / 100.0

        # Electrode positions (120 degrees apart)
        angles = [0, 120, 240]  # degrees
        electrode_positions = []

        # Calculate electrode positions with full geometry
        # Extend electrodes beyond refractory layer
        refractory_thickness = self.refractory_thickness_input.value()
        electrode_extension = self.electrode_extension_slider.value()
        total_electrode_length = (
            bath_radius + refractory_thickness + electrode_extension
        )

        for _, (depth, angle) in enumerate(zip(depths, angles, strict=False)):
            angle_rad = np.radians(angle)
            electrode_z = metal_height + glass_height / 2

            # Electrode tip position (inside vessel)
            x_tip = (bath_radius - depth) * np.cos(angle_rad)
            y_tip = (bath_radius - depth) * np.sin(angle_rad)

            # Electrode base/wall position (extended beyond refractory)
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

        # Draw the paths based on metal conductivity setting
        for i in range(3):
            j = (i + 1) % 3
            phase_key = f"{i + 1}-{j + 1}"

            # Get current values for this phase if available
            current_paths = (
                self.calculation_results.get("current_paths", {})
                if hasattr(self, "calculation_results") and self.calculation_results
                else {}
            )
            actual_currents = (
                self.calculation_results.get("actual_currents", {})
                if hasattr(self, "calculation_results") and self.calculation_results
                else {}
            )

            phase_current = actual_currents.get(phase_key, 0.0)
            phase_data = current_paths.get(phase_key, {})

            # Calculate current through each path
            direct_fraction = phase_data.get(
                "direct_fraction", 1.0 if not metal_conductive else 0.5
            )
            metal_fraction = phase_data.get(
                "metal_fraction", 0.0 if not metal_conductive else 0.5
            )

            direct_current = phase_current * direct_fraction
            metal_current = phase_current * metal_fraction if metal_conductive else 0.0

            # Calculate resistance values
            direct_resistance = self._get_path_resistance("direct_glass", i)
            metal_resistance = self._get_path_resistance("via_metal", i)

            # 1. Direct glass conduction path (always draw)
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

            # 2. Via-metal conduction path (only draw if metal conductivity is enabled)
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
        """
        Draw the correct trapezoidal prism path where:
        - The trapezoid is formed ONLY within the glass bath area (not in refractory)
        - Conductive paths start at the glass bath wall, not at electrode bases
        - Consists of vertical segments and a horizontal pie slice in metal
        """
        if self.electrode_ax is None:
            return
        ax = self.electrode_ax

        try:
            e1_tip = electrode1_pos["tip"]
            e2_tip = electrode2_pos["tip"]

            e1_wall = self._compute_wall_position(
                electrode1_pos,
                bath_radius,
            )
            e2_wall = self._compute_wall_position(
                electrode2_pos,
                bath_radius,
            )

            effective_height = conductive_height * self.config.vertical_spreading_factor
            electrode_z = (e1_tip[2] + e2_tip[2]) / 2

            faces = self._build_trapezoidal_prism(
                e1_wall,
                e1_tip,
                e2_tip,
                e2_wall,
                electrode_z,
                effective_height,
            )

            face_collection = Poly3DCollection(
                faces,
                alpha=alpha,
                facecolors=color,
                edgecolor="darkblue",
                linewidth=0.5,
            )
            ax.add_collection3d(face_collection)

            # Draw conductive path boundary lines
            if alpha > 0.3:
                for wall, tip in [(e1_wall, e1_tip), (e2_wall, e2_tip)]:
                    ax.plot(
                        [wall[0], tip[0]],
                        [wall[1], tip[1]],
                        [electrode_z, electrode_z],
                        "k-",
                        linewidth=2,
                        alpha=0.8,
                    )

            # Annotate current and resistance values
            mid_x = (e1_wall[0] + e2_wall[0]) / 2
            mid_y = (e1_wall[1] + e2_wall[1]) / 2

            self._annotate_path_value(
                ax,
                mid_x,
                mid_y,
                electrode_z + 1.5,
                current_value,
                "show_current_values_checkbox",
                "{:.0f}A",
                "lightyellow",
                "darkblue",
            )
            self._annotate_resistance_value(
                ax,
                mid_x,
                mid_y,
                electrode_z,
                resistance_value,
                current_value,
                "lightgreen",
                "darkgreen",
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception(
                "Error drawing correct trapezoidal path: %s",
                e,
            )

    @staticmethod
    def _compute_wall_position(
        electrode_pos: dict[str, Any],
        bath_radius: float,
    ) -> np.ndarray:
        """Compute glass bath wall intersection for an electrode."""
        angle = electrode_pos["angle"]
        tip_z = electrode_pos["tip"][2]
        return np.array(
            [
                bath_radius * np.cos(angle),
                bath_radius * np.sin(angle),
                tip_z,
            ]
        )

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
        z_top = electrode_z + effective_height / 2
        z_bottom = electrode_z - effective_height / 2

        # 8 vertices: bottom face (0-3), top face (4-7)
        v = [
            [wall1[0], wall1[1], z_bottom],
            [tip1[0], tip1[1], z_bottom],
            [tip2[0], tip2[1], z_bottom],
            [wall2[0], wall2[1], z_bottom],
            [wall1[0], wall1[1], z_top],
            [tip1[0], tip1[1], z_top],
            [tip2[0], tip2[1], z_top],
            [wall2[0], wall2[1], z_top],
        ]

        return [
            [v[0], v[1], v[2], v[3]],  # bottom
            [v[4], v[5], v[6], v[7]],  # top
            [v[0], v[1], v[5], v[4]],  # E1 side
            [v[1], v[2], v[6], v[5]],  # tip-to-tip
            [v[2], v[3], v[7], v[6]],  # E2 side
            [v[3], v[0], v[4], v[7]],  # wall-to-wall
        ]

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
        if not (
            hasattr(self, checkbox_name)
            and getattr(self, checkbox_name).isChecked()
            and value > 0
        ):
            return

        ax.text(
            mid_x,
            mid_y,
            mid_z,
            fmt.format(value),
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": bg_color,
                "alpha": 0.8,
            },
            fontsize=8,
            ha="center",
            va="center",
            color=text_color,
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
        if not (
            hasattr(self, "show_resistance_values_checkbox")
            and self.show_resistance_values_checkbox.isChecked()
            and resistance_value > 0
        ):
            return

        # Offset below current annotation when both visible
        offset = (
            -2.0
            if (
                hasattr(self, "show_current_values_checkbox")
                and self.show_current_values_checkbox.isChecked()
                and current_value > 0
            )
            else 1.5
        )
        mid_z = electrode_z + offset

        if resistance_value == float("inf"):
            text = "∞Ω"
        elif resistance_value >= 1.0:
            text = f"{resistance_value:.2f}Ω"
        else:
            text = f"{resistance_value:.3f}Ω"

        ax.text(
            mid_x,
            mid_y,
            mid_z,
            text,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": bg_color,
                "alpha": 0.8,
            },
            fontsize=8,
            ha="center",
            va="center",
            color=text_color,
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
        label: str = "",
        current_value: float = 0.0,
        resistance_value: float = 0.0,
    ) -> None:
        """Draw the correct 3-segment via-metal path with vertical extrusions only"""
        if self.electrode_ax is None:
            return
        ax = self.electrode_ax

        try:
            # Segment 1: Rectangular extrusion down from E1 glass portion
            self._draw_electrode_length_extrusion(
                electrode1_pos,
                metal_height,
                electrode_radius,
                bath_radius,
                direction="down",
                color=color,
                alpha=alpha,
            )

            # Segment 2: Through metal layer - IMPLIED by metal layer itself
            # No need to draw horizontal metal connection box

            # Segment 3: Rectangular extrusion up to E2 glass portion
            self._draw_electrode_length_extrusion(
                electrode2_pos,
                metal_height,
                electrode_radius,
                bath_radius,
                direction="up",
                color=color,
                alpha=alpha,
            )

            # Display current value if checkbox is enabled
            if (
                hasattr(self, "show_current_values_checkbox")
                and self.show_current_values_checkbox.isChecked()
                and current_value > 0
            ):
                # Calculate midpoint between electrodes for text placement
                e1_tip = electrode1_pos["tip"]
                e2_tip = electrode2_pos["tip"]
                mid_x = (e1_tip[0] + e2_tip[0]) / 2
                mid_y = (e1_tip[1] + e2_tip[1]) / 2
                mid_z = metal_height + 0.5  # Slightly above the metal layer

                # Display current value without decimal points
                current_text = f"{current_value:.0f}A"
                ax.text(
                    mid_x,
                    mid_y,
                    mid_z,
                    current_text,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "lightcoral",
                        "alpha": 0.8,
                    },
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="darkred",
                )

            # Display resistance value if checkbox is enabled
            if (
                hasattr(self, "show_resistance_values_checkbox")
                and self.show_resistance_values_checkbox.isChecked()
                and resistance_value > 0
            ):
                # Calculate position slightly offset from current display
                e1_tip = electrode1_pos["tip"]
                e2_tip = electrode2_pos["tip"]
                mid_x = (e1_tip[0] + e2_tip[0]) / 2
                mid_y = (e1_tip[1] + e2_tip[1]) / 2

                # Offset resistance display below current display if both are shown
                offset = (
                    -1.0
                    if (
                        hasattr(self, "show_current_values_checkbox")
                        and self.show_current_values_checkbox.isChecked()
                        and current_value > 0
                    )
                    else 0.5
                )
                mid_z = metal_height + offset

                # Display resistance value with appropriate precision
                if resistance_value == float("inf"):
                    resistance_text = "∞Ω"
                elif resistance_value >= 1.0:
                    resistance_text = f"{resistance_value:.2f}Ω"
                else:
                    resistance_text = f"{resistance_value:.3f}Ω"

                ax.text(
                    mid_x,
                    mid_y,
                    mid_z,
                    resistance_text,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "lightpink",
                        "alpha": 0.8,
                    },
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="darkred",
                )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error drawing correct via-metal path: %s", e)

    def _draw_electrode_length_extrusion(
        self,
        electrode_pos: dict[str, Any],
        metal_height: float,
        electrode_radius: float,
        bath_radius: float,
        direction: str,
        color: str,
        alpha: float,
    ) -> None:
        """Draw rectangular extrusion along electrode length within glass bath
        with horizontal spreading"""
        if self.electrode_ax is None:
            return
        ax = self.electrode_ax

        try:
            # Get glass wall position for this electrode
            angle = electrode_pos["angle"]
            wall_pos = np.array(
                [
                    bath_radius * np.cos(angle),
                    bath_radius * np.sin(angle),
                    electrode_pos["tip"][2],
                ]
            )

            # Get electrode tip
            tip_pos = electrode_pos["tip"]

            # Apply horizontal spreading factor
            effective_radius = (
                electrode_radius * self.config.horizontal_spreading_factor
            )

            # Calculate extrusion bounds
            electrode_z = electrode_pos["tip"][2]

            if direction == "down":
                z_start = electrode_z - electrode_radius
                z_end = metal_height
            else:  # up
                z_start = metal_height
                z_end = electrode_z - electrode_radius

            # Direction vector along electrode (within glass bath only)
            electrode_dir = tip_pos - wall_pos
            electrode_length = np.linalg.norm(electrode_dir[:2])  # Only x,y

            if electrode_length > 0:
                # Create perpendicular vector for width
                electrode_unit = electrode_dir[:2] / electrode_length
                # Get perpendicular in x-y plane
                perp = np.array([-electrode_unit[1], electrode_unit[0], 0])
                perp_scaled = perp * effective_radius

                # 8 vertices of the rectangular box
                vertices = []

                # Bottom face (at z_start)
                vertices.append(
                    wall_pos + perp_scaled + np.array([0, 0, z_start - wall_pos[2]])
                )
                vertices.append(
                    wall_pos - perp_scaled + np.array([0, 0, z_start - wall_pos[2]])
                )
                vertices.append(
                    tip_pos - perp_scaled + np.array([0, 0, z_start - tip_pos[2]])
                )
                vertices.append(
                    tip_pos + perp_scaled + np.array([0, 0, z_start - tip_pos[2]])
                )

                # Top face (at z_end)
                vertices.append(
                    wall_pos + perp_scaled + np.array([0, 0, z_end - wall_pos[2]])
                )
                vertices.append(
                    wall_pos - perp_scaled + np.array([0, 0, z_end - wall_pos[2]])
                )
                vertices.append(
                    tip_pos - perp_scaled + np.array([0, 0, z_end - tip_pos[2]])
                )
                vertices.append(
                    tip_pos + perp_scaled + np.array([0, 0, z_end - tip_pos[2]])
                )

                # Create faces
                faces = []
                # Bottom face
                faces.append([vertices[0], vertices[1], vertices[2], vertices[3]])
                # Top face
                faces.append([vertices[4], vertices[5], vertices[6], vertices[7]])
                # Side faces
                for i in range(4):
                    j = (i + 1) % 4
                    faces.append(
                        [vertices[i], vertices[j], vertices[j + 4], vertices[i + 4]]
                    )

                # Draw using Poly3DCollection
                face_collection = Poly3DCollection(
                    faces,
                    alpha=alpha,
                    facecolors=color,
                    edgecolor="darkred",
                    linewidth=0.5,
                )
                ax.add_collection3d(face_collection)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error drawing electrode length extrusion: %s", e)

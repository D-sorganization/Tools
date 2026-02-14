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
        """Update the 3D electrode visualization with new path geometry"""
        try:
            logger.debug("[DEBUG] _update_3d_visualization called")

            # Check if matplotlib is initialized
            if not getattr(self, "matplotlib_initialized", False):
                logger.debug(
                    "[DEBUG] Matplotlib not initialized, skipping visualization update"
                )
                return

            # Check if initialization is complete and widgets are available
            if not getattr(self, "_initialization_complete", False):
                logger.debug(
                    "[DEBUG] Initialization not complete, skipping visualization update"
                )
                return

            # Check if widgets are still valid before accessing them
            try:
                if (
                    not hasattr(self, "show_refractory_checkbox")
                    or not self.show_refractory_checkbox
                ):
                    logger.debug(
                        "[DEBUG] Widgets not available, skipping visualization update"
                    )
                    return

                # Test if we can access the checkbox state without error
                _ = self.show_refractory_checkbox.isChecked()
            except RuntimeError as e:
                logger.exception(
                    "[DEBUG] Widget access error: %s, skipping visualization update", e
                )
                return

            # Clear the plot
            if self.electrode_ax is not None:
                self.electrode_ax.clear()

            # Get current parameters
            bath_diameter = self.bath_diameter_input.value()
            bath_radius = bath_diameter / 2.0
            electrode_diameter = float(self.electrode_diameter_combo.currentText())
            electrode_radius = electrode_diameter / 2.0
            metal_height = self.metal_layer_height_input.value()
            glass_height = self.glass_layer_height_input.value()
            refractory_thickness = self.refractory_thickness_input.value()

            # Electrode depths
            depths = [
                self.depth_inputs[0].value(),
                self.depth_inputs[1].value(),
                self.depth_inputs[2].value(),
            ]

            # Draw components based on visibility settings with safe widget access
            def safe_checkbox_check(checkbox_name: str) -> bool:
                """Safe Checkbox Check method.

                Returns:
                    Checkbox state
                """
                try:
                    checkbox = getattr(self, checkbox_name, None)
                    if isinstance(checkbox, QCheckBox):
                        return checkbox.isChecked()
                    return False
                except (RuntimeError, AttributeError):
                    return False

            # Draw all components based on checkbox states
            if safe_checkbox_check("show_refractory_checkbox"):
                logger.debug("[DEBUG] Drawing refractory layer")
                self._draw_3d_refractory_layer(
                    bath_radius, glass_height + metal_height, refractory_thickness
                )

            if safe_checkbox_check("show_metal_shell_checkbox"):
                logger.debug("[DEBUG] Drawing metal shell")
                self._draw_3d_metal_shell(
                    bath_radius, glass_height + metal_height, refractory_thickness
                )

            # Check if metal conductivity is enabled before drawing metal layer
            metal_conductive = getattr(self, "metal_conductive_checkbox", None)
            metal_conductive_enabled = (
                metal_conductive.isChecked() if metal_conductive else True
            )

            if safe_checkbox_check("show_metal_checkbox") and metal_conductive_enabled:
                logger.debug("[DEBUG] Drawing metal layer")
                self._draw_3d_metal_layer(bath_radius, metal_height)

            if safe_checkbox_check("show_glass_checkbox"):
                logger.debug("[DEBUG] Drawing glass layer")
                self._draw_3d_glass_layer(bath_radius, metal_height, glass_height)

            if safe_checkbox_check("show_electrodes_checkbox"):
                logger.debug("[DEBUG] Drawing electrodes")
                self._draw_3d_electrodes(
                    depths, electrode_radius, bath_radius, metal_height, glass_height
                )

            if safe_checkbox_check("show_paths_checkbox"):
                logger.debug("[DEBUG] Drawing conductive paths")
                self._draw_3d_conductive_paths_new(
                    depths, electrode_radius, bath_radius, metal_height, glass_height
                )

            # Set labels and title based on user preference with safety check
            if safe_checkbox_check("show_axis_labels_checkbox"):
                if self.electrode_ax:
                    self.electrode_ax.set_xlabel("X (inches)")
                    self.electrode_ax.set_ylabel("Y (inches)")
                    if hasattr(self.electrode_ax, "set_zlabel"):
                        self.electrode_ax.set_zlabel("Height (inches)")
                    # Show tick marks and labels
                    self.electrode_ax.tick_params(
                        axis="x", which="both", bottom=True, top=False, labelbottom=True
                    )
                    self.electrode_ax.tick_params(
                        axis="y", which="both", left=True, right=False, labelleft=True
                    )
                    # For 3D z-axis ticks (if available)
                    if hasattr(self.electrode_ax, "zaxis"):
                        self.electrode_ax.zaxis.set_tick_params(labelleft=True)
            else:
                if self.electrode_ax:
                    self.electrode_ax.set_xlabel("")
                    self.electrode_ax.set_ylabel("")
                    if hasattr(self.electrode_ax, "set_zlabel"):
                        self.electrode_ax.set_zlabel("")
                # Hide tick marks and labels
            if hasattr(self, "electrode_ax") and self.electrode_ax:
                self.electrode_ax.tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
                self.electrode_ax.tick_params(
                    axis="y", which="both", left=False, right=False, labelleft=False
                )
                # For 3D z-axis ticks (if available)
                if hasattr(self.electrode_ax, "zaxis"):
                    self.electrode_ax.zaxis.set_tick_params(labelleft=False)

            # No title on main chart (per user request)
            if self.electrode_ax is not None:
                self.electrode_ax.set_title("")

            # Set equal aspect ratio for true scale
            extension_length = float(self.electrode_extension_slider.value())
            max_range = max(bath_radius + extension_length, glass_height + metal_height)

            # Apply current zoom level
            zoom_factor = self.zoom_slider.value() / 100.0
            scaled_range = max_range / zoom_factor * 1.1

            if self.electrode_ax:
                self.electrode_ax.set_xlim(-scaled_range, scaled_range)
                self.electrode_ax.set_ylim(-scaled_range, scaled_range)
                if hasattr(self.electrode_ax, "set_zlim"):
                    self.electrode_ax.set_zlim(
                        0, (glass_height + metal_height) / zoom_factor * 1.2
                    )

                # Set aspect ratio to 'equal' for true scale
                if hasattr(self.electrode_ax, "set_box_aspect"):
                    # For newer matplotlib versions
                    self.electrode_ax.set_box_aspect(
                        [1, 1, (glass_height + metal_height) / (2 * max_range)]
                    )

                # Set viewing angle for better perspective
                if hasattr(self.electrode_ax, "view_init"):
                    self.electrode_ax.view_init(elev=20, azim=45)

            # Refresh canvas
            logger.debug("[DEBUG] Drawing canvas...")
            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error updating 3D visualization: %s", e)

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
            # Get electrode tips (inside glass bath)
            e1_tip = electrode1_pos["tip"]
            e2_tip = electrode2_pos["tip"]

            # Calculate glass bath wall intersections for both electrodes
            # These are the actual starting points for conductive paths
            # For electrode 1: intersection at glass bath wall
            e1_angle = electrode1_pos["angle"]
            e1_wall_glass = np.array(
                [
                    bath_radius * np.cos(e1_angle),
                    bath_radius * np.sin(e1_angle),
                    e1_tip[2],
                ]
            )

            # For electrode 2: intersection at glass bath wall
            e2_angle = electrode2_pos["angle"]
            e2_wall_glass = np.array(
                [
                    bath_radius * np.cos(e2_angle),
                    bath_radius * np.sin(e2_angle),
                    e2_tip[2],
                ]
            )

            # Apply vertical spreading factor
            effective_height = conductive_height * self.config.vertical_spreading_factor

            # Calculate vertical extrusion bounds
            electrode_z = (e1_tip[2] + e2_tip[2]) / 2
            z_top = electrode_z + effective_height / 2
            z_bottom = electrode_z - effective_height / 2

            # Create 8 vertices of the 3D trapezoidal prism (ONLY in glass bath area)
            vertices = []

            # Bottom face vertices (trapezoid within glass bath only)
            vertices.append(
                [e1_wall_glass[0], e1_wall_glass[1], z_bottom]
            )  # 0: E1 glass wall bottom
            vertices.append([e1_tip[0], e1_tip[1], z_bottom])  # 1: E1 tip bottom
            vertices.append([e2_tip[0], e2_tip[1], z_bottom])  # 2: E2 tip bottom
            vertices.append(
                [e2_wall_glass[0], e2_wall_glass[1], z_bottom]
            )  # 3: E2 glass wall bottom

            # Top face vertices (same trapezoid, higher z)
            vertices.append(
                [e1_wall_glass[0], e1_wall_glass[1], z_top]
            )  # 4: E1 glass wall top
            vertices.append([e1_tip[0], e1_tip[1], z_top])  # 5: E1 tip top
            vertices.append([e2_tip[0], e2_tip[1], z_top])  # 6: E2 tip top
            vertices.append(
                [e2_wall_glass[0], e2_wall_glass[1], z_top]
            )  # 7: E2 glass wall top

            # Create faces for the trapezoidal prism (limited to glass bath area)
            faces = []

            # Bottom face (0-1-2-3)
            faces.append([vertices[0], vertices[1], vertices[2], vertices[3]])

            # Top face (4-5-6-7)
            faces.append([vertices[4], vertices[5], vertices[6], vertices[7]])

            # Side faces
            # Face along E1 (0-1-5-4) - from glass wall to tip
            faces.append([vertices[0], vertices[1], vertices[5], vertices[4]])
            # Face from tip to tip (1-2-6-5)
            faces.append([vertices[1], vertices[2], vertices[6], vertices[5]])
            # Face along E2 (2-3-7-6) - from tip to glass wall
            faces.append([vertices[2], vertices[3], vertices[7], vertices[6]])
            # Face from glass wall to glass wall (3-0-4-7)
            faces.append([vertices[3], vertices[0], vertices[4], vertices[7]])

            # Draw using Poly3DCollection
            face_collection = Poly3DCollection(
                faces,
                alpha=alpha,
                facecolors=color,
                edgecolor="darkblue",
                linewidth=0.5,
            )
            ax.add_collection3d(face_collection)

            # Draw conductive path boundaries within glass bath only
            if alpha > 0.3:
                # Draw E1 conductive length (from glass wall to tip)
                ax.plot(
                    [e1_wall_glass[0], e1_tip[0]],
                    [e1_wall_glass[1], e1_tip[1]],
                    [electrode_z, electrode_z],
                    "k-",
                    linewidth=2,
                    alpha=0.8,
                )
                # Draw E2 conductive length (from glass wall to tip)
                ax.plot(
                    [e2_wall_glass[0], e2_tip[0]],
                    [e2_wall_glass[1], e2_tip[1]],
                    [electrode_z, electrode_z],
                    "k-",
                    linewidth=2,
                    alpha=0.8,
                )

            # Display current value if checkbox is enabled
            if (
                hasattr(self, "show_current_values_checkbox")
                and self.show_current_values_checkbox.isChecked()
                and current_value > 0
            ):
                # Calculate midpoint for text placement
                mid_x = (e1_wall_glass[0] + e2_wall_glass[0]) / 2
                mid_y = (e1_wall_glass[1] + e2_wall_glass[1]) / 2
                mid_z = electrode_z + 1.5  # Slightly above the path

                # Display current value without decimal points
                current_text = f"{current_value:.0f}A"
                ax.text(
                    mid_x,
                    mid_y,
                    mid_z,
                    current_text,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "lightyellow",
                        "alpha": 0.8,
                    },
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="darkblue",
                )

            # Display resistance value if checkbox is enabled
            if (
                hasattr(self, "show_resistance_values_checkbox")
                and self.show_resistance_values_checkbox.isChecked()
                and resistance_value > 0
            ):
                # Calculate position slightly offset from current display
                mid_x = (e1_wall_glass[0] + e2_wall_glass[0]) / 2
                mid_y = (e1_wall_glass[1] + e2_wall_glass[1]) / 2

                # Offset resistance display below current display if both are shown
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
                        "facecolor": "lightgreen",
                        "alpha": 0.8,
                    },
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="darkgreen",
                )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error drawing correct trapezoidal path: %s", e)

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

"""ViaMetalMixin -- via-metal path drawing and electrode extrusion."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)


class ViaMetalMixin:
    """Mixin providing via-metal conductive path and electrode extrusion drawing."""

    # -- Attributes provided by the host class (declared for mypy) --
    config: Any
    electrode_ax: Any

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

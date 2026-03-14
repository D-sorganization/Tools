"""Visualization utilities for 3D electrode geometry rendering

This module contains all the 3D drawing and visualization functions
for electrode systems. Layer/vessel drawing is in visualization_layers.py.
"""

import logging
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .constants import (
    CYLINDER_CIRCUM_SEGMENTS,
    CYLINDER_LENGTH_SEGMENTS,
    CYLINDER_THETA_SEGMENTS,
    ELECTRODE_ANGLES_DEG,
    ELECTRODE_COLORS,
    SPHERE_U_RESOLUTION,
    SPHERE_V_RESOLUTION,
)
from .visualization_layers import ElectrodeLayersMixin

logger = logging.getLogger(__name__)


class ElectrodeVisualization(ElectrodeLayersMixin):
    """Handles all 3D visualization and drawing operations for electrode systems.

    Primitive drawing (cylinders, spheres, electrodes, current paths) lives here.
    Vessel layer drawing (metal, glass, refractory, shell) is inherited from
    :class:`ElectrodeLayersMixin`.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize with optional configuration object."""
        self.config = config

    # ================================================================
    # Primitive Cylinder Drawing
    # ================================================================

    def draw_cylinder(
        self,
        ax: Any,
        radius: float,
        height: float,
        z0: float,
        color: str,
        alpha: float = 1.0,
        linewidth: float = 0,
        wireframe: bool = False,
    ) -> None:
        """Draw a 3D cylinder."""
        if radius <= 0 or height <= 0:
            return
        theta = np.linspace(0, 2 * np.pi, CYLINDER_THETA_SEGMENTS)
        z = np.linspace(z0, z0 + height, 2)
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = radius * np.cos(theta_mesh)
        y_mesh = radius * np.sin(theta_mesh)

        if wireframe:
            ax.plot_wireframe(
                x_mesh,
                y_mesh,
                z_mesh,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
        else:
            ax.plot_surface(x_mesh, y_mesh, z_mesh, color=color, alpha=alpha)

    def draw_cylinder_between(
        self,
        ax: Any,
        base: Any,
        tip: Any,
        radius: float,
        color: str,
        alpha: float = 1.0,
    ) -> None:
        """Draw cylinder from base point to tip point."""
        base = np.array(base)
        tip = np.array(tip)
        direction = tip - base
        length = np.linalg.norm(direction)
        if length == 0:
            return
        theta = np.linspace(0, 2 * np.pi, CYLINDER_THETA_SEGMENTS)
        z = np.linspace(0, length, 10)
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = radius * np.cos(theta_mesh)
        y_mesh = radius * np.sin(theta_mesh)
        ax.plot_surface(
            x_mesh + base[0],
            y_mesh + base[1],
            z_mesh * direction[2] / length + base[2],
            color=color,
            alpha=alpha,
        )

    # ================================================================
    # Current Path Drawing
    # ================================================================

    def draw_trapezoidal_prism(
        self,
        ax: Any,
        pos1: dict[str, Any],
        pos2: dict[str, Any],
        bath_radius: float,
        glass_depth: float,
        color: str,
        alpha: float = 1.0,
    ) -> None:
        """Draw the real trapezoidal prism between two electrodes in glass."""
        e1_angle = pos1["angle"]
        e2_angle = pos2["angle"]
        z = pos1["tip"][2]
        wall1 = np.array(
            [bath_radius * np.cos(e1_angle), bath_radius * np.sin(e1_angle), z],
        )
        wall2 = np.array(
            [bath_radius * np.cos(e2_angle), bath_radius * np.sin(e2_angle), z],
        )
        tip1 = pos1["tip"]
        tip2 = pos2["tip"]
        verts = [wall1, wall2, tip2, tip1]
        verts_top = [v + np.array([0, 0, glass_depth]) for v in verts]
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],
            [verts_top[0], verts_top[1], verts_top[2], verts_top[3]],
            [verts[0], verts[1], verts_top[1], verts_top[0]],
            [verts[1], verts[2], verts_top[2], verts_top[1]],
            [verts[2], verts[3], verts_top[3], verts_top[2]],
            [verts[3], verts[0], verts_top[0], verts_top[3]],
        ]
        poly = Poly3DCollection(faces, color=color, alpha=alpha)
        ax.add_collection3d(poly)

    def draw_via_metal_path(
        self,
        ax: Any,
        pos1: dict[str, Any],
        pos2: dict[str, Any],
        bath_radius: float,
        metal_depth: float,
        glass_depth: float,
        color: str,
        alpha: float = 1.0,
    ) -> None:
        """Draw the real via-metal path as three segments."""
        tip1 = pos1["tip"]
        tip2 = pos2["tip"]
        # Segment 1: tip1 down to metal
        ax.plot(
            [tip1[0], tip1[0]],
            [tip1[1], tip1[1]],
            [tip1[2], metal_depth],
            color=color,
            alpha=alpha,
            linewidth=3,
        )
        # Segment 2: through metal horizontally
        ax.plot(
            [tip1[0], tip2[0]],
            [tip1[1], tip2[1]],
            [metal_depth, metal_depth],
            color=color,
            alpha=alpha,
            linewidth=3,
        )
        # Segment 3: up to tip2
        ax.plot(
            [tip2[0], tip2[0]],
            [tip2[1], tip2[1]],
            [metal_depth, tip2[2]],
            color=color,
            alpha=alpha,
            linewidth=3,
        )

    # ================================================================
    # Electrode Drawing
    # ================================================================

    def draw_3d_electrodes(
        self,
        ax: Any,
        depths: Any,
        electrode_radius: float,
        bath_radius: float,
        metal_height: float,
        glass_height: float,
        electrode_alpha: float = 0.8,
        extension_length: float = 5.0,
        show_electrode_labels: bool = False,
    ) -> None:
        """Draw the three electrodes as horizontal cylinders with spherical tips."""
        angles = ELECTRODE_ANGLES_DEG
        electrode_colors = ELECTRODE_COLORS

        for i, (depth, angle) in enumerate(zip(depths, angles, strict=False)):
            angle_rad = np.radians(angle)
            electrode_z = metal_height + glass_height - depth

            x_start = (bath_radius + extension_length) * np.cos(angle_rad)
            y_start = (bath_radius + extension_length) * np.sin(angle_rad)
            x_tip = (bath_radius - depth) * np.cos(angle_rad)
            y_tip = (bath_radius - depth) * np.sin(angle_rad)

            self.draw_horizontal_cylinder(
                ax,
                x_start,
                y_start,
                x_tip,
                y_tip,
                electrode_z,
                electrode_radius,
                electrode_colors[i],
                electrode_alpha,
                f"Electrode {i + 1}",
            )

            sphere_radius = electrode_radius * 1.2
            self.draw_electrode_sphere(
                ax,
                x_tip,
                y_tip,
                electrode_z,
                sphere_radius,
                electrode_colors[i],
                electrode_alpha,
            )

            if show_electrode_labels:
                mid_x = (x_start + x_tip) / 2
                mid_y = (y_start + y_tip) / 2
                if hasattr(ax, "text"):
                    ax.text(
                        mid_x,
                        mid_y,
                        electrode_z + 2,
                        f'E{i + 1}: {depth:.1f}" deep',
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "facecolor": "white",
                            "alpha": 0.8,
                        },
                        fontsize=10,
                        ha="center",
                    )

    def draw_horizontal_cylinder(
        self,
        ax: Any,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        z_center: float,
        radius: float,
        color: str,
        alpha: float,
        label: str,
    ) -> None:
        """Draw a horizontal cylindrical electrode with proper 3D geometry."""
        n_length = CYLINDER_LENGTH_SEGMENTS
        n_circum = CYLINDER_CIRCUM_SEGMENTS

        dx = x_end - x_start
        dy = y_end - y_start
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return

        dir_x = dx / length
        dir_y = dy / length
        perp1_x = -dir_y
        perp1_y = dir_x

        t = np.linspace(0, 1, n_length)
        theta = np.linspace(0, 2 * np.pi, n_circum)

        X = np.zeros((n_length, n_circum))
        Y = np.zeros((n_length, n_circum))
        Z = np.zeros((n_length, n_circum))

        for i, t_val in enumerate(t):
            center_x = x_start + t_val * dx
            center_y = y_start + t_val * dy
            for j, theta_val in enumerate(theta):
                offset_x = radius * np.cos(theta_val) * perp1_x
                offset_y = radius * np.cos(theta_val) * perp1_y
                offset_z = radius * np.sin(theta_val)
                X[i, j] = center_x + offset_x
                Y[i, j] = center_y + offset_y
                Z[i, j] = z_center + offset_z

        if hasattr(ax, "plot_surface"):
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)
        else:
            for i in range(0, n_length, 3):
                ax.plot(X[i, :], Y[i, :], Z[i, :], color=color, alpha=alpha)
            for j in range(0, n_circum, 4):
                ax.plot(X[:, j], Y[:, j], Z[:, j], color=color, alpha=alpha)

        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            [z_center, z_center],
            color="darkgray",
            linewidth=3,
            alpha=0.9,
        )

    def draw_electrode_sphere(
        self,
        ax: Any,
        x_center: float,
        y_center: float,
        z_center: float,
        radius: float,
        color: str,
        alpha: float,
    ) -> None:
        """Draw a spherical tip at the electrode end."""
        u = np.linspace(0, 2 * np.pi, SPHERE_U_RESOLUTION)
        v = np.linspace(0, np.pi, SPHERE_V_RESOLUTION)

        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center

        if hasattr(ax, "plot_surface"):
            ax.plot_surface(
                x_sphere,
                y_sphere,
                z_sphere,
                color=color,
                alpha=alpha,
                linewidth=0,
            )
        else:
            for i in range(0, len(u), 3):
                ax.plot(
                    x_sphere[i, :],
                    y_sphere[i, :],
                    z_sphere[i, :],
                    color=color,
                    alpha=alpha,
                )
            for j in range(0, len(v), 3):
                ax.plot(
                    x_sphere[:, j],
                    y_sphere[:, j],
                    z_sphere[:, j],
                    color=color,
                    alpha=alpha,
                )

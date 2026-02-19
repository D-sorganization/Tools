"""Visualization utilities for 3D electrode geometry rendering

This module contains all the 3D drawing and visualization functions
for electrode systems. Layer/vessel drawing is in visualization_layers.py.
"""

import logging
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .visualization_layers import ElectrodeLayersMixin

logger = logging.getLogger(__name__)


class ElectrodeVisualization(ElectrodeLayersMixin):
    """Handles all 3D visualization and drawing operations for electrode systems.

    Primitive drawing (cylinders, spheres, electrodes, current paths) lives here.
    Vessel layer drawing (metal, glass, refractory, shell) is inherited from
    :class:`ElectrodeLayersMixin`.
    """

    def __init__(self, ax: Any = None) -> None:
        """Initialize with optional matplotlib 3D axis."""
        self.ax = ax

    def set_axis(self, ax: Any) -> None:
        """Set the matplotlib 3D axis for drawing."""
        self.ax = ax

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
        theta = np.linspace(0, 2 * np.pi, 20)
        z = np.linspace(z0, z0 + height, 2)
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = radius * np.cos(theta_mesh)
        y_mesh = radius * np.sin(theta_mesh)

        if wireframe:
            ax.plot_wireframe(
                x_mesh, y_mesh, z_mesh,
                color=color, alpha=alpha, linewidth=linewidth,
            )
        else:
            ax.plot_surface(x_mesh, y_mesh, z_mesh, color=color, alpha=alpha)

    def draw_cylinder_between(
        self, ax: Any, base: Any, tip: Any, radius: float,
        color: str, alpha: float = 1.0,
    ) -> None:
        """Draw cylinder from base point to tip point."""
        base = np.array(base)
        tip = np.array(tip)
        direction = tip - base
        length = np.linalg.norm(direction)
        if length == 0:
            return
        theta = np.linspace(0, 2 * np.pi, 20)
        z = np.linspace(0, length, 10)
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = radius * np.cos(theta_mesh)
        y_mesh = radius * np.sin(theta_mesh)
        ax.plot_surface(
            x_mesh + base[0], y_mesh + base[1],
            z_mesh * direction[2] / length + base[2],
            color=color, alpha=alpha,
        )

    # ================================================================
    # Current Path Drawing
    # ================================================================

    def draw_trapezoidal_prism(
        self, ax: Any, pos1: dict[str, Any], pos2: dict[str, Any],
        bath_radius: float, glass_depth: float,
        color: str, alpha: float = 1.0,
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
        self, ax: Any, pos1: dict[str, Any], pos2: dict[str, Any],
        bath_radius: float, metal_depth: float, glass_depth: float,
        color: str, alpha: float = 1.0,
    ) -> None:
        """Draw the real via-metal path as three segments."""
        tip1 = pos1["tip"]
        tip2 = pos2["tip"]
        # Segment 1: tip1 down to metal
        ax.plot(
            [tip1[0], tip1[0]], [tip1[1], tip1[1]], [tip1[2], metal_depth],
            color=color, alpha=alpha, linewidth=3,
        )
        # Segment 2: through metal horizontally
        ax.plot(
            [tip1[0], tip2[0]], [tip1[1], tip2[1]], [metal_depth, metal_depth],
            color=color, alpha=alpha, linewidth=3,
        )
        # Segment 3: up to tip2
        ax.plot(
            [tip2[0], tip2[0]], [tip2[1], tip2[1]], [metal_depth, tip2[2]],
            color=color, alpha=alpha, linewidth=3,
        )

    def draw_correct_trapezoidal_path(
        self, ax: Any, electrode1_pos: dict, electrode2_pos: dict,
        glass_height: float, electrode_radius: float, bath_radius: float,
        horizontal_spreading_factor: float, color: str = "blue",
        alpha: float = 0.3, label: str = "", current_value: float = 0.0,
        show_current_values: bool = False,
    ) -> None:
        """Draw the correct trapezoidal glass path between electrodes."""
        try:
            wall1_pos, wall2_pos = self._electrode_wall_positions(
                electrode1_pos, electrode2_pos, bath_radius,
            )
            tip1_pos = electrode1_pos["tip"]
            tip2_pos = electrode2_pos["tip"]
            corners = [wall1_pos, wall2_pos, tip2_pos, tip1_pos]

            z_start = electrode1_pos["tip"][2] - electrode_radius
            z_end = z_start + glass_height

            faces = self._extrude_polygon(corners, z_start, z_end)
            face_collection = Poly3DCollection(
                faces, alpha=alpha, facecolors=color,
                edgecolor="darkblue", linewidth=0.5,
            )
            ax.add_collection3d(face_collection)

            if show_current_values and current_value > 0:
                self._label_midpoint(
                    ax, tip1_pos, tip2_pos,
                    z_start + glass_height / 2, current_value,
                    "lightblue", "darkblue",
                )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    def draw_correct_via_metal_path(
        self, ax: Any, electrode1_pos: dict, electrode2_pos: dict,
        metal_height: float, electrode_radius: float, bath_radius: float,
        horizontal_spreading_factor: float, color: str = "red",
        alpha: float = 0.3, label: str = "", current_value: float = 0.0,
        show_current_values: bool = False,
    ) -> None:
        """Draw the correct 3-segment via-metal path with vertical extrusions."""
        try:
            self.draw_electrode_length_extrusion(
                ax, electrode1_pos, metal_height, electrode_radius,
                bath_radius, horizontal_spreading_factor,
                direction="down", color=color, alpha=alpha,
            )
            self.draw_electrode_length_extrusion(
                ax, electrode2_pos, metal_height, electrode_radius,
                bath_radius, horizontal_spreading_factor,
                direction="up", color=color, alpha=alpha,
            )
            if show_current_values and current_value > 0:
                e1_tip = electrode1_pos["tip"]
                e2_tip = electrode2_pos["tip"]
                self._label_midpoint(
                    ax, e1_tip, e2_tip,
                    metal_height + 0.5, current_value,
                    "lightcoral", "darkred",
                )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    def draw_electrode_length_extrusion(
        self, ax: Any, electrode_pos: dict, metal_height: float,
        electrode_radius: float, bath_radius: float,
        horizontal_spreading_factor: float, direction: str,
        color: str, alpha: float,
    ) -> None:
        """Draw rectangular extrusion along electrode length within glass bath."""
        try:
            angle = electrode_pos["angle"]
            wall_pos = np.array([
                bath_radius * np.cos(angle),
                bath_radius * np.sin(angle),
                electrode_pos["tip"][2],
            ])
            tip_pos = electrode_pos["tip"]
            effective_radius = electrode_radius * horizontal_spreading_factor
            electrode_z = electrode_pos["tip"][2]

            if direction == "down":
                z_start = electrode_z - electrode_radius
                z_end = metal_height
            else:
                z_start = metal_height
                z_end = electrode_z - electrode_radius

            electrode_dir = tip_pos - wall_pos
            electrode_length = np.linalg.norm(electrode_dir[:2])

            if electrode_length <= 0:
                return

            electrode_unit = electrode_dir[:2] / electrode_length
            perp = np.array([-electrode_unit[1], electrode_unit[0], 0])
            perp_scaled = perp * effective_radius

            vertices = self._build_extrusion_vertices(
                wall_pos, tip_pos, perp_scaled, z_start, z_end,
            )
            faces = self._box_faces(vertices)

            face_collection = Poly3DCollection(
                faces, alpha=alpha, facecolors=color,
                edgecolor="darkred", linewidth=0.5,
            )
            ax.add_collection3d(face_collection)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    # ================================================================
    # Electrode Drawing
    # ================================================================

    def draw_3d_electrodes(
        self, ax: Any, depths: Any, electrode_radius: float,
        bath_radius: float, metal_height: float, glass_height: float,
        electrode_alpha: float = 0.8, extension_length: float = 5.0,
        show_electrode_labels: bool = False,
    ) -> None:
        """Draw the three electrodes as horizontal cylinders with spherical tips."""
        angles = [0, 120, 240]
        electrode_colors = ["silver", "#C0C0C0", "#E5E5E5"]

        for i, (depth, angle) in enumerate(zip(depths, angles, strict=False)):
            angle_rad = np.radians(angle)
            electrode_z = metal_height + glass_height / 2

            x_start = (bath_radius + extension_length) * np.cos(angle_rad)
            y_start = (bath_radius + extension_length) * np.sin(angle_rad)
            x_tip = (bath_radius - depth) * np.cos(angle_rad)
            y_tip = (bath_radius - depth) * np.sin(angle_rad)

            self.draw_horizontal_cylinder(
                ax, x_start, y_start, x_tip, y_tip,
                electrode_z, electrode_radius, electrode_colors[i],
                electrode_alpha, f"Electrode {i + 1}",
            )

            sphere_radius = electrode_radius * 1.2
            self.draw_electrode_sphere(
                ax, x_tip, y_tip, electrode_z,
                sphere_radius, electrode_colors[i], electrode_alpha,
            )

            if show_electrode_labels:
                mid_x = (x_start + x_tip) / 2
                mid_y = (y_start + y_tip) / 2
                if hasattr(ax, "text"):
                    ax.text(
                        mid_x, mid_y, electrode_z + 2,
                        f'E{i + 1}: {depth:.1f}" deep',
                        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                        fontsize=10, ha="center",
                    )

    def draw_horizontal_cylinder(
        self, ax: Any, x_start: float, y_start: float,
        x_end: float, y_end: float, z_center: float,
        radius: float, color: str, alpha: float, label: str,
    ) -> None:
        """Draw a horizontal cylindrical electrode with proper 3D geometry."""
        n_length = 30
        n_circum = 16

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
            [x_start, x_end], [y_start, y_end], [z_center, z_center],
            color="darkgray", linewidth=3, alpha=0.9,
        )

    def draw_electrode_sphere(
        self, ax: Any, x_center: float, y_center: float,
        z_center: float, radius: float, color: str, alpha: float,
    ) -> None:
        """Draw a spherical tip at the electrode end."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)

        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center

        if hasattr(ax, "plot_surface"):
            ax.plot_surface(
                x_sphere, y_sphere, z_sphere,
                color=color, alpha=alpha, linewidth=0,
            )
        else:
            for i in range(0, len(u), 3):
                ax.plot(x_sphere[i, :], y_sphere[i, :], z_sphere[i, :], color=color, alpha=alpha)
            for j in range(0, len(v), 3):
                ax.plot(x_sphere[:, j], y_sphere[:, j], z_sphere[:, j], color=color, alpha=alpha)

    # ================================================================
    # Internal helpers
    # ================================================================

    @staticmethod
    def _electrode_wall_positions(
        pos1: dict, pos2: dict, bath_radius: float,
    ) -> tuple[Any, Any]:
        """Calculate wall intersection positions for two electrodes."""
        angle1 = pos1["angle"]
        angle2 = pos2["angle"]
        wall1 = np.array([
            bath_radius * np.cos(angle1),
            bath_radius * np.sin(angle1),
            pos1["tip"][2],
        ])
        wall2 = np.array([
            bath_radius * np.cos(angle2),
            bath_radius * np.sin(angle2),
            pos2["tip"][2],
        ])
        return wall1, wall2

    @staticmethod
    def _extrude_polygon(
        corners: list[Any], z_start: float, z_end: float,
    ) -> list[list[Any]]:
        """Extrude a polygon between z_start and z_end, returning faces."""
        n = len(corners)
        bottom = [np.array([c[0], c[1], z_start]) for c in corners]
        top = [np.array([c[0], c[1], z_end]) for c in corners]

        faces = [bottom[:], top[:]]
        for i in range(n):
            j = (i + 1) % n
            faces.append([bottom[i], bottom[j], top[j], top[i]])
        return faces

    @staticmethod
    def _build_extrusion_vertices(
        wall_pos: Any, tip_pos: Any, perp_scaled: Any,
        z_start: float, z_end: float,
    ) -> list[Any]:
        """Build the 8 vertices of a rectangular extrusion box."""
        vertices = []
        z_offsets = [z_start, z_end]
        for z_val in z_offsets:
            for base, sign in [(wall_pos, 1), (wall_pos, -1), (tip_pos, -1), (tip_pos, 1)]:
                vertices.append(
                    base + sign * perp_scaled + np.array([0, 0, z_val - base[2]])
                )
        return vertices

    @staticmethod
    def _box_faces(vertices: list[Any]) -> list[list[Any]]:
        """Create 6 faces from 8 vertices (bottom 0-3, top 4-7)."""
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
        ]
        for i in range(4):
            j = (i + 1) % 4
            faces.append([vertices[i], vertices[j], vertices[j + 4], vertices[i + 4]])
        return faces

    @staticmethod
    def _label_midpoint(
        ax: Any, tip1: Any, tip2: Any, z: float,
        current_value: float, bg_color: str, text_color: str,
    ) -> None:
        """Display a current value label at the midpoint between two tips."""
        mid_x = (tip1[0] + tip2[0]) / 2
        mid_y = (tip1[1] + tip2[1]) / 2
        if hasattr(ax, "text"):
            ax.text(
                mid_x, mid_y, z, f"{current_value:.0f}A",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": bg_color, "alpha": 0.8},
                fontsize=8, ha="center", va="center", color=text_color,
            )

    # ...existing code...

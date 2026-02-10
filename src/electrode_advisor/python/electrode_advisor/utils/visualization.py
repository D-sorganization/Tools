"""Visualization utilities for 3D electrode geometry rendering

This module contains all the 3D drawing and visualization functions
extracted from the main electrode advisor to improve maintainability.
"""

import logging
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logger = logging.getLogger(__name__)


class ElectrodeVisualization:
    """Handles all 3D visualization and drawing operations for electrode systems"""

    def __init__(self, ax: Any = None) -> None:
        """Initialize with optional matplotlib 3D axis"""
        self.ax = ax

    def set_axis(self, ax: Any) -> None:
        """Set the matplotlib 3D axis for drawing"""
        self.ax = ax

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
        """Draw a 3D cylinder"""
        if radius <= 0 or height <= 0:
            return

        # Create cylinder coordinates
        theta = np.linspace(0, 2 * np.pi, 20)
        z = np.linspace(z0, z0 + height, 2)
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = radius * np.cos(theta_mesh)
        y_mesh = radius * np.sin(theta_mesh)

        # Draw cylinder surface
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
        """Draw cylinder from base point to tip point"""
        base = np.array(base)
        tip = np.array(tip)
        direction = tip - base
        length = np.linalg.norm(direction)

        if length == 0:
            return

        # Create cylinder along z-axis then transform
        theta = np.linspace(0, 2 * np.pi, 20)
        z = np.linspace(0, length, 10)
        theta_mesh, z_mesh = np.meshgrid(theta, z)
        x_mesh = radius * np.cos(theta_mesh)
        y_mesh = radius * np.sin(theta_mesh)

        # Transform to correct orientation
        # This is a simplified version - full rotation matrix would be more accurate
        ax.plot_surface(
            x_mesh + base[0],
            y_mesh + base[1],
            z_mesh * direction[2] / length + base[2],
            color=color,
            alpha=alpha,
        )

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
        """Draw the real trapezoidal prism between two electrodes in glass"""
        # Get wall intersection points
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
        # Extrude in Z by glass_depth
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
        """Draw the real via-metal path as three segments"""
        # Down from tip1 to metal
        tip1 = pos1["tip"]
        tip2 = pos2["tip"]
        e1_angle = pos1["angle"]
        e2_angle = pos2["angle"]
        np.array(
            [bath_radius * np.cos(e1_angle), bath_radius * np.sin(e1_angle), tip1[2]],
        )
        np.array(
            [bath_radius * np.cos(e2_angle), bath_radius * np.sin(e2_angle), tip2[2]],
        )
        # Segment 1: tip1 down to metal
        ax.plot(
            [tip1[0], tip1[0]],
            [tip1[1], tip1[1]],
            [tip1[2], metal_depth],
            color=color,
            alpha=alpha,
            linewidth=3,
        )
        # Segment 2: through metal horizontally (center to center)
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

    def draw_correct_trapezoidal_path(
        self,
        ax: Any,
        electrode1_pos: dict,
        electrode2_pos: dict,
        glass_height: float,
        electrode_radius: float,
        bath_radius: float,
        horizontal_spreading_factor: float,
        color: str = "blue",
        alpha: float = 0.3,
        label: str = "",
        current_value: float = 0.0,
        show_current_values: bool = False,
    ) -> None:
        """Draw the correct trapezoidal glass path between electrodes with horizontal spreading"""
        try:
            # Get glass wall positions for each electrode
            angle1 = electrode1_pos["angle"]
            angle2 = electrode2_pos["angle"]

            wall1_pos = np.array(
                [
                    bath_radius * np.cos(angle1),
                    bath_radius * np.sin(angle1),
                    electrode1_pos["tip"][2],
                ],
            )

            wall2_pos = np.array(
                [
                    bath_radius * np.cos(angle2),
                    bath_radius * np.sin(angle2),
                    electrode2_pos["tip"][2],
                ],
            )

            # Get electrode tips
            tip1_pos = electrode1_pos["tip"]
            tip2_pos = electrode2_pos["tip"]

            # Apply horizontal spreading factor to electrode radius
            electrode_radius * horizontal_spreading_factor

            # Define the 4 corner points of the trapezoidal path base
            corners = [wall1_pos, wall2_pos, tip2_pos, tip1_pos]

            # Extrude the trapezoid vertically through the entire glass height
            z_start = electrode1_pos["tip"][2] - electrode_radius
            z_end = z_start + glass_height

            # Create 8 vertices (4 bottom + 4 top)
            vertices = []

            # Bottom vertices (at electrode level minus radius)
            for corner in corners:
                vertices.append(np.array([corner[0], corner[1], z_start]))

            # Top vertices (at top of glass)
            for corner in corners:
                vertices.append(np.array([corner[0], corner[1], z_end]))

            # Create faces of the trapezoidal prism
            faces = []

            # Bottom face (vertices 0-3)
            faces.append([vertices[0], vertices[1], vertices[2], vertices[3]])

            # Top face (vertices 4-7)
            faces.append([vertices[4], vertices[5], vertices[6], vertices[7]])

            # Side faces
            for i in range(4):
                j = (i + 1) % 4
                faces.append(
                    [vertices[i], vertices[j], vertices[j + 4], vertices[i + 4]],
                )

            # Draw using Poly3DCollection
            face_collection = Poly3DCollection(
                faces,
                alpha=alpha,
                facecolors=color,
                edgecolor="darkblue",
                linewidth=0.5,
            )
            ax.add_collection3d(face_collection)

            # Display current value if enabled
            if show_current_values and current_value > 0:
                # Calculate midpoint between electrodes for text placement
                mid_x = (tip1_pos[0] + tip2_pos[0]) / 2
                mid_y = (tip1_pos[1] + tip2_pos[1]) / 2
                mid_z = z_start + glass_height / 2  # Middle of the glass layer

                # Display current value without decimal points
                current_text = f"{current_value:.0f}A"
                if hasattr(ax, "text"):
                    ax.text(
                        mid_x,
                        mid_y,
                        mid_z,
                        current_text,
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "facecolor": "lightblue",
                            "alpha": 0.8,
                        },
                        fontsize=8,
                        ha="center",
                        va="center",
                        color="darkblue",
                    )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    def draw_correct_via_metal_path(
        self,
        ax: Any,
        electrode1_pos: dict,
        electrode2_pos: dict,
        metal_height: float,
        electrode_radius: float,
        bath_radius: float,
        horizontal_spreading_factor: float,
        color: str = "red",
        alpha: float = 0.3,
        label: str = "",
        current_value: float = 0.0,
        show_current_values: bool = False,
    ) -> None:
        """Draw the correct 3-segment via-metal path with vertical extrusions only"""
        try:
            # Segment 1: Rectangular extrusion down from E1 glass portion
            self.draw_electrode_length_extrusion(
                ax,
                electrode1_pos,
                metal_height,
                electrode_radius,
                bath_radius,
                horizontal_spreading_factor,
                direction="down",
                color=color,
                alpha=alpha,
            )

            # Segment 2: Through metal layer - IMPLIED by metal layer itself
            # No need to draw horizontal metal connection box

            # Segment 3: Rectangular extrusion up to E2 glass portion
            self.draw_electrode_length_extrusion(
                ax,
                electrode2_pos,
                metal_height,
                electrode_radius,
                bath_radius,
                horizontal_spreading_factor,
                direction="up",
                color=color,
                alpha=alpha,
            )

            # Display current value if enabled
            if show_current_values and current_value > 0:
                # Calculate midpoint between electrodes for text placement
                e1_tip = electrode1_pos["tip"]
                e2_tip = electrode2_pos["tip"]
                mid_x = (e1_tip[0] + e2_tip[0]) / 2
                mid_y = (e1_tip[1] + e2_tip[1]) / 2
                mid_z = metal_height + 0.5  # Slightly above the metal layer

                # Display current value without decimal points
                current_text = f"{current_value:.0f}A"
                if hasattr(ax, "text"):
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

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    def draw_electrode_length_extrusion(
        self,
        ax,
        electrode_pos: dict,
        metal_height: float,
        electrode_radius: float,
        bath_radius: float,
        horizontal_spreading_factor: float,
        direction: str,
        color: str,
        alpha: float,
    ) -> None:
        """Draw rectangular extrusion along electrode length within glass
        bath with horizontal spreading"""
        try:
            # Get glass wall position for this electrode
            angle = electrode_pos["angle"]
            wall_pos = np.array(
                [
                    bath_radius * np.cos(angle),
                    bath_radius * np.sin(angle),
                    electrode_pos["tip"][2],
                ],
            )

            # Get electrode tip
            tip_pos = electrode_pos["tip"]

            # Apply horizontal spreading factor
            effective_radius = electrode_radius * horizontal_spreading_factor

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
                    wall_pos + perp_scaled + np.array([0, 0, z_start - wall_pos[2]]),
                )
                vertices.append(
                    wall_pos - perp_scaled + np.array([0, 0, z_start - wall_pos[2]]),
                )
                vertices.append(
                    tip_pos - perp_scaled + np.array([0, 0, z_start - tip_pos[2]]),
                )
                vertices.append(
                    tip_pos + perp_scaled + np.array([0, 0, z_start - tip_pos[2]]),
                )

                # Top face (at z_end)
                vertices.append(
                    wall_pos + perp_scaled + np.array([0, 0, z_end - wall_pos[2]]),
                )
                vertices.append(
                    wall_pos - perp_scaled + np.array([0, 0, z_end - wall_pos[2]]),
                )
                vertices.append(
                    tip_pos - perp_scaled + np.array([0, 0, z_end - tip_pos[2]]),
                )
                vertices.append(
                    tip_pos + perp_scaled + np.array([0, 0, z_end - tip_pos[2]]),
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
                        [vertices[i], vertices[j], vertices[j + 4], vertices[i + 4]],
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

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    def draw_3d_vessel(self, ax: Any, radius: float, total_height: float) -> None:
        """Draw the cylindrical vessel in 3D"""
        # Create cylinder wall
        theta = np.linspace(0, 2 * np.pi, 50)
        z_wall = np.linspace(0, total_height, 20)

        # Vessel wall (wireframe)
        for i in range(0, len(theta), 5):
            x_line = [radius * np.cos(theta[i])] * len(z_wall)
            y_line = [radius * np.sin(theta[i])] * len(z_wall)
            ax.plot(x_line, y_line, z_wall, "k-", alpha=0.3, linewidth=0.5)

        # Top and bottom circles
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)

        # Bottom circle
        ax.plot(x_circle, y_circle, 0, "k-", alpha=0.5, linewidth=2)

        # Top circle
        ax.plot(x_circle, y_circle, total_height, "k-", alpha=0.5, linewidth=2)

    def draw_3d_metal_layer(
        self, ax: Any, radius: float, height: float, metal_alpha: float = 0.6
    ) -> None:
        """Draw the metal layer as a fully shaded grey cylinder volume"""
        if height <= 0:
            return

        # Create metal volume representation (similar to glass layer approach)
        theta = np.linspace(0, 2 * np.pi, 30)
        z_metal = np.linspace(0, height, 8)

        # 1. Top surface of metal layer
        r = np.linspace(0, radius, 15)
        R, T = np.meshgrid(r, theta)
        X_top = R * np.cos(T)
        Y_top = R * np.sin(T)
        Z_top = np.ones_like(X_top) * height

        # 2. Bottom surface of metal layer
        X_bottom = R * np.cos(T)
        Y_bottom = R * np.sin(T)
        Z_bottom = np.zeros_like(X_bottom)

        # 3. Cylindrical side surface of metal volume
        T_cyl, Z_cyl = np.meshgrid(theta, z_metal)
        X_cyl = radius * np.cos(T_cyl)
        Y_cyl = radius * np.sin(T_cyl)

        # Metal color (grey/silver)
        metal_color = "#808080"

        # Draw all surfaces to create full volume
        if hasattr(ax, "plot_surface"):
            # Top surface
            ax.plot_surface(X_top, Y_top, Z_top, color=metal_color, alpha=metal_alpha)
            # Bottom surface
            ax.plot_surface(
                X_bottom,
                Y_bottom,
                Z_bottom,
                color=metal_color,
                alpha=metal_alpha,
            )
            # Cylindrical side surface
            ax.plot_surface(
                X_cyl,
                Y_cyl,
                Z_cyl,
                color=metal_color,
                alpha=metal_alpha * 0.9,
            )
        else:
            # Fallback to contour plot
            ax.contour(X_top, Y_top, Z_top, colors=metal_color, alpha=metal_alpha)
            ax.contour(
                X_bottom,
                Y_bottom,
                Z_bottom,
                colors=metal_color,
                alpha=metal_alpha,
            )

        # Edge lines for better definition
        x_edge = radius * np.cos(theta)
        y_edge = radius * np.sin(theta)

        # Top edge
        z_edge_top = np.ones_like(x_edge) * height
        ax.plot(x_edge, y_edge, z_edge_top, color="#606060", linewidth=2, alpha=0.9)

        # Bottom edge
        z_edge_bottom = np.zeros_like(x_edge)
        ax.plot(x_edge, y_edge, z_edge_bottom, color="#606060", linewidth=2, alpha=0.9)

    def draw_3d_glass_layer(
        self,
        ax: Any,
        radius: float,
        metal_height: float,
        glass_height: float,
        glass_alpha: float = 0.4,
    ) -> None:
        """Draw the full glass layer volume above the metal as translucent orange molten glass"""
        total_height = metal_height + glass_height

        # Create glass volume representation (translucent orange for molten glass)
        theta = np.linspace(0, 2 * np.pi, 30)
        z_glass = np.linspace(metal_height, total_height, 10)

        # 1. Top surface of glass
        r = np.linspace(0, radius, 15)
        R, T = np.meshgrid(r, theta)
        X_top = R * np.cos(T)
        Y_top = R * np.sin(T)
        Z_top = np.ones_like(X_top) * total_height

        # 2. Bottom surface of glass (top of metal layer)
        X_bottom = R * np.cos(T)
        Y_bottom = R * np.sin(T)
        Z_bottom = np.ones_like(X_bottom) * metal_height

        # 3. Cylindrical side surface of glass volume
        T_cyl, Z_cyl = np.meshgrid(theta, z_glass)
        X_cyl = radius * np.cos(T_cyl)
        Y_cyl = radius * np.sin(T_cyl)

        # Draw all surfaces with molten glass orange color
        if hasattr(ax, "plot_surface"):
            # Top surface
            ax.plot_surface(X_top, Y_top, Z_top, color="#FF8C00", alpha=glass_alpha)
            # Bottom surface (interface with metal)
            ax.plot_surface(
                X_bottom,
                Y_bottom,
                Z_bottom,
                color="#FF8C00",
                alpha=glass_alpha,
            )
            # Cylindrical side surface
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color="#FF8C00", alpha=glass_alpha)
        else:
            # Fallback to contour plot
            ax.contour(X_top, Y_top, Z_top, colors="#FF8C00", alpha=glass_alpha)
            ax.contour(
                X_bottom,
                Y_bottom,
                Z_bottom,
                colors="#FF8C00",
                alpha=glass_alpha,
            )

        # Edge lines for better definition
        x_edge_top = radius * np.cos(theta)
        y_edge_top = radius * np.sin(theta)
        z_edge_top = np.ones_like(x_edge_top) * total_height
        ax.plot(
            x_edge_top,
            y_edge_top,
            z_edge_top,
            color="#FF6500",
            linewidth=2,
            alpha=0.9,
        )

        # Bottom edge (metal-glass interface)
        z_edge_bottom = np.ones_like(x_edge_top) * metal_height
        ax.plot(
            x_edge_top,
            y_edge_top,
            z_edge_bottom,
            color="#FF6500",
            linewidth=1.5,
            alpha=0.7,
        )

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
        """Draw the three electrodes as horizontal cylinders with spherical tips"""
        # Electrode positions (120 degrees apart)
        angles = [0, 120, 240]  # degrees
        electrode_colors = ["silver", "#C0C0C0", "#E5E5E5"]

        for i, (depth, angle) in enumerate(zip(depths, angles, strict=False)):
            # Convert angle to radians
            angle_rad = np.radians(angle)

            # Electrode height (middle of glass layer)
            electrode_z = metal_height + glass_height / 2

            # Extended electrode - use slider value for extension length
            x_start = (bath_radius + extension_length) * np.cos(angle_rad)
            y_start = (bath_radius + extension_length) * np.sin(angle_rad)

            # Electrode tip position (inside vessel)
            x_tip = (bath_radius - depth) * np.cos(angle_rad)
            y_tip = (bath_radius - depth) * np.sin(angle_rad)

            # Draw the horizontal electrode cylinder
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

            # Draw spherical tip at the end (slightly larger for visibility)
            sphere_radius = electrode_radius * 1.2  # 20% larger than electrode
            self.draw_electrode_sphere(
                ax,
                x_tip,
                y_tip,
                electrode_z,
                sphere_radius,
                electrode_colors[i],
                electrode_alpha,
            )

            # Add depth annotation above electrode (if enabled)
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
        """Draw a horizontal cylindrical electrode with proper 3D geometry"""
        # Number of segments for smooth cylinder
        n_length = 30
        n_circum = 16

        # Direction vector along electrode
        dx = x_end - x_start
        dy = y_end - y_start
        length = np.sqrt(dx**2 + dy**2)

        if length == 0:
            return

        # Unit vectors
        dir_x = dx / length
        dir_y = dy / length

        # Perpendicular vectors for cylinder cross-section
        perp1_x = -dir_y
        perp1_y = dir_x

        # Create cylinder surface
        t = np.linspace(0, 1, n_length)
        theta = np.linspace(0, 2 * np.pi, n_circum)

        # Generate cylinder surface points
        X = np.zeros((n_length, n_circum))
        Y = np.zeros((n_length, n_circum))
        Z = np.zeros((n_length, n_circum))

        for i, t_val in enumerate(t):
            # Position along cylinder centerline
            center_x = x_start + t_val * dx
            center_y = y_start + t_val * dy
            center_z = z_center

            for j, theta_val in enumerate(theta):
                # Point on cylinder surface
                offset_x = radius * np.cos(theta_val) * perp1_x
                offset_y = radius * np.cos(theta_val) * perp1_y
                offset_z = radius * np.sin(theta_val)

                X[i, j] = center_x + offset_x
                Y[i, j] = center_y + offset_y
                Z[i, j] = center_z + offset_z

        # Draw cylinder surface
        if hasattr(ax, "plot_surface"):
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)
        else:
            # Fallback: draw wireframe cylinder
            for i in range(0, n_length, 3):
                ax.plot(X[i, :], Y[i, :], Z[i, :], color=color, alpha=alpha)
            for j in range(0, n_circum, 4):
                ax.plot(X[:, j], Y[:, j], Z[:, j], color=color, alpha=alpha)

        # Draw centerline for clarity
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
        """Draw a spherical tip at the electrode end"""
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)

        # Sphere coordinates
        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center

        # Draw sphere
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
            # Fallback: draw sphere wireframe
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

    def draw_3d_refractory_layer(
        self,
        ax: Any,
        inner_radius: float,
        total_height: float,
        thickness: float,
        refractory_alpha: float = 0.3,
    ) -> None:
        """Draw the refractory layer as a translucent light brown tube around the reactor"""
        try:
            outer_radius = inner_radius + thickness

            # Create refractory volume representation (translucent light brown)
            theta = np.linspace(0, 2 * np.pi, 30)
            z_ref = np.linspace(0, total_height, 10)

            # Outer cylindrical surface
            T_outer, Z_outer = np.meshgrid(theta, z_ref)
            X_outer = outer_radius * np.cos(T_outer)
            Y_outer = outer_radius * np.sin(T_outer)

            # Inner cylindrical surface
            X_inner = inner_radius * np.cos(T_outer)
            Y_inner = inner_radius * np.sin(T_outer)

            # Top annular surface
            r_annular = np.linspace(inner_radius, outer_radius, 8)
            R_annular, T_annular = np.meshgrid(r_annular, theta)
            X_top_annular = R_annular * np.cos(T_annular)
            Y_top_annular = R_annular * np.sin(T_annular)
            Z_top_annular = np.ones_like(X_top_annular) * total_height

            # Bottom annular surface
            X_bottom_annular = R_annular * np.cos(T_annular)
            Y_bottom_annular = R_annular * np.sin(T_annular)
            Z_bottom_annular = np.zeros_like(X_bottom_annular)

            # Light brown color for refractory
            refractory_color = "#D2B48C"  # Tan/light brown

            # Draw all surfaces
            if hasattr(ax, "plot_surface"):
                # Outer surface
                ax.plot_surface(
                    X_outer,
                    Y_outer,
                    Z_outer,
                    color=refractory_color,
                    alpha=refractory_alpha,
                )
                # Inner surface
                ax.plot_surface(
                    X_inner,
                    Y_inner,
                    Z_outer,
                    color=refractory_color,
                    alpha=refractory_alpha * 0.8,
                )
                # Top annular surface
                ax.plot_surface(
                    X_top_annular,
                    Y_top_annular,
                    Z_top_annular,
                    color=refractory_color,
                    alpha=refractory_alpha,
                )
                # Bottom annular surface
                ax.plot_surface(
                    X_bottom_annular,
                    Y_bottom_annular,
                    Z_bottom_annular,
                    color=refractory_color,
                    alpha=refractory_alpha,
                )
            else:
                # Fallback: draw wireframe
                for i in range(0, len(theta), 5):
                    # Outer surface lines
                    x_line_outer = [outer_radius * np.cos(theta[i])] * len(z_ref)
                    y_line_outer = [outer_radius * np.sin(theta[i])] * len(z_ref)
                    ax.plot(
                        x_line_outer,
                        y_line_outer,
                        z_ref,
                        color=refractory_color,
                        alpha=refractory_alpha * 0.5,
                        linewidth=1,
                    )
                    # Inner surface lines
                    x_line_inner = [inner_radius * np.cos(theta[i])] * len(z_ref)
                    y_line_inner = [inner_radius * np.sin(theta[i])] * len(z_ref)
                    ax.plot(
                        x_line_inner,
                        y_line_inner,
                        z_ref,
                        color=refractory_color,
                        alpha=refractory_alpha * 0.3,
                        linewidth=1,
                    )

            # Edge circles for definition
            x_outer_circle = outer_radius * np.cos(theta)
            y_outer_circle = outer_radius * np.sin(theta)

            # Top and bottom outer circles
            ax.plot(
                x_outer_circle,
                y_outer_circle,
                total_height,
                color="#8B4513",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.plot(
                x_outer_circle,
                y_outer_circle,
                0,
                color="#8B4513",
                linewidth=1.5,
                alpha=0.8,
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    def draw_3d_metal_shell(
        self,
        ax: Any,
        inner_radius: float,
        total_height: float,
        refractory_thickness: float,
        shell_alpha: float = 0.2,
    ) -> None:
        """Draw the metal vessel shell as a 1/2" thick cylinder outside the refractory"""
        try:
            # Metal shell is 1/2" thick outside the refractory
            shell_thickness = 0.5  # inches
            shell_inner_radius = inner_radius + refractory_thickness
            shell_outer_radius = shell_inner_radius + shell_thickness

            # Create shell volume representation (dark grey/black)
            theta = np.linspace(0, 2 * np.pi, 30)
            z_shell = np.linspace(0, total_height, 8)

            # Outer cylindrical surface
            T_outer, Z_outer = np.meshgrid(theta, z_shell)
            X_outer = shell_outer_radius * np.cos(T_outer)
            Y_outer = shell_outer_radius * np.sin(T_outer)

            # Inner cylindrical surface
            X_inner = shell_inner_radius * np.cos(T_outer)
            Y_inner = shell_inner_radius * np.sin(T_outer)

            # Top annular surface
            r_annular = np.linspace(shell_inner_radius, shell_outer_radius, 5)
            R_annular, T_annular = np.meshgrid(r_annular, theta)
            X_top_annular = R_annular * np.cos(T_annular)
            Y_top_annular = R_annular * np.sin(T_annular)
            Z_top_annular = np.ones_like(X_top_annular) * total_height

            # Bottom annular surface
            X_bottom_annular = R_annular * np.cos(T_annular)
            Y_bottom_annular = R_annular * np.sin(T_annular)
            Z_bottom_annular = np.zeros_like(X_bottom_annular)

            # Dark grey color for metal shell
            shell_color = "#2F2F2F"  # Dark grey

            # Draw all surfaces
            if hasattr(ax, "plot_surface"):
                # Outer surface
                ax.plot_surface(
                    X_outer,
                    Y_outer,
                    Z_outer,
                    color=shell_color,
                    alpha=shell_alpha,
                )
                # Inner surface
                ax.plot_surface(
                    X_inner,
                    Y_inner,
                    Z_outer,
                    color=shell_color,
                    alpha=shell_alpha * 0.8,
                )
                # Top annular surface
                ax.plot_surface(
                    X_top_annular,
                    Y_top_annular,
                    Z_top_annular,
                    color=shell_color,
                    alpha=shell_alpha,
                )
                # Bottom annular surface
                ax.plot_surface(
                    X_bottom_annular,
                    Y_bottom_annular,
                    Z_bottom_annular,
                    color=shell_color,
                    alpha=shell_alpha,
                )
            else:
                # Fallback: draw wireframe
                for i in range(0, len(theta), 5):
                    # Outer surface lines
                    x_line_outer = [shell_outer_radius * np.cos(theta[i])] * len(
                        z_shell,
                    )
                    y_line_outer = [shell_outer_radius * np.sin(theta[i])] * len(
                        z_shell,
                    )
                    ax.plot(
                        x_line_outer,
                        y_line_outer,
                        z_shell,
                        color=shell_color,
                        alpha=shell_alpha * 0.5,
                        linewidth=1,
                    )

            # Edge circles for definition
            x_outer_circle = shell_outer_radius * np.cos(theta)
            y_outer_circle = shell_outer_radius * np.sin(theta)

            # Top and bottom outer circles
            ax.plot(
                x_outer_circle,
                y_outer_circle,
                total_height,
                color="#1C1C1C",
                linewidth=2,
                alpha=0.9,
            )
            ax.plot(
                x_outer_circle,
                y_outer_circle,
                0,
                color="#1C1C1C",
                linewidth=2,
                alpha=0.9,
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    # ...existing code...

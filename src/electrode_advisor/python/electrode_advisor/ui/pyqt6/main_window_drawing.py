"""DrawingMixin -- vessel component drawing (metal, glass, electrodes, refractory, shell)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DrawingMixin:
    """Mixin providing 3D drawing for vessel components."""

    # -- Attributes provided by the host class (declared for mypy) --
    config: Any
    electrode_alpha_slider: Any
    electrode_ax: Any
    electrode_extension_slider: Any
    glass_alpha_slider: Any
    metal_alpha_slider: Any
    metal_shell_alpha_slider: Any
    refractory_alpha_slider: Any
    show_electrode_labels_checkbox: Any
    _draw_electrode_sphere: Any

    def _draw_3d_metal_layer(self, radius: float, height: float) -> None:
        """Draw the metal layer as a fully shaded grey cylinder volume"""
        if height <= 0 or self.electrode_ax is None:
            return

        # Get alpha from slider
        metal_alpha = self.metal_alpha_slider.value() / 100.0

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
        if self.electrode_ax is not None:
            # Top surface
            self.electrode_ax.plot_surface(
                X_top, Y_top, Z_top, color=metal_color, alpha=metal_alpha
            )
            # Bottom surface
            self.electrode_ax.plot_surface(
                X_bottom, Y_bottom, Z_bottom, color=metal_color, alpha=metal_alpha
            )
            # Cylindrical side surface
            self.electrode_ax.plot_surface(
                X_cyl, Y_cyl, Z_cyl, color=metal_color, alpha=metal_alpha * 0.9
            )

        # Edge lines for better definition
        x_edge = radius * np.cos(theta)
        y_edge = radius * np.sin(theta)

        # Top edge
        z_edge_top = np.ones_like(x_edge) * height
        self.electrode_ax.plot(
            x_edge, y_edge, z_edge_top, color="#606060", linewidth=2, alpha=0.9
        )

        # Bottom edge
        z_edge_bottom = np.zeros_like(x_edge)
        self.electrode_ax.plot(
            x_edge, y_edge, z_edge_bottom, color="#606060", linewidth=2, alpha=0.9
        )

    def _draw_3d_glass_layer(
        self, radius: float, metal_height: float, glass_height: float
    ) -> None:
        """Draw the full glass layer volume above the metal as translucent orange molten glass"""
        if self.electrode_ax is None:
            return
        total_height = metal_height + glass_height

        # Get alpha from slider
        glass_alpha = self.glass_alpha_slider.value() / 100.0

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
        if self.electrode_ax is not None:
            # Top surface
            self.electrode_ax.plot_surface(
                X_top, Y_top, Z_top, color="#FF8C00", alpha=glass_alpha
            )
            # Bottom surface (interface with metal)
            self.electrode_ax.plot_surface(
                X_bottom, Y_bottom, Z_bottom, color="#FF8C00", alpha=glass_alpha
            )
            # Cylindrical side surface
            self.electrode_ax.plot_surface(
                X_cyl, Y_cyl, Z_cyl, color="#FF8C00", alpha=glass_alpha
            )

        # Edge lines for better definition
        x_edge_top = radius * np.cos(theta)
        y_edge_top = radius * np.sin(theta)
        z_edge_top = np.ones_like(x_edge_top) * total_height
        self.electrode_ax.plot(
            x_edge_top, y_edge_top, z_edge_top, color="#FF6500", linewidth=2, alpha=0.9
        )

        # Bottom edge (metal-glass interface)
        z_edge_bottom = np.ones_like(x_edge_top) * metal_height
        self.electrode_ax.plot(
            x_edge_top,
            y_edge_top,
            z_edge_bottom,
            color="#FF6500",
            linewidth=1.5,
            alpha=0.7,
        )

    def _draw_3d_electrodes(
        self,
        depths: list[float],
        electrode_radius: float,
        bath_radius: float,
        metal_height: float,
        glass_height: float,
    ) -> None:
        """Draw the three electrodes as horizontal cylinders with spherical tips"""
        if self.electrode_ax is None:
            return
        # Get alpha from slider
        electrode_alpha = self.electrode_alpha_slider.value() / 100.0

        # Electrode positions (120 degrees apart)
        angles = [0, 120, 240]  # degrees
        electrode_colors = ["silver", "#C0C0C0", "#E5E5E5"]

        for i, (depth, angle) in enumerate(zip(depths, angles, strict=False)):
            # Convert angle to radians
            angle_rad = np.radians(angle)

            # Electrode height (middle of glass layer)
            electrode_z = metal_height + glass_height / 2

            # Extended electrode - use slider value for extension length
            extension_length = float(self.electrode_extension_slider.value())
            x_start = (bath_radius + extension_length) * np.cos(angle_rad)
            y_start = (bath_radius + extension_length) * np.sin(angle_rad)

            # Electrode tip position (inside vessel)
            x_tip = (bath_radius - depth) * np.cos(angle_rad)
            y_tip = (bath_radius - depth) * np.sin(angle_rad)

            # Draw the horizontal electrode cylinder
            self._draw_horizontal_cylinder(
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
            self._draw_electrode_sphere(
                x_tip,
                y_tip,
                electrode_z,
                sphere_radius,
                electrode_colors[i],
                electrode_alpha,
            )

            # Add depth annotation above electrode (if enabled)
            if self.show_electrode_labels_checkbox.isChecked():
                mid_x = (x_start + x_tip) / 2
                mid_y = (y_start + y_tip) / 2
                if self.electrode_ax is not None:
                    self.electrode_ax.text(
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

    def _draw_horizontal_cylinder(
        self,
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
        if self.electrode_ax is None:
            return
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
        if self.electrode_ax is not None:
            self.electrode_ax.plot_surface(
                X, Y, Z, color=color, alpha=alpha, linewidth=0
            )

        # Draw centerline for clarity
        self.electrode_ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            [z_center, z_center],
            color="darkgray",
            linewidth=3,
            alpha=0.9,
        )

    def _draw_3d_refractory_layer(
        self, inner_radius: float, total_height: float, thickness: float
    ) -> None:
        """Draw the refractory layer as a translucent light brown tube around the reactor"""
        if self.electrode_ax is None:
            return
        try:
            outer_radius = inner_radius + thickness

            # Get alpha from slider
            refractory_alpha = self.refractory_alpha_slider.value() / 100.0

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
            # Outer surface
            self.electrode_ax.plot_surface(
                X_outer,
                Y_outer,
                Z_outer,
                color=refractory_color,
                alpha=refractory_alpha,
            )
            # Inner surface
            self.electrode_ax.plot_surface(
                X_inner,
                Y_inner,
                Z_outer,
                color=refractory_color,
                alpha=refractory_alpha * 0.8,
            )
            # Top annular surface
            self.electrode_ax.plot_surface(
                X_top_annular,
                Y_top_annular,
                Z_top_annular,
                color=refractory_color,
                alpha=refractory_alpha,
            )
            # Bottom annular surface
            self.electrode_ax.plot_surface(
                X_bottom_annular,
                Y_bottom_annular,
                Z_bottom_annular,
                color=refractory_color,
                alpha=refractory_alpha,
            )

            # Edge circles for definition
            x_outer_circle = outer_radius * np.cos(theta)
            y_outer_circle = outer_radius * np.sin(theta)

            # Top and bottom outer circles
            self.electrode_ax.plot(
                x_outer_circle,
                y_outer_circle,
                total_height,
                color="#8B4513",
                linewidth=1.5,
                alpha=0.8,
            )
            self.electrode_ax.plot(
                x_outer_circle,
                y_outer_circle,
                0,
                color="#8B4513",
                linewidth=1.5,
                alpha=0.8,
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error drawing refractory layer: %s", e)

    def _draw_3d_metal_shell(
        self, inner_radius: float, total_height: float, refractory_thickness: float
    ) -> None:
        """Draw the metal vessel shell as a 1/2" thick cylinder outside the refractory"""
        if self.electrode_ax is None:
            return
        try:
            # Metal shell is 1/2" thick outside the refractory
            shell_thickness = 0.5  # inches
            shell_inner_radius = inner_radius + refractory_thickness
            shell_outer_radius = shell_inner_radius + shell_thickness

            # Get alpha from slider
            shell_alpha = self.metal_shell_alpha_slider.value() / 100.0

            # Create metal shell representation
            theta = np.linspace(0, 2 * np.pi, 30)
            z_shell = np.linspace(0, total_height, 10)

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

            # Metal shell color (dark grey)
            if self.config.colors is None:
                shell_color = "#444444"  # fallback
            else:
                mc = self.config.colors["metal_shell"]
                shell_color = mc.name() if hasattr(mc, "name") else str(mc)

            # Draw all surfaces
            self.electrode_ax.plot_surface(
                X_outer, Y_outer, Z_outer, color=shell_color, alpha=shell_alpha
            )
            # Inner surface
            self.electrode_ax.plot_surface(
                X_inner,
                Y_inner,
                Z_outer,
                color=shell_color,
                alpha=shell_alpha * 0.9,
            )
            # Top annular surface
            self.electrode_ax.plot_surface(
                X_top_annular,
                Y_top_annular,
                Z_top_annular,
                color=shell_color,
                alpha=shell_alpha,
            )
            # Bottom annular surface
            self.electrode_ax.plot_surface(
                X_bottom_annular,
                Y_bottom_annular,
                Z_bottom_annular,
                color=shell_color,
                alpha=shell_alpha,
            )

            # Edge circles for definition
            x_outer_circle = shell_outer_radius * np.cos(theta)
            y_outer_circle = shell_outer_radius * np.sin(theta)

            # Top and bottom outer circles
            self.electrode_ax.plot(
                x_outer_circle,
                y_outer_circle,
                total_height,
                color="#505050",
                linewidth=2,
                alpha=0.9,
            )
            self.electrode_ax.plot(
                x_outer_circle,
                y_outer_circle,
                0,
                color="#505050",
                linewidth=2,
                alpha=0.9,
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error drawing metal vessel shell: %s", e)

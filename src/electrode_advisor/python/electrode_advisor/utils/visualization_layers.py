"""Layer and vessel drawing methods for electrode visualization.

Extracted from ElectrodeVisualization to reduce class size.
Contains methods for drawing the cylindrical vessel, metal layer,
glass layer, refractory layer, and metal shell.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import SHELL_THICKNESS


class ElectrodeLayersMixin:
    """Mixin providing layer/vessel drawing for ElectrodeVisualization."""

    def draw_3d_vessel(self, ax: Any, radius: float, total_height: float) -> None:
        """Draw the cylindrical vessel in 3D."""
        theta = np.linspace(0, 2 * np.pi, 50)
        z_wall = np.linspace(0, total_height, 20)

        for i in range(0, len(theta), 5):
            x_line = [radius * np.cos(theta[i])] * len(z_wall)
            y_line = [radius * np.sin(theta[i])] * len(z_wall)
            ax.plot(x_line, y_line, z_wall, "k-", alpha=0.3, linewidth=0.5)

        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        ax.plot(x_circle, y_circle, 0, "k-", alpha=0.5, linewidth=2)
        ax.plot(x_circle, y_circle, total_height, "k-", alpha=0.5, linewidth=2)

    def draw_3d_metal_layer(
        self, ax: Any, radius: float, height: float, metal_alpha: float = 0.6
    ) -> None:
        """Draw the metal layer as a fully shaded grey cylinder volume."""
        if height <= 0:
            return
        theta = np.linspace(0, 2 * np.pi, 30)
        z_metal = np.linspace(0, height, 8)
        metal_color = "#808080"

        # Surface meshes
        top_x, top_y, top_z = self._disk_mesh(theta, radius, height)
        bot_x, bot_y, bot_z = self._disk_mesh(theta, radius, 0.0)
        cyl_x, cyl_y, cyl_z = self._cylinder_mesh(theta, z_metal, radius)

        self._draw_three_surfaces(
            ax,
            top_x,
            top_y,
            top_z,
            bot_x,
            bot_y,
            bot_z,
            cyl_x,
            cyl_y,
            cyl_z,
            metal_color,
            metal_alpha,
        )

        # Edge lines
        self._draw_edge_circles(ax, theta, radius, [height, 0.0], "#606060", 2, 0.9)

    def draw_3d_glass_layer(
        self,
        ax: Any,
        radius: float,
        metal_height: float,
        glass_height: float,
        glass_alpha: float = 0.4,
    ) -> None:
        """Draw the full glass layer volume above the metal as translucent orange."""
        total_height = metal_height + glass_height
        theta = np.linspace(0, 2 * np.pi, 30)
        z_glass = np.linspace(metal_height, total_height, 10)
        glass_color = "#FF8C00"

        top_x, top_y, top_z = self._disk_mesh(theta, radius, total_height)
        bot_x, bot_y, bot_z = self._disk_mesh(theta, radius, metal_height)
        cyl_x, cyl_y, cyl_z = self._cylinder_mesh(theta, z_glass, radius)

        self._draw_three_surfaces(
            ax,
            top_x,
            top_y,
            top_z,
            bot_x,
            bot_y,
            bot_z,
            cyl_x,
            cyl_y,
            cyl_z,
            glass_color,
            glass_alpha,
        )

        self._draw_edge_circles(
            ax,
            theta,
            radius,
            [total_height],
            "#FF6500",
            2,
            0.9,
        )
        self._draw_edge_circles(
            ax,
            theta,
            radius,
            [metal_height],
            "#FF6500",
            1.5,
            0.7,
        )

    def draw_3d_refractory_layer(
        self,
        ax: Any,
        inner_radius: float,
        total_height: float,
        thickness: float,
        refractory_alpha: float = 0.3,
    ) -> None:
        """Draw the refractory layer as a translucent light brown tube."""
        try:
            outer_radius = inner_radius + thickness
            refractory_color = "#D2B48C"

            self._draw_annular_volume(
                ax,
                inner_radius,
                outer_radius,
                total_height,
                refractory_color,
                refractory_alpha,
            )

            # Outer edge circles
            self._draw_edge_circles(
                ax,
                np.linspace(0, 2 * np.pi, 30),
                outer_radius,
                [total_height, 0.0],
                "#8B4513",
                1.5,
                0.8,
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
        """Draw the metal vessel shell as a 1/2 inch thick cylinder outside the refractory."""
        try:
            shell_thickness = SHELL_THICKNESS
            shell_inner_radius = inner_radius + refractory_thickness
            shell_outer_radius = shell_inner_radius + shell_thickness
            shell_color = "#2F2F2F"

            self._draw_annular_volume(
                ax,
                shell_inner_radius,
                shell_outer_radius,
                total_height,
                shell_color,
                shell_alpha,
            )

            self._draw_edge_circles(
                ax,
                np.linspace(0, 2 * np.pi, 30),
                shell_outer_radius,
                [total_height, 0.0],
                "#1C1C1C",
                2,
                0.9,
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _disk_mesh(theta: Any, radius: float, z_value: float) -> tuple[Any, Any, Any]:
        """Create a filled disk mesh at the given z height."""
        r = np.linspace(0, radius, 15)
        R, T = np.meshgrid(r, theta)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        Z = np.ones_like(X) * z_value
        return X, Y, Z

    @staticmethod
    def _cylinder_mesh(
        theta: Any, z_values: Any, radius: float
    ) -> tuple[Any, Any, Any]:
        """Create a cylindrical surface mesh."""
        T_cyl, Z_cyl = np.meshgrid(theta, z_values)
        X_cyl = radius * np.cos(T_cyl)
        Y_cyl = radius * np.sin(T_cyl)
        return X_cyl, Y_cyl, Z_cyl

    @staticmethod
    def _draw_three_surfaces(
        ax: Any,
        top_x: Any,
        top_y: Any,
        top_z: Any,
        bot_x: Any,
        bot_y: Any,
        bot_z: Any,
        cyl_x: Any,
        cyl_y: Any,
        cyl_z: Any,
        color: str,
        alpha: float,
    ) -> None:
        """Draw top, bottom, and cylindrical surfaces."""
        if hasattr(ax, "plot_surface"):
            ax.plot_surface(top_x, top_y, top_z, color=color, alpha=alpha)
            ax.plot_surface(bot_x, bot_y, bot_z, color=color, alpha=alpha)
            ax.plot_surface(cyl_x, cyl_y, cyl_z, color=color, alpha=alpha * 0.9)
        else:
            ax.contour(top_x, top_y, top_z, colors=color, alpha=alpha)
            ax.contour(bot_x, bot_y, bot_z, colors=color, alpha=alpha)

    @staticmethod
    def _draw_edge_circles(
        ax: Any,
        theta: Any,
        radius: float,
        z_levels: list[float],
        color: str,
        linewidth: float,
        alpha: float,
    ) -> None:
        """Draw horizontal edge circles at given z levels."""
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        for z in z_levels:
            ax.plot(x, y, z, color=color, linewidth=linewidth, alpha=alpha)

    def _draw_annular_volume(
        self,
        ax: Any,
        inner_radius: float,
        outer_radius: float,
        total_height: float,
        color: str,
        alpha: float,
    ) -> None:
        """Draw an annular (ring) volume between inner and outer radii."""
        theta = np.linspace(0, 2 * np.pi, 30)
        z_vals = np.linspace(0, total_height, 8 if total_height > 0 else 2)

        # Outer and inner cylindrical surfaces
        T_cyl, Z_cyl = np.meshgrid(theta, z_vals)
        X_outer = outer_radius * np.cos(T_cyl)
        Y_outer = outer_radius * np.sin(T_cyl)
        X_inner = inner_radius * np.cos(T_cyl)
        Y_inner = inner_radius * np.sin(T_cyl)

        # Annular top and bottom
        r_ann = np.linspace(inner_radius, outer_radius, 8)
        R_ann, T_ann = np.meshgrid(r_ann, theta)
        X_top = R_ann * np.cos(T_ann)
        Y_top = R_ann * np.sin(T_ann)
        Z_top = np.ones_like(X_top) * total_height
        X_bot = R_ann * np.cos(T_ann)
        Y_bot = R_ann * np.sin(T_ann)
        Z_bot = np.zeros_like(X_bot)

        if hasattr(ax, "plot_surface"):
            ax.plot_surface(X_outer, Y_outer, Z_cyl, color=color, alpha=alpha)
            ax.plot_surface(X_inner, Y_inner, Z_cyl, color=color, alpha=alpha * 0.8)
            ax.plot_surface(X_top, Y_top, Z_top, color=color, alpha=alpha)
            ax.plot_surface(X_bot, Y_bot, Z_bot, color=color, alpha=alpha)
        else:
            for i in range(0, len(theta), 5):
                x_outer_line = [outer_radius * np.cos(theta[i])] * len(z_vals)
                y_outer_line = [outer_radius * np.sin(theta[i])] * len(z_vals)
                ax.plot(
                    x_outer_line,
                    y_outer_line,
                    z_vals,
                    color=color,
                    alpha=alpha * 0.5,
                    linewidth=1,
                )

"""ViaMetalMixin -- via-metal path drawing and electrode extrusion.

Delegates duplicated drawing logic to :mod:`~electrode_advisor.utils.shared_drawing`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...utils.shared_drawing import (
    draw_electrode_length_extrusion,
    draw_via_metal_path,
)

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
        """Draw the correct 3-segment via-metal path with vertical extrusions.

        Delegates to :func:`~shared_drawing.draw_via_metal_path`.
        """
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
        """Draw rectangular extrusion along electrode length within glass bath.

        Delegates to :func:`~shared_drawing.draw_electrode_length_extrusion`.
        """
        if self.electrode_ax is None:
            return
        draw_electrode_length_extrusion(
            ax=self.electrode_ax,
            electrode_pos=electrode_pos,
            metal_height=metal_height,
            electrode_radius=electrode_radius,
            bath_radius=bath_radius,
            direction=direction,
            color=color,
            alpha=alpha,
            horizontal_spreading_factor=self.config.horizontal_spreading_factor,
        )

    def _draw_electrode_sphere(
        self,
        x_center: float,
        y_center: float,
        z_center: float,
        radius: float,
        color: Any,
        alpha: float,
    ) -> None:
        """Draw a spherical tip at the electrode end."""
        if self.electrode_ax is None:
            return
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)

        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x_center
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y_center
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z_center

        self.electrode_ax.plot_surface(
            x_sphere, y_sphere, z_sphere, color=color, alpha=alpha, linewidth=0
        )

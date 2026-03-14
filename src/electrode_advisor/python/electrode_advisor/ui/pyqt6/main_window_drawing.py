"""DrawingMixin -- vessel component drawing delegating to ElectrodeVisualization.

Fixes #1407-#1411: replaces duplicated drawing code with delegation to
ElectrodeLayersMixin (via self.visualization) and ElectrodeVisualization.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DrawingMixin:
    """Mixin providing 3D drawing for vessel components.

    Delegates all layer/electrode drawing to ``self.visualization``
    (an :class:`ElectrodeVisualization` instance set up in __init__).
    """

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
    visualization: Any

    def _draw_3d_metal_layer(self, radius: float, height: float) -> None:
        """Draw the metal layer, delegating to ElectrodeLayersMixin."""
        if height <= 0 or self.electrode_ax is None:
            return
        metal_alpha = self.metal_alpha_slider.value() / 100.0
        self.visualization.draw_3d_metal_layer(
            self.electrode_ax, radius, height, metal_alpha
        )

    def _draw_3d_glass_layer(
        self, radius: float, metal_height: float, glass_height: float
    ) -> None:
        """Draw the glass layer, delegating to ElectrodeLayersMixin."""
        if self.electrode_ax is None:
            return
        glass_alpha = self.glass_alpha_slider.value() / 100.0
        self.visualization.draw_3d_glass_layer(
            self.electrode_ax, radius, metal_height, glass_height, glass_alpha
        )

    def _draw_3d_refractory_layer(
        self, inner_radius: float, total_height: float, thickness: float
    ) -> None:
        """Draw the refractory layer, delegating to ElectrodeLayersMixin."""
        if self.electrode_ax is None:
            return
        refractory_alpha = self.refractory_alpha_slider.value() / 100.0
        self.visualization.draw_3d_refractory_layer(
            self.electrode_ax,
            inner_radius,
            total_height,
            thickness,
            refractory_alpha,
        )

    def _draw_3d_metal_shell(
        self,
        inner_radius: float,
        total_height: float,
        refractory_thickness: float,
    ) -> None:
        """Draw the metal shell, delegating to ElectrodeLayersMixin."""
        if self.electrode_ax is None:
            return
        shell_alpha = self.metal_shell_alpha_slider.value() / 100.0
        self.visualization.draw_3d_metal_shell(
            self.electrode_ax,
            inner_radius,
            total_height,
            refractory_thickness,
            shell_alpha,
        )

    def _draw_3d_electrodes(
        self,
        depths: list[float],
        electrode_radius: float,
        bath_radius: float,
        metal_height: float,
        glass_height: float,
    ) -> None:
        """Draw electrodes, delegating to ElectrodeVisualization."""
        if self.electrode_ax is None:
            return
        electrode_alpha = self.electrode_alpha_slider.value() / 100.0
        extension_length = float(self.electrode_extension_slider.value())
        show_labels = self.show_electrode_labels_checkbox.isChecked()
        self.visualization.draw_3d_electrodes(
            self.electrode_ax,
            depths,
            electrode_radius,
            bath_radius,
            metal_height,
            glass_height,
            electrode_alpha,
            extension_length,
            show_labels,
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
        """Draw a spherical tip, delegating to ElectrodeVisualization."""
        if self.electrode_ax is None:
            return
        self.visualization.draw_electrode_sphere(
            self.electrode_ax,
            x_center,
            y_center,
            z_center,
            radius,
            color,
            alpha,
        )

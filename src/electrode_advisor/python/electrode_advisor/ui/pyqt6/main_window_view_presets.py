"""ViewPresetsMixin -- view presets and color scheme methods.

Handles predefined view angles, color scheme changes, and transparency
value retrieval for the 3D visualization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...configs.color_schemes import get_color_scheme
from ...configs.view_presets import (
    DEFAULT_Z_SCALE_FACTOR,
    DEFAULT_ZOOM_SCALE_FACTOR,
    get_view_preset,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ViewPresetsMixin:
    """Mixin providing view preset and color scheme methods.

    Expected to be mixed into a QWidget subclass that defines:
    - ``self.electrode_ax``, ``self.electrode_canvas``
    - Various slider and checkbox attributes
    """

    def _set_view_preset(self, preset: str) -> None:
        """Set predefined view angles using view presets from config."""
        if self.electrode_ax is None:
            return

        try:
            view_angle = get_view_preset(preset)
            self.electrode_ax.view_init(elev=view_angle.elev, azim=view_angle.azim)

            # For default view, also reset pan/zoom
            if preset == "default":
                bath_diameter = self.bath_diameter_input.value()
                extension_length = float(self.electrode_extension_slider.value())
                glass_height = self.glass_layer_height_input.value()
                metal_height = self.metal_layer_height_input.value()
                max_range = max(
                    bath_diameter / 2 + extension_length, glass_height + metal_height
                )
                zoom_factor = self.zoom_slider.value() / 100.0
                scaled_range = max_range / zoom_factor * DEFAULT_ZOOM_SCALE_FACTOR
                self.electrode_ax.set_xlim(-scaled_range, scaled_range)
                self.electrode_ax.set_ylim(-scaled_range, scaled_range)
                if hasattr(self.electrode_ax, "set_zlim"):
                    self.electrode_ax.set_zlim(
                        0,
                        (glass_height + metal_height)
                        / zoom_factor
                        * DEFAULT_Z_SCALE_FACTOR,
                    )

            if self.electrode_canvas is not None:
                self.electrode_canvas.draw()
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error setting view preset: %s", e)

    def _on_color_scheme_changed(self, scheme: str) -> None:
        """Handle color scheme changes"""
        try:
            self.current_color_scheme = scheme
            # Trigger visualization update with new color scheme
            self._on_input_changed()
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error changing color scheme: %s", e)

    def _get_color_scheme_colors(self) -> list[str]:
        """Get colors based on current color scheme."""
        return get_color_scheme(self.current_color_scheme)

    def _get_transparency_values(self) -> dict[str, float]:
        """Get current transparency values from sliders"""
        return {
            "electrodes": self.electrode_alpha_slider.value() / 100.0,
            "glass": self.glass_alpha_slider.value() / 100.0,
            "metal": self.metal_alpha_slider.value() / 100.0,
            "paths": self.path_alpha_slider.value() / 100.0,
            "refractory": self.refractory_alpha_slider.value() / 100.0,
            "metal_shell": self.metal_shell_alpha_slider.value() / 100.0,
        }

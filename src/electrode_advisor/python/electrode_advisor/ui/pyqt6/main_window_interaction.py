"""InteractionMixin -- mouse and interaction mode handlers.

Handles scroll, mouse press/release/motion events, interaction mode
changes, and 3D view reset functionality.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InteractionMixin:
    """Mixin providing mouse interaction and view control methods.

    Expected to be mixed into a QWidget subclass that defines:
    - ``self.electrode_ax``, ``self.electrode_canvas``
    - ``self._update_3d_visualization``
    """

    def _on_scroll(self, event: Any) -> None:
        """Handle mouse scroll for zoom"""
        try:
            if event.inaxes == self.electrode_ax:
                # Zoom factor
                zoom_factor = 1.1 if event.button == "down" else 0.9

                # Update zoom slider
                current_zoom = self.zoom_slider.value()
                new_zoom = int(current_zoom * zoom_factor)
                new_zoom = max(50, min(200, new_zoom))  # Clamp to range
                self.zoom_slider.setValue(new_zoom)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error in scroll handler: %s", e)

    def _on_mouse_press(self, event: Any) -> None:
        """Handle mouse press for rotation and pan"""
        try:
            if event.inaxes == self.electrode_ax:
                self._mouse_pressed = True
                # Use pixel coordinates for consistent behavior
                self._last_mouse_pos = (
                    (event.x, event.y)
                    if event.x is not None and event.y is not None
                    else None
                )
                # Get 3D view angles for rotation mode
                if hasattr(self.electrode_ax, "elev") and hasattr(
                    self.electrode_ax, "azim"
                ):
                    self._last_elev = self.electrode_ax.elev
                    self._last_azim = self.electrode_ax.azim
                else:
                    # Fallback: use default angles if not available
                    self._last_elev = 20
                    self._last_azim = 45
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error in mouse press handler: %s", e)

    def _on_mouse_release(self, event: Any) -> None:
        """Handle mouse release"""
        try:
            self._mouse_pressed = False
            logger.info("Mouse released")
        except (ValueError, TypeError, ArithmeticError) as e:
            logger.exception("Error in mouse release handler: %s", e)

    def _on_mouse_motion(self, event: Any) -> None:
        """Handle mouse motion for rotation or panning based on interaction mode"""
        try:
            if (
                self._mouse_pressed
                and event.inaxes == self.electrode_ax
                and self._last_mouse_pos
                and event.x is not None
                and event.y is not None
            ):
                dx = event.x - self._last_mouse_pos[0]
                dy = event.y - self._last_mouse_pos[1]
                logger.info(
                    "Mouse motion: dx=%s, dy=%s, mode=%s", dx, dy, self.interaction_mode
                )
                if self.interaction_mode == "rotation":
                    # Only rotate if in rotation mode
                    if hasattr(self.electrode_ax, "view_init") and hasattr(
                        self, "_last_elev"
                    ):
                        rotation_sensitivity = 0.5
                        new_elev = self._last_elev + dy * rotation_sensitivity
                        new_azim = self._last_azim + dx * rotation_sensitivity
                        new_elev = max(-90, min(90, new_elev))
                        self.electrode_ax.view_init(elev=new_elev, azim=new_azim)
                        if self.electrode_canvas:
                            self.electrode_canvas.draw_idle()
                        self._last_elev = new_elev
                        self._last_azim = new_azim
                elif self.interaction_mode == "pan":
                    # Only pan if in pan mode
                    if hasattr(self.electrode_ax, "get_xlim"):
                        x_range = self.electrode_ax.get_xlim()
                        y_range = self.electrode_ax.get_ylim()
                        pan_factor = 0.01
                        x_shift = dx * pan_factor * (x_range[1] - x_range[0])
                        y_shift = dy * pan_factor * (y_range[1] - y_range[0])
                        self.electrode_ax.set_xlim(
                            x_range[0] - x_shift, x_range[1] - x_shift
                        )
                        self.electrode_ax.set_ylim(
                            y_range[0] + y_shift, y_range[1] + y_shift
                        )
                    if self.electrode_canvas is not None:
                        self.electrode_canvas.draw_idle()
                self._last_mouse_pos = (event.x, event.y)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error in mouse motion handler: %s", e)

    def _on_interaction_mode_changed(self) -> None:
        """Handle interaction mode change"""
        try:
            if self.rotation_mode_radio.isChecked():
                self.interaction_mode = "rotation"
            else:
                self.interaction_mode = "pan"
            logger.info("3D interaction mode changed to: %s", self.interaction_mode)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error changing interaction mode: %s", e)

    def _reset_3d_view(self) -> None:
        """Reset 3D view to default angle and zoom"""
        try:
            if self.electrode_canvas is None or self.electrode_ax is None:
                return

            if hasattr(self.electrode_ax, "view_init"):
                self.electrode_ax.view_init(elev=20, azim=45)

            # Reset zoom slider to 100%
            if hasattr(self, "zoom_slider"):
                self.zoom_slider.setValue(100)

            self.electrode_canvas.draw_idle()
            logger.info("3D view reset to default")
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Error resetting 3D view: %s", e)

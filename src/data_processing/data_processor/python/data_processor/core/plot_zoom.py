"""Mouse Wheel Zoom Support for Matplotlib Plots.

Provides interactive zoom functionality using the mouse wheel
for all plot types including 2D and 3D plots.

Features:
- Zoom centered on mouse cursor position
- Configurable zoom factors
- Support for 2D and 3D axes
- Smooth zoom animation option
- Keyboard modifier support (Ctrl for horizontal, Shift for vertical)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ZoomConfig:
    """Configuration for zoom behavior."""

    # Zoom factors
    zoom_in_factor: float = 1.2
    zoom_out_factor: float = 0.8

    # Behavior options
    center_on_cursor: bool = True
    smooth_animation: bool = False
    animation_duration_ms: int = 100

    # Axis constraints
    maintain_aspect_ratio: bool = False
    allow_horizontal_zoom: bool = True
    allow_vertical_zoom: bool = True

    # Limits
    min_zoom_range: float = 1e-10
    max_zoom_range: float = 1e10


class MouseWheelZoom:
    """Handles mouse wheel zoom for matplotlib figures.

    Supports both 2D and 3D plots with configurable behavior.
    """

    def __init__(self, config: ZoomConfig | None = None) -> None:
        """Initialize the zoom handler.

        Args:
            config: Zoom configuration options
        """
        self.config = config or ZoomConfig()
        self._connected_figures: dict[int, list[int]] = {}
        self._original_limits: dict[int, dict[str, tuple[float, float]]] = {}
        self._zoom_callbacks: list[Callable[[Any], None]] = []

    def connect(self, fig: Any) -> None:
        """Connect zoom handler to a matplotlib figure.

        Args:
            fig: Matplotlib figure object
        """
        fig_id = id(fig)

        if fig_id in self._connected_figures:
            # Already connected
            return

        # Connect scroll event
        cid = fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._connected_figures[fig_id] = [cid]

        # Store original limits for reset
        self._store_original_limits(fig)

        logger.debug(f"Connected zoom handler to figure {fig_id}")

    def disconnect(self, fig: Any) -> None:
        """Disconnect zoom handler from a matplotlib figure.

        Args:
            fig: Matplotlib figure object
        """
        fig_id = id(fig)

        if fig_id in self._connected_figures:
            for cid in self._connected_figures[fig_id]:
                fig.canvas.mpl_disconnect(cid)
            del self._connected_figures[fig_id]

        if fig_id in self._original_limits:
            del self._original_limits[fig_id]

        logger.debug(f"Disconnected zoom handler from figure {fig_id}")

    def reset_zoom(self, fig: Any) -> None:
        """Reset zoom to original limits.

        Args:
            fig: Matplotlib figure object
        """
        fig_id = id(fig)

        if fig_id not in self._original_limits:
            return

        for ax in fig.axes:
            ax_id = id(ax)
            if ax_id in self._original_limits[fig_id]:
                limits = self._original_limits[fig_id][ax_id]
                ax.set_xlim(limits["xlim"])
                ax.set_ylim(limits["ylim"])

                # Handle 3D axes
                if hasattr(ax, "set_zlim") and "zlim" in limits:
                    ax.set_zlim(limits["zlim"])

        fig.canvas.draw_idle()

    def add_zoom_callback(self, callback: Callable[[Any], None]) -> None:
        """Add a callback to be called after zoom.

        Args:
            callback: Function that receives the zoom event
        """
        self._zoom_callbacks.append(callback)

    def remove_zoom_callback(self, callback: Callable[[Any], None]) -> None:
        """Remove a zoom callback."""
        if callback in self._zoom_callbacks:
            self._zoom_callbacks.remove(callback)

    def _on_scroll(self, event: Any) -> None:
        """Handle scroll event for zooming."""
        if event.inaxes is None:
            return

        ax = event.inaxes

        # Determine zoom direction
        if event.button == "up":
            zoom_factor = self.config.zoom_in_factor
        elif event.button == "down":
            zoom_factor = self.config.zoom_out_factor
        else:
            return

        # Check for modifier keys
        horizontal_only = event.key == "control"
        vertical_only = event.key == "shift"

        # Handle 3D axes
        if hasattr(ax, "set_zlim"):
            self._zoom_3d(ax, event, zoom_factor)
        else:
            self._zoom_2d(ax, event, zoom_factor, horizontal_only, vertical_only)

        # Redraw
        ax.figure.canvas.draw_idle()

        # Notify callbacks
        for callback in self._zoom_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Zoom callback error: {e}")

    def _zoom_2d(
        self,
        ax: Any,
        event: Any,
        zoom_factor: float,
        horizontal_only: bool = False,
        vertical_only: bool = False,
    ) -> None:
        """Perform 2D zoom operation."""
        # Get current limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Get cursor position in data coordinates
        if self.config.center_on_cursor:
            x_center = event.xdata
            y_center = event.ydata
        else:
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

        # Handle None values (cursor outside data area)
        if x_center is None:
            x_center = (x_min + x_max) / 2
        if y_center is None:
            y_center = (y_min + y_max) / 2

        # Calculate new ranges
        x_range = (x_max - x_min) / zoom_factor
        y_range = (y_max - y_min) / zoom_factor

        # Clamp ranges
        x_range = np.clip(x_range, self.config.min_zoom_range, self.config.max_zoom_range)
        y_range = np.clip(y_range, self.config.min_zoom_range, self.config.max_zoom_range)

        # Calculate new limits centered on cursor
        if self.config.allow_horizontal_zoom and not vertical_only:
            # Maintain relative position of cursor
            x_ratio = (x_center - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0.5
            new_x_min = x_center - x_ratio * x_range
            new_x_max = x_center + (1 - x_ratio) * x_range
            ax.set_xlim(new_x_min, new_x_max)

        if self.config.allow_vertical_zoom and not horizontal_only:
            y_ratio = (y_center - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.5
            new_y_min = y_center - y_ratio * y_range
            new_y_max = y_center + (1 - y_ratio) * y_range
            ax.set_ylim(new_y_min, new_y_max)

    def _zoom_3d(self, ax: Any, event: Any, zoom_factor: float) -> None:
        """Perform 3D zoom operation."""
        # For 3D axes, we scale all three dimensions uniformly
        # centered on the current view center

        # Get current limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        z_min, z_max = ax.get_zlim()

        # Calculate centers
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Calculate new ranges
        x_range = (x_max - x_min) / zoom_factor
        y_range = (y_max - y_min) / zoom_factor
        z_range = (z_max - z_min) / zoom_factor

        # Clamp ranges
        x_range = np.clip(x_range, self.config.min_zoom_range, self.config.max_zoom_range)
        y_range = np.clip(y_range, self.config.min_zoom_range, self.config.max_zoom_range)
        z_range = np.clip(z_range, self.config.min_zoom_range, self.config.max_zoom_range)

        # Set new limits
        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
        ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
        ax.set_zlim(z_center - z_range / 2, z_center + z_range / 2)

    def _store_original_limits(self, fig: Any) -> None:
        """Store original axis limits for reset."""
        fig_id = id(fig)
        self._original_limits[fig_id] = {}

        for ax in fig.axes:
            ax_id = id(ax)
            limits = {
                "xlim": ax.get_xlim(),
                "ylim": ax.get_ylim(),
            }

            if hasattr(ax, "get_zlim"):
                limits["zlim"] = ax.get_zlim()

            self._original_limits[fig_id][ax_id] = limits


class InteractivePlotManager:
    """Manages interactive features for matplotlib plots.

    Combines mouse wheel zoom with other interactive features
    like pan and selection.
    """

    def __init__(self) -> None:
        """Initialize the plot manager."""
        self._zoom_handler = MouseWheelZoom()
        self._figures: dict[int, Any] = {}

    def setup_figure(
        self,
        fig: Any,
        enable_zoom: bool = True,
        enable_pan: bool = True,
        zoom_config: ZoomConfig | None = None,
    ) -> None:
        """Set up interactive features for a figure.

        Args:
            fig: Matplotlib figure object
            enable_zoom: Enable mouse wheel zoom
            enable_pan: Enable panning (already supported by toolbar)
            zoom_config: Configuration for zoom behavior
        """
        fig_id = id(fig)
        self._figures[fig_id] = fig

        if enable_zoom:
            if zoom_config:
                self._zoom_handler.config = zoom_config
            self._zoom_handler.connect(fig)

        # Enable tight layout updates on resize
        def on_resize(event: Any) -> None:
            try:
                fig.tight_layout()
            except Exception:
                pass

        fig.canvas.mpl_connect("resize_event", on_resize)

    def cleanup_figure(self, fig: Any) -> None:
        """Clean up interactive features from a figure.

        Args:
            fig: Matplotlib figure object
        """
        fig_id = id(fig)

        self._zoom_handler.disconnect(fig)

        if fig_id in self._figures:
            del self._figures[fig_id]

    def reset_all_zoom(self) -> None:
        """Reset zoom on all managed figures."""
        for fig in self._figures.values():
            self._zoom_handler.reset_zoom(fig)

    @property
    def zoom_handler(self) -> MouseWheelZoom:
        """Get the zoom handler."""
        return self._zoom_handler


def enable_wheel_zoom(
    fig: Any,
    config: ZoomConfig | None = None,
) -> MouseWheelZoom:
    """Convenience function to enable wheel zoom on a figure.

    Args:
        fig: Matplotlib figure
        config: Optional zoom configuration

    Returns:
        MouseWheelZoom handler (for customization)

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> zoom = enable_wheel_zoom(fig)
        >>> plt.show()
    """
    zoom = MouseWheelZoom(config)
    zoom.connect(fig)
    return zoom


def enable_wheel_zoom_all_figures(config: ZoomConfig | None = None) -> None:
    """Enable wheel zoom on all existing matplotlib figures.

    Args:
        config: Optional zoom configuration
    """
    import matplotlib.pyplot as plt

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        enable_wheel_zoom(fig, config)


__all__ = [
    "ZoomConfig",
    "MouseWheelZoom",
    "InteractivePlotManager",
    "enable_wheel_zoom",
    "enable_wheel_zoom_all_figures",
]

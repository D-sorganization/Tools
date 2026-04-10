"""
Overlay widgets: HelpOverlay, TransferPlanner, TooltipManager.
"""

from typing import Any

from ._base import PanelStyle


class HelpOverlay:
    """
    Overlay showing keyboard controls and help information.
    """

    def __init__(
        self, position: tuple[int, int] | None = None, style: PanelStyle | None = None
    ) -> None:
        """
        Initialize the help overlay.

        Args:
            position: Position or None for auto-placement
            style: Visual styling
        """
        self.position = position
        self.style = style or PanelStyle()
        self.visible = False
        self._controls: list[tuple[str, str]] = []

    def set_controls(self, controls: list[tuple[str, str]]) -> None:
        """
        Set the control bindings to display.

        Args:
            controls: List of (key, description) tuples
        """
        self._controls = controls

    def toggle(self) -> None:
        """Toggle visibility."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        return {
            "position": self.position,
            "controls": self._controls,
            "style": self.style,
            "visible": self.visible,
        }


class TransferPlanner:
    """
    UI for planning interplanetary transfers.

    Allows selection of origin, destination, and departure date.
    """

    def __init__(self, style: PanelStyle | None = None) -> None:
        """Initialize the transfer planner."""
        self.style = style or PanelStyle()
        self.visible = False
        self.origin: str | None = None
        self.destination: str | None = None
        self.departure_date: float | None = None
        self._transfer_info: dict[str, Any] = {}

    def set_origin(self, name: str) -> None:
        """Set origin body."""
        self.origin = name

    def set_destination(self, name: str) -> None:
        """Set destination body."""
        self.destination = name

    def set_transfer_info(self, info: dict[str, Any]) -> None:
        """Set transfer calculation results."""
        self._transfer_info = info

    def toggle(self) -> None:
        """Toggle visibility."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        return {
            "origin": self.origin,
            "destination": self.destination,
            "departure_date": self.departure_date,
            "transfer_info": self._transfer_info,
            "style": self.style,
            "visible": self.visible,
        }


class TooltipManager:
    """
    Manages tooltips for celestial bodies on hover.
    """

    def __init__(self, style: PanelStyle | None = None) -> None:
        """Initialize tooltip manager."""
        self.style = style or PanelStyle()
        self._active_tooltip: dict[str, Any] | None = None
        self._hover_time: float = 0.0
        self._show_delay: float = 0.5  # Seconds before showing

    def set_hover(
        self, body_name: str, position: tuple[int, int], info: dict[str, Any]
    ) -> None:
        """
        Set the currently hovered body.

        Args:
            body_name: Name of the body
            position: Screen position
            info: Information to display
        """
        self._active_tooltip = {"name": body_name, "position": position, "info": info}

    def clear_hover(self) -> None:
        """Clear the current hover."""
        self._active_tooltip = None
        self._hover_time = 0.0

    def update(self, delta_time: float) -> None:
        """Update tooltip timing."""
        if self._active_tooltip:
            self._hover_time += delta_time

    def should_show(self) -> bool:
        """Check if tooltip should be displayed."""
        return self._active_tooltip is not None and self._hover_time >= self._show_delay

    def get_render_data(self) -> dict[str, Any] | None:
        """Get tooltip data for rendering."""
        if self.should_show():
            return self._active_tooltip
        return None

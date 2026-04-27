"""HistoricalEventsPanel widget."""

from datetime import datetime
from typing import Any

from ._base import PanelStyle


class HistoricalEventsPanel:
    """
    Displays historical events related to the current date.

    Shows significant astronomical events, space missions, discoveries, etc.
    that occurred on or near the current simulation date.
    """

    def __init__(
        self,
        position: tuple[int, int] = (20, 450),
        width: int = 400,
        style: PanelStyle | None = None,
    ):
        """
        Initialize the historical events panel.

        Args:
            position: Top-left position (x, y)
            width: Panel width in pixels
            style: Visual styling
        """
        assert position is not None, "position must be provided"
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = False
        self._current_date: datetime | None = None
        self._events: list[dict[str, str]] = []

    def set_date(self, dt: datetime) -> None:
        """
        Set the current date and find relevant events.

        Args:
            dt: Current simulation date
        """
        assert dt is not None, "dt must be provided"
        self._current_date = dt
        self._events = self._find_events_for_date(dt)

    def _find_events_for_date(self, dt: datetime) -> list[dict[str, str]]:
        """
        Find historical events for the given date.

        Args:
            dt: Date to search

        Returns:
            List of event dictionaries with 'date', 'title', 'description'
        """
        try:
            from ...data.historical_events import get_events_for_date

            return get_events_for_date(dt, window_days=7)
        except ImportError:
            # Fallback to empty list if module not available
            return []

    def toggle(self) -> None:
        """Toggle visibility."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        return {
            "position": self.position,
            "width": self.width,
            "date": self._current_date,
            "events": self._events,
            "style": self.style,
            "visible": self.visible,
        }

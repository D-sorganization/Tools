"""
Date/time widgets: DateTimePicker, TimeNavigationPanel.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from ._base import PanelStyle


class DateTimePicker:
    """
    Interactive date/time picker for navigating to specific points in time.

    Allows users to manually input or select dates to see planetary positions
    at any point in history (within supported range).
    """

    def __init__(
        self,
        position: tuple[int, int] = (20, 100),
        style: PanelStyle | None = None,
        on_date_change: Callable[[datetime], None] | None = None,
    ):
        """
        Initialize the date/time picker.

        Args:
            position: Top-left position (x, y)
            style: Visual styling
            on_date_change: Callback when date is changed
        """
        assert position is not None, "position must be provided"
        self.position = position
        self.style = style or PanelStyle()
        self.visible = False
        self._current_date: datetime | None = None
        self._on_date_change = on_date_change
        self._editing_field: str | None = None  # 'year', 'month', 'day', 'hour'
        self._input_buffer: str = ""

    def toggle(self) -> None:
        """Toggle visibility of the picker."""
        self.visible = not self.visible

    def set_date(self, dt: datetime) -> None:
        """
        Set the current date.

        Args:
            dt: The datetime to display
        """
        assert dt is not None, "dt must be provided"
        self._current_date = dt

    def get_date(self) -> datetime | None:
        """Get the current selected date."""
        return self._current_date

    def handle_input(self, char: str) -> bool:
        """
        Handle text input for date editing.

        Args:
            char: Character input

        Returns:
            True if input was handled
        """
        assert char is not None, "char must be provided"
        if not self._editing_field:
            return False

        if char.isdigit():
            self._input_buffer += char
            return True
        elif char == "\r" or char == "\n":  # Enter
            self._apply_input()
            return True
        elif char == "\b":  # Backspace
            if self._input_buffer:
                self._input_buffer = self._input_buffer[:-1]
            return True

        return False

    def _apply_input(self) -> None:
        """Apply the input buffer to the current field."""
        if not self._input_buffer or not self._current_date:
            self._editing_field = None
            self._input_buffer = ""
            return

        try:
            value = int(self._input_buffer)

            if self._editing_field == "year":
                if 1800 <= value <= 2200:
                    self._current_date = self._current_date.replace(year=value)
            elif self._editing_field == "month":
                if 1 <= value <= 12:
                    self._current_date = self._current_date.replace(month=value)
            elif self._editing_field == "day":
                # Validate that the day exists in the current month/year
                from calendar import monthrange

                max_days = monthrange(
                    self._current_date.year, self._current_date.month
                )[1]
                if 1 <= value <= max_days:
                    from contextlib import suppress

                    with suppress(ValueError):
                        # Invalid date (e.g., Feb 30), ignore
                        self._current_date = self._current_date.replace(day=value)
            elif self._editing_field == "hour" and 0 <= value <= 23:
                self._current_date = self._current_date.replace(hour=value)

            if self._on_date_change:
                self._on_date_change(self._current_date)

        except ValueError:
            pass  # Invalid input, ignore

        self._editing_field = None
        self._input_buffer = ""

    def start_editing(self, field: str) -> None:
        """
        Start editing a specific date field.

        Args:
            field: Field name ('year', 'month', 'day', 'hour')
        """
        assert field is not None, "field must be provided"
        self._editing_field = field
        self._input_buffer = ""

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        return {
            "position": self.position,
            "date": self._current_date,
            "editing_field": self._editing_field,
            "input_buffer": self._input_buffer,
            "style": self.style,
            "visible": self.visible,
        }


class TimeNavigationPanel:
    """
    Panel with buttons for navigating through time.

    Provides quick navigation controls:
    - Jump forward/backward by day, week, month, year
    - Jump to specific dates (today, J2000, etc.)
    - Quick time warp presets
    """

    def __init__(
        self, position: tuple[int, int] = (20, 250), style: PanelStyle | None = None
    ):
        """
        Initialize the time navigation panel.

        Args:
            position: Top-left position (x, y)
            style: Visual styling
        """
        assert position is not None, "position must be provided"
        self.position = position
        self.style = style or PanelStyle()
        self.visible = True

        # Define navigation buttons
        self.buttons = [
            {"label": "◀◀ Year", "action": "prev_year", "tooltip": "Go back 1 year"},
            {"label": "◀ Month", "action": "prev_month", "tooltip": "Go back 1 month"},
            {"label": "◀ Week", "action": "prev_week", "tooltip": "Go back 1 week"},
            {"label": "◀ Day", "action": "prev_day", "tooltip": "Go back 1 day"},
            {
                "label": "Today",
                "action": "goto_today",
                "tooltip": "Jump to current date",
            },
            {
                "label": "J2000",
                "action": "goto_j2000",
                "tooltip": "Jump to J2000 epoch",
            },
            {"label": "Day ▶", "action": "next_day", "tooltip": "Go forward 1 day"},
            {"label": "Week ▶", "action": "next_week", "tooltip": "Go forward 1 week"},
            {
                "label": "Month ▶",
                "action": "next_month",
                "tooltip": "Go forward 1 month",
            },
            {"label": "Year ▶▶", "action": "next_year", "tooltip": "Go forward 1 year"},
        ]

    def toggle(self) -> None:
        """Toggle visibility."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        return {
            "position": self.position,
            "buttons": self.buttons,
            "style": self.style,
            "visible": self.visible,
        }

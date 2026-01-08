"""
UI Widgets
==========

Reusable UI components for the simulation overlay.

This module provides interactive widgets for controlling the simulation,
displaying information, and enhancing the educational experience.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class PanelStyle:
    """Styling for UI panels."""

    background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.7)
    text_color: tuple[int, int, int] = (220, 220, 220)
    title_color: tuple[int, int, int] = (255, 255, 100)
    border_color: tuple[float, float, float, float] = (0.3, 0.3, 0.4, 0.5)
    padding: int = 15
    line_height: int = 24
    font_size: int = 26  # Increased from 12
    title_font_size: int = 32  # Increased from 14


class InfoPanel:
    """
    Displays information about a selected object.

    Shows details like orbital parameters, physical properties,
    current position, etc.
    """

    def __init__(
        self,
        position: tuple[int, int] = (20, 20),
        width: int = 300,
        style: PanelStyle = None,
    ):
        """
        Initialize the info panel.

        Args:
            position: Top-left position (x, y)
            width: Panel width in pixels
            style: Visual styling
        """
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = True
        self._data: dict[str, Any] = {}
        self._title: str = ""

    def set_data(self, title: str, data: dict[str, Any]):
        """
        Set the data to display.

        Args:
            title: Panel title
            data: Dictionary of label -> value pairs
        """
        self._title = title
        self._data = data

    def clear(self):
        """Clear the panel data."""
        self._title = ""
        self._data = {}

    def toggle(self):
        """Toggle visibility."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """
        Get data formatted for rendering.

        Returns:
            Dictionary with render parameters
        """
        return {
            "position": self.position,
            "width": self.width,
            "title": self._title,
            "data": self._data,
            "style": self.style,
            "visible": self.visible,
        }


class StatusBar:
    """
    Status bar showing simulation state at bottom of screen.

    Displays time, speed, selected object, FPS, etc.
    """

    def __init__(self, style: PanelStyle = None):
        """Initialize the status bar."""
        self.style = style or PanelStyle()
        self.visible = True
        self._components: list[str] = []

    def set_time(self, time_str: str):
        """Set the time display."""
        self._time = time_str

    def set_speed(self, speed_str: str):
        """Set the time warp display."""
        self._speed = speed_str

    def set_selected(self, name: str):
        """Set the selected object name."""
        self._selected = name

    def set_fps(self, fps: float):
        """Set the FPS display."""
        self._fps = fps

    def set_paused(self, paused: bool):
        """Set paused state."""
        self._paused = paused

    def get_text(self) -> str:
        """Get formatted status bar text."""
        parts = []

        if hasattr(self, "_time"):
            parts.append(self._time)

        if hasattr(self, "_paused") and self._paused:
            parts.append("[PAUSED]")
        elif hasattr(self, "_speed"):
            parts.append(f"[{self._speed}]")

        if hasattr(self, "_selected") and self._selected:
            parts.append(f"Selected: {self._selected}")

        if hasattr(self, "_fps"):
            parts.append(f"FPS: {self._fps:.0f}")

        return "  |  ".join(parts)


class HelpOverlay:
    """
    Overlay showing keyboard controls and help information.
    """

    def __init__(self, position: tuple[int, int] = None, style: PanelStyle = None):
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

    def set_controls(self, controls: list[tuple[str, str]]):
        """
        Set the control bindings to display.

        Args:
            controls: List of (key, description) tuples
        """
        self._controls = controls

    def toggle(self):
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

    def __init__(self, style: PanelStyle = None):
        """Initialize the transfer planner."""
        self.style = style or PanelStyle()
        self.visible = False
        self.origin: str | None = None
        self.destination: str | None = None
        self.departure_date: float | None = None
        self._transfer_info: dict[str, Any] = {}

    def set_origin(self, name: str):
        """Set origin body."""
        self.origin = name

    def set_destination(self, name: str):
        """Set destination body."""
        self.destination = name

    def set_transfer_info(self, info: dict[str, Any]):
        """Set transfer calculation results."""
        self._transfer_info = info

    def toggle(self):
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

    def __init__(self, style: PanelStyle = None):
        """Initialize tooltip manager."""
        self.style = style or PanelStyle()
        self._active_tooltip: dict[str, Any] | None = None
        self._hover_time: float = 0.0
        self._show_delay: float = 0.5  # Seconds before showing

    def set_hover(
        self, body_name: str, position: tuple[int, int], info: dict[str, Any]
    ):
        """
        Set the currently hovered body.

        Args:
            body_name: Name of the body
            position: Screen position
            info: Information to display
        """
        self._active_tooltip = {"name": body_name, "position": position, "info": info}

    def clear_hover(self):
        """Clear the current hover."""
        self._active_tooltip = None
        self._hover_time = 0.0

    def update(self, delta_time: float):
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


class DateTimePicker:
    """
    Interactive date/time picker for navigating to specific points in time.

    Allows users to manually input or select dates to see planetary positions
    at any point in history (within supported range).
    """

    def __init__(
        self,
        position: tuple[int, int] = (20, 100),
        style: PanelStyle = None,
        on_date_change: Callable[[datetime], None] | None = None,
    ):
        """
        Initialize the date/time picker.

        Args:
            position: Top-left position (x, y)
            style: Visual styling
            on_date_change: Callback when date is changed
        """
        self.position = position
        self.style = style or PanelStyle()
        self.visible = False
        self._current_date: datetime | None = None
        self._on_date_change = on_date_change
        self._editing_field: str | None = None  # 'year', 'month', 'day', 'hour'
        self._input_buffer: str = ""

    def toggle(self):
        """Toggle visibility of the picker."""
        self.visible = not self.visible

    def set_date(self, dt: datetime):
        """
        Set the current date.

        Args:
            dt: The datetime to display
        """
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

    def _apply_input(self):
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

    def start_editing(self, field: str):
        """
        Start editing a specific date field.

        Args:
            field: Field name ('year', 'month', 'day', 'hour')
        """
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

    def __init__(self, position: tuple[int, int] = (20, 250), style: PanelStyle = None):
        """
        Initialize the time navigation panel.

        Args:
            position: Top-left position (x, y)
            style: Visual styling
        """
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

    def toggle(self):
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


class EducationalInfoPanel:
    """
    Enhanced info panel showing educational information about celestial bodies.

    Displays:
    - Physical properties
    - Orbital characteristics
    - Fun facts and educational content
    - Historical significance
    """

    def __init__(
        self,
        position: tuple[int, int] = (20, 20),
        width: int = 350,
        style: PanelStyle = None,
    ):
        """
        Initialize the educational info panel.

        Args:
            position: Top-left position (x, y)
            width: Panel width in pixels
            style: Visual styling
        """
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = True
        self._body_name: str | None = None
        self._properties: dict[str, Any] = {}
        self._fun_facts: list[str] = []
        self._current_fact_index: int = 0

    def set_body(
        self, name: str, properties: dict[str, Any], fun_facts: list[str] = None
    ):
        """
        Set the celestial body to display information about.

        Args:
            name: Body name
            properties: Dictionary of properties to display
            fun_facts: Optional list of educational fun facts
        """
        self._body_name = name
        self._properties = properties
        self._fun_facts = fun_facts or []
        self._current_fact_index = 0

    def cycle_fact(self):
        """Cycle to the next fun fact."""
        if self._fun_facts:
            self._current_fact_index = (self._current_fact_index + 1) % len(
                self._fun_facts
            )

    def get_current_fact(self) -> str | None:
        """Get the currently displayed fun fact."""
        if self._fun_facts:
            return self._fun_facts[self._current_fact_index]
        return None

    def toggle(self):
        """Toggle visibility."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        return {
            "position": self.position,
            "width": self.width,
            "body_name": self._body_name,
            "properties": self._properties,
            "current_fact": self.get_current_fact(),
            "fact_count": len(self._fun_facts),
            "fact_index": self._current_fact_index,
            "style": self.style,
            "visible": self.visible,
        }


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
        style: PanelStyle = None,
    ):
        """
        Initialize the historical events panel.

        Args:
            position: Top-left position (x, y)
            width: Panel width in pixels
            style: Visual styling
        """
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = False
        self._current_date: datetime | None = None
        self._events: list[dict[str, str]] = []

    def set_date(self, dt: datetime):
        """
        Set the current date and find relevant events.

        Args:
            dt: Current simulation date
        """
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
            from ..data.historical_events import get_events_for_date

            return get_events_for_date(dt, window_days=7)
        except ImportError:
            # Fallback to empty list if module not available
            return []

    def toggle(self):
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


@dataclass
class ImmersionTask:
    """A single task in the immersive learning checklist."""

    task_id: str
    title: str
    description: str
    is_complete: bool = False


class ImmersionChecklistPanel:
    """Curated list of activities to guide educational exploration."""

    def __init__(
        self,
        position: tuple[int, int] = (20, 250),
        width: int = 360,
        style: PanelStyle = None,
        tasks: list[ImmersionTask] | None = None,
    ):
        """Initialize the checklist panel."""
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = True
        self._tasks: dict[str, ImmersionTask] = {}
        self._initialize_tasks(tasks)

    def _initialize_tasks(self, tasks: list[ImmersionTask] | None):
        """Initialize checklist with default or provided tasks."""
        default_tasks = tasks or [
            ImmersionTask(
                task_id="select_body",
                title="Pick a world",
                description=(
                    "Use number keys or click to focus a planet and open its"
                    " fact sheet."
                ),
            ),
            ImmersionTask(
                task_id="navigate_time",
                title="Travel through time",
                description=(
                    "Use the date picker or time navigation hotkeys to see planetary"
                    " alignments."
                ),
            ),
            ImmersionTask(
                task_id="toggle_overlays",
                title="Tune the overlays",
                description=(
                    "Toggle orbits, labels, and the grid to compare scales and"
                    " visibility."
                ),
            ),
            ImmersionTask(
                task_id="historical_events",
                title="Explore mission history",
                description=(
                    "Open the historical events panel and jump to milestone dates."
                ),
            ),
            ImmersionTask(
                task_id="plan_transfer",
                title="Plot a transfer",
                description=(
                    "Plan an Earth→Mars Hohmann transfer to visualize interplanetary"
                    " travel."
                ),
            ),
        ]

        for task in default_tasks:
            self._tasks[task.task_id] = task

    def mark_complete(self, task_id: str):
        """Mark a checklist task as complete."""
        if task_id in self._tasks:
            self._tasks[task_id].is_complete = True

    def reset(self):
        """Reset all tasks to incomplete."""
        for task in self._tasks.values():
            task.is_complete = False

    def get_progress(self) -> tuple[int, int]:
        """Return number of completed tasks and total tasks."""
        completed = sum(1 for task in self._tasks.values() if task.is_complete)
        return completed, len(self._tasks)

    def toggle(self):
        """Toggle visibility of the checklist."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get data for rendering."""
        completed, total = self.get_progress()
        tasks = [
            {
                "title": task.title,
                "description": task.description,
                "completed": task.is_complete,
            }
            for task in self._tasks.values()
        ]

        return {
            "position": self.position,
            "width": self.width,
            "tasks": tasks,
            "progress": (completed, total),
            "style": self.style,
            "visible": self.visible,
        }


@dataclass
class Checkbox:
    label: str
    checked: bool
    action: str


class SettingsPanel:
    """
    Panel for configuring simulation settings.
    """

    def __init__(self, position: tuple[int, int] = (20, 500), style: PanelStyle = None):
        """Initialize the settings panel."""
        self.position = position
        self.style = style or PanelStyle()
        self.visible = False
        self.checkboxes: list[Checkbox] = []

    def add_checkbox(self, label: str, checked: bool, action: str):
        """Add a checkbox setting."""
        self.checkboxes.append(Checkbox(label, checked, action))

    def toggle_checkbox(self, index: int) -> str | None:
        """Toggle a checkbox by index."""
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].checked = not self.checkboxes[index].checked
            return self.checkboxes[index].action
        return None

    def toggle(self):
        """Toggle panel visibility."""
        self.visible = not self.visible

    def get_render_data(self) -> dict[str, Any]:
        """Get render data for settings panel."""
        return {
            "position": self.position,
            "checkboxes": self.checkboxes,
            "style": self.style,
            "visible": self.visible,
        }


class NavigationPanel:
    """
    Panel for changing navigation/interaction modes.
    """

    def __init__(self, position: tuple[int, int] = (20, 300), style: PanelStyle = None):
        """Initialize navigation panel."""
        self.position = position
        self.style = style or PanelStyle()
        self.visible = True
        self.modes = ["Orbit", "Pan", "Zoom"]
        self.current_mode_index = 0  # 0=Orbit

    def set_mode(self, mode_name: str):
        """Set the current navigation mode."""
        if mode_name in self.modes:
            self.current_mode_index = self.modes.index(mode_name)

    def cycle_mode(self) -> str:
        """Cycle to the next navigation mode."""
        self.current_mode_index = (self.current_mode_index + 1) % len(self.modes)
        return self.modes[self.current_mode_index]

    def get_current_mode(self) -> str:
        """Get the current navigation mode name."""
        return self.modes[self.current_mode_index]


@dataclass
class Tab:
    name: str
    content_renderer_key: str  # identify which renderer method to use


class SidebarPanel:
    """
    Combined Sidebar Panel (Right side)
    Contains tabs for: Info, Checklist, History
    """

    def __init__(
        self,
        position: tuple[int, int] = (0, 0),
        height: int = 600,
        style: PanelStyle = None,
    ):
        """Initialize sidebar panel."""
        self.position = position
        self.width = 380
        self.height = height
        self.style = style or PanelStyle()
        self.visible = True
        self.current_tab_index = 0
        self.tabs = [
            Tab("Info", "educational"),
            Tab("Planets", "planets"),
            Tab("Guide", "checklist"),
            Tab("History", "history"),
        ]

    def set_tab(self, index: int):
        """Set active tab index."""
        if 0 <= index < len(self.tabs):
            self.current_tab_index = index

    def handle_click(self, rel_x: int, rel_y: int) -> str | None:
        """Handle mouse click on sidebar."""
        # Simple tab hit detection
        tab_width = self.width // len(self.tabs)
        header_height = 35

        if rel_y < header_height:
            clicked_index = rel_x // tab_width
            self.set_tab(clicked_index)
            return "tab_changed"
        return None

    def get_render_data(self) -> dict[str, Any]:
        """Get render data for sidebar."""
        return {
            "position": self.position,
            "width": self.width,
            "height": self.height,
            "tabs": [t.name for t in self.tabs],
            "current_tab_index": self.current_tab_index,
            "current_content_key": self.tabs[
                self.current_tab_index
            ].content_renderer_key,
            "style": self.style,
            "visible": self.visible,
        }


@dataclass
class Button:
    label: str
    action: str
    width: int = 100


class UnifiedControlPanel:
    """
    Combined Bottom Control Panel
    Contains: Navigation Modes, View Settings, Time Controls
    """

    def __init__(
        self,
        position: tuple[int, int] = (0, 0),
        width: int = 800,
        style: PanelStyle = None,
    ):
        """Initialize unified control panel."""
        self.position = position
        self.width = width
        self.height = 140  # Increased height for more toggles
        self.style = style or PanelStyle()
        self.visible = True
        self.checkboxes: list[Checkbox] = []
        self.modes = ["Orbit", "Pan", "Zoom"]
        self.current_mode_index = 0
        self.buttons: list[Button] = []

    def add_checkbox(self, label: str, checked: bool, action: str):
        """Add a checkbox setting."""
        self.checkboxes.append(Checkbox(label, checked, action))

    def add_button(self, label: str, action: str):
        """Add a button."""
        self.buttons.append(Button(label, action))

    def toggle_checkbox(self, index: int) -> str | None:
        """Toggle checkbox by index."""
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].checked = not self.checkboxes[index].checked
            return self.checkboxes[index].action
        return None

    def set_mode(self, mode_name: str):
        """Set navigation mode."""
        if mode_name in self.modes:
            self.current_mode_index = self.modes.index(mode_name)

    def get_current_mode(self) -> str:
        """Get current navigation mode name."""
        return self.modes[self.current_mode_index]

    def get_render_data(self) -> dict[str, Any]:
        """Get render data for unified control panel."""
        return {
            "position": self.position,
            "width": self.width,
            "height": self.height,
            "checkboxes": self.checkboxes,
            "buttons": self.buttons,
            "modes": self.modes,
            "current_mode_index": self.current_mode_index,
            "style": self.style,
            "visible": self.visible,
        }

"""
Information display widgets: InfoPanel, StatusBar, EducationalInfoPanel.
"""

from typing import Any

from ._base import PanelStyle


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
        style: PanelStyle | None = None,
    ):
        """
        Initialize the info panel.

        Args:
            position: Top-left position (x, y)
            width: Panel width in pixels
            style: Visual styling
        """
        assert position is not None, "position must be provided"
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = True
        self._data: dict[str, Any] = {}
        self._title: str = ""

    def set_data(self, title: str, data: dict[str, Any]) -> None:
        """
        Set the data to display.

        Args:
            title: Panel title
            data: Dictionary of label -> value pairs
        """
        assert title is not None, "title must be provided"
        self._title = title
        self._data = data

    def clear(self) -> None:
        """Clear the panel data."""
        self._title = ""
        self._data = {}

    def toggle(self) -> None:
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

    def __init__(self, style: PanelStyle | None = None) -> None:
        """Initialize the status bar."""
        self.style = style or PanelStyle()
        self.visible = True
        self._components: list[str] = []

    def set_time(self, time_str: str) -> None:
        """Set the time display."""
        self._time = time_str

    def set_speed(self, speed_str: str) -> None:
        """Set the time warp display."""
        self._speed = speed_str

    def set_selected(self, name: str) -> None:
        """Set the selected object name."""
        self._selected = name

    def set_fps(self, fps: float) -> None:
        """Set the FPS display."""
        self._fps = fps

    def set_paused(self, paused: bool) -> None:
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
        style: PanelStyle | None = None,
    ):
        """
        Initialize the educational info panel.

        Args:
            position: Top-left position (x, y)
            width: Panel width in pixels
            style: Visual styling
        """
        assert position is not None, "position must be provided"
        self.position = position
        self.width = width
        self.style = style or PanelStyle()
        self.visible = True
        self._body_name: str | None = None
        self._properties: dict[str, Any] = {}
        self._fun_facts: list[str] = []
        self._current_fact_index: int = 0

    def set_body(
        self, name: str, properties: dict[str, Any], fun_facts: list[str] | None = None
    ) -> None:
        """
        Set the celestial body to display information about.

        Args:
            name: Body name
            properties: Dictionary of properties to display
            fun_facts: Optional list of educational fun facts
        """
        assert name is not None, "name must be provided"
        self._body_name = name
        self._properties = properties
        self._fun_facts = fun_facts or []
        self._current_fact_index = 0

    def cycle_fact(self) -> None:
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

    def toggle(self) -> None:
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

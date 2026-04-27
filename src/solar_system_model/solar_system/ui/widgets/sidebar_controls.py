"""
Sidebar and bottom control widgets:
Tab, SidebarPanel, Button, UnifiedControlPanel, MissionListPanel.
"""

from dataclasses import dataclass
from typing import Any

from ._base import PanelStyle
from .settings_nav import Checkbox


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
        style: PanelStyle | None = None,
    ):
        """Initialize sidebar panel."""
        assert position is not None, "position must be provided"
        self.position = position
        self.width = 380
        self.height = height
        self.style = style or PanelStyle()
        self.visible = True
        self.current_tab_index = 0
        self.tabs = [
            Tab("Info", "educational"),
            Tab("Missions", "missions"),
            Tab("History", "history"),
            Tab("Guide", "checklist"),
            Tab("Planets", "planets"),
        ]

    def set_tab(self, index: int) -> None:
        """Set active tab index."""
        if 0 <= index < len(self.tabs):
            self.current_tab_index = index

    def handle_click(self, rel_x: int, rel_y: int) -> str | None:
        """Handle mouse click on sidebar."""
        # Simple tab hit detection
        assert rel_x is not None, "rel_x must be provided"
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
        style: PanelStyle | None = None,
    ):
        """Initialize unified control panel."""
        assert position is not None, "position must be provided"
        self.position = position
        self.width = width
        self.height = 140  # Increased height for more toggles
        self.style = style or PanelStyle()
        self.visible = True
        self.checkboxes: list[Checkbox] = []
        self.modes = ["Orbit", "Pan", "Zoom"]
        self.current_mode_index = 0
        self.buttons: list[Button] = []

    def add_checkbox(self, label: str, checked: bool, action: str) -> None:
        """Add a checkbox setting."""
        self.checkboxes.append(Checkbox(label, checked, action))

    def add_button(self, label: str, action: str) -> None:
        """Add a button."""
        self.buttons.append(Button(label, action))

    def toggle_checkbox(self, index: int) -> str | None:
        """Toggle checkbox by index."""
        assert index is not None, "index must be provided"
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].checked = not self.checkboxes[index].checked
            return self.checkboxes[index].action
        return None

    def set_mode(self, mode_name: str) -> None:
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


class MissionListPanel:
    """Panel for selecting famous NASA missions."""

    def __init__(
        self, position: tuple[int, int] = (0, 0), style: PanelStyle | None = None
    ):
        assert position is not None, "position must be provided"
        self.position = position
        self.style = style or PanelStyle()
        self.visible = True
        self.scroll_offset = 0

    def get_render_data(self, missions_dict: dict[str, Any]) -> dict[str, Any]:
        assert missions_dict is not None, "missions_dict must be provided"
        missions_info = []
        for name, data in missions_dict.items():
            highlights = data.get("science_highlights", ())
            destinations = data.get("destinations", ())
            missions_info.append(
                {
                    "name": name,
                    "description": data.get("description", ""),
                    "launch_date": data.get("launch_date", ""),
                    "mission_type": data.get("mission_type", ""),
                    "destinations": ", ".join(destinations),
                    "science_highlights": "; ".join(highlights[:2]),
                }
            )

        return {
            "position": self.position,
            "missions": missions_info,
            "style": self.style,
            "visible": self.visible,
        }

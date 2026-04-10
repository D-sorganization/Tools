"""Settings/navigation widgets: Checkbox, SettingsPanel, NavigationPanel."""

from dataclasses import dataclass
from typing import Any

from ._base import PanelStyle


@dataclass
class Checkbox:
    label: str
    checked: bool
    action: str


class SettingsPanel:
    """
    Panel for configuring simulation settings.
    """

    def __init__(
        self, position: tuple[int, int] = (20, 500), style: PanelStyle | None = None
    ):
        """Initialize the settings panel."""
        assert position is not None, "position must be provided"
        self.position = position
        self.style = style or PanelStyle()
        self.visible = False
        self.checkboxes: list[Checkbox] = []

    def add_checkbox(self, label: str, checked: bool, action: str) -> None:
        """Add a checkbox setting."""
        self.checkboxes.append(Checkbox(label, checked, action))

    def toggle_checkbox(self, index: int) -> str | None:
        """Toggle a checkbox by index."""
        assert index is not None, "index must be provided"
        if 0 <= index < len(self.checkboxes):
            self.checkboxes[index].checked = not self.checkboxes[index].checked
            return self.checkboxes[index].action
        return None

    def toggle(self) -> None:
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

    def __init__(
        self, position: tuple[int, int] = (20, 300), style: PanelStyle | None = None
    ):
        """Initialize navigation panel."""
        assert position is not None, "position must be provided"
        self.position = position
        self.style = style or PanelStyle()
        self.visible = True
        self.modes = ["Orbit", "Pan", "Zoom"]
        self.current_mode_index = 0  # 0=Orbit

    def set_mode(self, mode_name: str) -> None:
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

"""GUI registration for Parametric URDF Builder.

Returns metadata for fleet launcher discovery. Immutable dict structure
ensures registration data cannot be accidentally mutated.
"""

from __future__ import annotations

from typing import Any

# Module-level constant — frozen via convention (no setter).
_GUI_INFO: dict[str, Any] = {
    "name": "Parametric URDF Builder",
    "tool_name": "urdf_builder_gui",
    "description": "Generate parametric URDF models for robotics applications",
    "category": "Robotics",
    "icon": "robot",
    "pyqt6": {
        "module": "urdf_builder_gui.ui.pyqt6.main_window",
        "class": "URDFBuilderWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "URDFBuilder",
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information.

    Returns a shallow copy to prevent callers from mutating the
    canonical registration dictionary.
    """
    return dict(_GUI_INFO)

"""GUI registration for shared typescript theme package.

Declares web: False and catalog_visible: False so this shared library package.json
is recognized as not an unreachable launcher tile in the tool registry.
"""

from __future__ import annotations

from typing import Any

GUI_INFO: dict[str, Any] = {
    "name": "TypeScript Theme",
    "tool_name": "typescript_theme",
    "description": "Shared TypeScript theme tokens and definitions",
    "category": "Theme",
    "catalog_visible": False,
    "maturity": "stable",
    "web": False,
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO

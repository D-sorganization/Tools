"""Registration stub for the static browser utilities under src/web_applications.

``calculator``, ``unit_converter`` and ``urdf_viewer`` run from a static file
server with no Python runtime; they are documented in the README and are
deliberately not launcher tiles. ``web: False`` records that decision for the
registry checker (Tools #4916) so their ``package.json`` is not reported as an
unreachable web app.
"""

from __future__ import annotations

from typing import Any

GUI_INFO: dict[str, Any] = {
    "name": "Browser utilities",
    "tool_name": "web_applications",
    "description": "Static browser utilities: calculator, unit_converter, urdf_viewer",
    "category": "Utilities",
    "catalog_visible": False,
    "maturity": "stable",
    "help": "src/web_applications/README.md",
    "web": False,
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO

"""Web theme bridge - loads themes from the shared theme-definitions.

Reads the canonical themes.json from src/shared/theme-definitions/ and
converts theme definitions into CSS custom properties for web frontends.
This ensures all web apps inherit the same themes as PyQt6 apps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the canonical theme definitions (relative to this file)
_THEMES_JSON_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "shared"
    / "theme-definitions"
    / "themes.json"  # noqa: E501
)


def load_themes() -> dict:
    """Load all themes from the shared themes.json.

    Returns:
        Parsed JSON data with 'themes', 'chartColors', etc.
    """
    if not _THEMES_JSON_PATH.exists():
        logger.warning("themes.json not found at %s", _THEMES_JSON_PATH)
        return {"themes": {}, "chartColors": []}

    with open(_THEMES_JSON_PATH, encoding="utf-8") as f:
        result: dict = json.load(f)
        return result


def get_theme_names() -> list[str]:
    """Return display names of all available themes."""
    data = load_themes()
    return [t["name"] for t in data.get("themes", {}).values()]


def get_theme_by_name(name: str) -> dict | None:
    """Look up a theme by its display name.

    Args:
        name: Display name (e.g., "Dark", "Dracula", "One Dark")

    Returns:
        Theme dict with 'name', 'colors', 'semantic', 'isDark', or None
    """
    data = load_themes()
    for theme_def in data.get("themes", {}).values():
        if theme_def["name"] == name:
            return dict(theme_def)
    return None


def get_default_theme_name() -> str:
    """Return the default theme name ('Dark')."""
    return "Dark"


def theme_to_css_vars(theme: dict) -> str:
    """Convert a theme dict into CSS custom property declarations.

    Maps the shared theme keys to CSS variables that the stylesheet uses.

    Args:
        theme: Theme dict from themes.json (with 'colors' and 'semantic')

    Returns:
        CSS string with --var declarations (without the :root{} wrapper)
    """
    colors = theme.get("colors", {})
    semantic = theme.get("semantic", {})
    is_dark = theme.get("isDark", False)

    lines = []
    lines.append(f"  color-scheme: {'dark' if is_dark else 'light'};")

    # Base color keys -> CSS vars
    _map = {
        "bg": "--bg",
        "group_bg": "--bg-card",
        "border": "--border",
        "text": "--text-primary",
        "text_secondary": "--text-secondary",
        "label": "--text-muted",
        "focus": "--border-focus",
        "input_bg": "--bg-input",
        "accent": "--accent",
        "title_bg": "--bg-elevated",
        "title_border": "--title-border",
        "table_header": "--table-header",
        "table_alt": "--table-alt",
        "button_hover": "--accent-hover",
    }
    for key, css_var in _map.items():
        if key in colors:
            lines.append(f"  {css_var}: {colors[key]};")

    # Semantic color keys
    _semantic_map = {
        "success": "--success",
        "warning": "--warning",
        "error": "--error",
        "info": "--info",
        "link": "--link",
        "link_hover": "--link-hover",
        "selection_bg": "--selection-bg",
        "selection_text": "--selection-text",
    }
    for key, css_var in _semantic_map.items():
        if key in semantic:
            lines.append(f"  {css_var}: {semantic[key]};")

    return "\n".join(lines)


def all_themes_as_css() -> str:
    """Generate CSS blocks for all themes.

    Returns a CSS string with :root (default theme) and
    [data-theme="ThemeName"] selectors for every theme.
    """
    data = load_themes()
    themes = data.get("themes", {})

    if not themes:
        return ""

    blocks = []

    # Find the default theme ("Dark")
    default_theme = None
    for theme_def in themes.values():
        if theme_def["name"] == "Dark":
            default_theme = theme_def
            break

    if default_theme is None:
        # Fallback to first theme
        default_theme = next(iter(themes.values()))

    # :root gets the default theme
    blocks.append(f":root {{\n{theme_to_css_vars(default_theme)}\n}}")

    # Each theme gets a data-theme attribute selector
    for theme_def in themes.values():
        name = theme_def["name"]
        css_vars = theme_to_css_vars(theme_def)
        blocks.append(f'[data-theme="{name}"] {{\n{css_vars}\n}}')

    return "\n\n".join(blocks)


def get_themes_for_api() -> list[dict]:
    """Return theme data formatted for the REST API.

    Returns:
        List of dicts with 'name', 'isDark', 'category'
    """
    data = load_themes()
    result = []
    result.extend(
        [
            {
                "id": theme_id,
                "name": theme_def["name"],
                "isDark": theme_def.get("isDark", False),
                "category": theme_def.get("category", ""),
            }
            for (theme_id, theme_def) in data.get("themes", {}).items()
        ]
    )
    return result

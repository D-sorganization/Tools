#!/usr/bin/env python3
"""Validate themes.json against the schema and check color consistency.

Usage:
    python scripts/validate_themes.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
THEMES_JSON = REPO_ROOT / "src" / "shared" / "theme-definitions" / "themes.json"
SCHEMA_JSON = REPO_ROOT / "src" / "shared" / "theme-definitions" / "theme-schema.json"

REQUIRED_BASE_KEYS = {
    "bg",
    "group_bg",
    "border",
    "text",
    "text_secondary",
    "label",
    "focus",
    "input_bg",
    "accent",
    "title_bg",
    "title_border",
    "table_header",
    "table_alt",
    "button_hover",
}

REQUIRED_SEMANTIC_KEYS = {
    "success",
    "warning",
    "error",
    "info",
    "link",
    "link_hover",
    "selection_bg",
    "selection_text",
}

HEX_PATTERN = re.compile(r"^#[0-9a-fA-F]{6,8}$")


def luminance(hex_color: str) -> float:
    """Calculate relative luminance of a hex color."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255
    g = int(h[2:4], 16) / 255
    b = int(h[4:6], 16) / 255
    return 0.299 * r + 0.587 * g + 0.114 * b


def contrast_ratio(fg: str, bg: str) -> float:
    """Calculate WCAG contrast ratio between two colors."""

    def srgb_to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    def relative_luminance(hex_color: str) -> float:
        h = hex_color.lstrip("#")[:6]
        r = srgb_to_linear(int(h[0:2], 16) / 255)
        g = srgb_to_linear(int(h[2:4], 16) / 255)
        b = srgb_to_linear(int(h[4:6], 16) / 255)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1 = relative_luminance(fg)
    l2 = relative_luminance(bg)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def validate() -> list[str]:
    """Validate themes.json and return list of errors."""
    errors: list[str] = []

    if not THEMES_JSON.exists():
        errors.append(f"themes.json not found: {THEMES_JSON}")
        return errors

    with open(THEMES_JSON, encoding="utf-8") as f:
        data = json.load(f)

    # Check top-level structure
    for key in ("version", "colorKeys", "chartColors", "themes"):
        if key not in data:
            errors.append(f"Missing top-level key: {key}")

    if errors:
        return errors

    # Check chart colors
    for i, color in enumerate(data.get("chartColors", [])):
        if not HEX_PATTERN.match(color):
            errors.append(f"Invalid chart color [{i}]: {color}")

    # Check each theme
    themes = data.get("themes", {})
    if not themes:
        errors.append("No themes defined")
        return errors

    for theme_id, theme_def in themes.items():
        prefix = f"Theme '{theme_id}'"

        # Check required fields
        for field in ("name", "category", "isDark", "colors", "semantic"):
            if field not in theme_def:
                errors.append(f"{prefix}: missing field '{field}'")

        if "colors" not in theme_def or "semantic" not in theme_def:
            continue

        # Check base color keys
        colors = theme_def["colors"]
        for key in REQUIRED_BASE_KEYS:
            if key not in colors:
                errors.append(f"{prefix}: missing base color '{key}'")
            elif not HEX_PATTERN.match(colors[key]):
                errors.append(f"{prefix}: invalid hex for '{key}': {colors[key]}")

        # Check semantic color keys
        semantic = theme_def["semantic"]
        for key in REQUIRED_SEMANTIC_KEYS:
            if key not in semantic:
                errors.append(f"{prefix}: missing semantic color '{key}'")
            elif not HEX_PATTERN.match(semantic[key]):
                errors.append(f"{prefix}: invalid hex for '{key}': {semantic[key]}")

        # Check isDark flag consistency
        if "bg" in colors:
            lum = luminance(colors["bg"])
            is_dark = theme_def.get("isDark", False)
            if is_dark and lum >= 0.5:
                errors.append(f"{prefix}: isDark=true but bg luminance={lum:.2f} (>=0.5)")
            elif not is_dark and lum < 0.5:
                errors.append(f"{prefix}: isDark=false but bg luminance={lum:.2f} (<0.5)")

        # WCAG AA contrast check (text on background, 4.5:1 minimum)
        if "text" in colors and "bg" in colors:
            ratio = contrast_ratio(colors["text"], colors["bg"])
            if ratio < 4.5:
                errors.append(f"{prefix}: text/bg contrast {ratio:.1f}:1 < 4.5:1 (WCAG AA)")

    return errors


def main() -> int:
    """Run validation and print results."""
    print(f"Validating {THEMES_JSON}...")
    errors = validate()

    if errors:
        print(f"\n{len(errors)} error(s) found:")
        for err in errors:
            print(f"  - {err}")
        return 1

    # Load and print summary
    with open(THEMES_JSON, encoding="utf-8") as f:
        data = json.load(f)

    themes = data.get("themes", {})
    print(f"  {len(themes)} themes validated successfully")
    print(f"  {len(data.get('chartColors', []))} chart colors")
    for tid, tdef in themes.items():
        dark = "dark" if tdef.get("isDark") else "light"
        print(f"    {tid}: {tdef['name']} ({dark}, {tdef.get('category', '?')})")
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

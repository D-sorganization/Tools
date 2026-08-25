"""Tests for shared theme palette and dynamic token resolution."""

from __future__ import annotations

import pytest

from src.shared.python.theme import (
    DARK_THEME,
    SEMANTIC_ALIASES,
    Colors,
    ThemePalette,
    get_current_colors,
)


@pytest.mark.unit
def test_theme_palette_dict_and_attr_access() -> None:
    assert "bg_elevated" in SEMANTIC_ALIASES
    assert isinstance(DARK_THEME, ThemePalette)
    palette = ThemePalette({"bg": "#123456", "group_bg": "#abcdef"})
    assert palette["bg"] == "#123456"
    assert palette.bg == "#123456"
    assert palette.bg_elevated == "#abcdef"
    assert palette.bg_surface == "#abcdef"


@pytest.mark.unit
def test_colors_dynamic_tokens() -> None:
    colors = Colors()
    assert isinstance(Colors.BG_BASE, str)
    assert Colors.BG_BASE.startswith("#")
    assert isinstance(colors.BG_BASE, str)
    assert colors.BG_BASE.startswith("#")


@pytest.mark.unit
def test_get_current_colors_returns_palette() -> None:
    current = get_current_colors()
    assert isinstance(current, ThemePalette)
    assert "bg" in current
    assert "text" in current

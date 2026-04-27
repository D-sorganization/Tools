from typing import Any

"""Tests for the theme module."""


from double_pendulum_golf.gui import theme


def test_theme_colors() -> Any:
    assert isinstance(theme.BG_DARKEST, str)
    assert theme.BG_DARKEST.startswith("#")


def test_theme_stylesheets() -> Any:
    assert "QGroupBox" in theme.STYLE_GROUP_BOX
    assert "QPushButton" in theme.STYLE_BUTTON


def test_theme_exports_all() -> Any:
    assert "STYLE_GROUP_BOX" in theme.__all__
    assert "SEVERITY_COLORS" in theme.__all__

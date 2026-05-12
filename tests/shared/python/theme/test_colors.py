"""Tests for the colors module."""

import pytest

from shared.python.theme.colors import (
    BUILTIN_THEMES,
    CHART_COLORS,
    THEME_COLOR_KEYS,
    get_rgba,
    is_dark_theme,
    is_valid_hex_color,
    normalise_hex_color,
)


class TestHexColorValidation:
    """Tests for hex color validation functions."""

    def test_valid_6_char_hex(self) -> None:
        """Test valid 6-character hex colors."""
        assert is_valid_hex_color("#ff0000")
        assert is_valid_hex_color("#FF0000")
        assert is_valid_hex_color("ff0000")
        assert is_valid_hex_color("#123abc")

    def test_valid_3_char_hex(self) -> None:
        """Test valid 3-character hex colors."""
        assert is_valid_hex_color("#f00")
        assert is_valid_hex_color("#FFF")
        assert is_valid_hex_color("abc")

    def test_valid_4_char_hex(self) -> None:
        """Test valid 4-character hex colors (#RGBA shorthand)."""
        assert is_valid_hex_color("#ff00")
        assert is_valid_hex_color("#abcd")

    def test_invalid_hex(self) -> None:
        """Test invalid hex colors."""
        assert not is_valid_hex_color("")
        assert not is_valid_hex_color("#gg0000")
        assert not is_valid_hex_color("#f")
        assert not is_valid_hex_color("#ff")
        assert not is_valid_hex_color("#fffff")
        assert not is_valid_hex_color("not-a-color")

    def test_normalise_6_char(self) -> None:
        """Test normalizing 6-character hex colors."""
        assert normalise_hex_color("#FF0000") == "#ff0000"
        assert normalise_hex_color("ff0000") == "#ff0000"
        assert normalise_hex_color("#123ABC") == "#123abc"

    def test_normalise_3_char(self) -> None:
        """Test normalizing 3-character hex colors."""
        assert normalise_hex_color("#f00") == "#ff0000"
        assert normalise_hex_color("abc") == "#aabbcc"

    def test_normalise_invalid_raises(self) -> None:
        """Test that invalid colors raise ValueError."""
        with pytest.raises(ValueError):
            normalise_hex_color("invalid")


class TestBuiltinThemes:
    """Tests for built-in theme definitions."""

    def test_all_themes_have_required_keys(self) -> None:
        """Test that all built-in themes have all required color keys."""
        for theme_name, theme in BUILTIN_THEMES.items():
            for key in THEME_COLOR_KEYS:
                assert key in theme, f"Theme '{theme_name}' missing key '{key}'"

    def test_all_theme_colors_are_valid_hex(self) -> None:
        """Test that all theme color values are valid hex colors."""
        for theme_name, theme in BUILTIN_THEMES.items():
            for key, value in theme.items():
                if key == "name":
                    continue
                # Handle special values like "white"
                if value.lower() in ("white", "black"):
                    continue
                assert is_valid_hex_color(
                    value
                ), f"Theme '{theme_name}' key '{key}' has invalid color '{value}'"

    def test_expected_themes_exist(self) -> None:
        """Test that expected themes are present."""
        expected = [
            "Light",
            "Dark",
            "Monokai",
            "Dracula",
            "One Dark",
            "MS Word",
            "MS Excel",
        ]
        for name in expected:
            assert name in BUILTIN_THEMES, f"Expected theme '{name}' not found"

    def test_theme_count(self) -> None:
        """Test minimum number of themes."""
        assert len(BUILTIN_THEMES) >= 12


class TestChartColors:
    """Tests for chart color definitions."""

    def test_chart_colors_are_valid_hex(self) -> None:
        """Test that all chart colors are valid hex colors."""
        for color in CHART_COLORS:
            assert is_valid_hex_color(color)

    def test_chart_colors_count(self) -> None:
        """Test minimum number of chart colors."""
        assert len(CHART_COLORS) >= 8


class TestGetRgba:
    """Tests for the get_rgba function."""

    def test_basic_conversion(self) -> None:
        """Test basic hex to RGBA conversion."""
        r, g, b, a = get_rgba("#ff0000")
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(0.0)
        assert b == pytest.approx(0.0)
        assert a == pytest.approx(1.0)

    def test_alpha_parameter(self) -> None:
        """Test alpha parameter."""
        _, _, _, a = get_rgba("#ff0000", alpha=0.5)
        assert a == pytest.approx(0.5)

    def test_8_char_hex_with_alpha(self) -> None:
        """Test 8-character hex color with embedded alpha."""
        _, _, _, a = get_rgba("#ff000080")  # 50% alpha
        assert a == pytest.approx(0.5, abs=0.01)


class TestIsDarkTheme:
    """Tests for the is_dark_theme function."""

    def test_light_theme_is_not_dark(self) -> None:
        """Test that Light theme is not dark."""
        assert not is_dark_theme("Light")

    def test_dark_theme_is_dark(self) -> None:
        """Test that Dark theme is dark."""
        assert is_dark_theme("Dark")

    def test_monokai_is_dark(self) -> None:
        """Test that Monokai theme is dark."""
        assert is_dark_theme("Monokai")

    def test_unknown_theme_returns_false(self) -> None:
        """Test that unknown theme returns False."""
        assert not is_dark_theme("NonexistentTheme")

"""Tests for plot theme definitions."""

from __future__ import annotations

import pytest
from plot_theme.themes import (
    DEFAULT_THEME,
    PLOT_THEMES,
    PlotTheme,
    get_theme,
    get_theme_names,
    register_theme,
)


class TestPlotTheme:
    """Tests for PlotTheme dataclass."""

    def test_plot_theme_creation(self) -> None:
        """Test creating a PlotTheme instance."""
        theme = PlotTheme(name="Test Theme")
        assert theme.name == "Test Theme"
        assert theme.figure_facecolor == "#ffffff"
        assert theme.primary_color == "#8B5CF6"

    def test_plot_theme_custom_values(self) -> None:
        """Test creating a PlotTheme with custom values."""
        theme = PlotTheme(
            name="Custom",
            figure_facecolor="#123456",
            primary_color="#abcdef",
            primary_colors=["#111111", "#222222"],
        )
        assert theme.figure_facecolor == "#123456"
        assert theme.primary_color == "#abcdef"
        assert theme.primary_colors == ["#111111", "#222222"]

    def test_get_color_cycle(self) -> None:
        """Test getting the full color cycle."""
        theme = PlotTheme(
            name="Test",
            primary_colors=["#111", "#222"],
            secondary_colors=["#333", "#444"],
            accent_colors=["#555", "#666"],
        )
        cycle = theme.get_color_cycle()
        assert cycle == ["#111", "#222", "#333", "#444", "#555", "#666"]

    def test_to_rcparams(self) -> None:
        """Test converting theme to rcParams."""
        theme = PlotTheme(
            name="Test",
            figure_facecolor="#ffffff",
            axes_facecolor="#fafafa",
            text_color="#333333",
        )
        params = theme.to_rcparams()

        assert params["figure.facecolor"] == "#ffffff"
        assert params["axes.facecolor"] == "#fafafa"
        assert params["text.color"] == "#333333"
        assert "axes.grid" in params


class TestPlotThemeRegistry:
    """Tests for theme registry functions."""

    def test_default_theme_exists(self) -> None:
        """Test that default theme is in registry."""
        assert DEFAULT_THEME in PLOT_THEMES

    def test_all_themes_valid(self) -> None:
        """Test that all registered themes are valid PlotTheme instances."""
        for name, theme in PLOT_THEMES.items():
            assert isinstance(theme, PlotTheme), f"{name} is not a PlotTheme"
            assert theme.name, f"{name} has no display name"
            assert theme.primary_color, f"{name} has no primary color"

    def test_get_theme_exact_name(self) -> None:
        """Test getting theme by exact name."""
        theme = get_theme("scientific_violet")
        assert theme.name == "Scientific Violet"

    def test_get_theme_case_insensitive(self) -> None:
        """Test getting theme with different case."""
        theme = get_theme("Scientific_Violet")
        assert theme.name == "Scientific Violet"

    def test_get_theme_with_hyphens(self) -> None:
        """Test getting theme with hyphens instead of underscores."""
        theme = get_theme("scientific-violet")
        assert theme.name == "Scientific Violet"

    def test_get_theme_with_spaces(self) -> None:
        """Test getting theme with spaces."""
        theme = get_theme("scientific violet")
        assert theme.name == "Scientific Violet"

    def test_get_theme_not_found(self) -> None:
        """Test getting non-existent theme raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_theme("nonexistent_theme")

    def test_get_theme_names(self) -> None:
        """Test getting all theme names."""
        names = get_theme_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "scientific_violet" in names
        assert names == sorted(names)  # Should be sorted

    def test_register_theme(self) -> None:
        """Test registering a custom theme."""
        custom = PlotTheme(name="My Custom Theme")
        register_theme("my_custom", custom)

        assert "my_custom" in PLOT_THEMES
        retrieved = get_theme("my_custom")
        assert retrieved.name == "My Custom Theme"

        # Cleanup
        del PLOT_THEMES["my_custom"]


class TestBuiltInThemes:
    """Tests for built-in theme configurations."""

    @pytest.mark.parametrize(
        "theme_name",
        [
            "scientific_violet",
            "scientific_violet_dark",
            "catppuccin_mocha",
            "catppuccin_latte",
            "vampire_dark",
            "nord",
            "solarized_dark",
            "solarized_light",
            "frost_dark",
            "gruvbox_dark",
            "material_dark",
            "tokyo_night",
            "classic_light",
            "classic_dark",
        ],
    )
    def test_theme_has_required_colors(self, theme_name: str) -> None:
        """Test that each theme has all required color attributes."""
        theme = get_theme(theme_name)

        # Check all required attributes
        assert theme.figure_facecolor.startswith("#")
        assert theme.axes_facecolor.startswith("#")
        assert theme.primary_color.startswith("#")
        assert theme.secondary_color.startswith("#")
        assert theme.accent_color.startswith("#")
        assert len(theme.primary_colors) >= 2
        assert len(theme.secondary_colors) >= 2
        assert len(theme.accent_colors) >= 2

    @pytest.mark.parametrize(
        "theme_name",
        [
            "scientific_violet",
            "catppuccin_mocha",
            "vampire_dark",
            "nord",
        ],
    )
    def test_theme_rcparams_valid(self, theme_name: str) -> None:
        """Test that theme generates valid rcParams."""
        theme = get_theme(theme_name)
        params = theme.to_rcparams()

        # Check key params exist
        assert "figure.facecolor" in params
        assert "axes.facecolor" in params
        assert "text.color" in params
        assert "grid.color" in params
        assert "lines.linewidth" in params

    def test_scientific_violet_theme(self) -> None:
        """Test Scientific Violet theme specific settings."""
        theme = get_theme("scientific_violet")

        # Should have light background
        assert theme.figure_facecolor.startswith("#f")
        assert theme.axes_facecolor.startswith("#e")

        # Should have purple primary
        assert "#9" in theme.primary_color or "#A" in theme.primary_color

        # Should have blue secondary
        assert "#3" in theme.secondary_color or "#6" in theme.secondary_color

    def test_dark_themes_have_dark_backgrounds(self) -> None:
        """Test that dark themes have appropriately dark backgrounds."""
        dark_themes = [
            "scientific_violet_dark",
            "catppuccin_mocha",
            "vampire_dark",
            "nord",
            "solarized_dark",
            "frost_dark",
            "gruvbox_dark",
            "material_dark",
            "tokyo_night",
            "classic_dark",
        ]

        for name in dark_themes:
            theme = get_theme(name)
            # Dark backgrounds should start with low hex values
            bg = theme.figure_facecolor.lower()
            # First digit after # should be 0-3 for dark themes
            first_digit = bg[1]
            assert first_digit in "0123", f"{name} doesn't have dark background: {bg}"

    def test_light_themes_have_light_backgrounds(self) -> None:
        """Test that light themes have appropriately light backgrounds."""
        light_themes = [
            "scientific_violet",
            "catppuccin_latte",
            "solarized_light",
            "classic_light",
        ]

        for name in light_themes:
            theme = get_theme(name)
            bg = theme.figure_facecolor.lower()
            # Light backgrounds should start with high hex values
            first_digit = bg[1]
            assert first_digit in "cdef", f"{name} doesn't have light background: {bg}"

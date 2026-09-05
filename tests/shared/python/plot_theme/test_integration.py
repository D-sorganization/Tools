"""Tests for plot theme integration helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from plot_theme.integration import (
    PlotThemeMixin,
    apply_plot_theme,
    get_theme_colors,
)
from plot_theme.manager import PlotThemeManager


def _matplotlib_available() -> bool:
    """Return True if matplotlib is importable."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_matplotlib = pytest.mark.skipif(
    not _matplotlib_available(), reason="matplotlib not installed"
)


class TestApplyPlotTheme:
    """Tests for apply_plot_theme function."""

    def test_returns_manager(self) -> None:
        """Test that apply_plot_theme returns a PlotThemeManager."""
        with patch("plot_theme.integration.get_plot_theme_manager") as mock_get:
            mock_manager = MagicMock(spec=PlotThemeManager)
            mock_get.return_value = mock_manager

            result = apply_plot_theme("scientific_violet")

            assert result is mock_manager
            mock_manager.set_theme.assert_called_once_with("scientific_violet")
            mock_manager.apply_to_matplotlib.assert_called_once()

    def test_without_theme_name(self) -> None:
        """Test apply_plot_theme without theme name."""
        with patch("plot_theme.integration.get_plot_theme_manager") as mock_get:
            mock_manager = MagicMock(spec=PlotThemeManager)
            mock_get.return_value = mock_manager

            apply_plot_theme()

            mock_manager.set_theme.assert_not_called()
            mock_manager.apply_to_matplotlib.assert_called_once()


class TestGetThemeColors:
    """Tests for get_theme_colors function."""

    def test_returns_color_dict(self) -> None:
        """Test that get_theme_colors returns a dictionary."""
        colors = get_theme_colors("scientific_violet")

        assert isinstance(colors, dict)
        assert "primary" in colors
        assert "secondary" in colors
        assert "accent" in colors
        assert "background" in colors
        assert "text" in colors

    def test_colors_are_hex(self) -> None:
        """Test that returned colors are hex strings."""
        colors = get_theme_colors("catppuccin_mocha")

        assert colors["primary"].startswith("#")
        assert colors["secondary"].startswith("#")
        assert colors["background"].startswith("#")

    def test_without_theme_name(self) -> None:
        """Test get_theme_colors without theme name uses current."""
        colors = get_theme_colors()
        assert "primary" in colors


class TestPlotThemeMixin:
    """Tests for PlotThemeMixin class."""

    def test_setup_plot_theme(self) -> None:
        """Test setting up plot theme via mixin."""

        class TestClass(PlotThemeMixin):
            pass

        obj = TestClass()
        manager = obj.setup_plot_theme(apply_immediately=False)

        assert manager is not None
        assert isinstance(manager, PlotThemeManager)

    def test_set_plot_theme(self) -> None:
        """Test setting plot theme via mixin."""

        class TestClass(PlotThemeMixin):
            pass

        obj = TestClass()
        obj.setup_plot_theme(apply_immediately=False)
        obj.set_plot_theme("vampire_dark")

        assert obj._plot_theme_manager.current_theme_name == "vampire_dark"

    def test_get_plot_colors(self) -> None:
        """Test getting plot colors via mixin."""

        class TestClass(PlotThemeMixin):
            pass

        obj = TestClass()
        obj.setup_plot_theme(apply_immediately=False)
        colors = obj.get_plot_colors()

        assert "primary" in colors
        assert "secondary" in colors

    def test_theme_change_callback(self) -> None:
        """Test that mixin receives theme change callbacks."""

        class TestClass(PlotThemeMixin):
            theme_changed_called = False

            def on_plot_theme_changed(self, theme: object) -> None:
                self.theme_changed_called = True

        obj = TestClass()
        obj.setup_plot_theme(apply_immediately=False)
        obj.set_plot_theme("nord")

        assert obj.theme_changed_called


class TestCreateThemedFigure:
    """Tests for create_themed_figure function."""

    @_skip_no_matplotlib
    def test_creates_figure_and_axes(self) -> None:
        """Test that create_themed_figure returns figure and axes."""
        from plot_theme.integration import create_themed_figure

        fig, ax = create_themed_figure(figsize=(8, 6))

        assert fig is not None
        assert ax is not None


class TestStyleAxis:
    """Tests for style_axis function."""

    @_skip_no_matplotlib
    def test_styles_axis(self) -> None:
        """Test that style_axis applies theme to axes."""
        import matplotlib.pyplot as plt
        from plot_theme.integration import style_axis

        fig, ax = plt.subplots()
        style_axis(ax, "scientific_violet")

        # Check that styling was applied (axes background)
        assert ax.get_facecolor() is not None

        plt.close(fig)

"""Tests for PlotThemeManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from plot_theme.manager import PlotThemeManager, get_plot_theme_manager
from plot_theme.themes import DEFAULT_THEME, PLOT_THEMES


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


class TestPlotThemeManager:
    """Tests for PlotThemeManager class."""

    @patch.object(PlotThemeManager, "_load_saved_theme")
    def test_manager_creation(self, mock_load: MagicMock) -> None:
        """Test creating a PlotThemeManager."""
        manager = PlotThemeManager()
        assert manager is not None
        assert manager.current_theme_name == DEFAULT_THEME

    @patch.object(PlotThemeManager, "_load_saved_theme")
    def test_current_theme_property(self, mock_load: MagicMock) -> None:
        """Test current_theme property returns PlotTheme."""
        manager = PlotThemeManager()
        theme = manager.current_theme
        assert theme is not None
        assert theme.name == PLOT_THEMES[DEFAULT_THEME].name

    def test_get_available_themes(self) -> None:
        """Test getting available theme names."""
        manager = PlotThemeManager()
        themes = manager.get_available_themes()
        assert isinstance(themes, list)
        assert len(themes) > 0
        assert DEFAULT_THEME in themes

    def test_get_theme_display_names(self) -> None:
        """Test getting theme display names."""
        manager = PlotThemeManager()
        names = manager.get_theme_display_names()
        assert isinstance(names, dict)
        assert DEFAULT_THEME in names
        assert names[DEFAULT_THEME] == PLOT_THEMES[DEFAULT_THEME].name

    def test_set_theme(self) -> None:
        """Test setting a theme."""
        manager = PlotThemeManager()
        manager.set_theme("catppuccin_mocha", save=False)
        assert manager.current_theme_name == "catppuccin_mocha"
        assert manager.current_theme.name == "Catppuccin Mocha"

    def test_set_theme_case_insensitive(self) -> None:
        """Test setting theme is case insensitive."""
        manager = PlotThemeManager()
        manager.set_theme("Catppuccin_Mocha", save=False)
        assert manager.current_theme_name == "catppuccin_mocha"

    def test_set_theme_invalid(self) -> None:
        """Test setting invalid theme raises KeyError."""
        manager = PlotThemeManager()
        with pytest.raises(KeyError):
            manager.set_theme("nonexistent")

    def test_theme_change_callback(self) -> None:
        """Test theme change callbacks are called."""
        manager = PlotThemeManager()
        callback = MagicMock()

        manager.add_theme_change_callback(callback)
        manager.set_theme("vampire_dark", save=False)

        callback.assert_called_once()
        # Check the theme was passed
        call_args = callback.call_args[0]
        assert call_args[0].name == "Vampire Dark"

    def test_remove_callback(self) -> None:
        """Test removing a callback."""
        manager = PlotThemeManager()
        callback = MagicMock()

        manager.add_theme_change_callback(callback)
        manager.remove_theme_change_callback(callback)
        manager.set_theme("vampire_dark", save=False)

        callback.assert_not_called()

    def test_get_colors(self) -> None:
        """Test getting theme colors."""
        manager = PlotThemeManager()
        colors = manager.get_colors()

        assert "primary" in colors
        assert "secondary" in colors
        assert "accent" in colors
        assert "background" in colors
        assert "text" in colors
        assert colors["primary"].startswith("#")

    def test_get_histogram_style(self) -> None:
        """Test getting histogram style kwargs."""
        manager = PlotThemeManager()
        style = manager.get_histogram_style()

        assert "color" in style
        assert "alpha" in style
        assert "edgecolor" in style
        assert style["color"].startswith("#")

    def test_get_scatter_style(self) -> None:
        """Test getting scatter style kwargs."""
        manager = PlotThemeManager()
        style = manager.get_scatter_style()

        assert "c" in style
        assert "s" in style
        assert "alpha" in style

    def test_get_line_style(self) -> None:
        """Test getting line style kwargs."""
        manager = PlotThemeManager()

        style0 = manager.get_line_style(0)
        style1 = manager.get_line_style(1)

        assert "color" in style0
        assert "linewidth" in style0
        # Different indices should give different colors
        assert style0["color"] != style1["color"]

    def test_get_fit_line_style(self) -> None:
        """Test getting fit line style kwargs."""
        manager = PlotThemeManager()
        style = manager.get_fit_line_style()

        assert "color" in style
        assert "linewidth" in style
        # Fit lines should use secondary color
        assert style["color"] == manager.current_theme.secondary_color


class TestPlotThemeManagerMatplotlib:
    """Tests for matplotlib integration."""

    @pytest.fixture
    def mock_matplotlib(self) -> MagicMock:
        """Create mock matplotlib."""
        with patch("shared.python.plot_theme.manager.mpl") as mock_mpl:
            mock_mpl.rcParams = {}
            yield mock_mpl

    @_skip_no_matplotlib
    def test_apply_to_matplotlib(self) -> None:
        """Test applying theme to matplotlib."""
        manager = PlotThemeManager()
        # Should not raise
        manager.apply_to_matplotlib()


class TestGetPlotThemeManager:
    """Tests for get_plot_theme_manager function."""

    def test_returns_manager(self) -> None:
        """Test that function returns a PlotThemeManager."""
        # Reset global manager
        import shared.python.plot_theme.manager as mgr_module

        mgr_module._manager = None

        manager = get_plot_theme_manager()
        assert isinstance(manager, PlotThemeManager)

    def test_returns_same_instance(self) -> None:
        """Test that function returns singleton."""
        import shared.python.plot_theme.manager as mgr_module

        mgr_module._manager = None

        manager1 = get_plot_theme_manager()
        manager2 = get_plot_theme_manager()
        assert manager1 is manager2

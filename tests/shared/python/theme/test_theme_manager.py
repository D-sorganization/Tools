"""Tests for the ThemeManager class."""

from unittest.mock import MagicMock, patch

import pytest

# Skip all tests in this module if PyQt6 is not available
pytest.importorskip("PyQt6")

from shared.python.theme.colors import BUILTIN_THEMES, THEME_COLOR_KEYS
from shared.python.theme.theme_manager import ThemeManager


@pytest.fixture
def mock_qsettings() -> MagicMock:
    """Create a mock QSettings."""
    mock = MagicMock()
    mock.value.return_value = "Light"
    return mock


@pytest.fixture
def theme_manager(mock_qsettings: MagicMock) -> ThemeManager:
    """Create a ThemeManager with mocked settings."""
    ThemeManager.reset_instance()
    with patch("shared.python.theme.theme_manager.QSettings", return_value=mock_qsettings):
        manager = ThemeManager()
    yield manager
    ThemeManager.reset_instance()


class TestThemeManagerSingleton:
    """Tests for the singleton pattern."""

    def test_instance_returns_same_object(self, mock_qsettings: MagicMock) -> None:
        """Test that instance() returns the same object."""
        ThemeManager.reset_instance()
        with patch("shared.python.theme.theme_manager.QSettings", return_value=mock_qsettings):
            manager1 = ThemeManager.instance()
            manager2 = ThemeManager.instance()
        assert manager1 is manager2
        ThemeManager.reset_instance()

    def test_reset_instance_clears_singleton(self, mock_qsettings: MagicMock) -> None:
        """Test that reset_instance clears the singleton."""
        ThemeManager.reset_instance()
        with patch("shared.python.theme.theme_manager.QSettings", return_value=mock_qsettings):
            manager1 = ThemeManager.instance()
            ThemeManager.reset_instance()
            manager2 = ThemeManager.instance()
        assert manager1 is not manager2
        ThemeManager.reset_instance()


class TestThemeQueries:
    """Tests for theme query methods."""

    def test_get_available_themes(self, theme_manager: ThemeManager) -> None:
        """Test getting available themes."""
        themes = theme_manager.get_available_themes()
        assert "Light" in themes
        assert "Dark" in themes
        assert len(themes) >= 12

    def test_get_builtin_themes(self, theme_manager: ThemeManager) -> None:
        """Test getting built-in themes."""
        themes = theme_manager.get_builtin_themes()
        assert themes == list(BUILTIN_THEMES.keys())

    def test_get_custom_theme_names_empty(self, theme_manager: ThemeManager) -> None:
        """Test getting custom theme names when none exist."""
        names = theme_manager.get_custom_theme_names()
        assert names == []

    def test_get_current_theme_name(self, theme_manager: ThemeManager) -> None:
        """Test getting current theme name."""
        name = theme_manager.get_current_theme_name()
        assert name == "Light"

    def test_get_current_colors(self, theme_manager: ThemeManager) -> None:
        """Test getting current theme colors."""
        colors = theme_manager.get_current_colors()
        assert isinstance(colors, dict)
        for key in THEME_COLOR_KEYS:
            assert key in colors

    def test_get_theme_colors_builtin(self, theme_manager: ThemeManager) -> None:
        """Test getting colors for a built-in theme."""
        colors = theme_manager.get_theme_colors("Dark")
        assert colors is not None
        assert colors["bg"] == BUILTIN_THEMES["Dark"]["bg"]

    def test_get_theme_colors_nonexistent(self, theme_manager: ThemeManager) -> None:
        """Test getting colors for a nonexistent theme."""
        colors = theme_manager.get_theme_colors("NonexistentTheme")
        assert colors is None


class TestThemeApplication:
    """Tests for theme application methods."""

    def test_change_theme_valid(self, theme_manager: ThemeManager) -> None:
        """Test changing to a valid theme."""
        theme_manager.change_theme("Dark")
        assert theme_manager.get_current_theme_name() == "Dark"

    def test_change_theme_invalid_ignored(self, theme_manager: ThemeManager) -> None:
        """Test that changing to invalid theme is ignored."""
        theme_manager.change_theme("NonexistentTheme")
        assert theme_manager.get_current_theme_name() == "Light"

    def test_theme_changed_signal_emitted(self, theme_manager: ThemeManager) -> None:
        """Test that themeChanged signal is emitted."""
        handler = MagicMock()
        theme_manager.themeChanged.connect(handler)
        theme_manager.change_theme("Dark")
        handler.assert_called_once_with("Dark")

    def test_get_theme_stylesheet(self, theme_manager: ThemeManager) -> None:
        """Test getting a theme stylesheet."""
        stylesheet = theme_manager.get_theme_stylesheet("Dark")
        assert isinstance(stylesheet, str)
        assert len(stylesheet) > 0
        # Check for some expected CSS content
        assert "background-color" in stylesheet

    def test_get_current_stylesheet(self, theme_manager: ThemeManager) -> None:
        """Test getting current theme stylesheet."""
        stylesheet = theme_manager.get_current_stylesheet()
        assert isinstance(stylesheet, str)
        assert len(stylesheet) > 0


class TestCustomThemes:
    """Tests for custom theme management."""

    def test_save_custom_theme(self, theme_manager: ThemeManager) -> None:
        """Test saving a custom theme."""
        colors = {key: "#ff0000" for key in THEME_COLOR_KEYS}
        name = theme_manager.save_custom_theme("MyTheme", colors)
        assert name == "MyTheme"
        assert "MyTheme" in theme_manager.get_custom_theme_names()

    def test_save_custom_theme_empty_name_raises(self, theme_manager: ThemeManager) -> None:
        """Test that empty theme name raises ValueError."""
        colors = {key: "#ff0000" for key in THEME_COLOR_KEYS}
        with pytest.raises(ValueError, match="empty"):
            theme_manager.save_custom_theme("", colors)

    def test_save_custom_theme_builtin_name_raises(self, theme_manager: ThemeManager) -> None:
        """Test that built-in theme name raises ValueError."""
        colors = {key: "#ff0000" for key in THEME_COLOR_KEYS}
        with pytest.raises(ValueError, match="conflicts"):
            theme_manager.save_custom_theme("Light", colors)

    def test_save_custom_theme_missing_colors_raises(self, theme_manager: ThemeManager) -> None:
        """Test that missing colors raise ValueError."""
        colors = {"bg": "#ff0000"}  # Missing other required keys
        with pytest.raises(ValueError, match="Missing"):
            theme_manager.save_custom_theme("MyTheme", colors)

    def test_delete_custom_theme(self, theme_manager: ThemeManager) -> None:
        """Test deleting a custom theme."""
        colors = {key: "#ff0000" for key in THEME_COLOR_KEYS}
        theme_manager.save_custom_theme("MyTheme", colors)
        result = theme_manager.delete_custom_theme("MyTheme")
        assert result is True
        assert "MyTheme" not in theme_manager.get_custom_theme_names()

    def test_delete_custom_theme_nonexistent(self, theme_manager: ThemeManager) -> None:
        """Test deleting a nonexistent custom theme."""
        result = theme_manager.delete_custom_theme("NonexistentTheme")
        assert result is False

    def test_delete_current_theme_switches_to_light(self, theme_manager: ThemeManager) -> None:
        """Test that deleting current theme switches to Light."""
        colors = {key: "#ff0000" for key in THEME_COLOR_KEYS}
        theme_manager.save_custom_theme("MyTheme", colors, apply_immediately=True)
        assert theme_manager.get_current_theme_name() == "MyTheme"
        theme_manager.delete_custom_theme("MyTheme")
        assert theme_manager.get_current_theme_name() == "Light"


class TestInheritance:
    """Tests for theme inheritance in sub-applications."""

    def test_inherit_option_available_with_context(self, mock_qsettings: MagicMock) -> None:
        """Test that Inherit option is available when app_context is set."""
        ThemeManager.reset_instance()
        with patch("shared.python.theme.theme_manager.QSettings", return_value=mock_qsettings):
            manager = ThemeManager(app_context="MyApp")
        themes = manager.get_available_themes()
        assert themes[0] == "Inherit"
        ThemeManager.reset_instance()

    def test_inherit_option_not_available_without_context(
        self, theme_manager: ThemeManager
    ) -> None:
        """Test that Inherit option is not available without app_context."""
        themes = theme_manager.get_available_themes()
        assert "Inherit" not in themes

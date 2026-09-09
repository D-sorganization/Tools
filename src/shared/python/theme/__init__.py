"""Fleet-wide shared theme management system.

This module provides a unified color theme system for all PyQt6 GUI applications
across the D-sorganization repository fleet.

Features:
- 12+ built-in themes (Light, Dark, Neon Warm Dark, Vampire Dark, Frost Dark, etc.)
- Custom theme support with persistence
- Theme inheritance for docked applications
- Qt stylesheet generation
- Matplotlib integration for consistent plotting colors
- Signal-based theme change notifications

Usage:
    from shared.python.theme import ThemeManager, get_theme_manager

    # Get singleton instance
    manager = get_theme_manager()

    # Get available themes
    themes = manager.get_available_themes()

    # Change theme
    manager.change_theme("Dark")

    # Apply to a window
    manager.apply_theme_to_window(my_window)

    # Connect to theme changes
    manager.themeChanged.connect(self.on_theme_changed)

    # Access current colors for custom styling
    colors = manager.get_current_colors()
    bg_color = colors["bg"]
"""

from typing import TYPE_CHECKING, Any

from .colors import (
    BUILTIN_THEMES,
    CHART_COLORS,
    SEMANTIC_COLOR_KEYS,
    THEME_COLOR_KEYS,
    get_matplotlib_colors,
    get_rgba,
    is_dark_theme,
    is_valid_hex_color,
    normalise_hex_color,
)
from .palette import (
    DARK_THEME,
    SEMANTIC_ALIASES,
    Colors,
    ThemePalette,
    get_current_colors,
)
from .protocols import StylesheetGenerator, ThemeProvider, ThemeSwitcher
from .stylesheets import generate_minimal_stylesheet, generate_stylesheet
from .typography import (
    CSS_FONT_DISPLAY,
    CSS_FONT_MONO,
    CSS_FONT_UI,
    FONT_STACK_DISPLAY,
    FONT_STACK_MONO,
    FONT_STACK_UI,
    FontSizes,
    FontWeights,
    Sizes,
    Weights,
    get_display_font,
    get_mono_font,
    get_qfont,
)

# PyQt6-dependent imports - only available when PyQt6 is installed
try:
    from .colors import get_qcolor
    from .dialogs import (
        ColorFieldEditor,
        ColorPickerButton,
        CustomThemeDialog,
        CustomThemeEditor,
        ThemeListItem,
        ThemeManagerDialog,
        ThemePreviewWidget,
    )
    from .font_manager import FontManager, get_font_manager
    from .integration import (
        ThemedWindowMixin,
        apply_theme_to_window,
        create_theme_menu,
        setup_themed_app,
    )
    from .responsive import (
        TextWidthSpec,
        configure_form_layout_for_readability,
        derive_text_candidates,
        readable_text_width,
        set_text_minimum_width,
        wrap_in_scroll_area,
    )
    from .theme_manager import ThemeManager, get_theme_manager
    from .zoom import (
        ApplicationZoomController,
        ZoomConfig,
        ZoomTokenSet,
        install_application_zoom,
        scale_px,
    )

    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False
    # Runtime fallbacks preserve the imported static types in both mypy modes.
    if not TYPE_CHECKING:
        ThemeManager = None
        get_theme_manager = None
        FontManager = None
        get_font_manager = None
        get_qcolor = None
        ThemedWindowMixin = None
        apply_theme_to_window = None
        create_theme_menu = None
        setup_themed_app = None
        ColorFieldEditor = None
        ColorPickerButton = None
        CustomThemeDialog = None
        CustomThemeEditor = None
        ThemeListItem = None
        ThemeManagerDialog = None
        ThemePreviewWidget = None
        TextWidthSpec = None
        configure_form_layout_for_readability = None
        derive_text_candidates = None
        readable_text_width = None
        set_text_minimum_width = None
        wrap_in_scroll_area = None
        ApplicationZoomController = None
        ZoomConfig = None
        ZoomTokenSet = None
        install_application_zoom = None
        scale_px = None


def _derive_full_palette(
    partial: dict[str, Any], theme_name: str | None = None
) -> dict[str, Any]:
    """Promote a partial colour dict into a full 60+ token palette."""
    from .api import ThemeColors

    _BASE_DEFAULTS: dict[str, Any] = {
        "bg": "#ffffff",
        "group_bg": "#f8f9fa",
        "input_bg": "#ffffff",
        "border": "#ced4da",
        "text": "#212529",
        "text_secondary": "#495057",
        "label": "#666e76",
        "focus": "#80bdff",
        "accent": "#5a8fc4",
        "title_bg": "#e3f2fd",
        "title_border": "#90caf9",
        "table_header": "#e9ecef",
        "table_alt": "#f8f9fa",
        "button_hover": "#4a7ba7",
    }
    merged: dict[str, Any] = {**_BASE_DEFAULTS, **partial}
    if theme_name and "name" not in merged:
        merged["name"] = theme_name
    try:
        palette: dict[str, Any] = ThemeColors(**merged).as_dict()
        return palette
    except Exception:  # noqa: BLE001
        return merged


__all__ = [
    # Protocols (no PyQt6 dependency)
    "StylesheetGenerator",
    "ThemeProvider",
    "ThemeSwitcher",
    # Theme manager (requires PyQt6)
    "ThemeManager",
    "get_theme_manager",
    # Font manager (requires PyQt6)
    "FontManager",
    "get_font_manager",
    # Integration helpers (requires PyQt6)
    "ThemedWindowMixin",
    "apply_theme_to_window",
    "create_theme_menu",
    "setup_themed_app",
    # Responsive sizing and zoom helpers (require PyQt6)
    "TextWidthSpec",
    "configure_form_layout_for_readability",
    "derive_text_candidates",
    "readable_text_width",
    "set_text_minimum_width",
    "wrap_in_scroll_area",
    "ApplicationZoomController",
    "ZoomConfig",
    "ZoomTokenSet",
    "install_application_zoom",
    "scale_px",
    # Dialogs (requires PyQt6)
    "ColorFieldEditor",
    "ColorPickerButton",
    "CustomThemeDialog",
    "CustomThemeEditor",
    "ThemeListItem",
    "ThemeManagerDialog",
    "ThemePreviewWidget",
    # Color utilities and Palette
    "BUILTIN_THEMES",
    "CHART_COLORS",
    "Colors",
    "DARK_THEME",
    "SEMANTIC_ALIASES",
    "SEMANTIC_COLOR_KEYS",
    "THEME_COLOR_KEYS",
    "ThemePalette",
    "get_current_colors",
    "get_matplotlib_colors",
    "get_qcolor",
    "get_rgba",
    "is_dark_theme",
    "is_valid_hex_color",
    "normalise_hex_color",
    # Typography
    "CSS_FONT_DISPLAY",
    "CSS_FONT_MONO",
    "CSS_FONT_UI",
    "FONT_STACK_DISPLAY",
    "FONT_STACK_MONO",
    "FONT_STACK_UI",
    "FontSizes",
    "FontWeights",
    "Sizes",
    "Weights",
    "get_display_font",
    "get_mono_font",
    "get_qfont",
    # Stylesheet generation
    "generate_minimal_stylesheet",
    "generate_stylesheet",
]

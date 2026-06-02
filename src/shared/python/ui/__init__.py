"""UI components for Golf Modeling Suite.

This package provides reusable UI components used across the application:

- Toast notifications: Non-blocking, auto-dismissing messages
- Shortcuts overlay: Modal keyboard shortcut reference
- Loading buttons: Buttons with loading state indicators
- Preferences dialog: User settings interface
- Recent models panel: Quick access to recently used models

Usage:
    from shared.python.ui import ToastManager, ShortcutsOverlay, LoadingButton

    # Toast notifications
    toast_manager = ToastManager(main_window)
    toast_manager.show_success("Done!")

    # Keyboard shortcuts
    overlay = ShortcutsOverlay(main_window)
    overlay.show()

    # Loading buttons
    btn = LoadingButton("Launch")
    btn.set_loading(True, "Launching...")

    # Recent models
    panel = RecentModelsPanel()
    panel.model_selected.connect(on_select)
"""

# PyQt6-dependent imports - only available when PyQt6 is installed.
# Guarded so the package imports cleanly in headless environments without
# PyQt6 (mirrors theme/__init__), exposing the Qt widgets only when present.
try:
    from .auto_complete import AutoCompleteLineEdit
    from .hover_copy_browser import HoverCopyTextBrowser

    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False
    AutoCompleteLineEdit = None  # type: ignore[assignment, misc]
    HoverCopyTextBrowser = None  # type: ignore[assignment, misc]

__all__ = [
    # Auto Complete
    "AutoCompleteLineEdit",
    # Hover Copy Text Browser
    "HoverCopyTextBrowser",
]

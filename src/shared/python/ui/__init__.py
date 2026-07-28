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

import importlib
from typing import Any

from .window_icon import (
    apply_window_icon,
    resolve_icon_path,
    set_app_user_model_id,
)

_PYQT6_AVAILABLE = False
_OPTIONAL_WIDGETS = {
    "AutoCompleteLineEdit": (".auto_complete", "AutoCompleteLineEdit"),
    "HoverCopyTextBrowser": (".hover_copy_browser", "HoverCopyTextBrowser"),
}


def _load_optional_widget(public_name: str) -> Any:
    """Return an optional Qt widget class, or None when its Qt stack is absent."""
    module_name, class_name = _OPTIONAL_WIDGETS[public_name]
    try:
        module = importlib.import_module(module_name, __name__)
    except (ImportError, OSError):
        globals()[public_name] = None
        return None

    widget = getattr(module, class_name)
    globals()[public_name] = widget
    globals()["_PYQT6_AVAILABLE"] = True
    return widget


def __getattr__(name: str) -> Any:
    """Lazily expose optional Qt widgets without importing unrelated submodules."""
    if name in _OPTIONAL_WIDGETS:
        return _load_optional_widget(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Auto Complete
    "AutoCompleteLineEdit",
    # Hover Copy Text Browser
    "HoverCopyTextBrowser",
    # Window icon / taskbar identity
    "apply_window_icon",
    "resolve_icon_path",
    "set_app_user_model_id",
]

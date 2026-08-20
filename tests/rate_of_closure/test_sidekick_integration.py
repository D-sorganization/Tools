"""Tests for Sidekick Unified Sidebar integration in RateOfClosureMainWindow."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow
from shared.python.gui_launcher.tools_sidebar_integration import NullToolsSidebar


def test_rate_of_closure_main_window_installs_sidekick_dock(qapp: QApplication) -> None:
    win = RateOfClosureMainWindow()
    try:
        assert win._sidekick_status.installed is True
        assert win._sidekick_status.sidebar is not None
        assert win._sidekick_status.dock is not None
        registry = win._sidekick_status.sidebar.registry
        assert registry.get("active_club") is not None
        assert registry.get("simulation_run") is not None
    finally:
        win.close()


def test_sidekick_toggle_visibility(qapp: QApplication) -> None:
    win = RateOfClosureMainWindow()
    try:
        win.show()
        dock = win._sidekick_status.dock
        if dock and hasattr(dock, "isVisible"):
            initial = dock.isVisible()
            win.toggle_sidekick_sidebar()
            assert dock.isVisible() != initial
            win.toggle_sidekick_sidebar()
            assert dock.isVisible() == initial
    finally:
        win.close()


def test_null_tools_sidebar_fallback(qapp: QApplication) -> None:
    fallback = NullToolsSidebar()
    assert fallback.sidekick_tokens == {}
    assert fallback.widget() is not None


def test_sidekick_context_provider(qapp: QApplication) -> None:
    win = RateOfClosureMainWindow()
    try:
        ctx = win._get_sidekick_context()
        assert isinstance(ctx, dict)
        assert ctx.get("tool_name") == "rate_of_closure"
        assert ctx.get("active_club") is not None
    finally:
        win.close()


def test_sidekick_themed_app_integration(qapp: QApplication) -> None:
    from shared.python.theme import setup_themed_app

    win = RateOfClosureMainWindow()
    try:
        setup_themed_app(qapp, win, settings_app="RateOfClosureTest")
        assert win._sidekick_status.installed is True
    finally:
        win.close()



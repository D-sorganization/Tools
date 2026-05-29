# ruff: noqa: E501
"""Loading contract tests for the Sidekick Function Generator (signal generator) tab.

Like the Data Processor, ``build_function_generator_tab`` swallows every
exception and falls back to a placeholder. These tests assert the real
Function Generator widget loads so a broken import chain cannot pass silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _qt_widgets():
    try:
        from sidekick.ui.tools_sidebar.qt_compat import QT_API, QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    if QT_API != "PyQt6":
        pytest.skip("Function Generator requires the PyQt6 UI backend")
    return QtWidgets


@pytest.fixture(autouse=True)
def _reset_theme_singleton():
    """Drop the shared ThemeManager singleton so a sidebar destroyed by a
    prior test cannot leave a dangling C++ parent behind."""
    try:
        from theme.theme_manager import ThemeManager
    except ImportError:
        yield
        return
    ThemeManager.reset_instance()
    yield
    ThemeManager.reset_instance()


def test_function_generator_registration_resolves() -> None:
    """The function_generator GUI registration must resolve its widget class."""
    import importlib

    registration = importlib.import_module("function_generator.gui_registration")
    info = registration.get_gui_info()
    pyqt_info = info["pyqt6"]
    module = importlib.import_module(pyqt_info["module"])
    assert hasattr(module, pyqt_info["class"])


def test_build_function_generator_tab_returns_real_widget(tmp_path: Path) -> None:
    QtWidgets = _qt_widgets()
    from sidekick.ui.tools_sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar import design_tokens as theme
    from sidekick.ui.tools_sidebar.default_tabs import build_function_generator_tab

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    widget = build_function_generator_tab(sidebar)

    assert widget.objectName() != theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME, (
        "Function Generator fell back to a placeholder; the underlying widget "
        "failed to import/instantiate."
    )
    assert widget.objectName() == theme.SIDEKICK_FUNCTION_GENERATOR_OBJECT_NAME

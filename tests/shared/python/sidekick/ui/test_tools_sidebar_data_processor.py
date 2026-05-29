# ruff: noqa: E501
"""Loading contract tests for the Sidekick Data Processor tab.

These tests guard against silent placeholder fallback: ``build_data_processor_tab``
catches every exception and returns a placeholder widget, so a genuinely broken
import chain (e.g. ``state_manager`` failing to import) is indistinguishable from
an optional dependency simply being absent. We assert the *real* tab is built.
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
        pytest.skip("Data Processor requires the PyQt6 UI backend")
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


def test_build_data_processor_tab_returns_real_widget(tmp_path: Path) -> None:
    QtWidgets = _qt_widgets()
    from sidekick.ui.tools_sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar import design_tokens as theme
    from sidekick.ui.tools_sidebar.data_processor_tab import (
        SidekickDataProcessorTab,
        build_data_processor_tab,
    )

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)

    widget = build_data_processor_tab(sidebar)

    # A placeholder fallback is the failure mode we are guarding against.
    assert widget.objectName() != theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME, (
        "Data Processor fell back to a placeholder; the underlying widget "
        "failed to import/instantiate."
    )
    assert isinstance(widget, SidekickDataProcessorTab)
    assert (
        widget.findChild(QtWidgets.QPushButton, "SidekickDataProcessorExportWorkspace")
        is not None
    )


def test_data_processor_widget_module_imports() -> None:
    """The embedded Data Processor widget must import without broken deps."""
    import importlib

    module = importlib.import_module("sidekick.ui.widgets.data_processor_widget")
    assert hasattr(module, "DataProcessorWidget")

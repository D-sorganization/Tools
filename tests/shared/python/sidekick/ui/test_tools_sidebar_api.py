"""Unit tests for the public ``tools_sidebar.api`` factory/install helpers.

``create_tools_sidebar`` and ``install_tools_sidebar`` are the stable host-facing
entry points. They build a real ``UnifiedToolsSidebar``, so the Qt scenarios are
consolidated into a single test: constructing multiple top-level widgets across
separate tests races their C++ teardown and can segfault. The Qt-free
host-rejection branch stays isolated.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _qt():
    try:
        from sidekick.ui.tools_sidebar.qt_compat import QT_API, QtWidgets
    except ImportError:
        pytest.skip("Qt widgets unavailable")
    if QT_API != "PyQt6":
        pytest.skip("Tools sidebar requires the PyQt6 backend")
    return QtWidgets


@pytest.fixture(autouse=True)
def _reset_theme_singleton():
    try:
        from theme.theme_manager import ThemeManager
    except ImportError:
        yield
        return
    ThemeManager.reset_instance()
    yield
    ThemeManager.reset_instance()


def test_install_rejects_non_dock_host() -> None:
    # Qt-free branch: a plain object has no addDockWidget.
    from sidekick.ui.tools_sidebar.api import install_tools_sidebar

    result = install_tools_sidebar(object())  # type: ignore[arg-type]
    assert result.installed is False
    assert "does not support docks" in result.reason
    assert result.sidebar is None


def test_create_and_install_sidebar_scenarios(tmp_path: Path) -> None:
    QtWidgets = _qt()
    from sidekick.ui.tools_sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar.api import (
        create_tools_sidebar,
        install_tools_sidebar,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # 1. Plain creation returns a real sidebar widget.
    sidebar = create_tools_sidebar(project_root=tmp_path)
    assert isinstance(sidebar, UnifiedToolsSidebar)

    # 2. A context provider is invoked during creation.
    calls: list[int] = []
    create_tools_sidebar(
        project_root=tmp_path, context_provider=lambda: calls.append(1) or "ctx"
    )
    assert calls

    # 3. A failing context provider is swallowed (optional surface).
    def boom() -> str:
        raise RuntimeError("provider exploded")

    tolerant = create_tools_sidebar(project_root=tmp_path, context_provider=boom)
    assert isinstance(tolerant, UnifiedToolsSidebar)

    # 4. Installing into a real main window returns an installed result.
    window = QtWidgets.QMainWindow()
    result = install_tools_sidebar(window, project_root=tmp_path, title="Tools")
    assert result.installed is True
    assert result.reason == "installed"
    assert result.sidebar is not None
    assert result.dock_widget is not None

    # Keep references alive until the single deterministic teardown.
    app.processEvents()
    window.close()

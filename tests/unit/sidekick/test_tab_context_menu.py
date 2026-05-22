"""Unit tests for sidekick.tab_context_menu (issues #3032, #2929).

The facade module re-exports from sidekick.ui.tools_sidebar.tab_context_menu.
Tests exercise the public API via the facade import path.  Qt-backed tests
are skipped when PyQt6 is not installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

SHARED = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))


@pytest.fixture(scope="module")
def tcm():  # type: ignore[no-untyped-def]
    """Import and return the tab_context_menu facade module."""
    return importlib.import_module("sidekick.tab_context_menu")


@pytest.fixture()
def live_sidebar(tmp_path: Path, qtbot):  # type: ignore[no-untyped-def]
    """Return a real sidebar so menu-construction failures stay test-visible."""
    pytest.importorskip("PyQt6")

    from sidekick.ui.tools_sidebar.qt_compat import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    qtbot.addWidget(sidebar)
    return sidebar


def test_tab_context_menu_all(tcm) -> None:  # type: ignore[no-untyped-def]
    """Facade exports exactly build_tab_context_menu and show_tab_context_menu."""
    assert set(tcm.__all__) == {"build_tab_context_menu", "show_tab_context_menu"}


def test_build_tab_context_menu_is_callable(tcm) -> None:  # type: ignore[no-untyped-def]
    """build_tab_context_menu is a callable."""
    assert callable(tcm.build_tab_context_menu)


def test_show_tab_context_menu_is_callable(tcm) -> None:  # type: ignore[no-untyped-def]
    """show_tab_context_menu is a callable."""
    assert callable(tcm.show_tab_context_menu)


@pytest.mark.gui
def test_show_tab_context_menu_no_tab_id_returns(tcm, qtbot) -> None:  # type: ignore[no-untyped-def]
    """show_tab_context_menu returns early if get_tab_id_at returns None."""
    pytest.importorskip("PyQt6")

    sidebar = MagicMock()
    sidebar.get_tab_id_at.return_value = None
    sidebar.tabs.tabBar.return_value.tabAt.return_value = -1

    # Should return None (early exit) without error
    result = tcm.show_tab_context_menu(sidebar, MagicMock())
    assert result is None


@pytest.mark.gui
def test_build_tab_context_menu_returns_qmenu(
    tcm,
    live_sidebar,
) -> None:  # type: ignore[no-untyped-def]
    """build_tab_context_menu returns a QMenu."""
    from PyQt6.QtWidgets import QMenu

    menu = tcm.build_tab_context_menu(live_sidebar, "calculator")
    assert isinstance(menu, QMenu)


@pytest.mark.gui
def test_build_tab_context_menu_close_action_present(
    tcm,
    live_sidebar,
) -> None:  # type: ignore[no-untyped-def]
    """build_tab_context_menu always includes a Close action."""
    menu = tcm.build_tab_context_menu(live_sidebar, "calculator")
    action_titles = [
        action.text().replace("&", "")
        for action in menu.actions()
        if not action.isSeparator()
    ]
    assert "Close" in action_titles

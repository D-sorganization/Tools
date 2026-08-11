"""Tests for sidekick tab right-click context menu (issues #3032, #2929).

Verifies that:
1. tab_context_menu.py exists at the stable top-level path.
2. The context menu registers rename, close, duplicate (pop_out), and
   the settings gear panel (selected_tab_panel.py).
3. The gear settings button is present in the sidebar.
4. selected_tab_panel.py exports the expected symbols.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt tab context menu tests run serially on Windows.",
        allow_module_level=True,
    )

pytest.importorskip("PyQt6")

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
_TEST_PKG = Path(__file__).resolve().parent  # tests/unit/sidekick/


def _fix_sidekick_import() -> None:
    """Ensure the production sidekick package is loaded (not the test shadow)."""
    shared_str = str(_SHARED)
    if shared_str not in sys.path:
        sys.path.insert(0, shared_str)
    else:
        sys.path.remove(shared_str)
        sys.path.insert(0, shared_str)

    test_dir = str(_TEST_PKG)
    top_mod = sys.modules.get("sidekick")
    if (
        top_mod is not None
        and getattr(top_mod, "__file__", None) is not None
        and test_dir in str(Path(top_mod.__file__).resolve().parent)
    ):
        del sys.modules["sidekick"]


@pytest.fixture(scope="module")
def tcm() -> Any:
    """Import and return the tab_context_menu facade module."""
    _fix_sidekick_import()
    return importlib.import_module("sidekick.tab_context_menu")


@pytest.fixture()
def live_sidebar(tmp_path: Path, qtbot: Any) -> Any:
    """Return a real sidebar so menu-construction failures stay test-visible."""
    _fix_sidekick_import()
    # Ensure application exists
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    sidebar = UnifiedToolsSidebar(project_root=tmp_path)
    qtbot.addWidget(sidebar)
    return sidebar


# ---------------------------------------------------------------------------
# Module-level smoke: confirm top-level re-exports are importable
# ---------------------------------------------------------------------------


def test_tab_context_menu_module_exists() -> None:
    """sidekick.tab_context_menu is importable at the stable top-level path."""
    _fix_sidekick_import()
    from sidekick.tab_context_menu import (  # noqa: F401
        build_tab_context_menu,
        show_tab_context_menu,
    )

    assert callable(build_tab_context_menu)
    assert callable(show_tab_context_menu)


def test_selected_tab_panel_module_exists() -> None:
    """sidekick.selected_tab_panel is importable at the stable top-level path."""
    _fix_sidekick_import()
    from sidekick.selected_tab_panel import (  # noqa: F401
        SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME,
        TabSettingsMixin,
        build_tab_settings_dialog,
        build_tab_settings_toolbar,
    )

    assert SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME == "SidekickActiveTabSettings"


def test_tab_context_menu_all(tcm: Any) -> None:
    """Facade exports exactly build_tab_context_menu and show_tab_context_menu."""
    assert set(tcm.__all__) == {"build_tab_context_menu", "show_tab_context_menu"}


def test_build_tab_context_menu_is_callable(tcm: Any) -> None:
    """build_tab_context_menu is a callable."""
    assert callable(tcm.build_tab_context_menu)


def test_show_tab_context_menu_is_callable(tcm: Any) -> None:
    """show_tab_context_menu is a callable."""
    assert callable(tcm.show_tab_context_menu)


@pytest.mark.gui
def test_show_tab_context_menu_no_tab_id_returns(tcm: Any, qtbot: Any) -> None:
    """show_tab_context_menu returns early if get_tab_id_at returns None."""
    sidebar = MagicMock()
    sidebar.get_tab_id_at.return_value = None
    sidebar.tabs.tabBar.return_value.tabAt.return_value = -1

    # Should return None (early exit) without error
    result = tcm.show_tab_context_menu(sidebar, MagicMock())
    assert result is None


@pytest.mark.gui
def test_build_tab_context_menu_returns_qmenu(
    tcm: Any,
    live_sidebar: Any,
) -> None:
    """build_tab_context_menu returns a QMenu."""
    from PyQt6.QtWidgets import QMenu

    menu = tcm.build_tab_context_menu(live_sidebar, "calculator")
    assert isinstance(menu, QMenu)


@pytest.mark.gui
def test_build_tab_context_menu_close_action_present(
    tcm: Any,
    live_sidebar: Any,
) -> None:
    """build_tab_context_menu always includes a Close action."""
    menu = tcm.build_tab_context_menu(live_sidebar, "calculator")
    action_titles = [
        action.text().replace("&", "")
        for action in menu.actions()
        if not action.isSeparator()
    ]
    assert "Close" in action_titles


# ---------------------------------------------------------------------------
# Context menu action set verification (no display needed)
# ---------------------------------------------------------------------------


def _make_sidebar_with_popout_tab(
    tmp_path: Path, qtbot: Any
) -> tuple[Any, str, Any, Any]:
    """Return (sidebar, tab_id, win, app) for a sidebar with one popout-capable tab."""
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar.tab_definition import SidebarTabDefinition

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    tab_def = SidebarTabDefinition(
        tab_id="ctx_test_tab",
        title="CtxTest",
        factory=lambda _sb: QtWidgets.QLabel("hello"),
        popout_enabled=True,
        duplicate_enabled=True,
    )
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[tab_def],
        parent=win,
    )
    sidebar.install_as_dock(win, title="Sidekick")
    win.show()
    return sidebar, "ctx_test_tab", win, app


def test_context_menu_has_rename_action(tmp_path: Path, qtbot: Any) -> None:
    """Context menu for a tab includes a Rename action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path, qtbot)
    menu = build_tab_context_menu(sidebar, tab_id)
    qtbot.addWidget(menu)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Rename" in action_texts, f"Expected 'Rename' in {action_texts}"


def test_context_menu_has_close_action(tmp_path: Path, qtbot: Any) -> None:
    """Context menu for a tab includes a Close action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path, qtbot)
    menu = build_tab_context_menu(sidebar, tab_id)
    qtbot.addWidget(menu)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Close" in action_texts, f"Expected 'Close' in {action_texts}"


def test_context_menu_has_pop_out_action(tmp_path: Path, qtbot: Any) -> None:
    """Context menu for a popout-enabled tab includes a Pop Out action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path, qtbot)
    menu = build_tab_context_menu(sidebar, tab_id)
    qtbot.addWidget(menu)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Pop Out" in action_texts, f"Expected 'Pop Out' in {action_texts}"


def test_context_menu_has_duplicate_action(tmp_path: Path, qtbot: Any) -> None:
    """Context menu for a duplicate-enabled tab includes a Duplicate action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path, qtbot)
    menu = build_tab_context_menu(sidebar, tab_id)
    qtbot.addWidget(menu)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Duplicate" in action_texts, f"Expected 'Duplicate' in {action_texts}"


def test_context_menu_has_minimize_action(tmp_path: Path, qtbot: Any) -> None:
    """Context menu includes a Minimize Sidebar action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path, qtbot)
    menu = build_tab_context_menu(sidebar, tab_id)
    qtbot.addWidget(menu)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert (
        "Minimize Sidebar" in action_texts
    ), f"Expected 'Minimize Sidebar' in {action_texts}"


# ---------------------------------------------------------------------------
# Selected-tab gear panel — settings button
# ---------------------------------------------------------------------------


def test_sidebar_has_settings_gear_button(tmp_path: Path, qtbot: Any) -> None:
    """Sidebar toolbar exposes a gear settings button with the expected objectName."""
    _fix_sidekick_import()
    from PyQt6.QtWidgets import QToolButton
    from sidekick.selected_tab_panel import (
        SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME,
    )

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path, qtbot)
    btn_name = SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME
    gear_btn = sidebar.findChild(QToolButton, btn_name)
    assert gear_btn is not None, f"Expected QToolButton with objectName {btn_name!r}"

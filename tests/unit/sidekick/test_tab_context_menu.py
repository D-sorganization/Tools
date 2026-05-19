"""Tests for sidekick tab right-click context menu (issue #2929).

Verifies that:
1. tab_context_menu.py exists at the stable top-level path
2. The context menu registers rename, close, duplicate (pop_out), and
   the settings gear panel (selected_tab_panel.py)
3. The gear settings button is present in the sidebar
4. selected_tab_panel.py exports the expected symbols

TDD: these tests verify the acceptance criteria from issue #2929.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

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


# ---------------------------------------------------------------------------
# Context menu action set verification (no display needed)
# ---------------------------------------------------------------------------


def _make_sidebar_with_popout_tab(tmp_path: Path):  # type: ignore[return]
    """Return (sidebar, tab_id, win, app) for a sidebar with one popout-capable tab."""
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar.tab_definition import SidebarTabDefinition

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
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


def test_context_menu_has_rename_action(tmp_path: Path) -> None:
    """Context menu for a tab includes a Rename action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path)
    menu = build_tab_context_menu(sidebar, tab_id)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Rename" in action_texts, f"Expected 'Rename' in {action_texts}"


def test_context_menu_has_close_action(tmp_path: Path) -> None:
    """Context menu for a tab includes a Close action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path)
    menu = build_tab_context_menu(sidebar, tab_id)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Close" in action_texts, f"Expected 'Close' in {action_texts}"


def test_context_menu_has_pop_out_action(tmp_path: Path) -> None:
    """Context menu for a popout-enabled tab includes a Pop Out action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path)
    menu = build_tab_context_menu(sidebar, tab_id)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    # Pop Out is the "redock" affordance for popping tabs out
    assert "Pop Out" in action_texts, f"Expected 'Pop Out' in {action_texts}"


def test_context_menu_has_duplicate_action(tmp_path: Path) -> None:
    """Context menu for a duplicate-enabled tab includes a Duplicate action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path)
    menu = build_tab_context_menu(sidebar, tab_id)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Duplicate" in action_texts, f"Expected 'Duplicate' in {action_texts}"


def test_context_menu_has_minimize_action(tmp_path: Path) -> None:
    """Context menu includes a Minimize Sidebar action."""
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar.tab_context_menu import build_tab_context_menu

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path)
    menu = build_tab_context_menu(sidebar, tab_id)
    action_texts = {a.text() for a in menu.actions() if a.text()}
    assert "Minimize Sidebar" in action_texts, (
        f"Expected 'Minimize Sidebar' in {action_texts}"
    )


# ---------------------------------------------------------------------------
# Selected-tab gear panel — settings button
# ---------------------------------------------------------------------------


def test_sidebar_has_settings_gear_button(tmp_path: Path) -> None:
    """Sidebar toolbar exposes a gear settings button with the expected objectName."""
    _fix_sidekick_import()
    from PyQt6.QtWidgets import QToolButton
    from sidekick.selected_tab_panel import (
        SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME,
    )

    sidebar, tab_id, win, _ = _make_sidebar_with_popout_tab(tmp_path)
    btn_name = SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME
    gear_btn = sidebar.findChild(QToolButton, btn_name)
    assert gear_btn is not None, f"Expected QToolButton with objectName {btn_name!r}"

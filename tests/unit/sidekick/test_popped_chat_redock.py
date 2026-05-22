"""Tests for re-dock affordance on popped-out chat windows (issue #2881).

TDD red phase: drives the pop_out_tab / redock_tab API extensions and the
"Re-dock" button on floating windows.

Import strategy: uses the same sidekick-shadow eviction as
``test_dock_close_affordances.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt popped chat redock tests run serially on Windows.",
        allow_module_level=True,
    )

pytest.importorskip("PyQt6")

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
_TEST_PKG = Path(__file__).resolve().parent  # tests/unit/sidekick/


def _fix_sidekick_import():
    """Remove test-package shadow and ensure production sidekick is loaded."""
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


def _get_classes():
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar
    from sidekick.ui.tools_sidebar.tab_definition import SidebarTabDefinition

    return UnifiedToolsSidebar, SidebarTabDefinition, QtWidgets


def _make_sidebar_with_popout_tab(tmp_path, qtbot, qt_app=None):
    """Return (sidebar, tab_id, floating_window, win, app)."""
    UnifiedToolsSidebar, SidebarTabDefinition, QtWidgets = _get_classes()

    app = qt_app or QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    tab_def = SidebarTabDefinition(
        tab_id="test_tab",
        title="Test",
        factory=lambda _sb: QtWidgets.QLabel("hello"),
        popout_enabled=True,
    )
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[tab_def],
        parent=win,
    )
    sidebar.install_as_dock(win, title="Sidekick")
    win.show()
    floating_win = sidebar.pop_out_tab("test_tab")
    if floating_win is not None:
        qtbot.addWidget(floating_win)
    return sidebar, "test_tab", floating_win, win, app


# ── Re-dock button presence ───────────────────────────────────────────────────


def test_popped_chat_has_redock_button(tmp_path, qtbot, qt_app):
    from PyQt6.QtWidgets import QPushButton

    sidebar, tab_id, floating_win, win, _ = _make_sidebar_with_popout_tab(
        tmp_path, qtbot, qt_app
    )
    assert floating_win is not None
    redock_btn = floating_win.findChild(QPushButton, "sidekick-redock")
    assert redock_btn is not None


# ── Re-dock behaviour ─────────────────────────────────────────────────────────


def test_redock_returns_tab_to_sidebar(tmp_path, qtbot, qt_app):
    from PyQt6.QtWidgets import QPushButton

    sidebar, tab_id, floating_win, win, _ = _make_sidebar_with_popout_tab(
        tmp_path, qtbot, qt_app
    )
    redock_btn = floating_win.findChild(QPushButton, "sidekick-redock")
    assert redock_btn is not None
    redock_btn.click()
    assert tab_id in sidebar.visible_tab_ids()


def test_redock_closes_floating_window(tmp_path, qtbot, qt_app):
    from PyQt6.QtWidgets import QPushButton

    sidebar, tab_id, floating_win, win, _ = _make_sidebar_with_popout_tab(
        tmp_path, qtbot, qt_app
    )
    redock_btn = floating_win.findChild(QPushButton, "sidekick-redock")
    assert redock_btn is not None
    redock_btn.click()
    assert not floating_win.isVisible()


# ── DbC: re_dock precondition ─────────────────────────────────────────────────


def test_redock_tab_precondition_not_floating_raises(tmp_path, qtbot, qt_app):
    UnifiedToolsSidebar, SidebarTabDefinition, QtWidgets = _get_classes()

    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    tab_def = SidebarTabDefinition(
        tab_id="test_tab2",
        title="Test2",
        factory=lambda _sb: QtWidgets.QLabel("hello"),
        popout_enabled=True,
    )
    sidebar = UnifiedToolsSidebar(
        project_root=tmp_path,
        tab_definitions=[tab_def],
        parent=win,
    )
    sidebar.install_as_dock(win, title="Sidekick")
    # test_tab2 is docked, not floating — re_dock should raise
    with pytest.raises(RuntimeError, match="not floating"):
        sidebar.re_dock("test_tab2")


# ── Last pop-out position memory ─────────────────────────────────────────────


def test_repop_remembers_last_position(tmp_path, qtbot, qt_app):
    sidebar, tab_id, win1, win, _ = _make_sidebar_with_popout_tab(
        tmp_path, qtbot, qt_app
    )
    # Move the window to a specific position before redocking
    win1.move(400, 300)
    pos1 = win1.pos()
    # redock
    sidebar.redock_tab(tab_id)
    # pop out again
    win2 = sidebar.pop_out_tab(tab_id)
    assert win2 is not None
    qtbot.addWidget(win2)
    assert abs(win2.pos().x() - pos1.x()) < 100  # within 100px tolerance

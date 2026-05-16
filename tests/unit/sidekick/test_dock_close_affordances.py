"""Tests for dock title-bar close/collapse chrome (issue #2881).

TDD red phase: these tests drive the DockTitleBar widget and the
toggle_collapsed / toggle_visibility / dock_title_widget API on
UnifiedToolsSidebar.

Import strategy: ``tests/unit/sidekick/`` is a Python package that shadows the
production ``sidekick`` package.  Each test function fixes this by evicting
the test-directory ``sidekick`` from ``sys.modules`` before importing
production code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
_TEST_PKG = Path(__file__).resolve().parent  # tests/unit/sidekick/


def _fix_sidekick_import():
    """Remove test-package shadow and ensure production sidekick is loaded.

    The ``tests/unit/sidekick/`` directory is a pytest test package whose name
    shadows the production ``sidekick`` package.  We remove only the bare
    ``sidekick`` top-level entry from sys.modules (not ``sidekick.conftest``
    or ``sidekick.test_*`` which pytest has already registered) then re-import
    from the correct location.
    """
    shared_str = str(_SHARED)
    if shared_str not in sys.path:
        sys.path.insert(0, shared_str)
    else:
        # Ensure it's first
        sys.path.remove(shared_str)
        sys.path.insert(0, shared_str)

    # Only evict the top-level 'sidekick' if it points to the test directory.
    test_dir = str(_TEST_PKG)
    top_mod = sys.modules.get("sidekick")
    if (
        top_mod is not None
        and getattr(top_mod, "__file__", None) is not None
        and test_dir in str(Path(top_mod.__file__).resolve().parent)
    ):
        del sys.modules["sidekick"]


def _get_classes():
    """Return production (UnifiedToolsSidebar, QtWidgets)."""
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    return UnifiedToolsSidebar, QtWidgets


def _make_sidebar(tmp_path):
    """Create a minimal sidebar with a QMainWindow parent for dock install."""
    UnifiedToolsSidebar, QtWidgets = _get_classes()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    sidebar = UnifiedToolsSidebar(project_root=tmp_path, parent=win)
    sidebar.install_as_dock(win, title="Sidekick")
    return sidebar, win, app


# ── Title-bar widget presence ─────────────────────────────────────────────────


def test_dock_title_widget_is_not_none(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    assert sidebar.dock_title_widget() is not None


def test_dock_title_bar_has_close_button(tmp_path):
    from PyQt6.QtWidgets import QPushButton

    sidebar, win, _ = _make_sidebar(tmp_path)
    close_btn = sidebar.dock_title_widget().findChild(QPushButton, "sidekick-close")
    assert close_btn is not None


def test_dock_title_bar_has_collapse_button(tmp_path):
    from PyQt6.QtWidgets import QPushButton

    sidebar, win, _ = _make_sidebar(tmp_path)
    collapse_btn = sidebar.dock_title_widget().findChild(
        QPushButton, "sidekick-collapse"
    )
    assert collapse_btn is not None


def test_close_button_tooltip_mentions_ctrl_b(tmp_path):
    from PyQt6.QtWidgets import QPushButton

    sidebar, win, _ = _make_sidebar(tmp_path)
    close_btn = sidebar.dock_title_widget().findChild(QPushButton, "sidekick-close")
    assert close_btn is not None
    assert "Ctrl+B" in close_btn.toolTip()


# ── Close behaviour ───────────────────────────────────────────────────────────


def test_dock_close_button_hides_dock(tmp_path):
    from PyQt6.QtWidgets import QPushButton

    sidebar, win, _ = _make_sidebar(tmp_path)
    win.show()
    assert sidebar.dock is not None
    close_btn = sidebar.dock_title_widget().findChild(QPushButton, "sidekick-close")
    assert close_btn is not None
    close_btn.click()
    assert sidebar.dock.isVisible() is False


# ── Collapse / expand ─────────────────────────────────────────────────────────


def test_is_collapsed_false_by_default(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    assert sidebar.is_collapsed() is False


def test_toggle_collapsed_collapses_to_icon_strip(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    win.show()
    sidebar.toggle_collapsed()
    assert sidebar.is_collapsed() is True
    assert sidebar.width() < 80


def test_toggle_collapsed_restores_width(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    win.show()
    sidebar.resize(320, 600)
    sidebar.toggle_collapsed()
    sidebar.toggle_collapsed()
    assert sidebar.is_collapsed() is False
    assert sidebar.width() >= 200  # tolerance: restored to at least 200


def test_collapse_button_click_toggles_collapsed(tmp_path):
    from PyQt6.QtWidgets import QPushButton

    sidebar, win, _ = _make_sidebar(tmp_path)
    win.show()
    collapse_btn = sidebar.dock_title_widget().findChild(
        QPushButton, "sidekick-collapse"
    )
    assert collapse_btn is not None
    collapse_btn.click()
    assert sidebar.is_collapsed() is True


# ── toggle_visibility (Ctrl+B) ────────────────────────────────────────────────


def test_toggle_visibility_hides_dock(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    win.show()
    assert sidebar.dock.isVisible()
    sidebar.toggle_visibility()
    assert sidebar.dock.isVisible() is False


def test_toggle_visibility_shows_dock(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    win.show()
    sidebar.toggle_visibility()  # hide
    sidebar.toggle_visibility()  # show again
    assert sidebar.dock.isVisible() is True

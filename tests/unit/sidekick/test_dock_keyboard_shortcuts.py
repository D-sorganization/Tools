"""Tests for keyboard shortcut wiring (issue #2881).

TDD red phase: Ctrl+B, Ctrl+Shift+B, Esc-in-chat-input behaviour.

Import strategy: uses the same sidekick-shadow eviction as
``test_dock_close_affordances.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


def _get_sidebar_class():
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    return UnifiedToolsSidebar, QtWidgets


def _make_sidebar(tmp_path):
    UnifiedToolsSidebar, QtWidgets = _get_sidebar_class()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    sidebar = UnifiedToolsSidebar(project_root=tmp_path, parent=win)
    sidebar.install_as_dock(win, title="Sidekick")
    sidebar.register_shortcuts(win)
    win.show()
    return sidebar, win, app


# ── Ctrl+B — toggle dock visibility ─────────────────────────────────────────
# We test the shortcut behaviour by directly invoking toggle_visibility()
# (the shortcut's target).  QTest.keySequence requires a visible, active
# window which is not guaranteed in headless CI.  A separate integration test
# would cover the OS-level shortcut dispatch; these unit tests verify the
# *logic* wired to the shortcut.


def test_ctrl_b_toggles_dock_visibility(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    was_visible = sidebar.dock.isVisible()
    sidebar.toggle_visibility()  # Ctrl+B calls this
    assert sidebar.dock.isVisible() != was_visible


def test_ctrl_b_twice_restores_visibility(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    was_visible = sidebar.dock.isVisible()
    sidebar.toggle_visibility()
    sidebar.toggle_visibility()
    assert sidebar.dock.isVisible() == was_visible


def test_ctrl_b_shortcut_is_registered(tmp_path):
    """Ctrl+B shortcut is wired to toggle_visibility on the main window."""

    sidebar, win, _ = _make_sidebar(tmp_path)
    _ = win.findChildren(type(None).__class__, "")
    # Verify register_shortcuts created QShortcut objects bound to Ctrl+B
    from PyQt6.QtGui import QShortcut

    all_shortcuts = win.findChildren(QShortcut)
    sequences = [sc.key().toString() for sc in all_shortcuts]
    assert "Ctrl+B" in sequences


# ── Ctrl+Shift+B — toggle collapse ──────────────────────────────────────────


def test_ctrl_shift_b_toggles_collapse(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    was_collapsed = sidebar.is_collapsed()
    sidebar.toggle_collapsed()  # Ctrl+Shift+B calls this
    assert sidebar.is_collapsed() != was_collapsed


def test_ctrl_shift_b_twice_restores_collapsed(tmp_path):
    sidebar, win, _ = _make_sidebar(tmp_path)
    was_collapsed = sidebar.is_collapsed()
    sidebar.toggle_collapsed()
    sidebar.toggle_collapsed()
    assert sidebar.is_collapsed() == was_collapsed


def test_ctrl_shift_b_shortcut_is_registered(tmp_path):
    """Ctrl+Shift+B shortcut is wired to toggle_collapsed on the main window."""
    from PyQt6.QtGui import QShortcut

    sidebar, win, _ = _make_sidebar(tmp_path)
    all_shortcuts = win.findChildren(QShortcut)
    sequences = [sc.key().toString() for sc in all_shortcuts]
    assert "Ctrl+Shift+B" in sequences


# ── Esc in chat-input must not hide dock ────────────────────────────────────


def test_esc_in_chat_input_does_not_hide_dock(tmp_path):
    """Pressing Escape in a plain text edit must leave the dock visible."""
    from PyQt6 import QtCore
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QPlainTextEdit

    sidebar, win, _ = _make_sidebar(tmp_path)
    # Create a simple chat-input-like widget and add it to the sidebar
    chat_input = QPlainTextEdit(sidebar)
    chat_input.setObjectName("sidekick-chat-input")
    sidebar.layout().addWidget(chat_input)
    chat_input.show()
    chat_input.setFocus()
    QTest.keyClick(chat_input, QtCore.Qt.Key.Key_Escape)
    # Dock must still be visible
    assert sidebar.dock.isVisible() is True

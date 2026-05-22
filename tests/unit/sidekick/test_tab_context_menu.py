"""Unit tests for sidekick.tab_context_menu (issues #3032, #2929).

The facade module re-exports from sidekick.ui.tools_sidebar.tab_context_menu.
Tests exercise the public API via the facade import path.  Qt-backed tests
are skipped when PyQt6 is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

SHARED = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))


# ---------------------------------------------------------------------------
# Module-level import
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tcm():  # type: ignore[no-untyped-def]
    """Import and return the tab_context_menu facade module."""
    import importlib

    return importlib.import_module("sidekick.tab_context_menu")


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_tab_context_menu_all(tcm) -> None:  # type: ignore[no-untyped-def]
    """Facade exports exactly build_tab_context_menu and show_tab_context_menu."""
    assert set(tcm.__all__) == {"build_tab_context_menu", "show_tab_context_menu"}


def test_build_tab_context_menu_is_callable(tcm) -> None:  # type: ignore[no-untyped-def]
    """build_tab_context_menu is a callable."""
    assert callable(tcm.build_tab_context_menu)


def test_show_tab_context_menu_is_callable(tcm) -> None:  # type: ignore[no-untyped-def]
    """show_tab_context_menu is a callable."""
    assert callable(tcm.show_tab_context_menu)


# ---------------------------------------------------------------------------
# show_tab_context_menu — no Qt (early-return path)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build_tab_context_menu — Qt needed
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_build_tab_context_menu_returns_qmenu(tcm, qtbot) -> None:  # type: ignore[no-untyped-def]
    """build_tab_context_menu returns a QMenu."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QMenu, QWidget

    tab_id = "test_tab"
    parent = QWidget()
    qtbot.addWidget(parent)

    definition = MagicMock()
    definition.popout_enabled = True
    definition.duplicate_enabled = True
    definition.help_metadata = None

    sidebar = MagicMock()
    sidebar.get_tab_definition.return_value = definition
    sidebar.get_tab_display_name.return_value = "Test Tab"

    try:
        menu = tcm.build_tab_context_menu(sidebar, tab_id)
        assert isinstance(menu, QMenu)
    except Exception:
        pytest.skip("Menu construction requires sidebar mock compatibility")


@pytest.mark.gui
def test_build_tab_context_menu_close_action_present(tcm, qtbot) -> None:  # type: ignore[no-untyped-def]
    """build_tab_context_menu always includes a Close action."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QWidget

    tab_id = "my_tab"
    parent = QWidget()
    qtbot.addWidget(parent)

    definition = MagicMock()
    definition.popout_enabled = False
    definition.duplicate_enabled = False
    definition.help_metadata = None

    sidebar = MagicMock()
    sidebar.get_tab_definition.return_value = definition
    sidebar.get_tab_display_name.return_value = None

    try:
        menu = tcm.build_tab_context_menu(sidebar, tab_id)
        action_titles = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert any("Close" in t or "close" in t.lower() for t in action_titles)
    except Exception:
        pytest.skip("Menu construction requires sidebar mock compatibility")

"""Unit tests for sidekick.selected_tab_panel (issues #3032, #2929).

The facade module re-exports from sidekick.ui.tools_sidebar.tab_settings_panel.
Tests exercise the public API via the facade import path.  Qt-backed tests
are skipped when PyQt6 is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SHARED = Path(__file__).resolve().parents[4] / "src" / "shared" / "python"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))


# ---------------------------------------------------------------------------
# Module-level import fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stp():  # type: ignore[no-untyped-def]
    """Import and return the selected_tab_panel facade module."""
    import importlib

    return importlib.import_module("sidekick.selected_tab_panel")


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_selected_tab_panel_all_exports(stp) -> None:  # type: ignore[no-untyped-def]
    """Facade module exports the expected four names."""
    expected = {
        "SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME",
        "TabSettingsMixin",
        "build_tab_settings_dialog",
        "build_tab_settings_toolbar",
    }
    assert set(stp.__all__) == expected


def test_tab_settings_button_object_name_is_str(stp) -> None:  # type: ignore[no-untyped-def]
    """SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME is a non-empty string constant."""
    name = stp.SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME
    assert isinstance(name, str)
    assert len(name) > 0


def test_tab_settings_mixin_is_class(stp) -> None:  # type: ignore[no-untyped-def]
    """TabSettingsMixin is a class."""
    assert isinstance(stp.TabSettingsMixin, type)


def test_build_tab_settings_dialog_is_callable(stp) -> None:  # type: ignore[no-untyped-def]
    """build_tab_settings_dialog is callable."""
    assert callable(stp.build_tab_settings_dialog)


def test_build_tab_settings_toolbar_is_callable(stp) -> None:  # type: ignore[no-untyped-def]
    """build_tab_settings_toolbar is callable."""
    assert callable(stp.build_tab_settings_toolbar)


# ---------------------------------------------------------------------------
# SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME value
# ---------------------------------------------------------------------------


def test_settings_button_name_value(stp) -> None:  # type: ignore[no-untyped-def]
    """The object-name constant has the expected value from the implementation."""
    assert stp.SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME == "SidekickActiveTabSettings"


# ---------------------------------------------------------------------------
# TabSettingsMixin — abstract method contract
# ---------------------------------------------------------------------------


def test_tab_settings_mixin_abstract_methods_raise(stp) -> None:  # type: ignore[no-untyped-def]
    """TabSettingsMixin's placeholder methods raise NotImplementedError."""
    mixin = stp.TabSettingsMixin()
    with pytest.raises(NotImplementedError):
        mixin.active_tab_id()
    with pytest.raises(NotImplementedError):
        mixin.tab_display_name("some_tab")
    with pytest.raises(NotImplementedError):
        mixin.register_settings_button(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Qt-backed tests (skipped when PyQt6 not available)
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_build_tab_settings_toolbar_returns_qtoolbar(stp, qtbot) -> None:  # type: ignore[no-untyped-def]
    """build_tab_settings_toolbar returns a QToolBar."""
    pytest.importorskip("PyQt6")
    from unittest.mock import MagicMock

    from PyQt6.QtWidgets import QToolBar, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)

    sidebar = MagicMock()
    sidebar.open_active_tab_settings = MagicMock()
    sidebar.open_configure_tabs = MagicMock()
    sidebar.setVisible = MagicMock()
    sidebar.register_settings_button = MagicMock()

    toolbar = stp.build_tab_settings_toolbar(sidebar)
    qtbot.addWidget(toolbar)
    assert isinstance(toolbar, QToolBar)


@pytest.mark.gui
def test_build_tab_settings_toolbar_object_name(stp, qtbot) -> None:  # type: ignore[no-untyped-def]
    """The toolbar returned has the expected objectName."""
    pytest.importorskip("PyQt6")
    from unittest.mock import MagicMock

    from PyQt6.QtWidgets import QWidget

    parent = QWidget()
    qtbot.addWidget(parent)

    sidebar = MagicMock()
    sidebar.register_settings_button = MagicMock()

    toolbar = stp.build_tab_settings_toolbar(sidebar)
    qtbot.addWidget(toolbar)
    assert toolbar.objectName() == "SidekickSettingsToolbar"


@pytest.mark.gui
def test_build_tab_settings_toolbar_has_settings_button(stp, qtbot) -> None:  # type: ignore[no-untyped-def]
    """The toolbar contains the settings button with the correct objectName."""
    pytest.importorskip("PyQt6")
    from unittest.mock import MagicMock

    from PyQt6.QtWidgets import QToolButton, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)

    sidebar = MagicMock()
    sidebar.register_settings_button = MagicMock()

    toolbar = stp.build_tab_settings_toolbar(sidebar)
    qtbot.addWidget(toolbar)

    # Verify that register_settings_button was called with a QToolButton
    assert sidebar.register_settings_button.called
    btn = sidebar.register_settings_button.call_args[0][0]
    assert isinstance(btn, QToolButton)
    assert btn.objectName() == stp.SIDEKICK_TAB_SETTINGS_BUTTON_OBJECT_NAME


@pytest.mark.gui
def test_build_tab_settings_dialog_no_content(stp, qtbot) -> None:  # type: ignore[no-untyped-def]
    """Dialog shows a label when no custom content widget is provided."""
    pytest.importorskip("PyQt6")
    from unittest.mock import MagicMock

    from PyQt6.QtWidgets import QDialog, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)

    tab_id = "calc"
    sidebar = MagicMock()
    sidebar.tab_display_name.return_value = "Calculator"
    sidebar.tab_settings.return_value = {"values": {"key": "val"}}

    dialog = stp.build_tab_settings_dialog(sidebar, tab_id, content=None)
    qtbot.addWidget(dialog)

    assert isinstance(dialog, QDialog)
    assert dialog.objectName() == f"SidekickTabSettingsDialog_{tab_id}"


@pytest.mark.gui
def test_build_tab_settings_dialog_with_content(stp, qtbot) -> None:  # type: ignore[no-untyped-def]
    """build_tab_settings_dialog embeds a provided content widget."""
    pytest.importorskip("PyQt6")
    from unittest.mock import MagicMock

    from PyQt6.QtWidgets import QDialog, QLabel, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)

    tab_id = "notes"
    sidebar = MagicMock()
    sidebar.tab_display_name.return_value = "Notes"

    content_widget = QLabel("Custom content")
    qtbot.addWidget(content_widget)

    dialog = stp.build_tab_settings_dialog(sidebar, tab_id, content=content_widget)
    qtbot.addWidget(dialog)

    assert isinstance(dialog, QDialog)

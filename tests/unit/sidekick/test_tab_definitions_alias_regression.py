"""Regression tests for the stale ``_tab_definitions`` alias bug.

``UnifiedToolsSidebar`` exposes ``self._tab_definitions`` as a *live alias*
into ``TabCollection._tab_definitions``. ``TabCollection.set_definitions``
previously **rebound** that dict (``self._tab_definitions = {...}``), orphaning
the sidebar's alias to the original empty dict. Visible tabs still rendered
(they use a different, in-place-mutated list), which masked two symptoms:

* the Configure Tabs dialog listed **no** tabs (it reads the alias), and
* the active-tab settings (gear) button was permanently disabled and
  ``open_active_tab_settings`` returned ``False``.

These tests pin the invariant (alias stays live) and both user-visible
symptoms so the regression cannot return.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit, pytest.mark.serial]

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt sidebar tests run serially on Windows.",
        allow_module_level=True,
    )

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
_TEST_PKG = Path(__file__).resolve().parent


def _fix_sidekick_import() -> None:
    shared_str = str(_SHARED)
    if shared_str in sys.path:
        sys.path.remove(shared_str)
    sys.path.insert(0, shared_str)
    top_mod = sys.modules.get("sidekick")
    if top_mod is not None and getattr(top_mod, "__file__", None) is not None:
        if str(_TEST_PKG) in str(Path(top_mod.__file__).resolve().parent):
            del sys.modules["sidekick"]


def _make_sidebar(tmp_path: Path, qtbot: Any) -> Any:
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    return UnifiedToolsSidebar(project_root=tmp_path, parent=win)


def test_tab_definitions_alias_stays_live(tmp_path: Path, qtbot: Any) -> None:
    """Precondition: sidebar constructed with default tabs.
    Postcondition: the sidebar's ``_tab_definitions`` alias is populated and
    is the *same object* as the collection's dict (root-cause invariant)."""
    sidebar = _make_sidebar(tmp_path, qtbot)

    assert sidebar._tab_definitions, "sidebar._tab_definitions is empty (stale alias)"
    # Same object identity — proves set_definitions mutated in place.
    assert sidebar._tab_definitions is sidebar._tab_collection._tab_definitions
    assert set(sidebar._tab_definitions) == set(sidebar.available_tab_ids())


def test_configure_tabs_dialog_lists_every_tab(tmp_path: Path, qtbot: Any) -> None:
    """Precondition: sidebar has default tabs.
    Postcondition: ConfigureTabsDialog shows one checkbox per available tab
    (Bug 3 — the list was empty)."""
    sidebar = _make_sidebar(tmp_path, qtbot)
    from sidekick.ui.tools_sidebar.tab_settings_panel import ConfigureTabsDialog

    dialog = ConfigureTabsDialog(sidebar)
    qtbot.addWidget(dialog)

    assert dialog.checkboxes, "Configure Tabs dialog listed no tabs"
    assert set(dialog.checkboxes) == set(sidebar.available_tab_ids())


def test_settings_button_enabled_for_tab_with_settings(
    tmp_path: Path, qtbot: Any
) -> None:
    """Precondition: the 'chat' tab declares settings and is active.
    Postcondition: the gear button is enabled (Bug 2 — it was always
    disabled because the alias was empty)."""
    sidebar = _make_sidebar(tmp_path, qtbot)
    chat_def = sidebar.get_tab_definition("chat")
    assert chat_def is not None
    assert chat_def.settings is not None

    assert sidebar.set_active_tab("chat")
    sidebar._refresh_settings_button()

    assert sidebar._settings_button is not None
    assert sidebar._settings_button.isEnabled() is True


def test_open_active_tab_settings_succeeds_without_blocking(
    tmp_path: Path, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Precondition: active tab declares settings; the modal dialog builder is
    stubbed so no event loop blocks.
    Postcondition: open_active_tab_settings returns True (Bug 2 — it returned
    False because the definition lookup hit the empty alias)."""
    sidebar = _make_sidebar(tmp_path, qtbot)
    assert sidebar.set_active_tab("chat")

    import sidekick.ui.tools_sidebar.tab_settings_panel as panel

    # A stand-in dialog whose modal-loop method is a no-op.
    fake_dialog = type("_FakeDialog", (), {})()
    fake_dialog.exec = lambda: 0
    monkeypatch.setattr(panel, "build_tab_settings_dialog", lambda *a, **k: fake_dialog)

    assert sidebar.open_active_tab_settings() is True

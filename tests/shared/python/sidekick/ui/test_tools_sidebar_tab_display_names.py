"""Unit tests for ``tools_sidebar.tab_display_names``.

``TabDisplayNameMixin`` maps stable tab ids to persisted user-facing names.
Only two methods touch Qt (``setTabText`` / ``QInputDialog``); the rest is pure
state logic. A lightweight fake host avoids needing a QApplication, and the
input dialog is monkeypatched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sidekick.ui.tools_sidebar import tab_display_names as tdn
from sidekick.ui.tools_sidebar.tab_display_names import TabDisplayNameMixin


class _FakeTabs:
    def __init__(self) -> None:
        self.texts: dict[int, str] = {}

    def setTabText(self, index: int, text: str) -> None:
        self.texts[index] = text


class _FakePopout:
    def __init__(self) -> None:
        self.title = ""

    def setWindowTitle(self, title: str) -> None:
        self.title = title


class _Host(TabDisplayNameMixin):
    def __init__(self) -> None:
        self._state = SimpleNamespace(tab_display_names={})
        self._tab_definitions = {
            "calc": SimpleNamespace(title="Calculator"),
            "notes": SimpleNamespace(title="Notes"),
        }
        self._tab_ids = ["calc", "notes"]
        self._popout_windows: dict = {}
        self.tabs = _FakeTabs()
        self.context_emits = 0

    def _emit_context(self) -> None:
        self.context_emits += 1


def test_default_display_name() -> None:
    assert _Host().tab_display_name("calc") == "Calculator"


def test_unknown_tab_display_name_raises() -> None:
    with pytest.raises(KeyError):
        _Host().tab_display_name("absent")


def test_rename_tab_sets_custom_name_and_refreshes() -> None:
    host = _Host()
    host.rename_tab("calc", "My Calc")
    assert host.tab_display_name("calc") == "My Calc"
    assert host._state.tab_display_names["calc"] == "My Calc"
    assert host.tabs.texts[0] == "My Calc"
    assert host.context_emits == 1


def test_rename_to_default_clears_override() -> None:
    host = _Host()
    host.rename_tab("calc", "Custom")
    host.rename_tab("calc", "Calculator")  # back to default title
    assert "calc" not in host._state.tab_display_names


def test_rename_empty_raises() -> None:
    host = _Host()
    with pytest.raises(ValueError, match="non-empty"):
        host.rename_tab("calc", "   ")


def test_rename_unknown_tab_raises() -> None:
    with pytest.raises(KeyError):
        _Host().rename_tab("absent", "x")


def test_reset_display_name_restores_default() -> None:
    host = _Host()
    host.rename_tab("calc", "Custom")
    host.reset_tab_display_name("calc")
    assert host.tab_display_name("calc") == "Calculator"
    assert host.context_emits == 2


def test_reset_unknown_tab_raises() -> None:
    with pytest.raises(KeyError):
        _Host().reset_tab_display_name("absent")


def test_refresh_updates_popout_window_title() -> None:
    host = _Host()
    popout = _FakePopout()
    host._popout_windows["calc"] = popout
    host.rename_tab("calc", "Renamed")
    assert popout.title == "Sidekick - Renamed"


def test_prompt_rename_applies_when_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    host = _Host()
    monkeypatch.setattr(
        tdn.QtWidgets.QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: ("Prompted", True)),
    )
    host._prompt_rename_tab("calc")
    assert host.tab_display_name("calc") == "Prompted"


def test_prompt_rename_ignored_when_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    host = _Host()
    monkeypatch.setattr(
        tdn.QtWidgets.QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: ("Ignored", False)),
    )
    host._prompt_rename_tab("calc")
    assert host.tab_display_name("calc") == "Calculator"

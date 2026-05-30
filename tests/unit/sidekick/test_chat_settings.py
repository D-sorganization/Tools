"""Tests for the Sidekick Chat-tab settings descriptor and panel.

Covers the fix for the dead sidebar ``⚙`` button on the Chat tab: the
Chat :class:`SidebarTabDefinition` now carries a settings descriptor whose
``widget_factory`` builds :class:`ChatSettingsPanel`.

The pure helpers (``coerce_chat_settings``, ``apply_chat_settings_to_dock``,
``credential_status``) are exercised headlessly; the Qt panel and the live
sidebar integration use the offscreen platform via pytest-qt's ``qapp``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.serial

if sys.platform == "win32" and os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "Qt chat-settings tests run serially on Windows.",
        allow_module_level=True,
    )

_SHARED = Path(__file__).resolve().parents[3] / "src" / "shared" / "python"
_TEST_PKG = Path(__file__).resolve().parent


def _fix_sidekick_import() -> None:
    """Evict the test-package ``sidekick`` shadow and load production code."""
    shared_str = str(_SHARED)
    if shared_str in sys.path:
        sys.path.remove(shared_str)
    sys.path.insert(0, shared_str)
    test_dir = str(_TEST_PKG)
    top_mod = sys.modules.get("sidekick")
    if top_mod is not None:
        top_mod_file = getattr(top_mod, "__file__", None)
        if top_mod_file is not None and test_dir in str(
            Path(top_mod_file).resolve().parent
        ):
            del sys.modules["sidekick"]


def _import_chat_settings() -> Any:
    _fix_sidekick_import()
    from sidekick.ui.tools_sidebar import chat_settings

    return chat_settings


# ─── Fakes ───────────────────────────────────────────────────────


class _FakeDock:
    def __init__(self, *, raise_on_switch: bool = False) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._raise = raise_on_switch

    def switch_provider(self, name: str, model: str, thinking_level: str) -> None:
        if self._raise:
            raise RuntimeError("boom")
        self.calls.append((name, model, thinking_level))


class _FakeCredentialManager:
    def __init__(self) -> None:
        self.keys: dict[str, str] = {}
        self.deleted: list[str] = []

    def store_api_key(self, provider: str, key: str) -> bool:
        self.keys[provider] = key
        return True

    def delete_api_key(self, provider: str) -> bool:
        self.deleted.append(provider)
        return self.keys.pop(provider, None) is not None

    def has_credentials(self, provider: str) -> bool:
        return provider in self.keys


class _FakeSidebar:
    def __init__(self, values: dict[str, Any] | None = None, dock: Any = None) -> None:
        self._values = dict(values or {})
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self._dock = dock

    def tab_settings(self, tab_id: str) -> dict[str, Any]:
        return {"schema_version": 1, "values": dict(self._values)}

    def update_tab_settings(
        self, tab_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        self.updated.append((tab_id, dict(values)))
        return values

    def chat_dock_widget(self) -> Any:
        return self._dock


# ─── Pure helper: coerce_chat_settings ───────────────────────────


def test_defaults_are_independent_copies() -> None:
    cs = _import_chat_settings()
    a = cs.chat_settings_defaults()
    a["provider"] = "mutated"
    b = cs.chat_settings_defaults()
    assert b["provider"] == "ollama"


def test_coerce_none_returns_defaults() -> None:
    cs = _import_chat_settings()
    assert cs.coerce_chat_settings(None) == cs.chat_settings_defaults()


def test_coerce_non_mapping_raises_typeerror() -> None:
    cs = _import_chat_settings()
    with pytest.raises(TypeError):
        cs.coerce_chat_settings(["not", "a", "mapping"])


def test_coerce_invalid_provider_falls_back() -> None:
    cs = _import_chat_settings()
    out = cs.coerce_chat_settings({"provider": "no-such-provider"})
    assert out["provider"] == "ollama"


def test_coerce_normalizes_case_and_whitespace() -> None:
    cs = _import_chat_settings()
    out = cs.coerce_chat_settings(
        {"provider": "  Anthropic ", "thinking_level": "HIGH"}
    )
    assert out["provider"] == "anthropic"
    assert out["thinking_level"] == "high"


def test_coerce_trims_model_and_ignores_blank() -> None:
    cs = _import_chat_settings()
    assert cs.coerce_chat_settings({"model": "  gpt-4o  "})["model"] == "gpt-4o"
    # blank model keeps the default rather than persisting an empty string
    assert cs.coerce_chat_settings({"model": "   "})["model"] == "llama3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (10, 500),  # below floor -> clamped up
        (5_000_000, 1_000_000),  # above ceiling -> clamped down
        ("not-a-number", 8000),  # non-int -> default
        (12345, 12345),  # valid passes through
    ],
)
def test_coerce_clamps_threshold(raw: Any, expected: int) -> None:
    cs = _import_chat_settings()
    out = cs.coerce_chat_settings({"auto_condense_threshold": raw})
    assert out["auto_condense_threshold"] == expected


def test_coerce_drops_unknown_keys() -> None:
    cs = _import_chat_settings()
    out = cs.coerce_chat_settings({"provider": "openai", "bogus": 123})
    assert "bogus" not in out
    assert set(out) == set(cs.chat_settings_defaults())


# ─── Pure helper: apply_chat_settings_to_dock ────────────────────


def test_apply_to_dock_none_returns_false() -> None:
    cs = _import_chat_settings()
    assert cs.apply_chat_settings_to_dock(None, {}) is False


def test_apply_to_dock_without_switch_returns_false() -> None:
    cs = _import_chat_settings()
    assert cs.apply_chat_settings_to_dock(object(), {}) is False


def test_apply_to_dock_calls_switch_provider() -> None:
    cs = _import_chat_settings()
    dock = _FakeDock()
    ok = cs.apply_chat_settings_to_dock(
        dock,
        {"provider": "anthropic", "model": "claude", "thinking_level": "medium"},
    )
    assert ok is True
    assert dock.calls == [("anthropic", "claude", "medium")]


def test_apply_to_dock_swallows_switch_error() -> None:
    cs = _import_chat_settings()
    dock = _FakeDock(raise_on_switch=True)
    assert cs.apply_chat_settings_to_dock(dock, {"provider": "openai"}) is False


# ─── Pure helper: credential_status ──────────────────────────────


def test_credential_status_none_manager_all_false() -> None:
    cs = _import_chat_settings()
    status = cs.credential_status(None)
    assert set(status) == set(cs.CHAT_PROVIDERS)
    assert not any(status.values())


def test_credential_status_queries_manager() -> None:
    cs = _import_chat_settings()
    mgr = _FakeCredentialManager()
    mgr.store_api_key("anthropic", "sk-test")
    status = cs.credential_status(mgr, ("anthropic", "openai"))
    assert status == {"anthropic": True, "openai": False}


def test_credential_status_swallows_backend_error() -> None:
    cs = _import_chat_settings()

    class _Boom:
        def has_credentials(self, provider: str) -> bool:
            raise RuntimeError("keyring down")

    status = cs.credential_status(_Boom(), ("openai",))
    assert status == {"openai": False}


# ─── Descriptor / store round trip ───────────────────────────────


def test_descriptor_round_trips_through_store() -> None:
    cs = _import_chat_settings()
    from sidekick.ui.tools_sidebar.settings import SidebarTabSettingsStore

    class _Def:
        tab_id = cs.CHAT_TAB_ID
        settings = cs.CHAT_TAB_SETTINGS

    class _State:
        tab_settings: dict[str, Any] = {}

    store = SidebarTabSettingsStore([_Def()], _State())
    saved = store.update_settings(
        cs.CHAT_TAB_ID, {"provider": "openai", "auto_condense_threshold": 9000}
    )
    assert saved["values"]["provider"] == "openai"
    assert saved["values"]["auto_condense_threshold"] == 9000
    materialized = store.settings_for(cs.CHAT_TAB_ID)["values"]
    assert materialized["model"] == "llama3"  # default filled in


# ─── Qt panel ────────────────────────────────────────────────────


def test_factory_builds_panel(qapp: Any) -> None:
    cs = _import_chat_settings()
    panel = cs.build_chat_settings_panel(_FakeSidebar(), cs.CHAT_TAB_ID)
    assert isinstance(panel, cs.ChatSettingsPanel)


def test_panel_rejects_missing_sidebar(qapp: Any) -> None:
    cs = _import_chat_settings()
    with pytest.raises(TypeError):
        cs.ChatSettingsPanel(None, cs.CHAT_TAB_ID)


def test_panel_rejects_blank_tab_id(qapp: Any) -> None:
    cs = _import_chat_settings()
    with pytest.raises(ValueError):
        cs.ChatSettingsPanel(_FakeSidebar(), "  ")


def test_panel_loads_current_values(qapp: Any) -> None:
    cs = _import_chat_settings()
    sidebar = _FakeSidebar(
        values={"provider": "anthropic", "model": "claude-x", "agent_mode": "plan"}
    )
    panel = cs.ChatSettingsPanel(
        sidebar, cs.CHAT_TAB_ID, credential_manager=_FakeCredentialManager()
    )
    collected = panel.collect()
    assert collected["provider"] == "anthropic"
    assert collected["model"] == "claude-x"
    assert collected["agent_mode"] == "plan"


def test_panel_save_persists_and_applies(qapp: Any) -> None:
    cs = _import_chat_settings()
    dock = _FakeDock()
    sidebar = _FakeSidebar(values={"provider": "openai", "model": "gpt-x"}, dock=dock)
    panel = cs.ChatSettingsPanel(
        sidebar, cs.CHAT_TAB_ID, credential_manager=_FakeCredentialManager()
    )
    panel._on_save()
    assert sidebar.updated, "settings were not persisted"
    tab_id, values = sidebar.updated[-1]
    assert tab_id == cs.CHAT_TAB_ID
    assert values["provider"] == "openai"
    assert dock.calls == [("openai", "gpt-x", "none")]
    assert "saved" in panel._status_label.text().lower()


def test_panel_reset_restores_defaults(qapp: Any) -> None:
    cs = _import_chat_settings()
    sidebar = _FakeSidebar(values={"provider": "anthropic", "model": "claude-x"})
    panel = cs.ChatSettingsPanel(
        sidebar, cs.CHAT_TAB_ID, credential_manager=_FakeCredentialManager()
    )
    panel._on_reset()
    assert panel.collect() == cs.chat_settings_defaults()


def test_panel_save_key_stores_credential(qapp: Any) -> None:
    cs = _import_chat_settings()
    mgr = _FakeCredentialManager()
    panel = cs.ChatSettingsPanel(_FakeSidebar(), cs.CHAT_TAB_ID, credential_manager=mgr)
    panel._set_combo(panel._key_provider_combo, "anthropic")
    panel._key_input.setText("sk-secret")
    panel._on_save_key()
    assert mgr.keys.get("anthropic") == "sk-secret"
    assert panel._key_input.text() == ""  # cleared after save


def test_panel_clear_key_deletes_credential(qapp: Any) -> None:
    cs = _import_chat_settings()
    mgr = _FakeCredentialManager()
    mgr.store_api_key("openai", "sk-secret")
    panel = cs.ChatSettingsPanel(_FakeSidebar(), cs.CHAT_TAB_ID, credential_manager=mgr)
    panel._set_combo(panel._key_provider_combo, "openai")
    panel._on_clear_key()
    assert "openai" in mgr.deleted
    assert "openai" not in mgr.keys


def test_panel_without_credential_manager_is_safe(
    qapp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cs = _import_chat_settings()
    monkeypatch.setattr(cs, "_default_credential_manager", lambda: None)
    panel = cs.ChatSettingsPanel(
        _FakeSidebar(), cs.CHAT_TAB_ID, credential_manager=None
    )
    # API-key handlers must no-op rather than crash.
    panel._on_save_key()
    panel._on_clear_key()
    assert "unavailable" in panel._key_status_label.text().lower()


# ─── Live sidebar integration ────────────────────────────────────


def _make_sidebar(tmp_path: Path, qtbot: Any) -> Any:
    _fix_sidekick_import()
    from PyQt6 import QtWidgets
    from sidekick.ui.tools_sidebar.sidebar import UnifiedToolsSidebar

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    qtbot.addWidget(win)
    sidebar = UnifiedToolsSidebar(project_root=tmp_path, parent=win)
    return sidebar


def test_chat_tab_declares_settings(tmp_path: Path, qtbot: Any) -> None:
    cs = _import_chat_settings()
    sidebar = _make_sidebar(tmp_path, qtbot)
    definition = sidebar.get_tab_definition("chat")
    assert definition is not None
    assert definition.settings is cs.CHAT_TAB_SETTINGS


def test_chat_dock_widget_accessor(tmp_path: Path, qtbot: Any) -> None:
    sidebar = _make_sidebar(tmp_path, qtbot)
    # The Chat tab is visible by default, so its widget should be built.
    assert sidebar.chat_dock_widget() is sidebar._tab_widgets.get("chat")


def test_gear_opens_chat_settings(tmp_path: Path, qtbot: Any) -> None:
    from unittest.mock import patch

    sidebar = _make_sidebar(tmp_path, qtbot)
    sidebar.set_active_tab("chat")
    with patch(
        "sidekick.ui.tools_sidebar.tab_settings_panel.build_tab_settings_dialog"
    ) as build_dialog:
        from unittest.mock import MagicMock

        build_dialog.return_value = MagicMock()
        assert sidebar.open_active_tab_settings() is True
        build_dialog.assert_called_once()

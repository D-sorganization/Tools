"""Focused tests for shared assistant-panel inline provider controls."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from logging import getLogger
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(ROOT / "src")]
sys.modules["src"] = src_pkg
logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = getLogger
logging_config.setup_logging = lambda *args, **kwargs: None
sys.modules["src.shared.python.logging_pkg"] = logging_pkg
sys.modules["src.shared.python.logging_pkg.logging_config"] = logging_config
config_pkg = types.ModuleType("src.shared.python.config")
environment = types.ModuleType("src.shared.python.config.environment")
environment.get_env = lambda _name, default=None, **_kwargs: default
environment.get_env_float = lambda _name, default=None, **_kwargs: default
sys.modules["src.shared.python.config"] = config_pkg
sys.modules["src.shared.python.config.environment"] = environment

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6.QtWidgets requires display server")
pytest.importorskip("pytestqt", reason="pytest-qt required for widget tests")


@pytest.fixture
def panel_harness(monkeypatch, qtbot):
    from src.shared.python.ai.gui import assistant_panel
    from src.shared.python.ai.gui.assistant_panel import AIAssistantPanel
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        AISettings,
        provider_default_model,
    )

    initial = AISettings(
        provider=AIProvider.OLLAMA,
        model=provider_default_model(AIProvider.OLLAMA),
        chat_mode="ask",
    )
    saved: list[tuple[AIProvider, str, str]] = []
    applied: list[AISettings] = []
    state = {"settings": initial}

    def fake_load(cls):
        return replace(state["settings"])

    def fake_save(self):
        state["settings"] = replace(self)
        saved.append((self.provider, self.model, self.chat_mode))

    def fake_apply_settings(self, settings):
        self._current_settings = settings
        self._sync_header_controls(settings)
        applied.append(replace(settings))

    monkeypatch.setattr(AISettings, "load", classmethod(fake_load))
    monkeypatch.setattr(AISettings, "save", fake_save)
    monkeypatch.setattr(AIAssistantPanel, "apply_settings", fake_apply_settings)
    monkeypatch.setattr(
        assistant_panel.ChatSessionManager,
        "list_sessions",
        lambda self: [],
    )

    panel = AIAssistantPanel()
    qtbot.addWidget(panel)
    return panel, saved, applied, state


def _combo_texts(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _combo_data(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def test_header_populates_provider_model_and_mode_controls(panel_harness):
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        provider_model_names,
    )

    panel, _saved, _applied, _state = panel_harness

    assert _combo_data(panel._provider_combo) == list(AIProvider)
    assert _combo_texts(panel._model_combo) == provider_model_names(AIProvider.OLLAMA)
    assert _combo_data(panel._mode_combo) == ["ask", "diagnose", "agent"]


def test_provider_change_repopulates_persists_and_reconnects(panel_harness):
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        provider_default_model,
        provider_model_names,
    )

    panel, saved, applied, _state = panel_harness

    panel._provider_combo.setCurrentIndex(
        panel._provider_combo.findData(AIProvider.OPENAI)
    )

    assert _combo_texts(panel._model_combo) == provider_model_names(AIProvider.OPENAI)
    assert panel._model_combo.currentText() == provider_default_model(AIProvider.OPENAI)
    assert saved[-1] == (
        AIProvider.OPENAI,
        provider_default_model(AIProvider.OPENAI),
        "ask",
    )
    assert applied[-1].provider == AIProvider.OPENAI
    assert applied[-1].model == provider_default_model(AIProvider.OPENAI)


def test_model_change_persists_and_reconnects(panel_harness):
    from src.shared.python.ai.gui.settings_dialog import AIProvider

    panel, saved, applied, _state = panel_harness
    panel._provider_combo.setCurrentIndex(
        panel._provider_combo.findData(AIProvider.OPENAI)
    )

    panel._model_combo.setCurrentIndex(panel._model_combo.findText("gpt-4o-mini"))

    assert saved[-1] == (AIProvider.OPENAI, "gpt-4o-mini", "ask")
    assert applied[-1].provider == AIProvider.OPENAI
    assert applied[-1].model == "gpt-4o-mini"


def test_mode_change_persists_without_adapter_reconnect(panel_harness):
    from src.shared.python.ai.gui.settings_dialog import AIProvider

    panel, saved, applied, _state = panel_harness
    reconnects_before = len(applied)

    panel._mode_combo.setCurrentIndex(panel._mode_combo.findData("diagnose"))

    assert saved[-1] == (AIProvider.OLLAMA, "llama3.1:8b", "diagnose")
    assert len(applied) == reconnects_before


def test_settings_dialog_loads_inline_provider_model_selection(panel_harness, qtbot):
    from src.shared.python.ai.gui.settings_dialog import AIProvider, AISettingsDialog

    panel, _saved, _applied, state = panel_harness
    panel._provider_combo.setCurrentIndex(
        panel._provider_combo.findData(AIProvider.OPENAI)
    )
    panel._model_combo.setCurrentIndex(panel._model_combo.findText("gpt-4o-mini"))

    dialog = AISettingsDialog()
    qtbot.addWidget(dialog)

    assert dialog._provider_combo.currentData() == AIProvider.OPENAI
    assert dialog._model_combo.currentText() == "gpt-4o-mini"
    assert state["settings"].provider == AIProvider.OPENAI
    assert state["settings"].model == "gpt-4o-mini"


def test_refreshed_models_update_inline_model_dropdown(panel_harness):
    panel, _saved, _applied, _state = panel_harness

    panel._on_chat_models_refreshed(
        [{"name": "llama3.1:8b"}, {"name": "qwen2.5-coder:7b"}]
    )

    assert _combo_texts(panel._model_combo) == ["llama3.1:8b", "qwen2.5-coder:7b"]
    assert panel._model_combo.currentText() == "llama3.1:8b"

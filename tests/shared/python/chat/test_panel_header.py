"""Unit tests for PanelHeaderController."""

from __future__ import annotations

import sys
import types
from logging import getLogger
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", src_pkg)

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = getLogger
logging_config.setup_logging = lambda *a, **k: None
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

config_pkg = types.ModuleType("src.shared.python.config")
environment = types.ModuleType("src.shared.python.config.environment")
environment.get_env = lambda _name, default=None, **_k: default
environment.get_env_float = lambda _name, default=None, **_k: default
sys.modules.setdefault("src.shared.python.config", config_pkg)
sys.modules.setdefault("src.shared.python.config.environment", environment)

pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture
def header(qapp):
    from src.shared.python.ai.gui._panel_header import PanelHeaderController
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        AISettings,
        provider_default_model,
    )

    settings = AISettings(
        provider=AIProvider.OLLAMA,
        model=provider_default_model(AIProvider.OLLAMA),
        chat_mode="ask",
    )
    return PanelHeaderController(settings)


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def test_header_populates_combos(header):
    from src.shared.python.ai.gui.settings_dialog import AIProvider

    data = [
        header.provider_combo.itemData(i) for i in range(header.provider_combo.count())
    ]
    assert data == list(AIProvider)
    assert header.mode_combo.count() == 3
    assert header.access_mode_combo.count() == 3


def test_provider_change_emits_signal(header):
    from src.shared.python.ai.gui.settings_dialog import AIProvider

    received: list = []
    header.provider_changed.connect(received.append)

    idx = header.provider_combo.findData(AIProvider.OPENAI)
    header.provider_combo.setCurrentIndex(idx)

    assert received and received[-1] == AIProvider.OPENAI


def test_update_models_replaces_options(header):
    header.update_models(["alpha", "beta", "gamma"])
    texts = [header.model_combo.itemText(i) for i in range(header.model_combo.count())]
    assert texts == ["alpha", "beta", "gamma"]


def test_set_status_updates_label(header):
    header.set_status("Working...")
    assert header.status_label.text() == "Working..."


def test_auto_index_checkbox_round_trip(header):
    received: list = []
    header.auto_index_toggled.connect(received.append)
    header.auto_index_checkbox.setChecked(True)
    assert header.auto_index_enabled() is True
    assert received and received[-1] is True


def test_sync_controls_does_not_emit(header):
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        AISettings,
        provider_default_model,
    )

    bursts: list = []
    header.provider_changed.connect(bursts.append)
    header.model_changed.connect(bursts.append)
    header.mode_changed.connect(bursts.append)

    header.sync_controls(
        AISettings(
            provider=AIProvider.ANTHROPIC,
            model=provider_default_model(AIProvider.ANTHROPIC),
            chat_mode="agent",
        )
    )

    assert bursts == []

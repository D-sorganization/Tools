"""Unit tests for AdapterLifecycleManager."""

from __future__ import annotations

import sys
import types
from logging import getLogger
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

src_pkg = sys.modules.setdefault("src", types.ModuleType("src"))
src_pkg.__path__ = [str(ROOT / "src")]
shared_pkg = sys.modules.setdefault("src.shared", types.ModuleType("src.shared"))
shared_pkg.__path__ = [str(ROOT / "src" / "shared")]
python_pkg = sys.modules.setdefault(
    "src.shared.python", types.ModuleType("src.shared.python")
)
python_pkg.__path__ = [str(ROOT / "src" / "shared" / "python")]
src_pkg.shared = shared_pkg
shared_pkg.python = python_pkg

logging_pkg = sys.modules.setdefault(
    "src.shared.python.logging_pkg",
    types.ModuleType("src.shared.python.logging_pkg"),
)
logging_config = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
logging_config.get_logger = getLogger
logging_config.setup_logging = lambda *a, **k: None
python_pkg.logging_pkg = logging_pkg
logging_pkg.logging_config = logging_config

config_pkg = sys.modules.setdefault(
    "src.shared.python.config", types.ModuleType("src.shared.python.config")
)
environment = sys.modules.setdefault(
    "src.shared.python.config.environment",
    types.ModuleType("src.shared.python.config.environment"),
)
environment.get_env = lambda _name, default=None, **_k: default
environment.get_env_float = lambda _name, default=None, **_k: default
python_pkg.config = config_pkg
config_pkg.environment = environment

pytest.importorskip("PyQt6.QtCore")


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def test_build_emits_failure_when_no_api_key(monkeypatch, qapp):
    from src.shared.python.ai.gui import _adapter_lifecycle
    from src.shared.python.ai.gui._adapter_lifecycle import AdapterLifecycleManager
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        AISettings,
        provider_default_model,
    )

    monkeypatch.setattr(_adapter_lifecycle, "AISettings", AISettings)
    # stub get_api_key to None so OpenAI build returns None
    settings_module = sys.modules["src.shared.python.ai.gui.settings_dialog"]
    monkeypatch.setattr(settings_module, "get_api_key", lambda *_a, **_k: None)

    mgr = AdapterLifecycleManager()
    seen: list = []
    mgr.adapter_changed.connect(lambda adapter, _id: seen.append(adapter))

    settings = AISettings(
        provider=AIProvider.OPENAI,
        model=provider_default_model(AIProvider.OPENAI),
        chat_mode="ask",
    )
    result = mgr.build(settings)
    assert result is None
    assert seen == [None]


def test_build_emits_system_message_for_unknown_provider(qapp):
    from src.shared.python.ai.gui._adapter_lifecycle import AdapterLifecycleManager
    from src.shared.python.ai.gui.settings_dialog import AISettings

    mgr = AdapterLifecycleManager()
    msgs: list[str] = []
    mgr.system_message.connect(msgs.append)

    # Construct a minimal-shape settings object whose provider is not handled.
    class FakeProvider:
        name = "FAKE"

    fake = AISettings.__new__(AISettings)
    fake.provider = FakeProvider()  # type: ignore[assignment]
    fake.model = "x"
    fake.chat_mode = "ask"

    mgr.build(fake)
    assert any("Could not connect" in m for m in msgs)


def test_construct_returns_none_for_unhandled_provider(qapp):
    from src.shared.python.ai.gui._adapter_lifecycle import AdapterLifecycleManager

    mgr = AdapterLifecycleManager()

    class Settings:
        class P:
            pass

        provider = P()
        model = ""
        chat_mode = "ask"

    assert mgr._construct(Settings()) is None  # noqa: SLF001

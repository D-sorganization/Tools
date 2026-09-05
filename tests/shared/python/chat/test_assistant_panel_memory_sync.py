"""Focused regressions for archived-memory sync behavior."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from logging import getLogger
from pathlib import Path

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
logging_config.setup_logging = lambda *args, **kwargs: None
python_pkg.logging_pkg = logging_pkg
logging_pkg.logging_config = logging_config

config_pkg = sys.modules.setdefault(
    "src.shared.python.config", types.ModuleType("src.shared.python.config")
)
environment = sys.modules.setdefault(
    "src.shared.python.config.environment",
    types.ModuleType("src.shared.python.config.environment"),
)
environment.get_env = lambda _name, default=None, **_kwargs: default
environment.get_env_float = lambda _name, default=None, **_kwargs: default
python_pkg.config = config_pkg
config_pkg.environment = environment

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6.QtWidgets requires display server")


def test_memory_sync_loads_archived_sessions_without_switching_context(
    monkeypatch, tmp_path
) -> None:
    """Archived sync must not emit session-loaded side effects into live UI."""
    from PyQt6.QtWidgets import QApplication

    from src.shared.python.ai.gui import assistant_panel
    from src.shared.python.ai.gui.assistant_panel import AIAssistantPanel
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        AISettings,
        provider_default_model,
    )
    from src.shared.python.ai.memory_manager import MemoryManager
    from src.shared.python.ai.types import ConversationContext

    app = QApplication.instance() or QApplication([])

    initial = AISettings(
        provider=AIProvider.OLLAMA,
        model=provider_default_model(AIProvider.OLLAMA),
        chat_mode="ask",
    )
    state = {"settings": initial}

    def fake_load(cls):
        return replace(state["settings"])

    def fake_save(self):
        state["settings"] = replace(self)

    def fake_apply_settings(self, settings):
        self._current_settings = settings
        self._sync_header_controls(settings)

    monkeypatch.setattr(AISettings, "load", classmethod(fake_load))
    monkeypatch.setattr(AISettings, "save", fake_save)
    monkeypatch.setattr(AIAssistantPanel, "apply_settings", fake_apply_settings)
    monkeypatch.setattr(
        assistant_panel.ChatSessionManager,
        "list_sessions",
        lambda self: [],
    )

    panel = AIAssistantPanel()
    panel._memory_manager = MemoryManager(tmp_path)
    panel._context = ConversationContext(session_id="active-session")
    panel._refresh_prompt_memory()

    archived_context = ConversationContext(session_id="archived-session")
    archived_context.add_user_message("Please remember I prefer concise summaries.")

    monkeypatch.setattr(
        panel._session_manager,
        "list_sessions",
        lambda: [
            {"id": "active-session", "archived": False},
            {"id": "archived-session", "archived": True},
        ],
    )

    load_calls: list[tuple[str, bool]] = []

    def fake_load_session(session_id: str, *, emit: bool = True):
        load_calls.append((session_id, emit))
        if session_id == "archived-session":
            return archived_context
        return None

    monkeypatch.setattr(panel._session_manager, "load_session", fake_load_session)

    panel._on_memory_sync_requested()

    memories = panel._context.metadata["prompt_memory"]["memories"]

    assert load_calls == [("archived-session", False)]
    assert panel._context.session_id == "active-session"
    assert len(memories) == 1
    assert memories[0]["content"] == "Please remember I prefer concise summaries."
    assert memories[0]["source"] == "archived-session:0"

    panel.deleteLater()
    app.processEvents()


def test_assistant_panel_loads_active_session_on_init(monkeypatch) -> None:
    """Regression for #4966: active sessions loaded on init must not crash
    accessing _messages.
    """
    from PyQt6.QtWidgets import QApplication

    from src.shared.python.ai.gui import assistant_panel
    from src.shared.python.ai.gui.assistant_panel import AIAssistantPanel
    from src.shared.python.ai.gui.settings_dialog import (
        AIProvider,
        AISettings,
        provider_default_model,
    )
    from src.shared.python.ai.types import ConversationContext

    app = QApplication.instance() or QApplication([])

    initial = AISettings(
        provider=AIProvider.OLLAMA,
        model=provider_default_model(AIProvider.OLLAMA),
        chat_mode="ask",
    )
    state = {"settings": initial}

    monkeypatch.setattr(
        AISettings, "load", classmethod(lambda cls: replace(state["settings"]))
    )
    monkeypatch.setattr(AISettings, "save", lambda self: None)
    monkeypatch.setattr(
        AIAssistantPanel,
        "apply_settings",
        lambda self, settings: setattr(self, "_current_settings", settings),
    )

    existing_context = ConversationContext(session_id="saved-session-1")
    existing_context.add_user_message("Hello past self")
    existing_context.add_assistant_message("Hello from the past")

    monkeypatch.setattr(
        assistant_panel.ChatSessionManager,
        "list_sessions",
        lambda self: [{"id": "saved-session-1", "archived": False}],
    )
    monkeypatch.setattr(
        assistant_panel.ChatSessionManager,
        "load_session",
        lambda self, session_id, emit=True: (
            self.session_loaded.emit(existing_context) if emit else None,
            existing_context,
        )[1],
    )

    panel = AIAssistantPanel()
    assert panel._context.session_id == "saved-session-1"
    assert panel._messages is not None
    # 1 system welcome message + 2 conversation messages + 1 stretch item
    # Check that MessageWidgets for the 2 conversation messages were added
    widgets = [
        panel._messages.message_layout.itemAt(i).widget()
        for i in range(panel._messages.message_layout.count())
        if panel._messages.message_layout.itemAt(i).widget() is not None
    ]
    assert len(widgets) >= 3  # welcome + 2 messages

    panel.deleteLater()
    app.processEvents()

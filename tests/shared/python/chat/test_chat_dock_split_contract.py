"""Characterization tests for the ChatDockWidget split (Tools #5641).

These pin the public surface that must not change when the dock's method
groups move into ``chat._qt`` mixins: the import path, the constructor
contract (Tools #4897), the Qt signals, the method resolution, and the
per-file line budget UpstreamDrift needs for a verbatim child copy.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from PyQt6.QtWidgets import QApplication, QDockWidget

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = [str(ROOT / "src")]
sys.modules.setdefault("src", _src_pkg)
for _ns in (
    "src.shared",
    "src.shared.python",
    "src.shared.python.chat",
    "src.shared.python.ai",
    "src.shared.python.ai.gui",
):
    _mod = types.ModuleType(_ns)
    _mod.__path__ = [str(ROOT.joinpath(*_ns.split(".")))]
    sys.modules.setdefault(_ns, _mod)
_logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
_logging_cfg = types.ModuleType("src.shared.python.logging_pkg.logging_config")
_logging_cfg.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_cfg.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", _logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", _logging_cfg)

from src.shared.python.chat import _chat_dock_widget_qt as dock_module  # noqa: E402
from src.shared.python.chat._chat_dock_widget_qt import (  # noqa: E402
    ChatConnectionConfig,
    ChatDockWidget,
    ChatIntegrationHooks,
    ChatPresentationConfig,
)

# Held at module level: a dropped QApplication reference crashes silently.
_APP = QApplication.instance() or QApplication([])

CHAT_DIR = ROOT / "src" / "shared" / "python" / "chat"
DOCK_FILE = CHAT_DIR / "_chat_dock_widget_qt.py"
SPLIT_FILES = [
    DOCK_FILE,
    CHAT_DIR / "_qt" / "streaming.py",
    CHAT_DIR / "_qt" / "terminal_mode.py",
    CHAT_DIR / "_qt" / "messaging.py",
]

# Every method that existed on the monolithic class and must still resolve.
METHODS = [
    "set_collapsed", "minimumSizeHint", "connection_diagnostics",
    "refresh_models", "index_codebase", "open_memory_panel",
    "set_send_button_state", "queued_messages", "queued_message_records",
    "steer_to_front", "switch_provider", "load_session", "breadcrumb_labels",
    "_connect", "_on_message", "_on_disconnected", "_enter_thinking_state",
    "_exit_thinking_state", "_flush_chunk_buffer", "_submit_or_queue",
    "_flush_queued_messages", "_on_send", "_on_steer", "_on_stop_agent",
    "_handle_slash_command", "_populate_shell_combo", "_populate_provider_combo",
    "_on_terminal_start", "_on_terminal_stop", "_on_terminal_input",
    "_set_terminal_runtime_available", "_sync_terminal_controls",
    "_append_terminal_line", "_on_mode_changed", "_current_mode",
    "_on_upload", "_on_screenshot", "_on_mic_toggle", "_on_voice_transcription",
    "_on_voice_error", "_send_ws", "_format_agent_label", "_add_bubble",
    "_populate_history", "_scroll_to_bottom", "_get_thread_markdown",
    "_copy_entire_thread", "_export_to_markdown", "_export_thread",
    "_condense_thread", "_build_session_snapshot", "_run_condense_local",
    "_refresh_token_indicator", "_request_review", "_handle_use_session",
    "_add_context_session", "_remove_context_session", "_refresh_breadcrumb",
    "_on_new_chat_clicked", "_on_toggle_history", "showEvent", "closeEvent",
]  # fmt: skip


class _FakeSessionManager:
    def list_sessions(self) -> list[dict[str, Any]]:
        return []


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def isValid(self) -> bool:  # noqa: N802
        return True

    def sendTextMessage(self, text: str) -> None:  # noqa: N802
        self.sent.append(json.loads(text))

    def close(self) -> None:
        pass


@pytest.fixture
def dock() -> ChatDockWidget:
    widget = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="split-test", app_name="split_test", session_id="s-1"
        ),
        integrations=ChatIntegrationHooks(session_manager=_FakeSessionManager()),
    )
    widget._socket = _FakeSocket()
    return widget


def test_public_import_path_reexports_names() -> None:
    for name in (
        "ChatDockWidget",
        "ChatConnectionConfig",
        "ChatPresentationConfig",
        "ChatIntegrationHooks",
        "ChatMessageBubble",
        "HistorySidebar",
        "QWebSocket",
        "QFileDialog",
        "QWidget",
        "QueuedMessage",
        "QueuePanel",
        "ThinkingIndicator",
    ):
        assert hasattr(dock_module, name), name
    assert issubclass(ChatDockWidget, QDockWidget)


def test_constructor_signature_is_pinned() -> None:
    sig = inspect.signature(ChatDockWidget.__init__)
    assert list(sig.parameters) == [
        "self",
        "connection",
        "presentation",
        "integrations",
        "parent",
    ]
    assert all(p.default is None for n, p in sig.parameters.items() if n != "self")


def test_config_dataclass_fields_are_pinned() -> None:
    def fields(cls: type) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    assert fields(ChatConnectionConfig) == [
        "server_url", "session_id", "ws_path_template",
        "app_context", "app_name", "project_root",
    ]  # fmt: skip
    assert fields(ChatPresentationConfig) == [
        "placeholder_text", "accent_color", "theme_provider", "auto_index_on_open",
    ]  # fmt: skip
    assert fields(ChatIntegrationHooks) == [
        "terminal_registry", "workspace_provider", "plot_request_sink",
        "session_manager", "memory_manager_factory",
    ]  # fmt: skip


@pytest.mark.parametrize("name", METHODS)
def test_every_method_still_resolves(name: str) -> None:
    assert callable(getattr(ChatDockWidget, name)), name


def test_properties_and_class_constants_resolve() -> None:
    for prop in ("collapsed", "input_state", "thinking_indicator", "current_provider"):
        assert isinstance(inspect.getattr_static(ChatDockWidget, prop), property)
    assert set(ChatDockWidget._SEND_BTN_STYLES) == {"idle", "awaiting", "stop"}


def test_signals_are_declared_on_the_dock_itself() -> None:
    assert "models_refreshed" in ChatDockWidget.__dict__
    assert "index_status_changed" in ChatDockWidget.__dict__


def test_signal_wiring_models_and_index_status(dock: ChatDockWidget) -> None:
    models: list[Any] = []
    statuses: list[dict[str, Any]] = []
    dock.models_refreshed.connect(models.append)
    dock.index_status_changed.connect(statuses.append)
    dock._on_message(json.dumps({"type": "model_list", "models": ["a", "b"]}))
    dock._on_message(json.dumps({"type": "index_status", "state": "complete"}))
    assert models == [["a", "b"]]
    assert statuses == [{"type": "index_status", "state": "complete"}]


def test_send_queue_and_stream_lifecycle(dock: ChatDockWidget) -> None:
    dock._submit_or_queue("hello")
    assert dock.input_state == "sending"
    assert dock._socket.sent[-1]["action"] == "send"
    assert dock._socket.sent[-1]["app_context"] == "split-test"
    dock._submit_or_queue("steer me")
    assert dock.queued_messages() == ["steer me"]
    assert dock.input_state == "awaiting"
    dock._on_message(json.dumps({"type": "complete"}))
    assert dock.queued_messages() == []
    assert dock._socket.sent[-1]["message"] == "steer me"
    dock._on_message(json.dumps({"type": "complete"}))
    assert dock.input_state == "idle"


def test_send_button_state_contract(dock: ChatDockWidget) -> None:
    dock.set_send_button_state("stop")
    assert dock._send_btn.text() == "Stop"
    with pytest.raises(ValueError):
        dock.set_send_button_state("bogus")


def test_terminal_runtime_toggles_mode_combo(dock: ChatDockWidget) -> None:
    dock._set_terminal_runtime_available(True)
    assert dock._mode_combo.findData("terminal") >= 0
    dock._set_terminal_runtime_available(False)
    assert dock._mode_combo.findData("terminal") < 0


def test_bubble_and_review_helpers(dock: ChatDockWidget) -> None:
    dock._populate_history([{"role": "user", "content": "hi"}])
    assert dock._message_layout.count() == 2  # one bubble + trailing stretch
    assert dock._format_agent_label().startswith("Agent")
    dock._request_review()
    assert dock._socket.sent[-1]["action"] == "request_review"


def test_split_files_stay_under_budget() -> None:
    for path in SPLIT_FILES:
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 1200, f"{path.name} has {lines} lines"
    assert len(DOCK_FILE.read_text(encoding="utf-8").splitlines()) <= 900


def test_mixins_are_ahead_of_qdockwidget_in_mro() -> None:
    names = [c.__name__ for c in ChatDockWidget.__mro__]
    for mixin in ("StreamingMixin", "TerminalModeMixin", "MessagingMixin"):
        assert names.index(mixin) < names.index("QDockWidget")

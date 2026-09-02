"""Tests for ChatDockWidget collapsed/expanded state and settings."""

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication

# Register src namespace packages so dotted imports resolve correctly
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
    _parts = _ns.split(".")
    _mod = types.ModuleType(_ns)
    _mod.__path__ = [str(ROOT.joinpath(*_parts))]
    sys.modules.setdefault(_ns, _mod)

import logging

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

from src.shared.python.chat._chat_dock_widget_qt import (
    ChatConnectionConfig,
    ChatDockWidget,
    ChatIntegrationHooks,
)


class _FakeSessionManager:
    def list_sessions(self) -> list[dict[str, Any]]:
        return []


def test_chat_dock_widget_accepts_injected_session_manager() -> None:
    """Hosts can provide session persistence without importing ai at module import."""
    _app = QApplication.instance() or QApplication([])
    manager = _FakeSessionManager()

    widget = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="test",
            app_name="test_chat_session_injection",
        ),
        integrations=ChatIntegrationHooks(session_manager=manager),
    )

    assert widget._session_manager is manager


def test_chat_dock_widget_collapse() -> None:
    """Test ChatDockWidget collapse and size hint overrides."""
    _app = QApplication.instance() or QApplication([])
    widget = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))

    # Test initial collapsed state
    initial_collapsed = widget.collapsed
    assert not initial_collapsed

    # Test collapse
    widget.set_collapsed(True)
    collapsed_after = widget.collapsed
    assert collapsed_after

    # All major widgets should be hidden
    assert widget._status_label.isHidden()
    assert widget._tools_btn.isHidden()
    assert widget._input_edit.isHidden()
    assert widget._send_btn.isHidden()

    # Test expand
    widget.set_collapsed(False)
    expanded_after = widget.collapsed
    assert not expanded_after
    assert not widget._status_label.isHidden()
    assert not widget._tools_btn.isHidden()
    assert not widget._input_edit.isHidden()
    assert not widget._send_btn.isHidden()

    # Test minimumSizeHint
    widget.set_collapsed(True)
    assert widget.minimumSizeHint() == QSize(56, 0)

    widget.set_collapsed(False)
    assert widget.minimumSizeHint() == QSize(320, 0)


def test_chat_dock_widget_switch_provider() -> None:
    """Test switch_provider and its DbC contracts."""
    _app = QApplication.instance() or QApplication([])
    widget = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))

    # Initial state
    assert widget._current_provider in ("gemini", "ollama")

    # Valid switch
    widget.switch_provider("openai", "gpt-4", "medium")
    assert widget._current_provider == "openai"
    assert widget._current_model == "gpt-4"
    assert widget._current_thinking_level == "medium"

    # Preconditions check
    with pytest.raises(ValueError, match="switch_provider: name must be non-empty"):
        widget.switch_provider("", "gpt-4", "medium")

    with pytest.raises(ValueError, match="switch_provider: model must be non-empty"):
        widget.switch_provider("openai", "   ", "medium")

    with pytest.raises(
        ValueError, match="switch_provider: thinking_level must be a string"
    ):
        widget.switch_provider("openai", "gpt-4", 123)  # type: ignore

    with pytest.raises(ValueError, match="switch_provider: thinking_level.*not in"):
        widget.switch_provider("openai", "gpt-4", "invalid_level")


def test_chat_dock_widget_apply_settings_change() -> None:
    """Test _apply_settings_change and _combo_for_field methods."""
    _app = QApplication.instance() or QApplication([])
    widget = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))

    # Test combo mapping
    assert widget._combo_for_field("provider") is widget._ai_provider_combo
    assert widget._combo_for_field("model") is widget._ai_model_combo
    assert widget._combo_for_field("thinking") is widget._ai_thinking_combo

    with pytest.raises(ValueError, match="unknown field"):
        widget._combo_for_field("invalid_field")

    # Test settings changes
    widget._apply_settings_change("provider", "anthropic")
    assert widget._current_provider == "anthropic"

    widget._apply_settings_change("model", "claude-3")
    assert widget._current_model == "claude-3"

    widget._apply_settings_change("thinking", "high")
    assert widget._current_thinking_level == "high"

    # Preconditions
    with pytest.raises(ValueError, match="unknown field"):
        widget._apply_settings_change("invalid_field", "value")

    with pytest.raises(ValueError, match="must be non-empty"):
        widget._apply_settings_change("provider", "  ")


def test_chat_dock_widget_mode_handling() -> None:
    """Test modes (chat vs terminal) and status syncing."""
    _app = QApplication.instance() or QApplication([])
    widget = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))

    assert widget._mode_combo.findData("terminal") == -1
    assert widget._current_mode() == "chat"

    widget._set_terminal_runtime_available(True)

    # Switch to terminal mode
    widget._mode_combo.setCurrentIndex(widget._mode_combo.findData("terminal"))
    assert widget._current_mode() == "terminal"

    # Switch back to chat mode
    widget._mode_combo.setCurrentIndex(widget._mode_combo.findData("chat"))
    assert widget._current_mode() == "chat"

    widget._set_terminal_runtime_available(False)
    assert widget._mode_combo.findData("terminal") == -1


def test_chat_dock_widget_session_info_enables_terminal_mode() -> None:
    """Terminal mode is available only when the server advertises a runtime."""
    _app = QApplication.instance() or QApplication([])
    widget = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))

    assert widget._mode_combo.findData("terminal") == -1

    widget._on_message(
        '{"type": "session_info", "session_id": "s1", '
        '"capabilities": {"terminal_runtime": true}}'
    )

    assert widget._mode_combo.findData("terminal") >= 0


def test_chat_dock_widget_adapter_refreshes() -> None:
    """Test refreshing combos dynamically when active adapter changes."""
    _app = QApplication.instance() or QApplication([])
    widget = ChatDockWidget(connection=ChatConnectionConfig(app_context="test"))

    mock_adapter = MagicMock()
    mock_adapter.list_models.return_value = ["model1", "model2"]

    mock_caps = MagicMock()
    mock_level = MagicMock()
    mock_level.label = "Low Limit"
    mock_level.name = "low"
    mock_caps.available_levels = [mock_level]
    mock_adapter.thinking_capabilities.return_value = mock_caps

    with patch.object(widget, "_get_active_ai_adapter", return_value=mock_adapter):
        widget._refresh_ai_model_combo()
        assert widget._ai_model_combo.count() == 2
        assert widget._ai_model_combo.itemText(0) == "model1"
        assert widget._ai_model_combo.itemData(0) == "model1"

        widget._refresh_ai_thinking_combo()
        assert widget._ai_thinking_combo.count() == 1
        assert widget._ai_thinking_combo.itemText(0) == "Low Limit"
        assert widget._ai_thinking_combo.itemData(0) == "low"

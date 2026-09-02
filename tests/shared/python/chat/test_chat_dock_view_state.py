"""Regression tests for Tools #3332 chat dock view-state ownership."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.shared.python.chat._chat_dock_widget_qt import (
    ChatConnectionConfig,
    ChatDockWidget,
)
from src.shared.python.chat._qt.ui_builder import ChatDockView

ROOT = Path(__file__).resolve().parents[4]
UI_BUILDER = ROOT / "src" / "shared" / "python" / "chat" / "_qt" / "ui_builder.py"
AI_DROPDOWNS = ROOT / "src" / "shared" / "python" / "chat" / "_qt" / "ai_dropdowns.py"


def test_ui_builder_uses_view_state_instead_of_private_dock_injection() -> None:
    source = UI_BUILDER.read_text(encoding="utf-8")

    assert not re.search(r"dock\._[A-Za-z][A-Za-z0-9_]*\s*=", source)
    assert "def mirror_chat_dock_view" in source


def test_ai_dropdowns_use_view_state_for_combo_widgets() -> None:
    source = AI_DROPDOWNS.read_text(encoding="utf-8")

    assert "view.ai_provider_combo" in source
    assert "view.ai_model_combo" in source
    assert "view.ai_thinking_combo" in source
    assert not re.search(r"dock\._ai_(provider|model|thinking)_combo", source)


def test_chat_dock_widget_exposes_populated_view_state() -> None:
    _app = QApplication.instance() or QApplication([])
    widget = ChatDockWidget(
        connection=ChatConnectionConfig(
            app_context="test", app_name="test_chat_view_state"
        )
    )

    assert isinstance(widget._view, ChatDockView)
    for field_info in fields(ChatDockView):
        value = getattr(widget._view, field_info.name)
        assert value is not None, f"ChatDockView.{field_info.name} was not populated"
        assert getattr(widget, f"_{field_info.name}") is value

"""Focused regressions for chat history sidebar interactions."""

from __future__ import annotations

import sys
import types
from datetime import datetime
from logging import getLogger
from pathlib import Path

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

import pytest

pytest.importorskip("PyQt6.QtWidgets", reason="PyQt6.QtWidgets requires display server")


def test_sidebar_refresh_keeps_typed_session_items(tmp_path: Path, monkeypatch):
    """Sidebar selections keep the session metadata object expected by menus."""
    from PyQt6.QtWidgets import QApplication

    from src.shared.python.ai.gui.history_sidebar import (
        ChatHistorySidebar,
        SessionListWidgetItem,
    )
    from src.shared.python.ai.gui.session_manager import ChatSessionManager

    manager = ChatSessionManager(storage_dir=tmp_path)
    monkeypatch.setattr(
        manager,
        "list_sessions",
        lambda: [
            {
                "id": "session-1",
                "title": "Session One",
                "snippet": "Please remember concise summaries.",
                "archived": False,
                "timestamp": datetime(2026, 5, 13, 12, 0, 0),
            }
        ],
    )

    app = QApplication.instance() or QApplication([])
    sidebar = ChatHistorySidebar(manager)

    selected: list[str] = []
    sidebar.session_selected.connect(selected.append)

    item = sidebar._active_list.item(0)

    assert isinstance(item, SessionListWidgetItem)

    sidebar._on_item_clicked(item)

    assert selected == ["session-1"]

    sidebar.deleteLater()
    app.processEvents()

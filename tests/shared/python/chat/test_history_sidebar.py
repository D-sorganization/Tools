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

shared_pkg = types.ModuleType("src.shared")
shared_pkg.__path__ = [str(ROOT / "src" / "shared")]
python_pkg = types.ModuleType("src.shared.python")
python_pkg.__path__ = [str(ROOT / "src" / "shared" / "python")]
ai_pkg = types.ModuleType("src.shared.python.ai")
ai_pkg.__path__ = [str(ROOT / "src" / "shared" / "python" / "ai")]
ai_gui_pkg = types.ModuleType("src.shared.python.ai.gui")
ai_gui_pkg.__path__ = [str(ROOT / "src" / "shared" / "python" / "ai" / "gui")]
sys.modules["src.shared"] = shared_pkg
sys.modules["src.shared.python"] = python_pkg
sys.modules["src.shared.python.ai"] = ai_pkg
sys.modules["src.shared.python.ai.gui"] = ai_gui_pkg

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = getLogger
logging_config.setup_logging = lambda *args, **kwargs: None
sys.modules["src.shared.python.logging_pkg"] = logging_pkg
sys.modules["src.shared.python.logging_pkg.logging_config"] = logging_config

config_pkg = types.ModuleType("src.shared.python.config")
environment = types.ModuleType("src.shared.python.config.environment")
environment.get_env = lambda _name, default=None: default
environment.get_env_float = lambda _name, default=0.0: default
sys.modules["src.shared.python.config"] = config_pkg
sys.modules["src.shared.python.config.environment"] = environment

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


def test_sidebar_rows_wrap_text_and_expose_inline_actions(tmp_path: Path, monkeypatch):
    """History rows keep long text readable and expose archive/delete buttons."""
    from PyQt6.QtWidgets import QApplication

    from src.shared.python.ai.gui.history_sidebar import (
        ChatHistorySidebar,
        SessionListItemWidget,
    )
    from src.shared.python.ai.gui.session_manager import ChatSessionManager

    manager = ChatSessionManager(storage_dir=tmp_path)
    monkeypatch.setattr(
        manager,
        "list_sessions",
        lambda: [
            {
                "id": "session-long",
                "title": (
                    "A very long chat history title that should wrap instead "
                    "of clipping"
                ),
                "snippet": (
                    "This is a long archived chat snippet with enough detail to "
                    "prove that the row widget uses wrapped labels."
                ),
                "archived": False,
                "timestamp": datetime(2026, 5, 13, 12, 0, 0),
            }
        ],
    )

    archived: list[tuple[str, bool]] = []
    deleted: list[str] = []
    monkeypatch.setattr(
        manager,
        "archive_session",
        lambda sid, flag: archived.append((sid, flag)),
    )

    app = QApplication.instance() or QApplication([])
    sidebar = ChatHistorySidebar(manager)
    monkeypatch.setattr(sidebar, "_confirm_delete", deleted.append)

    item = sidebar._active_list.item(0)
    row = sidebar._active_list.itemWidget(item)

    assert isinstance(row, SessionListItemWidget)
    assert row._title_label.wordWrap()
    assert row._snippet_label.wordWrap()
    assert item.sizeHint().height() >= 92

    assert row._archive_toggle_btn is not None
    assert row._delete_btn is not None
    assert row._archive_toggle_btn.autoRaise()
    assert row._delete_btn.autoRaise()

    row._archive_toggle_btn.click()
    row._delete_btn.click()

    assert archived == [("session-long", True)]
    assert deleted == ["session-long"]

    sidebar.deleteLater()
    app.processEvents()

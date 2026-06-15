"""Tests for ChatHistorySidebar widget collapsed/expanded functionality.

Also tests session management.
"""

import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QMessageBox

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
    _registered = sys.modules.setdefault(_ns, _mod)
    _parent_name, _, _child_name = _ns.rpartition(".")
    if _parent_name:
        setattr(sys.modules[_parent_name], _child_name, _registered)

import logging

logging_pkg = types.ModuleType("src.shared.python.logging_pkg")
logging_config = types.ModuleType("src.shared.python.logging_pkg.logging_config")
logging_config.get_logger = logging.getLogger  # type: ignore[attr-defined]
logging_config.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("src.shared.python.logging_pkg", logging_pkg)
sys.modules.setdefault("src.shared.python.logging_pkg.logging_config", logging_config)

from src.shared.python.ai.gui.history_sidebar import (
    ChatHistorySidebar,
    SessionListItemWidget,
    SessionListWidgetItem,
)
from src.shared.python.ai.gui.session_manager import ChatSessionManager


def test_session_list_widget_item_no_timestamp() -> None:
    """Test SessionListWidgetItem behavior with missing or invalid timestamp."""
    _app = QApplication.instance() or QApplication([])
    item = SessionListWidgetItem({"title": "No Time"})
    assert "No Time" in item.text()
    assert "Unknown" in item.text()


def test_chat_history_sidebar_collapse() -> None:
    """Test ChatHistorySidebar collapse and size hint overrides."""
    _app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ChatSessionManager(storage_dir=Path(tmpdir))
        sidebar = ChatHistorySidebar(manager)

        # Test initial collapsed state
        initial_collapsed = sidebar.collapsed
        assert not initial_collapsed

        # Test collapsing
        sidebar.set_collapsed(True)
        collapsed_after = sidebar.collapsed
        assert collapsed_after
        assert sidebar._header.isHidden()
        assert sidebar._tabs.isHidden()

        # Test expanding
        sidebar.set_collapsed(False)
        expanded_after = sidebar.collapsed
        assert not expanded_after
        assert not sidebar._header.isHidden()
        assert not sidebar._tabs.isHidden()

        # Test minimumSizeHint
        sidebar.set_collapsed(True)
        assert sidebar.minimumSizeHint() == QSize(56, 0)

        sidebar.set_collapsed(False)
        assert sidebar.minimumSizeHint() == QSize(320, 0)


def test_chat_history_sidebar_lists_and_actions() -> None:
    """Test ChatHistorySidebar handles session loading, styling, and action triggers."""
    _app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        # Write mock session files: one active, one archived
        active_sess = {
            "session_id": "active_id",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello user active chat",
                    "timestamp": "2026-05-21T12:00:00",
                }
            ],
            "metadata": {"title": "Active Chat Title", "archived": False},
        }
        archived_sess = {
            "session_id": "archived_id",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello user archived chat",
                    "timestamp": "2026-05-21T12:05:00",
                }
            ],
            "metadata": {"title": "Archived Chat Title", "archived": True},
        }

        with open(storage_path / "active_id.json", "w", encoding="utf-8") as f:
            json.dump(active_sess, f)
        with open(storage_path / "archived_id.json", "w", encoding="utf-8") as f:
            json.dump(archived_sess, f)

        manager = ChatSessionManager(storage_dir=storage_path)
        sidebar = ChatHistorySidebar(manager)

        # Assert active list and archive list are populated
        assert sidebar._active_list.count() == 1
        assert sidebar._archive_list.count() == 1

        active_item = sidebar._active_list.item(0)
        assert isinstance(active_item, SessionListWidgetItem)
        assert active_item.session_data["id"] == "active_id"

        active_widget = sidebar._active_list.itemWidget(active_item)
        assert isinstance(active_widget, SessionListItemWidget)
        assert active_widget._title_label.text() == "Active Chat Title"

        # Test selecting item
        selected_sessions: list[str] = []
        sidebar.session_selected.connect(selected_sessions.append)
        sidebar._on_item_clicked(active_item)
        assert selected_sessions == ["active_id"]

        # Test archive toggle click
        with patch.object(manager, "archive_session") as mock_archive:
            assert active_widget._archive_toggle_btn is not None
            active_widget._archive_toggle_btn.click()
            mock_archive.assert_called_once_with("active_id", True)

        # Test delete click (No response)
        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.No
        ):
            with patch.object(manager, "delete_session") as mock_delete:
                assert active_widget._delete_btn is not None
                active_widget._delete_btn.click()
                mock_delete.assert_not_called()

        # Test delete click (Yes response)
        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
        ):
            with patch.object(manager, "delete_session") as mock_delete:
                assert active_widget._delete_btn is not None
                active_widget._delete_btn.click()
                mock_delete.assert_called_once_with("active_id")


def test_chat_history_sidebar_context_menu() -> None:
    """Test right click context menu action mapping and execution."""
    _app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)
        active_sess = {
            "session_id": "active_id",
            "messages": [
                {"role": "user", "content": "test", "timestamp": "2026-05-21T12:00:00"}
            ],
            "metadata": {"title": "Active Chat", "archived": False},
        }
        with open(storage_path / "active_id.json", "w", encoding="utf-8") as f:
            json.dump(active_sess, f)

        manager = ChatSessionManager(storage_dir=storage_path)
        sidebar = ChatHistorySidebar(manager)

        # Call context menu with item
        active_item = sidebar._active_list.item(0)

        # We mock QMenu.exec to not show menu blockingly but execute target triggers
        mock_menu = MagicMock()
        mock_action = MagicMock()
        mock_menu.addAction.return_value = mock_action

        with patch(
            "src.shared.python.ai.gui.history_sidebar.QMenu", return_value=mock_menu
        ):
            sidebar._show_context_menu(
                sidebar._active_list.visualItemRect(active_item).center(),
                sidebar._active_list,
            )

            # Verify we created the menu and added expected actions
            assert mock_menu.addAction.call_count >= 2
            mock_menu.exec.assert_called_once()

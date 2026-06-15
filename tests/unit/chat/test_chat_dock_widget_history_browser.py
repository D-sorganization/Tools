"""RED tests for Tools issue #2872 chat dock widget UI additions.

The tests bypass ``QDockWidget.__init__`` so they run headless on CI
where no display server exists.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("PyQt6.QtWebSockets")

from chat._chat_dock_widget_qt import ChatDockWidget  # noqa: E402


def _build_widget() -> ChatDockWidget:
    widget = ChatDockWidget.__new__(ChatDockWidget)
    widget._app_context = "test"
    widget._app_name = "test_app"
    widget._loaded_context_sessions = []
    widget._session_manager = MagicMock()
    widget._breadcrumb_widget = None
    widget._breadcrumb_callbacks = []
    widget._search_query = ""
    return widget


# ───────────────────────── sidebar split ───────────────────────────


class TestHistorySidebarSections:
    def test_sidebar_has_active_and_archived_sections(self) -> None:
        from chat._chat_dock_widget_qt import HistorySidebar

        manager = MagicMock()
        manager.list_sessions.return_value = [
            {"id": "a", "title": "Active 1", "archived": False, "snippet": "x"},
            {"id": "b", "title": "Archived 1", "archived": True, "snippet": "y"},
        ]
        manager.is_archived.side_effect = lambda sid: sid == "b"

        with patch(
            "chat._chat_dock_widget_qt.QWidget.__init__",
            return_value=None,
        ):
            sidebar = HistorySidebar.__new__(HistorySidebar)
            sidebar._manager = manager
            sidebar._active_ids = []
            sidebar._archived_ids = []
            sidebar._refresh_data()

        assert "a" in sidebar._active_ids
        assert "b" in sidebar._archived_ids


# ───────────────────────── search filter ───────────────────────────


class TestSearchFilter:
    def test_search_input_filters_sessions(self) -> None:
        from chat._chat_dock_widget_qt import HistorySidebar

        manager = MagicMock()
        manager.search_sessions.return_value = [
            {
                "id": "a",
                "title": "pendulum tuning",
                "archived": False,
                "snippet": "x",
            }
        ]
        manager.is_archived.return_value = False

        with patch(
            "chat._chat_dock_widget_qt.QWidget.__init__",
            return_value=None,
        ):
            sidebar = HistorySidebar.__new__(HistorySidebar)
            sidebar._manager = manager
            sidebar._active_ids = []
            sidebar._archived_ids = []
            sidebar.set_search_query("pendulum")

        manager.search_sessions.assert_called_once()
        assert sidebar._active_ids == ["a"]


# ───────────────────────── restore button ──────────────────────────


class TestRestoreButton:
    def test_restore_button_calls_unarchive(self) -> None:
        from chat._chat_dock_widget_qt import HistorySidebar

        manager = MagicMock()
        with patch(
            "chat._chat_dock_widget_qt.QWidget.__init__",
            return_value=None,
        ):
            sidebar = HistorySidebar.__new__(HistorySidebar)
            sidebar._manager = manager
            sidebar._active_ids = []
            sidebar._archived_ids = ["archived-id"]
            sidebar._on_restore_clicked("archived-id")

        manager.unarchive_session.assert_called_once_with("archived-id")


# ───────────────────────── export click ────────────────────────────


class TestExportButton:
    def test_export_button_triggers_export_call(self, tmp_path: Path) -> None:
        from chat._chat_dock_widget_qt import HistorySidebar

        manager = MagicMock()
        manager.export_session.return_value = "# exported"

        with patch(
            "chat._chat_dock_widget_qt.QWidget.__init__",
            return_value=None,
        ):
            sidebar = HistorySidebar.__new__(HistorySidebar)
            sidebar._manager = manager
            sidebar._active_ids = []
            sidebar._archived_ids = []

        target = tmp_path / "out.md"
        with patch(
            "chat._chat_dock_widget_qt.QFileDialog.getSaveFileName",
            return_value=(str(target), "Markdown (*.md)"),
        ):
            sidebar._on_export_clicked("sid-1", "markdown")

        manager.export_session.assert_called_once_with("sid-1", "markdown")
        assert target.read_text(encoding="utf-8") == "# exported"


# ───────────────────────── slash command ───────────────────────────


class TestSlashUseSession:
    def test_slash_use_session_command_parses_id(self) -> None:
        widget = _build_widget()
        widget._session_manager.list_sessions.return_value = [
            {"id": "abc-123", "title": "Pendulum", "archived": False, "snippet": ""}
        ]
        widget._session_manager.load_context_from.return_value = "ctx"

        resolved = widget._resolve_use_session_target("abc-123")

        assert resolved == "abc-123"

    def test_slash_use_session_command_parses_title(self) -> None:
        widget = _build_widget()
        widget._session_manager.list_sessions.return_value = [
            {
                "id": "abc-123",
                "title": "Pendulum Run",
                "archived": False,
                "snippet": "",
            }
        ]

        resolved = widget._resolve_use_session_target("Pendulum Run")

        assert resolved == "abc-123"


# ───────────────────────── breadcrumbs ─────────────────────────────


class TestBreadcrumb:
    def test_breadcrumb_shows_loaded_sessions(self) -> None:
        widget = _build_widget()
        widget._session_manager.list_sessions.return_value = [
            {"id": "s1", "title": "Pendulum", "archived": False, "snippet": ""},
            {"id": "s2", "title": "Reactor", "archived": False, "snippet": ""},
        ]
        widget._session_manager.load_context_from.return_value = "CONTEXT"

        widget._add_context_session("s1")
        widget._add_context_session("s2")

        labels = widget.breadcrumb_labels()
        assert "Pendulum" in labels
        assert "Reactor" in labels

    def test_breadcrumb_remove_chip_drops_session(self) -> None:
        widget = _build_widget()
        widget._session_manager.list_sessions.return_value = [
            {"id": "s1", "title": "Pendulum", "archived": False, "snippet": ""},
            {"id": "s2", "title": "Reactor", "archived": False, "snippet": ""},
        ]
        widget._session_manager.load_context_from.return_value = "CONTEXT"

        widget._add_context_session("s1")
        widget._add_context_session("s2")
        widget._remove_context_session("s1")

        assert widget._loaded_context_sessions == ["s2"]
        labels = widget.breadcrumb_labels()
        assert "Pendulum" not in labels
        assert "Reactor" in labels

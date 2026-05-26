# ruff: noqa: E501
"""``HistorySidebar`` widget — Tools issue #2872 session-history pane.

Extracted from the monolithic ``_chat_dock_widget_qt`` module. The public
name is re-exported from the parent module so the historical import path
(``from chat._chat_dock_widget_qt import HistorySidebar``) keeps working.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QFileDialog, QWidget

logger = logging.getLogger(__name__)


class HistorySidebar(QWidget):
    """Sidebar listing active + archived sessions with search/export.

    The widget talks only to a :class:`ChatSessionManager` (Law of
    Demeter — no direct ``session.context.metadata`` access).

    Attributes:
        _manager: Session manager used for every persistence call.
        _active_ids: Ordered list of active session ids currently shown.
        _archived_ids: Ordered list of archived session ids currently shown.
    """

    def __init__(
        self,
        manager: Any,
        parent: QWidget | None = None,
    ) -> None:
        if manager is None:  # DbC precondition
            raise ValueError("manager must be provided")
        super().__init__(parent)
        self._manager = manager
        self._active_ids: list[str] = []
        self._archived_ids: list[str] = []
        self._refresh_data()

    def _refresh_data(self) -> None:
        """Reload the active/archived id lists from the manager."""
        self._active_ids = []
        self._archived_ids = []
        for info in self._manager.list_sessions():
            sid = info.get("id")
            if sid is None:
                continue
            try:
                archived = self._manager.is_archived(sid)
            except KeyError:
                continue
            if archived:
                self._archived_ids.append(sid)
            else:
                self._active_ids.append(sid)

    def set_search_query(self, query: str) -> None:
        """Apply a search query and re-bucket results.

        Args:
            query: Substring query (case-insensitive). Empty string
                clears the filter.

        Pre:
            ``query`` is a string (may be empty).
        Post:
            ``self._active_ids`` and ``self._archived_ids`` reflect the
            current filter state.
        """
        if query is None:
            raise ValueError("query must be provided")
        if not query.strip():
            self._refresh_data()
            return
        hits = self._manager.search_sessions(query)
        self._active_ids = []
        self._archived_ids = []
        for info in hits:
            sid = info.get("id")
            if sid is None:
                continue
            if info.get("archived"):
                self._archived_ids.append(sid)
            else:
                self._active_ids.append(sid)

    def _on_restore_clicked(self, session_id: str) -> None:
        """Restore an archived session via the manager (LOD-clean)."""
        self._manager.unarchive_session(session_id)
        self._refresh_data()

    def _on_export_clicked(self, session_id: str, fmt: str) -> None:
        """Export ``session_id`` as ``fmt`` to a user-selected file."""
        if fmt not in ("markdown", "json"):
            raise ValueError(f"Unsupported export format: {fmt!r}")
        suffix = "md" if fmt == "markdown" else "json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session",
            f"{session_id}.{suffix}",
            (
                "Markdown Files (*.md);;All Files (*)"
                if fmt == "markdown"
                else "JSON Files (*.json);;All Files (*)"
            ),
        )
        if not path:
            return
        try:
            payload = self._manager.export_session(session_id, fmt)
            Path(path).write_text(payload, encoding="utf-8")
        except (OSError, KeyError, ValueError):
            logger.exception("export of session %s failed", session_id)

# ruff: noqa: E501
"""``HistorySidebar`` widget — Tools issue #2872 session-history pane.

Extracted from the monolithic ``_chat_dock_widget_qt`` module. The public
name is re-exported from the parent module so the historical import path
(``from chat._chat_dock_widget_qt import HistorySidebar``) keeps working.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .styling import get_theme_colors

logger = logging.getLogger(__name__)


class SessionListWidgetItem(QListWidgetItem):
    """Custom list item to hold session metadata."""

    def __init__(self, session_data: dict[str, Any]) -> None:
        super().__init__()
        self.session_data = session_data

        # Format the text
        title = session_data.get("title", "New Chat")
        dt = session_data.get("timestamp")

        if dt and dt != datetime.min:
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = "Unknown"

        self.setText(f"{title}\n{time_str}")
        self.setToolTip(session_data.get("snippet", ""))


class SessionListItemWidget(QFrame):
    """Readable wrapped chat-history row with icon-only actions."""

    def __init__(
        self,
        session_data: dict[str, Any],
        *,
        archived: bool,
        on_archive_toggle: Any,
        on_delete: Any,
        colors: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session_data = session_data
        self._colors = colors
        self._archive_toggle_btn: QToolButton | None = None
        self._delete_btn: QToolButton | None = None
        self._setup_ui(
            archived=archived,
            on_archive_toggle=on_archive_toggle,
            on_delete=on_delete,
        )

    def _setup_ui(
        self,
        *,
        archived: bool,
        on_archive_toggle: Any,
        on_delete: Any,
    ) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(8)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)

        title = str(self.session_data.get("title") or "New Chat")
        snippet = str(self.session_data.get("snippet") or "")
        dt = self.session_data.get("timestamp")
        if dt and dt != datetime.min:
            time_str = dt.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = "Unknown"

        self._title_label = QLabel(title)
        self._title_label.setWordWrap(True)
        self._title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        text_layout.addWidget(self._title_label)

        self._meta_label = QLabel(time_str)
        text_layout.addWidget(self._meta_label)

        self._snippet_label = QLabel(snippet)
        self._snippet_label.setWordWrap(True)
        self._snippet_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        text_layout.addWidget(self._snippet_label)

        layout.addLayout(text_layout, stretch=1)

        actions = QVBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        self._archive_toggle_btn = self._make_icon_button(
            "↩" if archived else "⇣",
            "Restore conversation" if archived else "Archive conversation",
            on_archive_toggle,
        )
        self._delete_btn = self._make_icon_button("×", "Delete conversation", on_delete)
        actions.addWidget(self._archive_toggle_btn)
        actions.addWidget(self._delete_btn)
        actions.addStretch()
        layout.addLayout(actions)

        self._apply_style()
        self.setMinimumHeight(86)

    def _make_icon_button(self, text: str, tooltip: str, callback: Any) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFixedSize(24, 24)
        button.clicked.connect(callback)
        return button

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            SessionListItemWidget {{
                background: transparent;
                border: none;
            }}
            QLabel {{
                background: transparent;
                color: {self._colors["text_color"]};
            }}
            QLabel#meta {{
                color: {self._colors["text_muted"]};
            }}
            QToolButton {{
                background: transparent;
                border: none;
                color: {self._colors["text_muted"]};
                font-size: 16px;
                padding: 0;
            }}
            QToolButton:hover {{
                color: {self._colors["accent"]};
            }}
        """)
        self._title_label.setStyleSheet("font-weight: 600;")
        self._meta_label.setObjectName("meta")
        self._meta_label.setStyleSheet(f"color: {self._colors['text_muted']};")
        self._snippet_label.setStyleSheet(f"color: {self._colors['text_muted']};")


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

        self._setup_ui()
        self.refresh_theme()
        self.refresh_lists()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title
        self._title_label = QLabel("Conversations")
        layout.addWidget(self._title_label)

        # Search Bar
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search history...")
        self._search_edit.textChanged.connect(self._on_search_changed)
        layout.addWidget(self._search_edit)

        # Tabs for Active / Archived
        self._tabs = QTabWidget()

        # Active List
        self._active_list = QListWidget()
        self._setup_list_widget(self._active_list)
        self._tabs.addTab(self._active_list, "Active")

        # Archived List
        self._archive_list = QListWidget()
        self._setup_list_widget(self._archive_list)
        self._tabs.addTab(self._archive_list, "Archived")

        layout.addWidget(self._tabs)

    def _setup_list_widget(self, list_widget: QListWidget) -> None:
        list_widget.setWordWrap(True)
        list_widget.setUniformItemSizes(False)
        list_widget.itemClicked.connect(self._on_item_clicked)
        # Context menu for right click
        list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        list_widget.customContextMenuRequested.connect(
            lambda pos, lw=list_widget: self._show_context_menu(pos, lw)
        )

    def refresh_theme(self) -> None:
        """Dynamically update colors based on the current theme."""
        # Find the parent ChatDockWidget to access theme provider
        dock = self.parent()
        while dock is not None and not hasattr(dock, "_theme_provider"):
            dock = dock.parent()

        theme_provider = dock._theme_provider if dock is not None else None
        colors = get_theme_colors(theme_provider)
        bg_primary = colors.get("bg", "#1e1e1e")
        bg_alt = colors.get("group_bg", "#2d2d2d")
        text_color = colors.get("text", "#e0e0e0")
        text_muted = colors.get("text_secondary", "#888")
        accent = colors.get("accent", "#ffaa33")
        border = colors.get("border", "#444")

        self._theme_colors = {
            "bg_base": bg_primary,
            "bg_alt": bg_alt,
            "text_color": text_color,
            "text_muted": text_muted,
            "accent": accent,
            "border": border,
        }

        # Apply layout/sidebar styling
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_primary};
                color: {text_color};
            }}
            QLineEdit {{
                background-color: {bg_alt};
                color: {text_color};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 4px;
            }}
            QTabWidget::pane {{
                border: 1px solid {border};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background: {bg_alt};
                color: {text_muted};
                padding: 4px 8px;
                border: 1px solid {border};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: {bg_primary};
                color: {accent};
                font-weight: bold;
            }}
        """)

        if hasattr(self, "_title_label"):
            self._title_label.setStyleSheet(
                f"font-weight: bold; color: {text_color}; font-size: 11px;"
            )

        if hasattr(self, "_active_list"):
            self._apply_list_style(self._active_list)
        if hasattr(self, "_archive_list"):
            self._apply_list_style(self._archive_list)

    def _apply_list_style(self, list_widget: QListWidget) -> None:
        bg_base = self._theme_colors["bg_base"]
        bg_alt = self._theme_colors["bg_alt"]
        text_color = self._theme_colors["text_color"]
        border = self._theme_colors["border"]
        accent = self._theme_colors["accent"]
        list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {bg_base};
                color: {text_color};
                border: none;
                outline: 0;
            }}
            QListWidget::item {{
                border-bottom: 1px solid {border};
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: transparent;
                border-left: 3px solid {accent};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {bg_alt};
            }}
        """)

    def _on_search_changed(self, text: str) -> None:
        self.refresh_lists()

    def refresh_lists(self) -> None:
        """Reload sessions from the manager and update UI lists."""
        query = (
            self._search_edit.text().strip() if hasattr(self, "_search_edit") else ""
        )
        if query:
            self.set_search_query(query)
        else:
            self._refresh_data()

        self._active_list.clear()
        self._archive_list.clear()

        # Let's get metadata for all sessions
        sessions = self._manager.list_sessions()
        sessions_map = {s["id"]: s for s in sessions}

        # Re-populate Active list
        for sid in self._active_ids:
            s = sessions_map.get(sid)
            if not s:
                continue
            item = SessionListWidgetItem(s)
            row = SessionListItemWidget(
                s,
                archived=False,
                on_archive_toggle=lambda checked=False,
                session_id=sid: self._on_archive_toggle(session_id, True),
                on_delete=lambda checked=False, session_id=sid: self._confirm_delete(
                    session_id
                ),
                colors=self._theme_colors,
                parent=self,
            )
            item.setSizeHint(QSize(220, max(92, row.sizeHint().height())))
            self._active_list.addItem(item)
            self._active_list.setItemWidget(item, row)

        # Re-populate Archived list
        for sid in self._archived_ids:
            s = sessions_map.get(sid)
            if not s:
                continue
            item = SessionListWidgetItem(s)
            row = SessionListItemWidget(
                s,
                archived=True,
                on_archive_toggle=lambda checked=False,
                session_id=sid: self._on_archive_toggle(session_id, False),
                on_delete=lambda checked=False, session_id=sid: self._confirm_delete(
                    session_id
                ),
                colors=self._theme_colors,
                parent=self,
            )
            item.setSizeHint(QSize(220, max(92, row.sizeHint().height())))
            self._archive_list.addItem(item)
            self._archive_list.setItemWidget(item, row)

    def _on_archive_toggle(self, session_id: str, archive: bool) -> None:
        self._manager.archive_session(session_id, archive)
        self.refresh_lists()

    def _confirm_delete(self, session_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            "Are you sure you want to permanently delete this conversation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._manager.delete_session(session_id)
            self.refresh_lists()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if isinstance(item, SessionListWidgetItem):
            session_id = item.session_data["id"]
            dock = self.parent()
            # Search upwards for ChatDockWidget if parent is not it (due to splitter layout)
            while dock is not None and not hasattr(dock, "load_session"):
                dock = dock.parent()
            if dock is not None:
                dock.load_session(session_id)

    def _show_context_menu(self, pos: Any, list_widget: QListWidget) -> None:
        item = list_widget.itemAt(pos)
        if not item or not isinstance(item, SessionListWidgetItem):
            return

        menu = QMenu(self)
        session_id = item.session_data["id"]
        is_archived = item.session_data.get("archived", False)

        if is_archived:
            restore_action = menu.addAction("Restore")
            if restore_action is not None:
                restore_action.triggered.connect(
                    lambda: self._on_archive_toggle(session_id, False)
                )
        else:
            archive_action = menu.addAction("Archive")
            if archive_action is not None:
                archive_action.triggered.connect(
                    lambda: self._on_archive_toggle(session_id, True)
                )

        menu.addSeparator()
        delete_action = menu.addAction("Delete Permanently")
        if delete_action is not None:
            delete_action.triggered.connect(lambda: self._confirm_delete(session_id))

        menu.exec(list_widget.mapToGlobal(pos))

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
        self.refresh_lists()

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

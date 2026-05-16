"""Lightweight AI Chat dock widget for embedding in any PyQt6 application.

Connects to a FastAPI server's WebSocket chat endpoint and provides a
minimal streaming chat interface. Shares conversation context across
all windows via a common session ID persisted to disk.

This widget is fully portable — it depends only on PyQt6 and json,
with no application-specific imports.

Usage::

    from chat import ChatDockWidget

    dock = ChatDockWidget(
        app_context="gasification",
        app_name="integrated_process_simulator",
        accent_color="#3498db",
        parent=main_window,
    )
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWebSockets import QWebSocket
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ._theme_protocol import ThemeProviderProtocol, _DefaultDarkTheme
from .chat_dock_widget import (
    _DEFAULT_SERVER,
    _read_shared_session_id,
    _session_file_path,
    _write_shared_session_id,
)
from .terminal_contracts import TerminalProviderRegistry
from .terminal_providers import build_default_terminal_provider_registry
from .voice_input_manager import VoiceInputManager

logger = logging.getLogger(__name__)


def _get_theme_colors(
    theme_provider: ThemeProviderProtocol | None = None,
) -> dict[str, str]:
    """Get the current theme colors from the injected provider.

    Falls back to :class:`_DefaultDarkTheme` so the widget never depends
    on ``theme.theme_manager`` being importable (Tools issue #2766).
    """
    provider: ThemeProviderProtocol = theme_provider or _DefaultDarkTheme()
    try:
        colors: dict[str, str] = provider.get_current_colors()
        return colors
    except Exception:  # noqa: BLE001 - defensive: a misbehaving provider
        # must not crash the widget
        colors = _DefaultDarkTheme().get_current_colors()
        return colors


class ChatMessageBubble(QFrame):
    """Compact message bubble for chat display."""

    def __init__(
        self,
        role: str,
        content: str,
        accent_color: str = "#FF8800",
        parent: QWidget | None = None,
        theme_provider: ThemeProviderProtocol | None = None,
    ) -> None:
        if role is None:
            raise ValueError("role must be provided")
        super().__init__(parent)
        self._role = role
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        colors = _get_theme_colors(theme_provider)
        text_primary = colors.get("text", "#e0e0e0")
        bg_alt = colors.get("group_bg", "#2d2d2d")
        bg_secondary = colors.get("input_bg", "#252526")

        # Role label
        user_style = f"font-size: 10px; font-weight: bold; color: {accent_color};"
        ai_color = colors.get("accent", "#58a6ff")
        ai_style = f"font-size: 10px; font-weight: bold; color: {ai_color};"
        role_label = QLabel("You" if role == "user" else "AI")
        role_label.setStyleSheet(user_style if role == "user" else ai_style)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(role_label)
        header_row.addStretch()

        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setToolTip("Copy message to clipboard")
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setStyleSheet(
            "QPushButton { background-color: transparent; "
            f"color: {colors.get('text_secondary', '#888')}; "
            "border: none; font-size: 10px; padding: 0px; }"
            f"QPushButton:hover {{ color: {colors.get('text', '#e0e0e0')}; }}"
        )
        self._copy_btn.clicked.connect(self._copy_to_clipboard)
        header_row.addWidget(self._copy_btn)

        layout.addLayout(header_row)

        # Content
        self._content_label = QLabel(content)
        self._content_label.setWordWrap(True)
        self._content_label.setTextFormat(Qt.TextFormat.PlainText)
        self._content_label.setStyleSheet(f"color: {text_primary}; font-size: 12px;")
        layout.addWidget(self._content_label)

        bg = bg_alt if role == "user" else bg_secondary
        self.setStyleSheet(
            f"ChatMessageBubble {{ background-color: {bg}; border-radius: 6px; }}"
        )

    def set_content(self, text: str) -> None:
        """Replace the content text."""
        if text is None:
            raise ValueError("text must be provided")
        self._content = text
        self._content_label.setText(text)

    def append_content(self, text: str) -> None:
        """Append text to existing content."""
        if text is None:
            raise ValueError("text must be provided")
        self._content += text
        self._content_label.setText(self._content)

    def _copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._content)


class ChatDockWidget(QDockWidget):
    """Lightweight chat dock widget that connects to a FastAPI chat server.

    Uses QWebSocket for real-time streaming. All instances share the same
    conversation session via a file-persisted session ID.

    Args:
        app_context: Name of the module/context this widget is embedded in.
        app_name: Application identifier for session file storage.
        server_url: WebSocket server base URL.
        session_id: Explicit session ID (None = use shared or create new).
        ws_path_template: WebSocket path template with ``{session_id}`` placeholder.
        placeholder_text: Placeholder text for the input field.
        accent_color: Primary accent color for styling.
        auto_index_on_open: When True, send an ``index_codebase`` action on
            connect so the chat backend rebuilds its codemap before the user
            starts typing. Tools issue #2549 / PR #2567.
        terminal_registry: Registry used to populate shell/provider dropdowns.
        theme_provider: Object implementing ``get_current_colors()`` to drive
            widget styling. Defaults to :class:`_DefaultDarkTheme` so the
            widget is fully portable and does not require ``theme`` to be on
            ``sys.path`` (Tools issue #2766). Pass an app-specific manager
            (e.g. ``theme.theme_manager.get_theme_manager()``) to honor the
            host application's theme.
        parent: Parent widget.
    """

    # Class-level session for in-process sharing. All reads/writes are
    # serialized through ``_SHARED_SESSION_LOCK`` in ``chat_dock_widget``
    # to prevent the multi-window race described in Tools issue #2753.
    _shared_session_id: str | None = None
    _session_lock: threading.Lock = threading.Lock()

    @classmethod
    def _get_shared_session_id(cls) -> str | None:
        """Return the shared session ID under the class lock."""
        with cls._session_lock:
            return cls._shared_session_id

    @classmethod
    def _set_shared_session_id(cls, val: str | None) -> None:
        """Set the shared session ID under the class lock."""
        with cls._session_lock:
            cls._shared_session_id = val

    # Signals for external UI integration
    models_refreshed = pyqtSignal(list)
    index_status_changed = pyqtSignal(dict)

    def __init__(
        self,
        app_context: str = "unknown",
        app_name: str = "shared_chat",
        server_url: str = _DEFAULT_SERVER,
        session_id: str | None = None,
        ws_path_template: str = "/api/ws/chat/{session_id}",
        placeholder_text: str = "Ask a question...",
        accent_color: str = "#FF8800",
        auto_index_on_open: bool = False,
        project_root: str | Path | None = None,
        terminal_registry: TerminalProviderRegistry | None = None,
        theme_provider: ThemeProviderProtocol | None = None,
        parent: QWidget | None = None,
    ) -> None:
        if app_context is None:
            raise ValueError("app_context must be provided")
        super().__init__("AI Chat", parent)
        self._app_context = app_context
        self._app_name = app_name
        self._server_url = server_url.rstrip("/")
        self._ws_path_template = ws_path_template
        self._accent_color = accent_color
        self._placeholder_text = placeholder_text
        self._auto_index_on_open = bool(auto_index_on_open)
        self._project_root = (
            Path(project_root).resolve() if project_root else Path.cwd()
        )
        self._terminal_registry = (
            terminal_registry or build_default_terminal_provider_registry()
        )
        self._theme_provider: ThemeProviderProtocol = (
            theme_provider or _DefaultDarkTheme()
        )
        self._is_streaming = False
        self._current_bubble: ChatMessageBubble | None = None
        self._terminal_session_id: str | None = None
        self._voice_manager = VoiceInputManager()
        self._terminal_start_pending = False
        self._socket: QWebSocket | None = None
        self._session_file = _session_file_path(app_name)
        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._connect)

        # Resolve session ID: explicit > class-level > file > "new"
        if session_id:
            ChatDockWidget._set_shared_session_id(session_id)
        elif not ChatDockWidget._get_shared_session_id():
            ChatDockWidget._set_shared_session_id(
                _read_shared_session_id(self._session_file)
            )

        self._setup_ui()
        self._connect_on_show = True

    def _setup_ui(self) -> None:
        colors = _get_theme_colors(self._theme_provider)
        bg_primary = colors.get("bg", "#1e1e1e")
        bg_alt = colors.get("group_bg", "#2d2d2d")
        text_primary = colors.get("text", "#e0e0e0")
        text_secondary = colors.get("text_secondary", "#888")
        border = colors.get("border", "#444")
        button_hover = colors.get("button_hover", "#ffaa33")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Status bar
        status_row = QHBoxLayout()
        self._status_label = QLabel("Connecting...")
        self._status_label.setStyleSheet(f"color: {text_secondary}; font-size: 10px;")
        status_row.addWidget(self._status_label, stretch=1)

        self._tools_btn = QPushButton("Tools")
        self._tools_btn.setToolTip("Chat tools and actions")
        self._tools_menu = QMenu(self)
        self._action_copy_thread = self._tools_menu.addAction("Copy Entire Thread")
        self._action_export_thread = self._tools_menu.addAction("Export to Markdown...")
        self._action_condense_thread = self._tools_menu.addAction("Condense Thread")
        self._action_request_review = self._tools_menu.addAction(
            "Request Agent Review..."
        )
        if self._action_copy_thread is not None:
            self._action_copy_thread.triggered.connect(self._copy_entire_thread)
        if self._action_export_thread is not None:
            self._action_export_thread.triggered.connect(self._export_to_markdown)
        if self._action_condense_thread is not None:
            self._action_condense_thread.triggered.connect(self._condense_thread)
        if self._action_request_review is not None:
            self._action_request_review.triggered.connect(self._request_review)
        self._tools_btn.setMenu(self._tools_menu)
        status_row.addWidget(self._tools_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.setToolTip("Close chat")
        self._close_btn.clicked.connect(self.close)
        status_row.addWidget(self._close_btn)
        layout.addLayout(status_row)

        mode_row = QHBoxLayout()
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Chat", "chat")
        self._mode_combo.addItem("Terminal", "terminal")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)

        self._shell_combo = QComboBox()
        self._populate_shell_combo()
        self._shell_combo.currentIndexChanged.connect(self._on_terminal_shell_changed)
        mode_row.addWidget(self._shell_combo)

        self._provider_combo = QComboBox()
        self._populate_provider_combo()
        mode_row.addWidget(self._provider_combo)

        self._terminal_start_btn = QPushButton("Start")
        self._terminal_start_btn.clicked.connect(self._on_terminal_start)
        mode_row.addWidget(self._terminal_start_btn)

        self._terminal_stop_btn = QPushButton("Stop")
        self._terminal_stop_btn.clicked.connect(self._on_terminal_stop)
        mode_row.addWidget(self._terminal_stop_btn)

        layout.addLayout(mode_row)

        # Message scroll area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._message_container = QWidget()
        self._message_layout = QVBoxLayout(self._message_container)
        self._message_layout.setContentsMargins(2, 2, 2, 2)
        self._message_layout.setSpacing(4)
        self._message_layout.addStretch()
        self._scroll_area.setWidget(self._message_container)

        self._terminal_output = QPlainTextEdit()
        self._terminal_output.setReadOnly(True)
        self._terminal_output.setStyleSheet(
            "QPlainTextEdit {"
            f"  background-color: {bg_alt}; color: {text_primary};"
            f"  border: 1px solid {border}; border-radius: 4px;"
            "  font-family: Consolas, monospace; font-size: 12px; padding: 4px;"
            "}"
        )

        self._content_stack = QStackedWidget()
        self._content_stack.addWidget(self._scroll_area)
        self._content_stack.addWidget(self._terminal_output)
        layout.addWidget(self._content_stack, stretch=1)

        # Input row
        input_row = QHBoxLayout()
        self._input_edit = QPlainTextEdit()
        self._input_edit.setMaximumHeight(50)
        self._input_edit.setPlaceholderText(self._placeholder_text)
        self._input_edit.setStyleSheet(
            "QPlainTextEdit {"
            f"  background-color: {bg_alt}; color: {text_primary};"
            f"  border: 1px solid {border}; border-radius: 4px;"
            "  font-size: 12px; padding: 4px;"
            "}"
        )
        input_row.addWidget(self._input_edit, stretch=1)

        self._upload_btn = QPushButton("📎")
        self._upload_btn.setToolTip("Upload file")
        self._upload_btn.setFixedWidth(28)
        self._upload_btn.setStyleSheet(
            "QPushButton {"
            f"  background-color: {bg_alt}; color: {text_primary};"
            "  border-radius: 4px; padding: 4px;"
            "}"
            f"QPushButton:hover {{ background-color: {border}; }}"
        )
        self._upload_btn.clicked.connect(self._on_upload)
        input_row.addWidget(self._upload_btn)

        self._screenshot_btn = QPushButton("📸")
        self._screenshot_btn.setToolTip("Capture screenshot")
        self._screenshot_btn.setFixedWidth(28)
        self._screenshot_btn.setStyleSheet(
            "QPushButton {"
            f"  background-color: {bg_alt}; color: {text_primary};"
            "  border-radius: 4px; padding: 4px;"
            "}"
            f"QPushButton:hover {{ background-color: {border}; }}"
        )
        self._screenshot_btn.clicked.connect(self._on_screenshot)
        input_row.addWidget(self._screenshot_btn)

        self._mic_btn = QPushButton("\U0001f3a4")
        self._mic_btn.setToolTip("Voice input (Ctrl+Shift+V)")
        self._mic_btn.setFixedWidth(28)
        self._mic_btn.setStyleSheet(
            "QPushButton {"
            f"  background-color: {bg_alt}; color: {text_primary};"
            "  border-radius: 4px; padding: 4px;"
            "}"
            f"QPushButton:hover {{ background-color: {border}; }}"
        )
        self._mic_btn.clicked.connect(self._on_mic_toggle)
        input_row.addWidget(self._mic_btn)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(55)
        self._send_btn.setStyleSheet(
            "QPushButton {"
            f"  background-color: {self._accent_color}; color: black;"
            "  border-radius: 4px; font-weight: bold; padding: 4px;"
            "}"
            f"QPushButton:hover {{ background-color: {button_hover}; }}"
            "QPushButton:disabled { background-color: #555; color: #888; }"
        )
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._send_btn)

        layout.addLayout(input_row)
        self.setWidget(container)
        self._on_mode_changed()

        # Dock widget styling
        self.setStyleSheet(
            f"QDockWidget {{ background-color: {bg_primary}; color: {text_primary}; }}"
            "QDockWidget::title {"
            f"  background-color: {self._accent_color}; color: black;"
            "  padding: 6px; font-weight: bold;"
            "}"
        )
        self._scroll_area.setStyleSheet(
            f"QScrollArea {{ background-color: {bg_primary}; border: none; }}"
        )
        self._message_container.setStyleSheet(f"background-color: {bg_primary};")

        # Keyboard shortcut for voice input
        shortcut = QShortcut(QKeySequence("Ctrl+Shift+V"), self)
        shortcut.activated.connect(self._on_mic_toggle)

        # Wire voice manager callbacks
        self._voice_manager.connect_transcription(self._on_voice_transcription)
        self._voice_manager.connect_error(self._on_voice_error)

    # ── WebSocket connection ─────────────────────────────────────────

    def _connect(self) -> None:
        """Establish WebSocket connection to the chat server."""
        if self._socket is not None:
            self._socket.close()
            self._socket.deleteLater()

        self._socket = QWebSocket()
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.textMessageReceived.connect(self._on_message)

        sid = ChatDockWidget._get_shared_session_id() or "new"
        path = self._ws_path_template.replace("{session_id}", sid)
        url = QUrl(f"{self._server_url}{path}")
        self._status_label.setText("Connecting...")
        self._socket.open(url)

    def _on_connected(self) -> None:
        self._status_label.setText("Connected")
        self._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
        self.refresh_models()
        if self._auto_index_on_open:
            self.index_codebase()

    def refresh_models(self) -> None:
        self._send_ws({"action": "refresh_models"})

    def index_codebase(self) -> None:
        self._send_ws({"action": "index_codebase"})

    def _on_disconnected(self) -> None:
        self._status_label.setText("Disconnected - retrying in 3s...")
        self._status_label.setStyleSheet("color: #f85149; font-size: 10px;")
        self._is_streaming = False
        self._send_btn.setEnabled(True)
        self._reconnect_timer.start(3000)

    def _on_message(self, raw: str) -> None:
        """Handle incoming WebSocket message."""
        if raw is None:
            raise ValueError("raw must be provided")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        msg_type = data.get("type")

        if msg_type == "session_info":
            sid = data.get("session_id", "")
            ChatDockWidget._set_shared_session_id(sid)
            _write_shared_session_id(sid, self._session_file)
            self._send_ws({"action": "history"})

        elif msg_type == "chunk":
            content = data.get("content", "")
            if self._current_bubble:
                self._current_bubble.append_content(content)
                self._scroll_to_bottom()

        elif msg_type == "complete":
            self._is_streaming = False
            self._send_btn.setEnabled(True)
            self._current_bubble = None
            sid = data.get("session_id")
            if sid:
                ChatDockWidget._set_shared_session_id(sid)
                _write_shared_session_id(sid, self._session_file)

        elif msg_type == "session_created":
            sid = data.get("session_id", "")
            ChatDockWidget._set_shared_session_id(sid)
            _write_shared_session_id(sid, self._session_file)

        elif msg_type == "history":
            self._populate_history(data.get("messages", []))

        elif msg_type == "model_list":
            models = data.get("models", [])
            if isinstance(models, list):
                self.models_refreshed.emit(models)

        elif msg_type == "index_status":
            self.index_status_changed.emit(dict(data))
            state = data.get("state")
            if state == "running":
                files = data.get("files_parsed", 0)
                self._status_label.setText(f"Indexing codebase ({files} files)...")
            elif state == "complete":
                self._status_label.setText("Connected")
                self._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
            elif state == "error":
                detail = data.get("error", "Unknown indexing error")
                self._status_label.setText(f"Index error: {detail}")
                self._status_label.setStyleSheet("color: #f85149; font-size: 10px;")

        elif msg_type == "error":
            detail = data.get("detail", "Unknown error")
            self._status_label.setText(f"Error: {detail}")
            self._is_streaming = False
            self._send_btn.setEnabled(True)

        elif msg_type == "terminal_session":
            session = data.get("session", {})
            self._terminal_session_id = session.get("session_id")
            self._terminal_start_pending = False
            state = session.get("state", "unknown")
            self._status_label.setText(f"Terminal {state}")
            if self._terminal_session_id:
                self._append_terminal_line(f"[terminal] session {state}")
            if state in {"stopped", "exited", "error"}:
                self._terminal_session_id = None
            self._sync_terminal_controls()

        elif msg_type == "terminal_events":
            for event in data.get("events", []):
                self._append_terminal_line(event.get("data", ""))

        elif msg_type == "terminal_ack":
            self._status_label.setText("Terminal input sent")

    # ── UI actions ───────────────────────────────────────────────────

    def _on_send(self) -> None:
        text = self._input_edit.toPlainText().strip()
        if not text or self._is_streaming:
            return

        if self._current_mode() == "terminal":
            self._on_terminal_input(text)
            return

        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        self._input_edit.clear()
        self._add_bubble("user", text)

        self._is_streaming = True
        self._send_btn.setEnabled(False)
        self._current_bubble = self._add_bubble("assistant", "")

        self._send_ws(
            {
                "action": "send",
                "message": text,
                "app_context": self._app_context,
            }
        )

    def _handle_slash_command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0][1:].lower()
        self._input_edit.clear()
        self._add_bubble("user", text)
        self._is_streaming = True
        self._send_btn.setEnabled(False)
        self._current_bubble = self._add_bubble(
            "assistant", f"Starting workflow: {cmd}..."
        )
        self._send_ws(
            {
                "action": "skill_invoke",
                "skill_id": cmd,
                "app_context": self._app_context,
            }
        )

    def _populate_shell_combo(self) -> None:
        self._shell_combo.clear()
        for shell in self._terminal_registry.shells():
            self._shell_combo.addItem(shell.display_name, shell.id)

    def _populate_provider_combo(self) -> None:
        shell_id = str(self._shell_combo.currentData() or "")
        providers = self._terminal_registry.providers_for_shell(shell_id)
        current_provider = self._provider_combo.currentData()
        self._provider_combo.blockSignals(True)
        try:
            self._provider_combo.clear()
            for provider in providers:
                self._provider_combo.addItem(provider.display_name, provider.id)
            if current_provider:
                idx = self._provider_combo.findData(current_provider)
                if idx >= 0:
                    self._provider_combo.setCurrentIndex(idx)
        finally:
            self._provider_combo.blockSignals(False)
        self._sync_terminal_controls()

    def _on_terminal_shell_changed(self, _index: int) -> None:
        self._populate_provider_combo()

    def _on_terminal_start(self) -> None:
        if self._terminal_session_id or self._terminal_start_pending:
            self._append_terminal_line("[terminal] session already active")
            return
        if (
            not self._shell_combo.currentData()
            or not self._provider_combo.currentData()
        ):
            self._append_terminal_line("[terminal] select a shell and provider first")
            return
        self._terminal_start_pending = True
        self._sync_terminal_controls()
        self._terminal_output.clear()
        self._append_terminal_line("[terminal] starting...")
        self._send_ws(
            {
                "action": "terminal_start",
                "project_root": str(self._project_root),
                "shell_id": self._shell_combo.currentData(),
                "provider_id": self._provider_combo.currentData(),
                "app_context": self._app_context,
            }
        )

    def _on_terminal_stop(self) -> None:
        if not self._terminal_session_id:
            self._append_terminal_line("[terminal] start a session first")
            return
        self._send_ws(
            {
                "action": "terminal_stop",
                "terminal_session_id": self._terminal_session_id,
            }
        )

    def _on_terminal_input(self, text: str) -> None:
        if not self._terminal_session_id:
            self._append_terminal_line("[terminal] start a session first")
            return
        self._input_edit.clear()
        self._append_terminal_line(f"> {text}")
        self._send_ws(
            {
                "action": "terminal_input",
                "terminal_session_id": self._terminal_session_id,
                "text": f"{text}\n",
            }
        )

    def _on_upload(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Attach File", "", "All Files (*)"
        )
        if file_path:
            path = Path(file_path)
            try:
                data = path.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                self._send_ws(
                    {"action": "file_upload", "filename": path.name, "content": b64}
                )
                self._add_bubble("user", f"[Uploaded file: {path.name}]")
            except (OSError, ValueError) as exc:
                self._status_label.setText(f"Upload failed: {exc}")

    def _on_screenshot(self) -> None:
        app = QApplication.instance()
        if not app:
            return
        parent = self.parentWidget()
        from PyQt6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        pixmap = parent.grab() if parent else (screen.grabWindow(0) if screen else None)  # type: ignore[arg-type]
        if not pixmap:
            return
        from PyQt6.QtCore import QBuffer, QByteArray, QIODevice

        ba = QByteArray()
        buffer = QBuffer(ba)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        b64 = base64.b64encode(ba.data()).decode("ascii")
        self._send_ws(
            {"action": "file_upload", "filename": "screenshot.png", "content": b64}
        )
        self._add_bubble("user", "[Captured screenshot]")

    def _on_mic_toggle(self) -> None:
        if self._voice_manager.is_recording:
            self._voice_manager.stop()
            self._mic_btn.setText("\U0001f3a4")
            self._mic_btn.setToolTip("Voice input (Ctrl+Shift+V)")
        else:
            self._voice_manager.start()
            self._mic_btn.setText("[REC]")
            self._mic_btn.setToolTip("Recording... click to stop (Ctrl+Shift+V)")
            self._status_label.setText("Listening...")

    def _on_voice_transcription(self, text: str) -> None:
        self._mic_btn.setText("\U0001f3a4")
        self._mic_btn.setToolTip("Voice input (Ctrl+Shift+V)")
        cursor = self._input_edit.textCursor()
        cursor.insertText(text)
        self._input_edit.setTextCursor(cursor)
        self._status_label.setText("Transcription complete")

    def _on_voice_error(self, message: str) -> None:
        self._mic_btn.setText("\U0001f3a4")
        self._mic_btn.setToolTip("Voice input (Ctrl+Shift+V)")
        self._status_label.setText(f"Voice: {message}")

    def _on_mode_changed(self) -> None:
        is_terminal = self._current_mode() == "terminal"
        self._content_stack.setCurrentIndex(1 if is_terminal else 0)
        self._shell_combo.setVisible(is_terminal)
        self._provider_combo.setVisible(is_terminal)
        self._terminal_start_btn.setVisible(is_terminal)
        self._terminal_stop_btn.setVisible(is_terminal)
        self._sync_terminal_controls()
        placeholder = (
            "Type terminal input..." if is_terminal else self._placeholder_text
        )
        self._input_edit.setPlaceholderText(placeholder)

    def _current_mode(self) -> str:
        mode = self._mode_combo.currentData()
        return str(mode or "chat")

    def _sync_terminal_controls(self) -> None:
        if not hasattr(self, "_terminal_start_btn"):
            return
        active = bool(self._terminal_session_id)
        pending = bool(self._terminal_start_pending)
        startable = (
            not active
            and not pending
            and bool(self._shell_combo.currentData())
            and bool(self._provider_combo.currentData())
        )
        self._terminal_start_btn.setEnabled(startable)
        self._terminal_stop_btn.setEnabled(active)
        self._shell_combo.setEnabled(not active and not pending)
        self._provider_combo.setEnabled(not active and not pending)

    def _append_terminal_line(self, text: str) -> None:
        if text:
            self._terminal_output.appendPlainText(text)

    def _send_ws(self, payload: dict) -> None:
        if self._socket and self._socket.isValid():
            self._socket.sendTextMessage(json.dumps(payload))

    def _add_bubble(self, role: str, content: str) -> ChatMessageBubble:
        """Add a message bubble to the scroll area."""
        if role is None:
            raise ValueError("role must be provided")
        bubble = ChatMessageBubble(
            role,
            content,
            accent_color=self._accent_color,
            theme_provider=self._theme_provider,
        )
        # Insert before the stretch item at the end
        count = self._message_layout.count()
        self._message_layout.insertWidget(count - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _populate_history(self, messages: list[dict]) -> None:
        """Clear and rebuild message bubbles from history."""
        if messages is None:
            raise ValueError("messages must be provided")
        while self._message_layout.count() > 1:
            item = self._message_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                self._add_bubble(role, content)

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            10,
            lambda: (
                scrollbar.setValue(scrollbar.maximum())
                if (scrollbar := self._scroll_area.verticalScrollBar()) is not None
                else None
            ),
        )

    def _get_thread_markdown(self) -> str:
        lines = []
        for i in range(self._message_layout.count()):
            item = self._message_layout.itemAt(i)
            if item:
                widget = item.widget()
                if isinstance(widget, ChatMessageBubble):
                    role_str = "You" if widget._role == "user" else "AI"
                    lines.append(f"**{role_str}**:\n\n{widget._content}\n")
        return "\n".join(lines)

    def _copy_entire_thread(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._get_thread_markdown())
            self._status_label.setText("Thread copied to clipboard")
            self._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")

    def _export_to_markdown(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Chat Thread",
            str(self._project_root / "chat_export.md"),
            "Markdown Files (*.md);;All Files (*)",
        )
        if path:
            try:
                Path(path).write_text(self._get_thread_markdown(), encoding="utf-8")
                self._status_label.setText("Exported successfully")
                self._status_label.setStyleSheet("color: #3fb950; font-size: 10px;")
            except OSError as exc:
                self._status_label.setText(f"Export error: {exc}")
                self._status_label.setStyleSheet("color: #f85149; font-size: 10px;")

    def _condense_thread(self) -> None:
        self._status_label.setText("Condensing thread...")
        self._send_ws({"action": "condense", "app_context": self._app_context})

    def _request_review(self) -> None:
        self._status_label.setText("Review requested...")
        self._send_ws({"action": "request_review", "provider": "openai"})

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if getattr(self, "_connect_on_show", False):
            self._connect_on_show = False
            self._connect()

    def closeEvent(self, event: Any) -> None:
        self._reconnect_timer.stop()
        if self._socket:
            self._socket.close()
        super().closeEvent(event)

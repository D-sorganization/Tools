# ruff: noqa: E501
"""UI-construction helper for :class:`ChatDockWidget`.

Extracts the ~340-line body of ``ChatDockWidget._setup_ui`` into a free
function so the parent module fits in the repo's 1500-line budget.

The function operates on a ``ChatDockWidget`` instance and mutates it in
place — it sets the same private attributes the method always set, in
the same order, so the public behaviour is byte-for-byte identical to
the inline version. Tests that patch ``_setup_ui`` itself are unaffected
because the public ``_setup_ui`` method on the class delegates here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .._thinking_indicator import ThinkingIndicator
from .input import install_enter_submit
from .styling import get_theme_colors

if TYPE_CHECKING:
    pass


def build_chat_dock_ui(dock: Any) -> None:
    """Build the full chat dock widget UI on ``dock``.

    DbC:
        Pre: ``dock`` must be a ``QDockWidget`` subclass with all
        configuration attributes already set (``_accent_color``,
        ``_placeholder_text``, ``_theme_provider``, etc.).
        Post: every widget attribute the original ``_setup_ui`` set is
        present on ``dock``; ``dock.setWidget(...)`` has been called.
    """
    colors = get_theme_colors(dock._theme_provider)
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
    dock._status_label = QLabel("Connecting...")
    dock._status_label.setStyleSheet(f"color: {text_secondary}; font-size: 10px;")
    status_row.addWidget(dock._status_label, stretch=1)

    dock._tools_btn = QPushButton("Tools")
    dock._tools_btn.setToolTip("Chat tools and actions")
    dock._tools_menu = QMenu(dock)
    dock._action_copy_thread = dock._tools_menu.addAction("Copy Entire Thread")
    export_menu = dock._tools_menu.addMenu("Export Thread")
    dock._action_export_markdown = (
        export_menu.addAction("Markdown...") if export_menu is not None else None
    )
    dock._action_export_text = (
        export_menu.addAction("Plain Text...") if export_menu is not None else None
    )
    dock._action_export_html = (
        export_menu.addAction("HTML...") if export_menu is not None else None
    )
    condense_menu = dock._tools_menu.addMenu("Condense Thread")
    dock._action_condense_keep_recent = (
        condense_menu.addAction("Keep recent...") if condense_menu is not None else None
    )
    dock._action_condense_semantic = (
        condense_menu.addAction("Semantic summary...")
        if condense_menu is not None
        else None
    )
    dock._action_condense_pinned = (
        condense_menu.addAction("Pinned anchor...")
        if condense_menu is not None
        else None
    )
    # Backwards-compat alias for tests that introspect _action_export_thread.
    dock._action_export_thread = dock._action_export_markdown
    dock._action_condense_thread = dock._action_condense_keep_recent
    dock._action_request_review = dock._tools_menu.addAction("Request Agent Review...")
    dock._action_manage_memory = dock._tools_menu.addAction("Manage Memory...")
    if dock._action_manage_memory is not None:
        dock._action_manage_memory.triggered.connect(dock.open_memory_panel)
    if dock._action_copy_thread is not None:
        dock._action_copy_thread.triggered.connect(dock._copy_entire_thread)
    if dock._action_export_markdown is not None:
        dock._action_export_markdown.triggered.connect(
            lambda: dock._export_thread("markdown", "Markdown Files (*.md)", ".md")
        )
    if dock._action_export_text is not None:
        dock._action_export_text.triggered.connect(
            lambda: dock._export_thread("text", "Text Files (*.txt)", ".txt")
        )
    if dock._action_export_html is not None:
        dock._action_export_html.triggered.connect(
            lambda: dock._export_thread("html", "HTML Files (*.html)", ".html")
        )
    if dock._action_condense_keep_recent is not None:
        dock._action_condense_keep_recent.triggered.connect(
            lambda: dock._run_condense_local("keep_recent")
        )
    if dock._action_condense_semantic is not None:
        dock._action_condense_semantic.triggered.connect(
            lambda: dock._run_condense_local("semantic_summary")
        )
    if dock._action_condense_pinned is not None:
        dock._action_condense_pinned.triggered.connect(
            lambda: dock._run_condense_local("pinned_anchor")
        )
    if dock._action_request_review is not None:
        dock._action_request_review.triggered.connect(dock._request_review)
    dock._tools_btn.setMenu(dock._tools_menu)

    # Token-budget indicator + condense-now button.
    dock._token_indicator = QLabel("0 tok")
    dock._token_indicator.setToolTip(
        "Approximate token count for the current thread. "
        "When it exceeds the auto-condense threshold the thread will "
        "be condensed automatically."
    )
    dock._token_indicator.setStyleSheet(f"color: {text_secondary}; font-size: 10px;")
    status_row.addWidget(dock._token_indicator)
    dock._auto_condense_threshold = 8000

    layout.addLayout(status_row)

    mode_row = QHBoxLayout()
    dock._build_ai_dropdowns(mode_row)

    dock._mode_combo = QComboBox()
    dock._mode_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    dock._mode_combo.setMinimumWidth(0)
    dock._mode_combo.addItem("Chat", "chat")
    dock._mode_combo.addItem("Terminal", "terminal")
    dock._mode_combo.currentIndexChanged.connect(dock._on_mode_changed)
    mode_row.addWidget(dock._mode_combo)

    dock._shell_combo = QComboBox()
    dock._shell_combo.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
    )
    dock._shell_combo.setMinimumWidth(0)
    dock._populate_shell_combo()
    dock._shell_combo.currentIndexChanged.connect(dock._on_terminal_shell_changed)
    mode_row.addWidget(dock._shell_combo)

    dock._provider_combo = QComboBox()
    dock._provider_combo.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
    )
    dock._provider_combo.setMinimumWidth(0)
    dock._populate_provider_combo()
    mode_row.addWidget(dock._provider_combo)

    dock._terminal_start_btn = QPushButton("Start")
    dock._terminal_start_btn.clicked.connect(dock._on_terminal_start)
    mode_row.addWidget(dock._terminal_start_btn)

    dock._terminal_stop_btn = QPushButton("Stop")
    dock._terminal_stop_btn.clicked.connect(dock._on_terminal_stop)
    mode_row.addWidget(dock._terminal_stop_btn)

    # Message scroll area
    dock._scroll_area = QScrollArea()
    dock._scroll_area.setWidgetResizable(True)
    dock._scroll_area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    dock._message_container = QWidget()
    dock._message_layout = QVBoxLayout(dock._message_container)
    dock._message_layout.setContentsMargins(2, 2, 2, 2)
    dock._message_layout.setSpacing(4)
    dock._message_layout.addStretch()
    dock._scroll_area.setWidget(dock._message_container)

    dock._terminal_output = QPlainTextEdit()
    dock._terminal_output.setReadOnly(True)
    dock._terminal_output.setStyleSheet(
        "QPlainTextEdit {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        f"  border: 1px solid {border}; border-radius: 4px;"
        "  font-family: Consolas, monospace; font-size: 12px; padding: 4px;"
        "}"
    )

    dock._content_stack = QStackedWidget()
    dock._content_stack.addWidget(dock._scroll_area)
    dock._content_stack.addWidget(dock._terminal_output)
    layout.addWidget(dock._content_stack, stretch=1)

    # Thinking indicator — animated "Sidekick is thinking ●●●" pulser.
    # Placed between the message stack and the input row so it sits
    # immediately above whatever the user is typing, making the active
    # state immediately discoverable.
    dock._thinking_indicator = ThinkingIndicator(
        parent=dock,
        theme_provider=dock._theme_provider,
        accent_color=dock._accent_color,
    )
    layout.addWidget(dock._thinking_indicator)

    # Input row
    input_row = QHBoxLayout()
    dock._input_edit = QPlainTextEdit()
    install_enter_submit(dock._input_edit, dock._on_send)
    dock._input_edit.setMinimumHeight(60)
    dock._input_edit.setMaximumHeight(150)
    dock._input_edit.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding
    )
    dock._input_edit.setPlaceholderText(dock._placeholder_text)
    dock._input_edit.setStyleSheet(
        "QPlainTextEdit {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        f"  border: 1px solid {border}; border-radius: 4px;"
        "  font-size: 12px; padding: 4px;"
        "}"
    )
    layout.addWidget(dock._input_edit)

    # Tools on the far left
    dock._tools_btn.setFixedWidth(50)
    dock._tools_btn.setSizePolicy(
        QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
    )
    dock._tools_btn.setStyleSheet(
        "QPushButton {"
        f"  background-color: {bg_alt}; color: {text_primary};"
        "  border-radius: 4px; padding: 4px;"
        "}"
        f"QPushButton:hover {{ background-color: {border}; }}"
    )
    input_row.addWidget(dock._tools_btn)

    dock._upload_btn = _make_icon_button(
        "+", "Upload file", bg_alt, text_primary, border, dock._on_upload
    )
    input_row.addWidget(dock._upload_btn)

    dock._screenshot_btn = _make_icon_button(
        "⛶", "Capture screenshot", bg_alt, text_primary, border, dock._on_screenshot
    )
    input_row.addWidget(dock._screenshot_btn)

    dock._mic_btn = _make_icon_button(
        "\U0001f3a4",
        "Voice input (Ctrl+Shift+V)",
        bg_alt,
        text_primary,
        border,
        dock._on_mic_toggle,
    )
    input_row.addWidget(dock._mic_btn)

    input_row.addStretch()

    dock._agent_mode_combo = QComboBox()
    dock._agent_mode_combo.setSizePolicy(
        QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
    )
    dock._agent_mode_combo.addItem("Agent", "agent")
    dock._agent_mode_combo.addItem("Plan", "plan")
    dock._agent_mode_combo.addItem("Ask", "ask")
    input_row.addWidget(dock._agent_mode_combo)

    # Send, Steer, Stop on the right side
    send_style = (
        "QPushButton {"
        f"  background-color: {dock._accent_color}; color: black;"
        "  border-radius: 4px; font-weight: bold; padding: 4px;"
        "}"
        f"QPushButton:hover {{ background-color: {button_hover}; }}"
        "QPushButton:disabled { background-color: #555; color: #888; }"
    )
    dock._send_btn = _make_action_button(
        "Send", "Send message", 55, send_style, dock._on_send
    )
    input_row.addWidget(dock._send_btn)

    dock._steer_btn = _make_action_button(
        "Steer", "Queue message", 50, send_style, dock._on_steer
    )
    input_row.addWidget(dock._steer_btn)

    dock._stop_agent_btn = _make_action_button(
        "Stop", "Stop response", 50, send_style, dock._on_stop_agent
    )
    input_row.addWidget(dock._stop_agent_btn)

    layout.addLayout(input_row)
    layout.addLayout(mode_row)
    dock.setWidget(container)
    dock._on_mode_changed()

    # Dock widget styling
    dock.setStyleSheet(
        f"QDockWidget {{ background-color: {bg_primary}; color: {text_primary}; }}"
        "QDockWidget::title {"
        f"  background-color: {dock._accent_color}; color: black;"
        "  padding: 6px; font-weight: bold;"
        "}"
    )
    dock._scroll_area.setStyleSheet(
        f"QScrollArea {{ background-color: {bg_primary}; border: none; }}"
    )
    dock._message_container.setStyleSheet(f"background-color: {bg_primary};")

    # Keyboard shortcut for voice input
    shortcut = QShortcut(QKeySequence("Ctrl+Shift+V"), dock)
    shortcut.activated.connect(dock._on_mic_toggle)

    # Wire voice manager callbacks
    dock._voice_manager.connect_transcription(dock._on_voice_transcription)
    dock._voice_manager.connect_error(dock._on_voice_error)


def _make_icon_button(
    text: str,
    tooltip: str,
    bg: str,
    fg: str,
    border: str,
    on_clicked: Any,
) -> QPushButton:
    """DRY helper for the small fixed-width icon buttons (upload, screenshot, mic)."""
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFixedWidth(28)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    btn.setStyleSheet(
        "QPushButton {"
        f"  background-color: {bg}; color: {fg};"
        "  border-radius: 4px; padding: 4px;"
        "}"
        f"QPushButton:hover {{ background-color: {border}; }}"
    )
    btn.clicked.connect(on_clicked)
    return btn


def _make_action_button(
    text: str,
    tooltip: str,
    width: int,
    style: str,
    on_clicked: Any,
) -> QPushButton:
    """DRY helper for the right-aligned action buttons (Send/Steer/Stop)."""
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFixedWidth(width)
    btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
    btn.setStyleSheet(style)
    btn.clicked.connect(on_clicked)
    return btn

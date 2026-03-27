"""Help system components for the Unified Tools Launcher.

This module provides:
- HelpDialog: A modal dialog for displaying markdown documentation
- HelpButton: A small help button that shows context-sensitive help
- TooltipManager: Manager for enhanced tooltips on complex inputs
- HelpManager: Singleton for managing help content and navigation
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


# Module-level singleton holder (avoids mutable global + global keyword)
class _HelpManagerHolder:
    instance: HelpManager | None = None


def get_help_manager() -> HelpManager:
    """Get the singleton HelpManager instance."""
    if _HelpManagerHolder.instance is None:
        _HelpManagerHolder.instance = HelpManager()
    return _HelpManagerHolder.instance


def load_help_from_file(file_path: str | Path) -> str:
    """Load help content from a markdown file.

    Args:
        file_path: Path to the markdown file

    Returns:
        The file contents as a string, or error message if not found
    """
    path = Path(file_path)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            return f"Error reading help file: {e}"
    return f"Help file not found: {path}"


class _MarkdownState:
    """Mutable state tracked while converting markdown to HTML."""

    __slots__ = (
        "html_lines",
        "in_code_block",
        "in_list",
        "in_table",
        "table_has_header",
    )

    def __init__(self) -> None:
        self.html_lines: list[str] = []
        self.in_code_block: bool = False
        self.in_list: bool = False
        self.in_table: bool = False
        self.table_has_header: bool = False

    def close_list(self) -> None:
        if self.in_list:
            self.html_lines.append("</ul>")
            self.in_list = False


def _markdown_to_html(markdown_text: str) -> str:
    """Convert basic markdown to HTML for display in QTextBrowser.

    Supports headers, bold, italic, code blocks, lists, links,
    horizontal rules, and basic tables.

    Args:
        markdown_text: Markdown formatted text

    Returns:
        HTML formatted text
    """
    state = _MarkdownState()

    for line in markdown_text.split("\n"):
        if _handle_code_block(line, state):
            continue
        if _handle_horizontal_rule(line, state):
            continue
        if _handle_table_line(line, state):
            continue
        if _handle_header(line, state):
            continue
        if _handle_list_item(line, state):
            continue
        _handle_paragraph(line, state)

    # Close any open tags
    if state.in_list:
        state.html_lines.append("</ul>")
    if state.in_table:
        state.html_lines.append("</table>")
    if state.in_code_block:
        state.html_lines.append("</pre>")

    return "\n".join(state.html_lines)


def _handle_code_block(line: str, s: _MarkdownState) -> bool:
    """Handle fenced code block start/end and code block content."""
    if not (line is not None):
        raise ValueError("line must be provided")
    if line.strip().startswith("```"):
        if s.in_code_block:
            s.html_lines.append("</pre>")
            s.in_code_block = False
        else:
            s.html_lines.append(
                '<pre style="background-color: #2d2d2d; color: #f0f0f0; '
                'padding: 10px; border-radius: 4px; font-family: Consolas, monospace;">'
            )
            s.in_code_block = True
        return True
    if s.in_code_block:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s.html_lines.append(escaped)
        return True
    return False


def _handle_horizontal_rule(line: str, s: _MarkdownState) -> bool:
    """Handle --- and *** horizontal rules."""
    if not (line is not None):
        raise ValueError("line must be provided")
    if line.strip() in ("---", "***"):
        s.close_list()
        s.html_lines.append("<hr>")
        return True
    return False


def _handle_table_line(line: str, s: _MarkdownState) -> bool:
    """Handle markdown table rows and separators."""
    if not (line is not None):
        raise ValueError("line must be provided")
    if "|" in line and not line.strip().startswith("|--"):
        if "|--" in line or "| --" in line or "|:--" in line:
            s.table_has_header = True
            return True

        if not s.in_table:
            s.html_lines.append(
                '<table style="border-collapse: collapse; width: 100%; '
                'margin: 10px 0;">'
            )
            s.in_table = True
            s.table_has_header = False

        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            tag = "th" if not s.table_has_header else "td"
            style = 'style="border: 1px solid #555; padding: 8px; text-align: left;"'
            row = (
                "<tr>" + "".join(f"<{tag} {style}>{c}</{tag}>" for c in cells) + "</tr>"
            )
            s.html_lines.append(row)
            if tag == "th":
                s.table_has_header = True
        return True

    if s.in_table:
        s.html_lines.append("</table>")
        s.in_table = False
    return False


_HEADER_MAP: list[tuple[str, str, str, str]] = [
    ("####", "h4", "#89b4fa", "15px 0 8px 0"),
    ("###", "h3", "#89b4fa", "18px 0 10px 0"),
    ("##", "h2", "#cba6f7", "20px 0 12px 0"),
    ("#", "h1", "#f5c2e7", "25px 0 15px 0"),
]


def _handle_header(line: str, s: _MarkdownState) -> bool:
    """Handle # through #### headers."""
    if not (line is not None):
        raise ValueError("line must be provided")
    for prefix, tag, color, margin in _HEADER_MAP:
        if line.startswith(prefix):
            s.close_list()
            content = line[len(prefix) :].strip()
            s.html_lines.append(
                f'<{tag} style="color: {color}; margin: {margin};">{content}</{tag}>'
            )
            return True
    return False


def _handle_list_item(line: str, s: _MarkdownState) -> bool:
    """Handle unordered and ordered list items."""
    if not (line is not None):
        raise ValueError("line must be provided")
    stripped = line.strip()
    if stripped.startswith("- ") or stripped.startswith("* "):
        if not s.in_list:
            s.html_lines.append('<ul style="margin: 5px 0; padding-left: 25px;">')
            s.in_list = True
        content = _process_inline_formatting(stripped[2:])
        s.html_lines.append(f"<li>{content}</li>")
        return True

    if re.match(r"^\d+\.\s", stripped):
        if not s.in_list:
            s.html_lines.append('<ol style="margin: 5px 0; padding-left: 25px;">')
            s.in_list = True
        content = _process_inline_formatting(re.sub(r"^\d+\.\s", "", stripped))
        s.html_lines.append(f"<li>{content}</li>")
        return True

    if s.in_list and not stripped:
        s.html_lines.append("</ul>")
        s.in_list = False
    return False


def _handle_paragraph(line: str, s: _MarkdownState) -> None:
    """Handle regular paragraphs and blank lines."""
    if not (line is not None):
        raise ValueError("line must be provided")
    stripped = line.strip()
    if stripped:
        content = _process_inline_formatting(stripped)
        s.html_lines.append(
            f'<p style="margin: 8px 0; line-height: 1.5;">{content}</p>'
        )
    elif not s.in_list:
        s.html_lines.append("<br>")


def _process_inline_formatting(text: str) -> str:
    """Process inline markdown formatting (bold, italic, code, links).

    Args:
        text: Text with inline markdown

    Returns:
        Text with HTML formatting
    """
    # Links [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" style="color: #89dceb;">\1</a>',
        text,
    )

    # Inline code `code`
    text = re.sub(
        r"`([^`]+)`",
        r'<code style="background-color: #45475a; padding: 2px 5px; '
        r'border-radius: 3px; font-family: Consolas, monospace;">\1</code>',
        text,
    )

    # Bold **text**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

    # Italic *text*
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    return text


class HelpDialog(QDialog):
    """Modal dialog for displaying help documentation.

    Supports markdown content with navigation history and topic links.
    """

    topicRequested = pyqtSignal(str)  # Emitted when user clicks a topic link

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        content: str,
        topics: dict[str, str] | None = None,
    ) -> None:
        """Initialize the help dialog.

        Args:
            parent: Parent widget
            title: Dialog title
            content: Markdown content to display
            topics: Optional dict mapping topic IDs to content for navigation
        """
        if not (title is not None):
            raise ValueError("title must be provided")
        super().__init__(parent)
        self.setWindowTitle(f"Help - {title}")
        self.resize(900, 650)
        self.setModal(True)

        self.topics = topics or {}
        self.history: list[str] = []
        self.history_index = -1
        self.current_topic = title

        self._setup_ui()
        self._apply_theme()
        self._display_content(title, content)

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Navigation bar
        nav_bar = QHBoxLayout()

        self.back_btn = QPushButton("<")
        self.back_btn.setFixedSize(30, 30)
        self.back_btn.setToolTip("Back")
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setEnabled(False)
        nav_bar.addWidget(self.back_btn)

        self.forward_btn = QPushButton(">")
        self.forward_btn.setFixedSize(30, 30)
        self.forward_btn.setToolTip("Forward")
        self.forward_btn.clicked.connect(self._go_forward)
        self.forward_btn.setEnabled(False)
        nav_bar.addWidget(self.forward_btn)

        nav_bar.addSpacing(10)

        self.topic_combo = QComboBox()
        self.topic_combo.setMinimumWidth(200)
        if self.topics:
            self.topic_combo.addItems(sorted(self.topics.keys()))
            self.topic_combo.currentTextChanged.connect(self._on_topic_selected)
        else:
            self.topic_combo.setEnabled(False)
        nav_bar.addWidget(self.topic_combo)

        nav_bar.addStretch()

        self.title_label = QLabel()
        self.title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        nav_bar.addWidget(self.title_label)

        layout.addLayout(nav_bar)

        # Content area with optional sidebar
        if self.topics and len(self.topics) > 5:
            splitter = QSplitter(Qt.Orientation.Horizontal)

            # Topic list sidebar
            self.topic_list = QListWidget()
            self.topic_list.setMaximumWidth(200)
            for topic in sorted(self.topics.keys()):
                item = QListWidgetItem(topic)
                self.topic_list.addItem(item)
            self.topic_list.itemClicked.connect(
                lambda item: self.navigate_to(item.text())
            )
            splitter.addWidget(self.topic_list)

            # Content browser
            self.content_browser = QTextBrowser()
            self.content_browser.setOpenExternalLinks(True)
            splitter.addWidget(self.content_browser)

            splitter.setSizes([180, 700])
            layout.addWidget(splitter)
        else:
            self.topic_list = None  # type: ignore[assignment]
            self.content_browser = QTextBrowser()
            self.content_browser.setOpenExternalLinks(True)
            layout.addWidget(self.content_browser)

        # Button bar
        button_bar = QHBoxLayout()
        button_bar.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(100)
        close_btn.clicked.connect(self.accept)
        button_bar.addWidget(close_btn)

        layout.addLayout(button_bar)

        # Keyboard shortcuts
        QShortcut(QKeySequence("Escape"), self, self.close)
        QShortcut(QKeySequence("Alt+Left"), self, self._go_back)
        QShortcut(QKeySequence("Alt+Right"), self, self._go_forward)

    def _apply_theme(self) -> None:
        """Apply Catppuccin Mocha dark theme to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QLabel {
                color: #cdd6f4;
            }
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #585b70;
            }
            QPushButton:pressed {
                background-color: #313244;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #6c7086;
            }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox:hover {
                border-color: #89b4fa;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cdd6f4;
                margin-right: 5px;
            }
            QListWidget {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                background-color: #45475a;
            }
            QListWidget::item:hover {
                background-color: #313244;
            }
            QTextBrowser {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 10px;
            }
            QSplitter::handle {
                background-color: #45475a;
            }
        """)

    def _display_content(self, title: str, content: str) -> None:
        """Display content in the browser.

        Args:
            title: Topic title
            content: Markdown content
        """
        if not (title is not None):
            raise ValueError("title must be provided")
        self.title_label.setText(title)
        html_content = _markdown_to_html(content)
        self.content_browser.setHtml(html_content)

        # Update combo box selection
        if self.topics and title in self.topics:
            self.topic_combo.blockSignals(True)
            self.topic_combo.setCurrentText(title)
            self.topic_combo.blockSignals(False)

        # Update list selection
        if self.topic_list:
            items = self.topic_list.findItems(title, Qt.MatchFlag.MatchExactly)
            if items:
                self.topic_list.setCurrentItem(items[0])

    def navigate_to(self, topic: str) -> None:
        """Navigate to a specific topic.

        Args:
            topic: Topic ID/title to navigate to
        """
        if not (topic is not None):
            raise ValueError("topic must be provided")
        if topic not in self.topics:
            return

        # Add to history
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]

        self.history.append(topic)
        self.history_index = len(self.history) - 1
        self._update_nav_buttons()

        self.current_topic = topic
        self._display_content(topic, self.topics[topic])
        self.topicRequested.emit(topic)

    def _on_topic_selected(self, topic: str) -> None:
        """Handle topic selection from combo box."""
        if topic != self.current_topic:
            self.navigate_to(topic)

    def _go_back(self) -> None:
        """Navigate to previous topic in history."""
        if self.history_index > 0:
            self.history_index -= 1
            topic = self.history[self.history_index]
            self.current_topic = topic
            self._display_content(topic, self.topics.get(topic, ""))
            self._update_nav_buttons()

    def _go_forward(self) -> None:
        """Navigate to next topic in history."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            topic = self.history[self.history_index]
            self.current_topic = topic
            self._display_content(topic, self.topics.get(topic, ""))
            self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        """Update navigation button states."""
        self.back_btn.setEnabled(self.history_index > 0)
        self.forward_btn.setEnabled(self.history_index < len(self.history) - 1)


class HelpButton(QToolButton):
    """A small help button that shows context-sensitive help when clicked.

    Designed to be placed next to input fields or controls.
    """

    def __init__(
        self,
        topic_id: str,
        parent: QWidget | None = None,
        tooltip: str = "Click for help",
    ) -> None:
        """Initialize the help button.

        Args:
            topic_id: ID of the help topic to display
            parent: Parent widget
            tooltip: Tooltip text for the button
        """
        if not (topic_id is not None):
            raise ValueError("topic_id must be provided")
        super().__init__(parent)
        self.topic_id = topic_id

        self.setText("?")
        self.setToolTip(tooltip)
        self.setFixedSize(QSize(20, 20))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._apply_style()
        self.clicked.connect(self._show_help)

    def _apply_style(self) -> None:
        """Apply button styling."""
        self.setStyleSheet("""
            QToolButton {
                background-color: #45475a;
                color: #89b4fa;
                border: 1px solid #585b70;
                border-radius: 10px;
                font-weight: bold;
                font-size: 12px;
            }
            QToolButton:hover {
                background-color: #585b70;
                border-color: #89b4fa;
            }
            QToolButton:pressed {
                background-color: #313244;
            }
        """)

    def _show_help(self) -> None:
        """Show the help dialog for this topic."""
        manager = get_help_manager()
        manager.show_topic(self.topic_id, self)


class TooltipManager:
    """Manager for enhanced tooltips on widgets.

    Provides consistent tooltip styling and content management
    across the application.
    """

    def __init__(self) -> None:
        """Initialize the tooltip manager."""
        self._tooltips: dict[str, str] = {}
        self._widgets: dict[str, list[QWidget]] = {}

    def register_tooltip(self, key: str, text: str) -> None:
        """Register a tooltip text for a key.

        Args:
            key: Unique identifier for the tooltip
            text: Tooltip text (can include basic HTML)
        """
        self._tooltips[key] = text

    def register_widget(self, widget: QWidget, key: str) -> None:
        """Register a widget to receive a tooltip.

        Args:
            widget: Widget to apply tooltip to
            key: Tooltip key (must be registered first)
        """
        if not (widget is not None):
            raise ValueError("widget must be provided")
        if key not in self._widgets:
            self._widgets[key] = []
        self._widgets[key].append(widget)

        if key in self._tooltips:
            self._apply_tooltip(widget, self._tooltips[key])

    def _apply_tooltip(self, widget: QWidget, text: str) -> None:
        """Apply formatted tooltip to widget.

        Args:
            widget: Target widget
            text: Tooltip text
        """
        # Wrap in styled HTML for consistent appearance
        if not (widget is not None):
            raise ValueError("widget must be provided")
        styled_text = f"""
            <div style="
                background-color: #313244;
                color: #cdd6f4;
                padding: 8px;
                border-radius: 4px;
                max-width: 300px;
            ">
                {text}
            </div>
        """
        widget.setToolTip(styled_text)

    def update_tooltip(self, key: str, text: str) -> None:
        """Update a tooltip and refresh all registered widgets.

        Args:
            key: Tooltip key
            text: New tooltip text
        """
        if not (key is not None):
            raise ValueError("key must be provided")
        self._tooltips[key] = text
        if key in self._widgets:
            for widget in self._widgets[key]:
                self._apply_tooltip(widget, text)

    def get_tooltip(self, key: str) -> str | None:
        """Get tooltip text for a key.

        Args:
            key: Tooltip key

        Returns:
            Tooltip text or None if not found
        """
        return self._tooltips.get(key)


class HelpManager:
    """Singleton manager for the help system.

    Manages help content loading, caching, and display.
    """

    def __init__(self) -> None:
        """Initialize the help manager."""
        self._topics: dict[str, str] = {}
        self._topic_files: dict[str, Path] = {}
        self._help_dir: Path | None = None
        self._user_manual: str = ""
        self._category_mappings: dict[str, str] = {}
        self._tooltip_manager = TooltipManager()

    @property
    def tooltip_manager(self) -> TooltipManager:
        """Get the tooltip manager instance."""
        return self._tooltip_manager

    def set_help_directory(self, path: str | Path) -> None:
        """Set the help files directory.

        Args:
            path: Path to the help directory
        """
        if not (path is not None):
            raise ValueError("path must be provided")
        self._help_dir = Path(path)
        self._scan_help_files()

    def set_user_manual_path(self, path: str | Path) -> None:
        """Set the path to the user manual.

        Args:
            path: Path to USER_MANUAL.md
        """
        if not (path is not None):
            raise ValueError("path must be provided")
        manual_path = Path(path)
        if manual_path.exists():
            self._user_manual = load_help_from_file(manual_path)

    def _scan_help_files(self) -> None:
        """Scan help directory for markdown files."""
        if not self._help_dir or not self._help_dir.exists():
            return

        for md_file in self._help_dir.glob("*.md"):
            topic_id = md_file.stem
            self._topic_files[topic_id] = md_file

    def register_topic(self, topic_id: str, content: str) -> None:
        """Register a help topic with content.

        Args:
            topic_id: Unique topic identifier
            content: Markdown content
        """
        self._topics[topic_id] = content

    def register_category_mapping(self, category: str, topic_id: str) -> None:
        """Map a tool category to a help topic.

        Args:
            category: Tool category name
            topic_id: Help topic ID
        """
        self._category_mappings[category] = topic_id

    def get_topic_content(self, topic_id: str) -> str:
        """Get content for a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            Topic content or error message
        """
        # Check registered topics first
        if not (topic_id is not None):
            raise ValueError("topic_id must be provided")
        if topic_id in self._topics:
            return self._topics[topic_id]

        # Check topic files
        if topic_id in self._topic_files:
            content = load_help_from_file(self._topic_files[topic_id])
            self._topics[topic_id] = content  # Cache it
            return content

        return f"Help topic '{topic_id}' not found."

    def get_all_topics(self) -> dict[str, str]:
        """Get all available topics.

        Returns:
            Dict mapping topic IDs to content
        """
        # Load all topic files
        for topic_id in self._topic_files:
            if topic_id not in self._topics:
                self._topics[topic_id] = load_help_from_file(
                    self._topic_files[topic_id]
                )

        return self._topics.copy()

    def get_user_manual(self) -> str:
        """Get the user manual content.

        Returns:
            User manual markdown content
        """
        return self._user_manual

    def show_topic(self, topic_id: str, parent: QWidget | None = None) -> None:
        """Show a help dialog for a specific topic.

        Args:
            topic_id: Topic to display
            parent: Parent widget for the dialog
        """
        if not (topic_id is not None):
            raise ValueError("topic_id must be provided")
        content = self.get_topic_content(topic_id)
        all_topics = self.get_all_topics()

        # Use topic_id as title, replacing underscores with spaces
        title = topic_id.replace("_", " ").title()

        dialog = HelpDialog(parent, title, content, all_topics)
        dialog.exec()

    def show_user_manual(self, parent: QWidget | None = None) -> None:
        """Show the user manual dialog.

        Args:
            parent: Parent widget for the dialog
        """
        if not self._user_manual:
            QMessageBox.warning(
                parent,
                "User Manual Not Found",
                "The user manual could not be loaded.\n"
                "Please ensure docs/USER_MANUAL.md exists.",
            )
            return

        all_topics = self.get_all_topics()
        dialog = HelpDialog(parent, "User Manual", self._user_manual, all_topics)
        dialog.exec()

    def show_category_help(self, category: str, parent: QWidget | None = None) -> None:
        """Show help for a tool category.

        Args:
            category: Category name
            parent: Parent widget
        """
        if category in self._category_mappings:
            self.show_topic(self._category_mappings[category], parent)
        else:
            # Show generic category help
            content = f"""# {category}

This category contains tools for {category.lower()}.

Select a specific tool from the launcher to see its description,
or consult the User Manual for detailed documentation.
"""
            dialog = HelpDialog(parent, category, content, self.get_all_topics())
            dialog.exec()

    def show_about_dialog(self, parent: QWidget | None = None) -> None:
        """Show the About dialog.

        Args:
            parent: Parent widget
        """
        about_text = """
# About Unified Tools Launcher

**Version:** 1.0

The Unified Tools Launcher provides a central interface for accessing
all tools in the Tools repository.

## Features

- **Tabbed Interface**: Tools organized by category
- **Plugin System**: Automatic tool discovery
- **Help System**: Context-sensitive documentation
- **Theme Support**: Multiple color themes available

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F1 | Open Help |
| Ctrl+L | Launch selected tool |
| Ctrl+Tab | Next category tab |
| Ctrl+Q | Quit launcher |

## Credits

Built with Python 3.11+ and PyQt6.

Part of the Tools Monorepo.
"""
        dialog = HelpDialog(parent, "About", about_text)
        dialog.resize(600, 500)
        dialog.exec()


def create_help_menu_actions(
    parent: QWidget,
    help_manager: HelpManager,
) -> list[QAction]:
    """Create standard help menu actions.

    Args:
        parent: Parent widget for the actions
        help_manager: HelpManager instance

    Returns:
        List of QAction objects for the help menu
    """
    if not (parent is not None):
        raise ValueError("parent must be provided")
    actions: list[QAction] = []

    # User Manual action
    manual_action = QAction("User Manual", parent)
    manual_action.setShortcut(QKeySequence("F1"))
    manual_action.triggered.connect(lambda: help_manager.show_user_manual(parent))
    actions.append(manual_action)

    # Tool Help action (shows current category help)
    tool_help_action = QAction("Tool Help...", parent)
    tool_help_action.triggered.connect(
        lambda: help_manager.show_topic("getting_started", parent)
    )
    actions.append(tool_help_action)

    # Separator (None represents separator)
    actions.append(None)  # type: ignore[arg-type]

    # About action
    about_action = QAction("About", parent)
    about_action.triggered.connect(lambda: help_manager.show_about_dialog(parent))
    actions.append(about_action)

    return actions

"""Collapsible category group component for organizing tools by category."""

from typing import Any

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from tools.gui.components.tool_card import ToolCard


class CollapsibleCategoryGroup(QFrame):
    """A collapsible group of tools organized by category."""

    def __init__(
        self,
        category_name: str,
        tools: list[dict[str, Any]],
        launch_callback: Any,
    ) -> None:
        """Initialize the collapsible category group.

        Args:
            category_name: Name of the category.
            tools: List of tool configuration dicts for this category.
            launch_callback: Callback function when a tool is launched.
        """
        super().__init__()
        self.category_name = category_name
        self.tools = tools
        self.launch_callback = launch_callback
        self.is_expanded = True
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the collapsible category UI."""
        self.setFrameStyle(QFrame.Shape.NoFrame)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header with category name and expand/collapse button
        header = QFrame()
        header.setObjectName("categoryHeader")
        header.setStyleSheet("""
            #categoryHeader {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 0px;
            }
            #categoryHeader:hover {
                background-color: #efefef;
            }
            """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(0)

        # Category label with count
        label_layout = QVBoxLayout()
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(2)

        title_label = QLabel(self.category_name)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setObjectName("categoryTitle")
        label_layout.addWidget(title_label)

        count_label = QLabel(
            f"{len(self.tools)} tool{'s' if len(self.tools) != 1 else ''}"
        )
        count_label.setFont(QFont("Segoe UI", 9))
        count_label.setStyleSheet("color: #666;")
        label_layout.addWidget(count_label)

        header_layout.addLayout(label_layout)

        # Make header clickable
        header.mousePressEvent = self.on_header_clicked

        main_layout.addWidget(header)

        # Content area (tools grid)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ddd;
                border-top: none;
            }
            """)

        self.content_widget = QWidget()
        self.content_layout = QGridLayout(self.content_widget)
        self.content_layout.setSpacing(15)
        self.content_layout.setContentsMargins(15, 15, 15, 15)

        # Populate tools
        cols = 2
        for i, tool_info in enumerate(self.tools):
            card = ToolCard(tool_info, self.launch_callback)
            row = i // cols
            col = i % cols
            self.content_layout.addWidget(card, row, col)

        # Push items to top-left
        self.content_layout.setRowStretch(self.content_layout.rowCount(), 1)
        self.content_layout.setColumnStretch(cols, 1)

        self.scroll_area.setWidget(self.content_widget)
        main_layout.addWidget(self.scroll_area)

    def on_header_clicked(self, event: Any) -> None:
        """Handle header click to toggle expansion."""
        self.toggle_expansion()

    def toggle_expansion(self) -> None:
        """Toggle the expanded state of the category."""
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.scroll_area.show()
        else:
            self.scroll_area.hide()

    def set_expanded(self, expanded: bool) -> None:
        """Set the expanded state directly.

        Args:
            expanded: Whether the category should be expanded.
        """
        self.is_expanded = expanded
        if expanded:
            self.scroll_area.show()
        else:
            self.scroll_area.hide()

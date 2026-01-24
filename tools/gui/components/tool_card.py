"""ToolCard component for the Unified Launcher."""

from collections.abc import Callable
from typing import Any

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


class ToolCard(QFrame):
    """A card widget representing a single launchable tool."""

    def __init__(
        self,
        tool_info: dict[str, Any],
        launch_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        super().__init__()
        self.tool_info = tool_info
        self.launch_callback = launch_callback
        self.setup_ui()

    def setup_ui(self) -> None:
        """Initialize the card UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            ToolCard {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
            }
            ToolCard:hover {
                border: 1px solid #2196F3;
                background-color: #f8fbff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        name = self.tool_info.get("name", "Unknown Tool")
        title = QLabel(name)
        title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333;
        """)
        header.addWidget(title)
        header.addStretch()

        # Type badge
        tool_type = self.tool_info.get("type", "unknown")
        badge = QLabel(f" {tool_type.upper()} ")
        badge.setStyleSheet(f"""
            background-color: {self._get_type_color(tool_type)};
            color: white;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
            padding: 2px;
        """)
        header.addWidget(badge)
        layout.addLayout(header)

        # Description
        desc_text = self.tool_info.get("desc", "No description available.")
        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(desc)

        # Path
        path_text = self.tool_info.get("path", "")
        path_lbl = QLabel(path_text)
        path_lbl.setWordWrap(True)
        path_lbl.setStyleSheet("""
            color: #999;
            font-family: monospace;
            font-size: 10px;
        """)
        layout.addWidget(path_lbl)
        layout.addStretch()

        # Launch Button
        btn = QPushButton("Launch Tool")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: self.launch_callback(self.tool_info))
        btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(btn)

    def _get_type_color(self, tool_type: str) -> str:
        """Get badge color based on tool type."""
        colors = {
            "python": "#4CAF50",  # Green
            "matlab": "#FF9800",  # Orange
            "web": "#2196F3",  # Blue
            "browser": "#9C27B0",  # Purple
        }
        return colors.get(tool_type.lower(), "#9E9E9E")  # Grey default

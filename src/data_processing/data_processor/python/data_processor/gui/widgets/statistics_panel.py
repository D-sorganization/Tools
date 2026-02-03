"""Statistics display panel widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


class StatisticsPanel(QWidget):
    """Panel for displaying and calculating signal statistics."""

    # Signals
    calculate_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the statistics panel."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        self._add_title(layout)
        self._add_calculate_button(layout)
        self._add_stats_display(layout)

    def _add_title(self, layout: QVBoxLayout) -> None:
        """Add title label."""
        title = QLabel("Signal Statistics")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

    def _add_calculate_button(self, layout: QVBoxLayout) -> None:
        """Add calculate button."""
        self.calculate_button = QPushButton("Calculate Statistics")
        layout.addWidget(self.calculate_button)

    def _add_stats_display(self, layout: QVBoxLayout) -> None:
        """Add statistics display area."""
        self.stats_display = QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMinimumHeight(200)
        layout.addWidget(self.stats_display)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.calculate_button.clicked.connect(self._on_calculate_clicked)

    def _on_calculate_clicked(self) -> None:
        """Handle calculate button click."""
        self.calculate_requested.emit()

    def set_statistics(self, stats: dict[str, dict[str, Any]]) -> None:
        """Set statistics to display."""
        text = self._format_statistics(stats)
        self.stats_display.setText(text)

    def _format_statistics(self, stats: dict[str, dict[str, Any]]) -> str:
        """Format statistics dictionary as text."""
        lines = ["=== Signal Statistics ===\n"]

        for signal_name, signal_stats in stats.items():
            lines.append(f"\n{signal_name}:")
            lines.extend(self._format_signal_stats(signal_stats))

        return "\n".join(lines)

    def _format_signal_stats(self, stats: dict[str, Any]) -> list[str]:
        """Format statistics for a single signal."""
        lines = []
        for key, value in stats.items():
            formatted_value = self._format_stat_value(value)
            lines.append(f"  {key}: {formatted_value}")
        return lines

    def _format_stat_value(self, value: Any) -> str:
        """Format a single statistic value."""
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def clear(self) -> None:
        """Clear the statistics display."""
        self.stats_display.clear()

    def append_text(self, text: str) -> None:
        """Append text to the display."""
        self.stats_display.append(text)

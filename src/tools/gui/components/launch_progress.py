"""Launch progress indicator for the Unified Tools Launcher.

Displays visual feedback during tool launch with spinner animation,
status message, and timeout handling.
"""

from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class LaunchProgressDialog(QDialog):
    """Modal dialog showing launch progress with spinner and timeout."""

    # Spinner animation frames
    SPINNER_FRAMES = ["|", "/", "-", "\\"]

    def __init__(
        self,
        parent: Any,
        tool_name: str,
        timeout_seconds: int = 300,  # 5 minutes
    ) -> None:
        """Initialize the launch progress dialog.

        Args:
            parent: Parent widget.
            tool_name: Name of the tool being launched.
            timeout_seconds: Maximum time to wait before timing out.
        """
        super().__init__(parent)
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = 0
        self.spinner_index = 0
        self.setup_ui()
        self.setup_timers()

    def setup_ui(self) -> None:
        """Set up the progress dialog UI."""
        self.setWindowTitle(f"Launching {self.tool_name}")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.resize(400, 200)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
                border-radius: 8px;
            }
            QLabel#titleLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
            QLabel#statusLabel {
                font-size: 12px;
                color: #666;
            }
            QLabel#spinnerLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2196F3;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)

        self.title_label = QLabel(f"Starting {self.tool_name}...")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(self.title_label)

        spinner_status_layout = QVBoxLayout()
        self.spinner_label = QLabel(self.SPINNER_FRAMES[0])
        self.spinner_label.setObjectName("spinnerLabel")
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinner_status_layout.addWidget(self.spinner_label)

        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinner_status_layout.addWidget(self.status_label)
        layout.addLayout(spinner_status_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.timeout_seconds)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.time_label = QLabel(f"Time: 0/{self.timeout_seconds}s")
        self.time_label.setObjectName("statusLabel")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel)
        layout.addWidget(cancel_button)

    def setup_timers(self) -> None:
        """Set up animation and timeout timers."""
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self.update_spinner)
        self.spinner_timer.start(100)  # Update spinner every 100ms

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(1000)  # Update progress every second

    def update_spinner(self) -> None:
        """Update the spinner animation frame."""
        self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_FRAMES)
        self.spinner_label.setText(self.SPINNER_FRAMES[self.spinner_index])

    def update_progress(self) -> None:
        """Update the progress bar and check for timeout."""
        self.elapsed_seconds += 1
        self.progress_bar.setValue(self.elapsed_seconds)
        self.time_label.setText(f"Time: {self.elapsed_seconds}/{self.timeout_seconds}s")

        if self.elapsed_seconds >= self.timeout_seconds:
            self.on_timeout()

    def on_timeout(self) -> None:
        """Handle timeout event."""
        self.spinner_timer.stop()
        self.progress_timer.stop()
        self.status_label.setText("Timeout: Tool launch took too long")
        self.title_label.setText(f"Timeout launching {self.tool_name}")

    def on_cancel(self) -> None:
        """Handle cancel button click."""
        self.spinner_timer.stop()
        self.progress_timer.stop()
        self.reject()

    def on_success(self) -> None:
        """Mark the launch as successful."""
        self.spinner_timer.stop()
        self.progress_timer.stop()
        self.spinner_label.setText("✓")
        self.status_label.setText("Tool launched successfully!")
        self.title_label.setText(f"Successfully launched {self.tool_name}")

        # Auto-close after 2 seconds
        QTimer.singleShot(2000, self.accept)

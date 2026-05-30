"""Error notification dialog for the Unified Tools Launcher.

Provides user-friendly error notifications when tool launches fail,
with suggestions for remediation based on error type.
"""

from typing import Any

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from shared.python.theme.integration import ThemedDialogMixin
from shared.python.ui import HoverCopyTextBrowser
from tools.launch_utils import (
    LaunchError,
    PlatformError,
    SecurityError,
    ToolNotFoundError,
)


class ErrorNotificationDialog(ThemedDialogMixin, QDialog):
    """Modal dialog for displaying tool launch errors with suggestions."""

    def __init__(
        self,
        parent: Any,
        tool_name: str,
        error: Exception,
    ) -> None:
        """Initialize the error notification dialog.

        Args:
            parent: Parent widget.
            tool_name: Name of the tool that failed to launch.
            error: The exception that was raised.
        """
        super().__init__(parent)
        self.tool_name = tool_name
        self.error = error
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the error notification UI."""
        self.setWindowTitle(f"Launch Failed: {self.tool_name}")
        self.setModal(True)
        self.resize(500, 400)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel#titleLabel {
                font-size: 14px;
                font-weight: bold;
                color: #d32f2f;
            }
            QLabel#subtitleLabel {
                font-size: 11px;
                color: #666;
            }
            QTextEdit, QTextBrowser {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            """)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"Failed to launch: {self.tool_name}")
        title.setObjectName("titleLabel")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel("An error occurred during tool launch. See details below.")
        subtitle.setObjectName("subtitleLabel")
        layout.addWidget(subtitle)

        error_type = QLabel(f"Error Type: {type(self.error).__name__}")
        error_type.setFont(QFont("Segoe UI", 10))
        layout.addWidget(error_type)

        error_msg_label = QLabel("Error Message:")
        error_msg_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(error_msg_label)

        error_text = HoverCopyTextBrowser()
        error_text.setReadOnly(True)
        error_text.setPlainText(str(self.error))
        error_text.setMaximumHeight(120)
        layout.addWidget(error_text)

        suggestions_label = QLabel("Suggestions:")
        suggestions_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(suggestions_label)

        suggestions_text = HoverCopyTextBrowser()
        suggestions_text.setReadOnly(True)
        suggestions_text.setPlainText(self._get_suggestions())
        suggestions_text.setMaximumHeight(120)
        layout.addWidget(suggestions_text)

        layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def _get_suggestions(self) -> str:
        """Generate suggestions based on the error type."""
        error_msg = str(self.error).lower()

        if isinstance(self.error, ToolNotFoundError):
            if "not found" in error_msg:
                return (
                    "Tool file not found:\n"
                    "• Check that the tool path in tools.json is correct\n"
                    "• Verify the file exists at the specified location\n"
                    "• Try updating tools.json or reinstalling the tool"
                )
            elif "python" in error_msg:
                return (
                    "Python executable not found:\n"
                    "• Ensure Python 3.11+ is installed\n"
                    "• Add Python to your system PATH\n"
                    "• Restart the launcher after installing Python"
                )

        if isinstance(self.error, SecurityError):
            return (
                "Security error:\n"
                "• The tool path contains invalid characters\n"
                "• Path traversal protection may have blocked this tool\n"
                "• Contact the administrator for more information"
            )

        if isinstance(self.error, PlatformError):
            return (
                "Platform error:\n"
                "• This tool may not be supported on your platform\n"
                "• Check the tool requirements documentation\n"
                f"• Current platform: {self._get_platform_info()}"
            )

        if isinstance(self.error, LaunchError):
            if "permission" in error_msg:
                return (
                    "Permission denied:\n"
                    "• Check that you have read/execute permissions\n"
                    "• Run as administrator if needed\n"
                    "• Check file ownership and permissions"
                )
            elif "not found" in error_msg:
                return (
                    "Required component not found:\n"
                    "• A required runtime or dependency is missing\n"
                    "• Check tool documentation for dependencies\n"
                    "• Install missing dependencies and try again"
                )

        return (
            "General troubleshooting:\n"
            "• Check the launcher log file (unified_launcher.log)\n"
            "• Verify tool configuration in tools.json\n"
            "• Try launching from command line for more details\n"
            "• Contact support if the problem persists"
        )

    def _get_platform_info(self) -> str:
        """Get the current platform information."""
        import sys

        if sys.platform == "win32":
            return "Windows"
        elif sys.platform == "darwin":
            return "macOS"
        else:
            return "Linux"

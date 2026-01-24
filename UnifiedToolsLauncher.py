"""
Unified Tools Launcher - Modern PyQt6-based launcher for the Tools repository.

This is the PRIMARY and RECOMMENDED launcher for accessing all tools.
Provides a clean, tabbed interface for launching Python, MATLAB, and web tools.
"""

import html
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable

if sys.version_info < (3, 10):  # noqa: UP036
    print(
        "Critical Error: UnifiedToolsLauncher requires Python 3.10 or newer.",
        file=sys.stderr,
    )
    print(f"Current version: {sys.version}", file=sys.stderr)
    sys.exit(1)
from pathlib import Path
from typing import Any

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def validate_tools_config(
    tools_dict: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Validate and sanitize tools configuration.

    Args:
        tools_dict: Dictionary of tool categories and lists of tools.

    Returns:
        Validated dictionary with invalid entries removed.
    """
    valid_tools = {}
    allowed_types = {"python", "matlab", "web", "browser", "bat"}

    for category, tool_list in tools_dict.items():
        valid_list = []
        for tool in tool_list:
            # Validate type
            tool_type = tool.get("type")
            if tool_type not in allowed_types:
                print(
                    f"Warning: Skipping tool '{tool.get('name')}' with invalid type '{tool_type}'",
                    file=sys.stderr,
                )
                continue

            # Validate path for directory traversal
            path_str = str(tool.get("path", ""))
            if ".." in path_str:
                print(
                    f"Warning: Skipping tool '{tool.get('name')}' with suspicious path '{path_str}'",
                    file=sys.stderr,
                )
                continue

            valid_list.append(tool)

        if valid_list:
            valid_tools[category] = valid_list

    return valid_tools


# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
REPO_ROOT = Path(__file__).parent.absolute()
ensure_utils_in_path()

# Use shared path setup utility for consistency
try:
    from utils.path_setup import setup_python_path

    setup_python_path(repo_root=REPO_ROOT)
except ImportError:
    # Fallback if shared utility not available
    ensure_utils_in_path()

# Import compatibility shim early to verify environment and provide friendly errors
try:
    from utils.compatibility import UTC, StrEnum  # noqa: F401
except ImportError as e:
    print(
        "Critical Error: Failed to import compatibility shim.",
        file=sys.stderr,
    )
    print(
        "This may indicate a Python version incompatibility or missing dependencies.",
        file=sys.stderr,
    )
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    print(
        "\nPlease ensure you are using Python 3.10 or newer.",
        file=sys.stderr,
    )
    sys.exit(1)

# Load Tool Definitions using PluginManager
try:
    from core.plugin_manager import PluginManager

    plugin_manager = PluginManager(REPO_ROOT)
    # Convert PluginManager tool objects to simple dicts so the existing UI code
    # (which relies on dict keys) does not need a massive refactor right now.
    plugin_manager.load_tools()
    TOOLS = {}
    for cat, tool_list in plugin_manager.tools.items():
        TOOLS[cat] = [
            {"name": t.name, "path": t.path, "type": t.type, "desc": t.desc}
            for t in tool_list
        ]
except Exception as e:
    # Fallback if core logic missing (should not happen with sys.path fix)
    sys.stderr.write(
        f"Critical: PluginManager failed ({e}). Using fallback JSON load.\n"
    )
    TOOLS_FILE = REPO_ROOT / "tools.json"
    TOOLS = {}
    if TOOLS_FILE.exists():
        try:
            TOOLS = safe_read_json(TOOLS_FILE, default=None)
        except Exception as e:
            sys.stderr.write(f"Error loading tools.json: {e}\n")
        # Fallback to empty or default if needed

# Validate loaded tools
TOOLS = validate_tools_config(TOOLS)


# =============================================================================
# STYLING
# =============================================================================
STYLE_SHEET = """
QMainWindow {
    background-color: #1a1b26;
}
QTabWidget::pane {
    border: 1px solid #414868;
    background-color: #1a1b26;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #24283b;
    color: #c0caf5;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #7aa2f7;
    color: #1a1b26;
    font-weight: bold;
}
QGroupBox {
    border: 1px solid #414868;
    border-radius: 6px;
    margin-top: 20px;
    background-color: #24283b;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #7aa2f7;
    font-weight: bold;
}
QPushButton {
    background-color: #7aa2f7;
    color: #1a1b26;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
    text-align: left;
}
QPushButton:hover {
    background-color: #bb9af7;
}
QPushButton:pressed {
    background-color: #7dcfff;
}
QLabel {
    color: #c0caf5;
}
QLabel#DescLabel {
    color: #565f89;
    font-style: italic;
    font-size: 11px;
}
QTextEdit {
    background-color: #0f0f14;
    color: #9ece6a;
    border: 1px solid #414868;
    border-radius: 4px;
    font-family: Consolas, monospace;
}
QCheckBox {
    color: #c0caf5;
    spacing: 5px;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
}
"""


# =============================================================================
# LAUNCHER LOGIC
# =============================================================================
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Check if exists
        full_path = REPO_ROOT / self.tool_info["path"]
        exists = full_path.exists()

        # Dependency check (simplified)
        if exists:
            if self.tool_info.get("type") == "python":
                # Assuming python is available if we are running this script
                pass
            elif self.tool_info.get("type") == "matlab":
                # Basic check if matlab is in path?
                pass

        # Button
        # Sanitize name to prevent HTML injection
        safe_name = html.escape(self.tool_info.get("name", "Unknown"))
        btn_text = f"🚀 {safe_name}"
        self.btn = QPushButton(btn_text)
        self.btn.clicked.connect(lambda: self.launch_callback(self.tool_info))
        self.btn.setEnabled(exists)

        if not exists:
            self.btn.setStyleSheet("background-color: #f7768e; color: #1a1b26;")
            self.btn.setText(f"❌ {safe_name} (Missing)")
            self.btn.setToolTip(f"File not found: {full_path}")

        layout.addWidget(self.btn)

        # Description
        # Sanitize description to prevent HTML injection
        safe_desc = html.escape(self.tool_info.get("desc", ""))
        desc = QLabel(safe_desc)
        desc.setObjectName("DescLabel")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Path
        path_lbl = QLabel(str(self.tool_info["path"]))
        path_lbl.setStyleSheet("color: #414868; font-size: 10px;")
        layout.addWidget(path_lbl)


class UnifiedLauncher(QMainWindow):
    """Main launcher window with tabbed interface for all tools."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Antigravity Unified Tools Launcher")
        self.resize(1000, 700)
        self.setStyleSheet(STYLE_SHEET)

        # Icon
        icon_path = REPO_ROOT / "assets" / "tools_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.setup_ui()

    def _create_header_layout(self) -> QHBoxLayout:
        """Create the header layout with title and debug checkbox."""
        header_layout = QHBoxLayout()
        header = QLabel("🛠️ Unified Tools Repository")
        header.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #7aa2f7; margin-bottom: 10px;"
        )
        header_layout.addWidget(header)
        header_layout.addStretch()

        self.debug_mode = QCheckBox("Debug Mode")
        self.debug_mode.setToolTip("Enable verbose output when launching tools")
        header_layout.addWidget(self.debug_mode)
        return header_layout

    def _create_log_area(self) -> QFrame:
        """Create the activity log area widget."""
        log_group = QFrame()
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(0, 10, 0, 0)

        lbl = QLabel("Activity Log")
        lbl.setStyleSheet("font-weight: bold;")
        log_layout.addWidget(lbl)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        log_layout.addWidget(self.log_area)
        return log_group

    def _create_tool_tabs(self) -> QTabWidget:
        """Create the tabbed interface for tool categories."""
        tabs = QTabWidget()
        for category, tools in TOOLS.items():
            tab = QWidget()
            self.setup_category_tab(tab, tools)
            tabs.addTab(tab, category)
        return tabs

    def setup_ui(self) -> None:
        """Set up the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        main_layout.addLayout(self._create_header_layout())

        self.tabs = self._create_tool_tabs()
        main_layout.addWidget(self.tabs)

        main_layout.addWidget(self._create_log_area())

        if not TOOLS:
            self.log("❌ Warning: tools.json not found or empty.")
            QMessageBox.warning(
                self,
                "Configuration Error",
                "Could not load tool definitions from tools.json.\nThe launcher will be empty.",
            )

    def setup_category_tab(self, tab: QWidget, tools: list[dict[str, Any]]) -> None:
        """Set up a tab for a category of tools."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")

        content_widget = QWidget()
        layout = QGridLayout(content_widget)
        layout.setSpacing(15)

        row = 0
        col = 0
        max_cols = 3

        for tool in tools:
            card = ToolCard(tool, self.launch_tool)
            layout.addWidget(card, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Push content up
        layout.setRowStretch(row + 1, 1)

        scroll.setWidget(content_widget)

        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)

    def log(self, message: str) -> None:
        """Log a message to the activity log area."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
        cursor = self.log_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_area.setTextCursor(cursor)

    def _validate_and_sanitize_path(self, path_str: str) -> Path:
        """
        Validate and sanitize tool path to prevent path traversal attacks.

        Args:
            path_str: Path string from tool_info

        Returns:
            Validated and sanitized Path object

        Raises:
            ValueError: If path is invalid or outside repository
        """
        # Convert to Path and resolve to absolute
        try:
            path = Path(path_str)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid path format: {path_str}") from e

        # Resolve to absolute path to prevent relative path tricks
        try:
            full_path = (REPO_ROOT / path).resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve path: {path_str}") from e

        # Ensure path is within repository root (prevent path traversal)
        repo_root_abs = REPO_ROOT.resolve()
        try:
            # Use relative_to to ensure path is actually within repo
            full_path.relative_to(repo_root_abs)
        except ValueError:
            raise ValueError(
                f"Security Alert: Path outside repository: {full_path}"
            ) from None

        # Additional validation: ensure path exists and is a file
        if not full_path.exists():
            raise ValueError(f"Tool file not found: {full_path}")

        if not full_path.is_file():
            raise ValueError(f"Path is not a file: {full_path}")

        return full_path

    def _launch_python_tool(
        self, path: Path, tool_info: dict[str, Any], is_debug: bool
    ) -> None:
        """Launch a Python tool."""
        args = [sys.executable, str(path)]
        try:
            if is_debug:
                process = subprocess.Popen(
                    args,
                    cwd=path.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.log("✅ Process started (Python)")
                self.log("DEBUG: Process PID: " + str(process.pid))
            else:
                process = subprocess.Popen(
                    args,
                    cwd=path.parent,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.log("✅ Process started (Python)")
                import time

                time.sleep(0.5)
                if process.poll() is not None:
                    error_msg = (
                        f"Tool exited immediately (exit code: {process.returncode})\n\n"
                        f"Tool: {tool_info.get('name', 'Unknown')}\n"
                        f"Path: {path}\n\n"
                        "Check the tool's requirements and dependencies."
                    )
                    self.log(f"❌ {error_msg}")
                    QMessageBox.warning(self, "Tool Launch Warning", error_msg)

        except FileNotFoundError:
            error_msg = (
                f"Python executable not found: {sys.executable}\n\n"
                "Please ensure Python is installed and in your system PATH."
            )
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Launch Error", error_msg)
            return
        except PermissionError:
            error_msg = (
                f"Permission denied: Cannot execute {path}\n\n"
                "Please check file permissions or run with appropriate privileges."
            )
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Launch Error", error_msg)
            return
        except OSError as e:
            error_msg = (
                f"Failed to start Python process: {e}\n\n"
                f"Tool: {tool_info.get('name', 'Unknown')}\n"
                f"Path: {path}"
            )
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Launch Error", error_msg)
            return

    def _launch_matlab_tool(
        self, path: Path, tool_info: dict[str, Any], is_debug: bool
    ) -> None:
        """Launch a MATLAB tool."""
        self.log("ℹ️ Attempting to launch MATLAB...")
        sanitized_path = str(path).replace("'", "''")
        matlab_script = f"run('{sanitized_path}');"
        cmd_list = ["matlab", "-nosplash", "-nodesktop", "-r", matlab_script]
        try:
            process = subprocess.Popen(
                cmd_list,
                cwd=path.parent,
                stdout=subprocess.DEVNULL if not is_debug else None,
                stderr=subprocess.DEVNULL if not is_debug else None,
            )
            self.log("✅ MATLAB command sent")
            if is_debug:
                self.log(f"DEBUG: MATLAB process PID: {process.pid}")
        except FileNotFoundError:
            error_msg = (
                f"MATLAB not found in system PATH.\n\n"
                f"Tool: {tool_info.get('name', 'Unknown')}\n"
                f"Path: {path}\n\n"
                "Attempting to open file in default editor..."
            )
            self.log(f"⚠️ {error_msg}")
            QMessageBox.warning(self, "MATLAB Not Found", error_msg)
            self._open_file_with_default_app(path, tool_info)

    def _open_file_with_default_app(
        self, path: Path, tool_info: dict[str, Any]
    ) -> None:
        """Open a file with the system's default application."""
        try:
            if hasattr(os, "startfile"):
                os.startfile(path)  # type: ignore[attr-defined]
                self.log("Opened file in default editor (Windows)")
            else:
                subprocess.Popen(["xdg-open", str(path)])
                self.log("Opened file with xdg-open (Linux/macOS)")
        except OSError as open_error:
            final_error = (
                f"Could not open file: {open_error}\n\n"
                f"Tool: {tool_info.get('name', 'Unknown')}\n"
                f"Path: {path}"
            )
            self.log(f"❌ {final_error}")
            QMessageBox.critical(self, "File Open Error", final_error)
            return

    def _launch_browser_tool(self, path: Path) -> None:
        """Launch a web/browser tool."""
        try:
            uri = path.as_uri()
            if not (
                uri.startswith("file://") or uri.startswith(("http://", "https://"))
            ):
                raise ValueError(f"Invalid URI scheme: {uri}")
            webbrowser.open(uri)
            self.log("✅ Opened in default browser")
        except (ValueError, OSError) as e:
            error_msg = f"Failed to open browser: {e}"
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Browser Error", error_msg)
            return

    def _launch_batch_tool(
        self, path: Path, tool_info: dict[str, Any], is_debug: bool
    ) -> None:
        """Launch a Windows batch script."""
        if path.suffix.lower() not in [".bat", ".cmd"]:
            raise ValueError(
                "Security: File must be .bat or .cmd to execute as batch script"
            )
        try:
            process = subprocess.Popen(
                ["cmd.exe", "/c", str(path)],
                cwd=path.parent,
                stdout=subprocess.DEVNULL if not is_debug else None,
                stderr=subprocess.DEVNULL if not is_debug else None,
            )
            self.log("✅ Batch script executed")
            if is_debug:
                self.log(f"DEBUG: Batch process PID: {process.pid}")
        except FileNotFoundError:
            error_msg = (
                f"Windows command processor (cmd.exe) not found.\n\n"
                f"Tool: {tool_info.get('name', 'Unknown')}\n"
                f"Path: {path}\n\n"
                "This tool requires Windows."
            )
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Platform Error", error_msg)
        except OSError as e:
            error_msg = (
                f"Failed to execute batch script: {e}\n\n"
                f"Tool: {tool_info.get('name', 'Unknown')}\n"
                f"Path: {path}"
            )
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Launch Error", error_msg)

    def launch_tool(self, tool_info: dict[str, Any]) -> None:
        """Launch the specified tool based on its type."""
        type_ = tool_info["type"]
        is_debug = self.debug_mode.isChecked()

        self.log(f"Launching {tool_info['name']}...")

        try:
            try:
                path = self._validate_and_sanitize_path(tool_info["path"])
            except ValueError as e:
                error_msg = f"Path validation failed: {e}"
                self.log(f"❌ {error_msg}")
                QMessageBox.critical(self, "Security Error", error_msg)
                return

            self.log(f"Path: {path}")

            if is_debug:
                self.log(f"DEBUG: Mode enabled. Launching {type_} tool.")

            if type_ == "python":
                self._launch_python_tool(path, tool_info, is_debug)
            elif type_ == "matlab":
                self._launch_matlab_tool(path, tool_info, is_debug)
            elif type_ in ("web", "browser"):
                self._launch_browser_tool(path)
            elif type_ == "bat":
                self._launch_batch_tool(path, tool_info, is_debug)
            else:
                error_msg = (
                    f"Unknown tool type: {type_}\n\n"
                    f"Tool: {tool_info.get('name', 'Unknown')}\n"
                    f"Supported types: python, matlab, web, browser, bat"
                )
                self.log(f"❌ {error_msg}")
                QMessageBox.warning(self, "Unknown Tool Type", error_msg)

        except ValueError as e:
            error_msg = (
                f"Path validation failed: {e}\n\n"
                f"Tool: {tool_info.get('name', 'Unknown')}\n"
                f"Path: {tool_info.get('path', 'Unknown')}"
            )
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Security Error", error_msg)
        except OSError as e:
            error_msg = (
                f"Unexpected error: {e}\n\n"
                f"Tool: {tool_info.get('name', 'Unknown')}\n"
                f"Type: {type_}\n\n"
                "Please check the activity log for more details."
            )
            self.log(f"❌ {error_msg}")
            QMessageBox.critical(self, "Launch Error", error_msg)


# =============================================================================
# ENTRY POINT
# =============================================================================
def main() -> None:
    """Entry point for the Unified Tools Launcher application."""
    app = QApplication(sys.argv)

    # Set app style
    app.setStyle("Fusion")

    # Run
    window = UnifiedLauncher()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

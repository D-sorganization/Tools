"""
Unified Tools Launcher - Modern PyQt6-based launcher for the Tools repository.

This is the PRIMARY and RECOMMENDED launcher for accessing all tools.
Provides a clean, tabbed interface for launching Python, MATLAB, and web tools.
"""

import json
import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable
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

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(REPO_ROOT / "python" / "src"))

# try to import compatibility shim to verify environment
try:
    from utils.compatibility import UTC, StrEnum  # noqa: F401
except ImportError:
    # If this fails, we are likely in a very broken state
    pass

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
    sys.stderr.write(f"Critical: PluginManager failed ({e}). Using fallback JSON load.\n")
    TOOLS_FILE = REPO_ROOT / "tools.json"
    TOOLS = {}
    if TOOLS_FILE.exists():
        try:
            with open(TOOLS_FILE) as f:
                TOOLS = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Error loading tools.json: {e}\n")
        # Fallback to empty or default if needed


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
        btn_text = f"🚀 {self.tool_info['name']}"
        self.btn = QPushButton(btn_text)
        self.btn.clicked.connect(lambda: self.launch_callback(self.tool_info))
        self.btn.setEnabled(exists)

        if not exists:
            self.btn.setStyleSheet("background-color: #f7768e; color: #1a1b26;")
            self.btn.setText(f"❌ {self.tool_info['name']} (Missing)")
            self.btn.setToolTip(f"File not found: {full_path}")

        layout.addWidget(self.btn)

        # Description
        desc = QLabel(self.tool_info["desc"])
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

    def setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
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

        main_layout.addLayout(header_layout)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        for category, tools in TOOLS.items():
            tab = QWidget()
            self.setup_category_tab(tab, tools)
            self.tabs.addTab(tab, category)

        # Status Log
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

        main_layout.addWidget(log_group)

        # Check for tools
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

    def launch_tool(self, tool_info: dict[str, Any]) -> None:
        """Launch the specified tool based on its type."""
        path = REPO_ROOT / tool_info["path"]
        type_ = tool_info["type"]
        is_debug = self.debug_mode.isChecked()

        self.log(f"Launching {tool_info['name']}...")
        self.log(f"Path: {path}")

        if is_debug:
            self.log(f"DEBUG: Mode enabled. Launching {type_} tool.")

        try:
            if type_ == "python":
                args = [sys.executable, str(path)]
                if is_debug:
                    # Could add a verbose flag if the tool supports it
                    pass
                subprocess.Popen(args, cwd=path.parent)
                self.log("✅ Process started (Python)")

            elif type_ == "matlab":
                self.log("ℹ️ Attempting to launch MATLAB...")
                # Build MATLAB command safely without shell=True
                # Using list form to avoid shell injection vulnerabilities
                matlab_script = f"run('{str(path).replace(chr(39), chr(39)+chr(39))}');"
                cmd_list = ["matlab", "-nosplash", "-nodesktop", "-r", matlab_script]
                try:
                    subprocess.Popen(cmd_list, cwd=path.parent)
                    self.log("✅ MATLAB command sent")
                except FileNotFoundError:
                    # MATLAB not in PATH, try opening file directly
                    # Use hasattr pattern for Windows-specific startfile
                    if hasattr(os, "startfile"):
                        os.startfile(path)
                        self.log("⚠️ MATLAB not in PATH, opened file in default editor")
                    else:
                        subprocess.Popen(["xdg-open", str(path)])
                        self.log("⚠️ MATLAB not in PATH, opened file with xdg-open")

            elif type_ == "web" or type_ == "browser":
                webbrowser.open(path.as_uri())
                self.log("✅ Opened in default browser")

            elif type_ == "bat":
                # Use cmd.exe explicitly instead of shell=True for security
                subprocess.Popen(["cmd.exe", "/c", str(path)], cwd=path.parent)
                self.log("✅ Batch script executed")

            else:
                self.log(f"❌ Unknown type: {type_}")

        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            QMessageBox.critical(
                self, "Launch Error", f"Failed to launch tool:\n{str(e)}"
            )


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

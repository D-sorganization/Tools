"""
Unified Tools Launcher - Modern PyQt6-based launcher for the Tools repository.

This is the PRIMARY and RECOMMENDED launcher for accessing all tools.
Provides a clean, tabbed interface for launching Python, MATLAB, and web tools.
"""

import html
import sys
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

# Use shared path setup utility for consistency
# Manually add src/python/src to sys.path to ensure we can import utils
sys.path.append(str(REPO_ROOT / "src" / "python" / "src"))

try:
    from utils.path_setup import setup_python_path

    setup_python_path(repo_root=REPO_ROOT)
except ImportError:
    # Fallback if shared utility not available
    try:
        from utils.path_helpers import ensure_utils_in_path

        ensure_utils_in_path()
    except ImportError:
        # Last resort fallback
        pass

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
            from utils.file_utils import safe_read_json

            TOOLS = safe_read_json(TOOLS_FILE, default=None)
        except ImportError:
            # Fallback to direct json.load

            try:
                TOOLS = safe_read_json(TOOLS_FILE, default=None)
            except Exception as e:
                sys.stderr.write(f"Error loading tools.json: {e}\n")
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
        exists = False
        try:
            # Basic existence check for UI state
            # We recreate the path logic briefly just for the button state
            # Real validation happens on launch
            p = Path(self.tool_info.get("path", ""))
            if not p.is_absolute():
                p = REPO_ROOT / p
            exists = p.exists()
        except Exception:
            exists = False

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
            self.btn.setToolTip(f"File not found: {self.tool_info.get('path')}")

        layout.addWidget(self.btn)

        # Description
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

    def launch_tool(self, tool_info: dict[str, Any]) -> None:
        """Launch the specified tool using shared utils."""
        # Late import to prevent circular dependency issues during startup
        from tools.launch_utils import (
            launch_tool,
            LaunchError,
            SecurityError,
            ToolNotFoundError,
        )

        is_debug = self.debug_mode.isChecked()
        name = tool_info.get("name", "Unknown")

        self.log(f"Launching {name}...")

        try:
            launch_tool(
                tool_info=tool_info,
                repo_root=REPO_ROOT,
                is_debug=is_debug,
                log_func=self.log,
            )
        except (LaunchError, SecurityError, ToolNotFoundError) as e:
            self.log(f"❌ Error: {e}")
            QMessageBox.critical(self, "Launch Error", str(e))
        except Exception as e:
            self.log(f"❌ Unexpected Error: {e}")
            QMessageBox.critical(
                self, "Critical Error", f"An unexpected error occurred:\n{e}"
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

"""Main window for the Unified Tools Launcher."""

import html
import queue
import threading
from datetime import datetime
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tools.gui.components.tool_card import ToolCard
from tools.launch_utils import (
    LaunchError,
    PlatformError,
    SecurityError,
    ToolNotFoundError,
    get_repo_root,
    launch_tool,
)


class UnifiedLauncher(QMainWindow):
    """Main launcher window with tabbed interface for all tools."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Antigravity Unified Tools Launcher")
        self.resize(1000, 700)
        self.repo_root = get_repo_root()

        # Set window icon if available
        icon_path = self.repo_root / "tools_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.setup_ui()
        self.setup_log_consumer()

    def setup_ui(self) -> None:
        """Set up the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        main_layout.addLayout(self._create_header_layout())
        self.tabs = self._create_tool_tabs()
        main_layout.addWidget(self.tabs, stretch=1)

        self.log_area = self._create_log_area()
        main_layout.addWidget(self.log_area, stretch=0)

        # Status Bar
        self.statusBar().showMessage(f"Repository Root: {self.repo_root}")

    def _create_header_layout(self) -> QHBoxLayout:
        """Create the header layout with title and debug checkbox."""
        header_layout = QHBoxLayout()

        title = QLabel("Unified Tools Launcher")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.debug_mode = QCheckBox("Debug Mode (Verbose Logs)")
        self.debug_mode.setFont(QFont("Segoe UI", 10))
        header_layout.addWidget(self.debug_mode)

        return header_layout

    def _create_log_area(self) -> QTextEdit:
        """Create the activity log area widget."""
        log_area = QTextEdit()
        log_area.setReadOnly(True)
        log_area.setMaximumHeight(150)
        log_area.setPlaceholderText("Activity log will appear here...")
        log_area.setStyleSheet(
            """
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 10pt;
                padding: 5px;
            }
        """
        )
        return log_area

    def _create_tool_tabs(self) -> QTabWidget:
        """Create the tabbed interface for tool categories."""
        tabs = QTabWidget()
        tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #ddd;
                background: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #2196F3;
                font-weight: bold;
            }
        """
        )

        from tools.config_loader import CATEGORY_ORDER, load_tools_config

        tools_config = load_tools_config(self.repo_root)

        if not tools_config:
            self.log(
                "Warning: No tools configuration found (tools.json missing or invalid)."
            )
            no_tools = QLabel(
                "No tools configuration found.\nPlease ensure tools.json exists in the repository root."
            )
            no_tools.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tabs.addTab(no_tools, "Error")
            return tabs

        # Sort categories to keep UI consistent
        sorted_cats = sorted(
            tools_config.keys(),
            key=lambda x: CATEGORY_ORDER.index(x) if x in CATEGORY_ORDER else 999,
        )

        for category in sorted_cats:
            cat_tab = QWidget()
            try:
                if self.setup_category_tab(cat_tab, tools_config[category]):
                    tabs.addTab(cat_tab, category)
            except Exception as e:
                self.log(f"Error creating tab for {category}: {e}")

        return tabs

    def setup_category_tab(self, tab: QWidget, tools: list[dict[str, Any]]) -> bool:
        """Set up a tab for a category of tools."""
        if not tools:
            return False

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(20)
        grid.setContentsMargins(20, 20, 20, 20)

        # Responsive grid logic
        cols = 2
        for i, tool_info in enumerate(tools):
            card = ToolCard(tool_info, self.launch_tool_wrapper)
            row = i // cols
            col = i % cols
            grid.addWidget(card, row, col)

        # Push items to top-left
        grid.setRowStretch(grid.rowCount(), 1)
        grid.setColumnStretch(cols, 1)

        scroll.setWidget(container)

        layout = QVBoxLayout(tab)
        layout.addWidget(scroll)
        return True

    def log(self, message: str) -> None:
        """Log a message to the activity log area (thread-safe)."""
        self.log_queue.put(message)

    def setup_log_consumer(self) -> None:
        """Consume logs from queue and update UI."""

        def process_queue() -> None:
            while not self.log_queue.empty():
                msg = self.log_queue.get()
                timestamp = datetime.now().strftime("%H:%M:%S")
                # Basic HTML escaping just in case
                safe_msg = html.escape(msg)

                color = "black"
                if "Error" in msg or "Fail" in msg:
                    color = "#d32f2f"  # Red
                elif "Success" in msg:
                    color = "#388E3C"  # Green

                self.log_area.append(
                    f'<span style="color:#888">[{timestamp}]</span> <span style="color:{color}">{safe_msg}</span>'
                )

        self.timer = QTimer()
        self.timer.timeout.connect(process_queue)
        self.timer.start(100)

    def launch_tool_wrapper(self, tool_info: dict[str, Any]) -> None:
        """Wrapper to launch tool in background thread."""
        is_debug = self.debug_mode.isChecked()

        def run_launch() -> None:
            try:
                self.log(f"🚀 Launching: {tool_info.get('name')}...")
                launch_tool(
                    tool_info=tool_info,
                    repo_root=self.repo_root,
                    is_debug=is_debug,
                    log_func=self.log,
                )
            except (LaunchError, SecurityError, ToolNotFoundError, PlatformError) as e:
                self.log(f"❌ Launch Error: {e}")
                # We can't show message box from thread easily, but log is visible
            except Exception as e:
                self.log(f"❌ Unexpected Error: {e}")

        # Launch in thread to keep UI responsive
        threading.Thread(target=run_launch, daemon=True).start()

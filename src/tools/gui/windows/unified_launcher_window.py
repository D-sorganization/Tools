"""Main window for the Unified Tools Launcher."""

import html
import queue
import threading
from datetime import datetime
from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shared.python.theme import ThemedWindowMixin
from tools.gui.components.tool_card import ToolCard
from tools.launch_utils import (
    LaunchError,
    PlatformError,
    SecurityError,
    ToolNotFoundError,
    get_repo_root,
    launch_tool,
)

# Import help system components
try:
    from python.src.help import get_help_manager
    from python.src.help.help_content import initialize_help_manager

    HELP_AVAILABLE = True
except ImportError:
    HELP_AVAILABLE = False


class UnifiedLauncher(ThemedWindowMixin, QMainWindow):
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

        # Initialize theme support with Theme menu
        self.setup_theme_support(settings_app="UnifiedToolsLauncher")

        # Initialize help system
        self._setup_help_system()

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
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(f"Repository Root: {self.repo_root}")

    def _create_header_layout(self) -> QHBoxLayout:
        """Create the header layout with title and debug checkbox."""
        header_layout = QHBoxLayout()

        title = QLabel("Unified Tools Launcher")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setObjectName("titleLabel")  # For theme-specific styling
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
        log_area.setObjectName("activityLog")  # For theme-specific styling
        return log_area

    def _create_tool_tabs(self) -> QTabWidget:
        """Create the tabbed interface for tool categories."""
        tabs = QTabWidget()
        tabs.setObjectName("toolTabs")  # For theme-specific styling

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
            except (KeyError, ValueError, TypeError) as e:
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
            except (KeyError, ValueError, TypeError) as e:
                self.log(f"❌ Unexpected Error: {e}")

        # Launch in thread to keep UI responsive
        threading.Thread(target=run_launch, daemon=True).start()

    def _setup_help_system(self) -> None:
        """Initialize the help system with menu and shortcuts."""
        if not HELP_AVAILABLE:
            self.log("Warning: Help system not available (module import failed)")
            return

        try:
            # Initialize help content
            initialize_help_manager()

            # Get the help manager instance
            self.help_manager = get_help_manager()

            # Set paths for help content
            help_dir = self.repo_root / "docs" / "help"
            user_manual_path = self.repo_root / "docs" / "USER_MANUAL.md"

            self.help_manager.set_help_directory(help_dir)
            self.help_manager.set_user_manual_path(user_manual_path)

            # Create Help menu
            self._create_help_menu()

            # Set up F1 shortcut
            f1_shortcut = QShortcut(QKeySequence("F1"), self)
            f1_shortcut.activated.connect(self._show_user_manual)

            self.log("Help system initialized successfully")
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            self.log(f"Warning: Failed to initialize help system: {e}")
            self.help_manager = None

    def _create_help_menu(self) -> None:
        """Create the Help menu."""
        menubar = self.menuBar()
        if menubar is None:
            return

        help_menu = QMenu("&Help", self)

        # User Manual action (F1)
        manual_action = QAction("User &Manual", self)
        manual_action.setShortcut(QKeySequence("F1"))
        manual_action.setStatusTip("Open the User Manual")
        manual_action.triggered.connect(self._show_user_manual)
        help_menu.addAction(manual_action)

        # Tool Help action
        tool_help_action = QAction("&Tool Help...", self)
        tool_help_action.setStatusTip("Show help for the current tool category")
        tool_help_action.triggered.connect(self._show_tool_help)
        help_menu.addAction(tool_help_action)

        # Getting Started
        getting_started_action = QAction("&Getting Started", self)
        getting_started_action.setStatusTip("Show getting started guide")
        getting_started_action.triggered.connect(self._show_getting_started)
        help_menu.addAction(getting_started_action)

        help_menu.addSeparator()

        # About action
        about_action = QAction("&About", self)
        about_action.setStatusTip("About Unified Tools Launcher")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        menubar.addMenu(help_menu)

    def _show_user_manual(self) -> None:
        """Show the user manual dialog."""
        if self.help_manager:
            self.help_manager.show_user_manual(self)

    def _show_tool_help(self) -> None:
        """Show help for the currently selected tool category."""
        if not self.help_manager:
            return

        # Get the current tab name (category)
        current_index = self.tabs.currentIndex()
        if current_index >= 0:
            category = self.tabs.tabText(current_index)
            self.help_manager.show_category_help(category, self)

    def _show_getting_started(self) -> None:
        """Show the getting started guide."""
        if self.help_manager:
            self.help_manager.show_topic("getting_started", self)

    def _show_about(self) -> None:
        """Show the about dialog."""
        if self.help_manager:
            self.help_manager.show_about_dialog(self)

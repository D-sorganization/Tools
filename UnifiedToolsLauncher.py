#!/usr/bin/env python3
"""
Unified Tools Launcher - Modern PyQt6-based launcher for the Tools repository.

This is the PRIMARY and RECOMMENDED launcher for accessing all tools.
Provides a clean, tabbed interface for launching Python, MATLAB, and web tools.
"""

import sys

from PyQt6.QtWidgets import QApplication

from tools.logger import setup_logging

# Configure logging
logger = setup_logging(__name__, "unified_launcher.log")


def main() -> None:
    """Entry point for the Unified Tools Launcher application."""
    try:
        from tools.gui.windows.unified_launcher_window import UnifiedLauncher
    except ImportError as e:
        logger.error(f"Failed to import GUI components: {e}")
        # Could show a simple Tkinter error box here if PyQt imports fail entirely
        print(
            f"CRITICAL ERROR: Failed to load launcher components: {e}", file=sys.stderr
        )
        sys.exit(1)

    app = QApplication(sys.argv)

    # Optional: Load stylesheet or platform specific tweaks
    app.setStyle("Fusion")

    try:
        from tools.gui.windows.unified_launcher_window import UnifiedLauncher
        from tools.ui_utils import set_qt_icon

        window = UnifiedLauncher()
        set_qt_icon(window)
        window.show()
    except Exception as e:
        logger.error(f"Failed to create window: {e}")
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

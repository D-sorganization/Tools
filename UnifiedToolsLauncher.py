#!/usr/bin/env python3
"""
Unified Tools Launcher - Modern PyQt6-based launcher for the Tools repository.

This is the PRIMARY and RECOMMENDED launcher for accessing all tools.
Provides a clean, tabbed interface for launching Python, MATLAB, and web tools.
"""

import logging
import sys
from pathlib import Path

# Ensure Python 3.11+
if sys.version_info < (3, 11):
    print("CRITICAL: UnifiedToolsLauncher requires Python 3.11 or higher.", file=sys.stderr)
    print(f"Current version: {sys.version}", file=sys.stderr)
    sys.exit(1)

from PyQt6.QtWidgets import QApplication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("unified_launcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add src to path to resolve tools package
SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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

    window = UnifiedLauncher()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

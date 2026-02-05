#!/usr/bin/env python3
"""Launch the Data Processor PyQt6 GUI.

This script provides a standalone launcher for the Data Processor application
with dependency checking and helpful error messages.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if required dependencies are installed.

    Returns:
        Tuple of (all_ok, missing_packages)
    """
    required = {
        "PyQt6": "pip install PyQt6",
        "pandas": "pip install pandas",
        "numpy": "pip install numpy",
    }

    missing = []
    for package, install_cmd in required.items():
        if find_spec(package) is None:
            missing.append(f"{package} ({install_cmd})")

    return len(missing) == 0, missing


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("Data Processor - PyQt6 GUI Launcher")
    print("=" * 60)

    # Check dependencies
    ok, missing = check_dependencies()
    if not ok:
        print("\nMissing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install the missing packages and try again.")
        return 1

    # Set up path for module import
    src_path = Path(__file__).parent / "python"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    # Also add the shared utils path (for utils.path_helpers, etc.)
    tools_root = Path(__file__).parent.parent.parent.parent
    utils_path = tools_root / "src" / "python" / "src"
    if utils_path.exists():
        sys.path.insert(0, str(utils_path))

    try:
        from data_processor.ui.pyqt6.main_window import main as gui_main

        print("\nStarting Data Processor GUI...")
        gui_main()
        return 0

    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        print("\nTrying alternative launch method...")

        # Try running as module
        cmd = [sys.executable, "-m", "data_processor.ui.pyqt6.main_window"]
        return subprocess.call(cmd, cwd=src_path)

    except Exception as e:
        logger.error(f"Failed to launch GUI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

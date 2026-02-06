#!/usr/bin/env python3
"""Launch script for Glass Bath FEA PyQt6 GUI.

This script provides the entry point for launching the Glass Bath FEA
desktop application using PyQt6.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def check_dependencies() -> list[str]:
    """Check for required dependencies.

    Returns:
        List of missing dependency names.
    """
    missing = []

    if importlib.util.find_spec("PyQt6") is None:
        missing.append("PyQt6")

    if importlib.util.find_spec("numpy") is None:
        missing.append("numpy")

    if importlib.util.find_spec("scipy") is None:
        missing.append("scipy")

    return missing


def main() -> int:
    """Launch the Glass Bath FEA application.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Check dependencies
    missing = check_dependencies()
    if missing:
        sys.stderr.write(f"Missing required dependencies: {', '.join(missing)}\n")
        sys.stderr.write("Install with: pip install PyQt6 numpy scipy\n")
        return 1

    # Add paths
    tools_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(tools_root / "src"))

    # Launch application
    try:
        from glass_bath_fea.ui.pyqt6.main_window import main as run_app

        run_app()
        return 0
    except Exception as e:
        sys.stderr.write(f"Error launching application: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

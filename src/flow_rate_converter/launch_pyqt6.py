#!/usr/bin/env python3
"""
Flow Rate Converter - Standalone PyQt6 Launcher
================================================

Launch the Flow Rate Converter as a standalone application.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    missing = []

    try:
        import PyQt6  # noqa: F401
    except ImportError:
        missing.append("PyQt6")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def setup_path() -> None:
    """Add necessary paths for imports."""
    # Add the python package directory
    package_dir = Path(__file__).parent / "python"
    if package_dir.exists():
        sys.path.insert(0, str(package_dir))

    # Add shared modules
    shared_dir = Path(__file__).parent.parent / "shared" / "python"
    if shared_dir.exists():
        sys.path.insert(0, str(shared_dir))


def main() -> int:
    """Main entry point."""
    print("Flow Rate Converter")
    print("=" * 40)
    print()

    if not check_dependencies():
        return 1

    print("Starting application...")
    print()

    setup_path()

    try:
        from flow_rate_converter.ui.pyqt6.main_window import main as run_app

        run_app()
        return 0
    except ImportError as e:
        print(f"Error importing GUI: {e}")
        return 1
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

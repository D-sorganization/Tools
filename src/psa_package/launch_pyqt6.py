#!/usr/bin/env python3
"""
PSA Package - Standalone PyQt6 Launcher
========================================

Launch the Two-Stage PSA System Analysis GUI as a standalone application.
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

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def setup_path() -> None:
    """Add necessary paths for imports."""
    # Add shared modules
    shared_dir = Path(__file__).parent.parent / "shared" / "python"
    if shared_dir.exists():
        sys.path.insert(0, str(shared_dir))


def main() -> int:
    """Main entry point."""
    print("PSA Package - Two-Stage PSA System Analysis")
    print("=" * 50)
    print()

    if not check_dependencies():
        return 1

    print("Starting application...")
    print()

    setup_path()

    try:
        from upstream_drift_tools.process_calculators.psa_package.psa_gui import main as run_app

        run_app()
        return 0
    except ImportError as e:
        print(f"Error importing PSA GUI: {e}")
        print("\nEnsure the upstream_drift_tools package is available.")
        return 1
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Steam Engine Calculator - Standalone Launcher
==============================================

Launch the Steam Engine Calculator as a standalone PyQt6 application.
"""

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

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    # Check optional dependencies
    try:
        import CoolProp  # noqa: F401

        print("CoolProp: Available (high-accuracy calculations)")
    except ImportError:
        print("CoolProp: Not installed (optional - pip install CoolProp)")

    try:
        import cantera  # noqa: F401

        print("Cantera: Available")
    except ImportError:
        print("Cantera: Not installed (optional)")

    return True


def setup_path() -> None:
    """Add necessary paths for imports."""
    # Bootstrap imports for development mode (before pip install -e .)
    _repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_repo_root / "src" / "shared" / "python"))
    from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

    ensure_paths(_repo_root)


def main() -> None:
    """Main entry point."""
    print("Steam Engine Calculator")
    print("=" * 40)
    print()

    if not check_dependencies():
        sys.exit(1)

    print()
    print("Starting application...")
    print()

    setup_path()

    from steam_engine_calculator.ui.pyqt6.main_window import main as run_app

    run_app()


if __name__ == "__main__":
    main()

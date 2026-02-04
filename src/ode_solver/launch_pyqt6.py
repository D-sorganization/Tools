#!/usr/bin/env python3
"""
ODE Solver - PyQt6 Launcher
===========================

Launch the ODE Solver as a standalone PyQt6 application.
"""

from __future__ import annotations

import sys


def check_dependencies() -> list[str]:
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
        import scipy  # noqa: F401
    except ImportError:
        missing.append("scipy")

    try:
        import sympy  # noqa: F401
    except ImportError:
        missing.append("sympy")

    return missing


def main() -> int:
    """Main entry point for the ODE Solver GUI."""
    print("ODE Solver - PyQt6 GUI")
    print("=" * 50)
    print()

    missing = check_dependencies()
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return 1

    try:
        from PyQt6.QtWidgets import QApplication

        from ode_solver.ui.pyqt6.main_window import ODESolverWindow

        app = QApplication(sys.argv)
        window = ODESolverWindow()
        window.show()
        return app.exec()
    except ImportError as e:
        print(f"Error importing GUI components: {e}")
        print("\nMake sure the package is installed correctly.")
        return 1
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

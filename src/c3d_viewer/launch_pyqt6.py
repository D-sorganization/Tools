#!/usr/bin/env python3
"""
C3D Motion Capture Viewer - PyQt6 Launcher
==========================================

Launch the C3D Motion Capture Viewer as a standalone PyQt6 application.
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

    return missing


def main() -> int:
    """Main entry point for the C3D Motion Capture Viewer GUI."""
    print("C3D Motion Capture Viewer - PyQt6 GUI")
    print("=" * 50)
    print()

    missing = check_dependencies()
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return 1

    # Check optional dependencies
    try:
        import ezc3d  # noqa: F401

        print("ezc3d: Available")
    except ImportError:
        print("ezc3d: Not available (demo mode)")
        print("  Install with: pip install ezc3d")

    print()

    try:
        from PyQt6.QtWidgets import QApplication

        from c3d_viewer.ui.pyqt6.main_window import C3DViewerWindow

        app = QApplication(sys.argv)
        window = C3DViewerWindow()
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

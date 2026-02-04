#!/usr/bin/env python3
"""Launch script for Scrubber Calculator PyQt6 application."""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths for imports
MODULE_DIR = Path(__file__).parent
TOOLS_ROOT = MODULE_DIR.parent.parent
sys.path.insert(0, str(MODULE_DIR / "python"))
sys.path.insert(0, str(TOOLS_ROOT / "src"))


def main() -> int:
    """Launch the Scrubber Calculator PyQt6 application."""
    from scrubber_calculator.ui.pyqt6.main_window import main as run_app

    run_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())

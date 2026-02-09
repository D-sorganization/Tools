#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Signal Processing Studio.

Combines Function Generator, Signal Toolkit, and Polynomial Generator
into a single unified application.
"""

from __future__ import annotations

import logging
import sys
from importlib.util import find_spec
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT / "src" / "function_generator" / "python"))
sys.path.insert(0, str(_REPO_ROOT / "src" / "signal_processing_studio" / "python"))


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if required dependencies are installed."""
    required = {
        "PyQt6": "pip install PyQt6",
        "matplotlib": "pip install matplotlib",
        "numpy": "pip install numpy",
        "scipy": "pip install scipy",
        "sympy": "pip install sympy",
    }

    missing = []
    for package, install_cmd in required.items():
        if find_spec(package) is None:
            missing.append(f"{package} ({install_cmd})")

    return len(missing) == 0, missing


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("Signal Processing Studio - PyQt6 Launcher")
    print("=" * 60)

    ok, missing = check_dependencies()
    if not ok:
        print("\nMissing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install the missing packages and try again.")
        return 1

    try:
        from signal_processing_studio.main_window import main as studio_main

        print("\nStarting Signal Processing Studio...")
        return studio_main()

    except ImportError as e:
        logger.error(f"Failed to import Studio module: {e}")
        return 1

    except Exception as e:
        logger.error(f"Failed to launch Studio: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Launch script for Scrubber Calculator React web application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).parent
WEB_DIR = MODULE_DIR / "web"


def main() -> int:
    """Launch the Scrubber Calculator React web application."""
    if not WEB_DIR.exists():
        print(f"Error: Web directory not found: {WEB_DIR}")
        return 1

    # Check if node_modules exists
    node_modules = WEB_DIR / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies...")
        install_result = subprocess.run(
            ["npm", "install"],
            cwd=WEB_DIR,
            shell=False,
        )
        if install_result.returncode != 0:
            print("Error: Failed to install dependencies")
            return 1

    # Run the dev server
    print("Starting Scrubber Calculator web application...")
    print("Open http://localhost:5177 in your browser")
    dev_result = subprocess.run(
        ["npm", "run", "dev"],
        cwd=WEB_DIR,
        shell=False,
    )
    return dev_result.returncode


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Standalone launcher for Financial Calculator React web application.

This launcher starts a development server for the React application.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Main entry point for the Financial Calculator web launcher."""
    web_dir = Path(__file__).parent / "web"

    if not web_dir.exists():
        print(f"Error: Web directory not found at {web_dir}")
        return 1

    # Check if node_modules exists
    node_modules = web_dir / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=web_dir,
            shell=True,
        )
        if result.returncode != 0:
            print("Failed to install dependencies")
            return 1

    # Start dev server
    print("Starting Financial Calculator web application...")
    print("Open http://localhost:5173 in your browser")

    try:
        subprocess.run(
            ["npm", "run", "dev"],
            cwd=web_dir,
            shell=True,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")

    return 0


if __name__ == "__main__":
    sys.exit(main())

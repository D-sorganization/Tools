#!/usr/bin/env python3
"""Standalone launcher for Baghouse Calculator React web application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Main entry point for the Baghouse Calculator web launcher."""
    web_dir = Path(__file__).parent / "web"

    if not web_dir.exists():
        print(f"Error: Web directory not found at {web_dir}")
        return 1

    node_modules = web_dir / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies...")
        result = subprocess.run(["npm", "install"], cwd=web_dir, shell=False)
        if result.returncode != 0:
            return 1

    print("Starting Baghouse Calculator web application...")
    print("Open http://localhost:5173 in your browser")

    try:
        subprocess.run(["npm", "run", "dev"], cwd=web_dir, shell=False)
    except KeyboardInterrupt:
        print("\nShutting down...")

    return 0


if __name__ == "__main__":
    sys.exit(main())

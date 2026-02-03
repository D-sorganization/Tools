#!/usr/bin/env python3
"""Standalone React web launcher for Syngas Compression Calculator.

This launcher starts a development server for the React-based
syngas compression calculator web application.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def check_dependencies() -> list[str]:
    """Check for required dependencies."""
    missing = []

    # Check for Node.js
    try:
        subprocess.run(
            ["node", "--version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("node")

    # Check for npm
    try:
        subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            check=True,
            shell=True,  # Required on Windows
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("npm")

    return missing


def install_dependencies(web_dir: Path) -> bool:
    """Install npm dependencies if needed."""
    node_modules = web_dir / "node_modules"
    if not node_modules.exists():
        print("Installing npm dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=str(web_dir),
            shell=True,  # Required on Windows
        )
        return result.returncode == 0
    return True


def main() -> int:
    """Launch the Syngas Compression Calculator React application."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install Node.js from https://nodejs.org/")
        return 1

    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        print(f"Web directory not found: {web_dir}")
        return 1

    # Install dependencies if needed
    if not install_dependencies(web_dir):
        print("Failed to install npm dependencies")
        return 1

    print("Starting Syngas Compression Calculator web application...")
    print("Opening http://localhost:5173 in your browser...")

    # Open browser after a short delay
    def open_browser() -> None:
        import time
        time.sleep(2)
        webbrowser.open("http://localhost:5173")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start development server
    env = os.environ.copy()
    result = subprocess.run(
        ["npm", "run", "dev"],
        cwd=str(web_dir),
        shell=True,  # Required on Windows
        env=env,
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

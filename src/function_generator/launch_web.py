#!/usr/bin/env python3
"""Standalone React web launcher for Function Generator.

This launcher starts a development server for the React-based
function generator web application.
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

    try:
        subprocess.run(
            ["node", "--version"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("node")

    try:
        subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            check=True,
            shell=True,
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
            shell=True,
        )
        return result.returncode == 0
    return True


def main() -> int:
    """Launch the Function Generator React application."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install Node.js from https://nodejs.org/")
        return 1

    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        print(f"Web directory not found: {web_dir}")
        return 1

    if not install_dependencies(web_dir):
        print("Failed to install npm dependencies")
        return 1

    print("Starting Function Generator web application...")
    print("Opening http://localhost:5174 in your browser...")

    def open_browser() -> None:
        import time
        time.sleep(2)
        webbrowser.open("http://localhost:5174")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    env = os.environ.copy()
    result = subprocess.run(
        ["npm", "run", "dev"],
        cwd=str(web_dir),
        shell=True,
        env=env,
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

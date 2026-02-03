#!/usr/bin/env python3
"""Standalone React web launcher for Pressure Drop Calculator."""

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
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("node")
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True, shell=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("npm")
    return missing


def install_dependencies(web_dir: Path) -> bool:
    """Install npm dependencies if needed."""
    if not (web_dir / "node_modules").exists():
        print("Installing npm dependencies...")
        result = subprocess.run(["npm", "install"], cwd=str(web_dir), shell=True)
        return result.returncode == 0
    return True


def main() -> int:
    """Launch the Pressure Drop Calculator React application."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        return 1

    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        print(f"Web directory not found: {web_dir}")
        return 1

    if not install_dependencies(web_dir):
        return 1

    print("Starting Pressure Drop Calculator web application...")

    def open_browser() -> None:
        import time
        time.sleep(2)
        webbrowser.open("http://localhost:5175")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    result = subprocess.run(["npm", "run", "dev"], cwd=str(web_dir), shell=True, env=os.environ.copy())
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

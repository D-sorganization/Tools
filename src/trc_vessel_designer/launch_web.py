#!/usr/bin/env python3
"""Standalone launcher for TRC Vessel Designer Web GUI."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import sleep


def check_node() -> bool:
    """Check if Node.js is available."""
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_npm() -> bool:
    """Check if npm is available."""
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_dependencies(web_path: Path) -> bool:
    """Install npm dependencies if needed."""
    node_modules = web_path / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies...")
        result = subprocess.run(["npm", "install"], cwd=web_path, shell=True)
        return result.returncode == 0
    return True


def main() -> int:
    """Main entry point for the TRC Vessel Designer Web GUI."""
    if not check_node():
        print("Error: Node.js is not installed or not in PATH")
        return 1

    if not check_npm():
        print("Error: npm is not installed or not in PATH")
        return 1

    web_path = Path(__file__).parent / "web"
    if not web_path.exists():
        print(f"Error: Web directory not found at {web_path}")
        return 1

    if not install_dependencies(web_path):
        print("Error: Failed to install dependencies")
        return 1

    port = 3002
    print(f"Starting TRC Vessel Designer Web GUI on port {port}...")

    env = os.environ.copy()
    env["PORT"] = str(port)

    process = subprocess.Popen(["npm", "run", "dev"], cwd=web_path, env=env, shell=True)

    sleep(2)
    webbrowser.open(f"http://localhost:{port}")

    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())

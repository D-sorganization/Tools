#!/usr/bin/env python3
"""Launch the Data Processor React Web GUI.

This script starts the Vite development server for the React-based
Data Processor web application.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
import time


def check_node() -> bool:
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            shell=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_dependencies(web_path: Path) -> bool:
    """Check if npm dependencies are installed."""
    return (web_path / "node_modules").exists()


def install_dependencies(web_path: Path) -> bool:
    """Install npm dependencies."""
    print("Installing dependencies...")
    result = subprocess.run(
        ["npm", "install"],
        cwd=web_path,
        shell=True,
    )
    return result.returncode == 0


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("Data Processor - React Web GUI Launcher")
    print("=" * 60)

    # Check Node.js
    if not check_node():
        print("\nNode.js is not installed or not in PATH.")
        print("Please install Node.js from https://nodejs.org/")
        return 1

    # Get web project path
    web_path = Path(__file__).parent / "web"
    if not web_path.exists():
        print(f"\nWeb project not found at: {web_path}")
        return 1

    # Check package.json
    if not (web_path / "package.json").exists():
        print(f"\npackage.json not found in: {web_path}")
        return 1

    # Install dependencies if needed
    if not check_dependencies(web_path):
        print("\nNode modules not found. Installing dependencies...")
        if not install_dependencies(web_path):
            print("Failed to install dependencies.")
            return 1
        print("Dependencies installed successfully.\n")

    # Start dev server
    port = 3000
    print(f"\nStarting development server on port {port}...")
    print(f"Web UI will be available at: http://localhost:{port}")
    print("\nPress Ctrl+C to stop the server.\n")

    # Set environment
    env = os.environ.copy()
    env["PORT"] = str(port)

    # Start the dev server
    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=web_path,
        shell=True,
        env=env,
    )

    # Wait a moment then open browser
    time.sleep(3)
    webbrowser.open(f"http://localhost:{port}")

    try:
        return process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        process.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())

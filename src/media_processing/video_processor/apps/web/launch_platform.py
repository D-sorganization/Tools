#!/usr/bin/env python3
"""
Cross-platform launcher for Video Processor Platform.
Replaces launch_platform.bat for better portability.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Video Processor Platform."""
    script_dir = Path(__file__).parent.absolute()

    # Check if Node.js is available
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Node.js is not installed or not in PATH")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)

    # Check if package.json exists
    package_json = script_dir / "package.json"
    if not package_json.exists():
        print("ERROR: package.json not found")
        print(f"Expected at: {package_json}")
        sys.exit(1)

    # Install dependencies if node_modules doesn't exist
    node_modules = script_dir / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies...")
        try:
            subprocess.run(["npm", "install"], cwd=script_dir, check=True)
        except subprocess.CalledProcessError:
            print("ERROR: Failed to install dependencies")
            sys.exit(1)

    # Launch the platform
    print("Starting Video Processor Platform...")
    try:
        subprocess.run(["npm", "run", "dev"], cwd=script_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to start platform (exit code {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPlatform stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

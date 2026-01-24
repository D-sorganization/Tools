#!/usr/bin/env python3
"""
Cross-platform launcher for Video Processor Platform.
Replaces launch_platform.bat for better portability.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Video Processor Platform from the current project directory.

    This function performs several checks and commands in sequence:

    1. Verifies that the ``node`` executable (Node.js) is available on ``PATH``
       by running ``node --version``.
    2. Ensures that a ``package.json`` file exists in the same directory as
       this launcher script.
    3. If a ``node_modules`` directory is not present in that directory, runs
       ``npm install`` to install JavaScript dependencies.
    4. Starts the development server for the Video Processor Platform by
       running ``npm run dev`` in the script directory.

    Environment requirements:
    - Node.js and npm must be installed and discoverable on the system ``PATH``.
    - This script must reside in a directory that contains a valid
      ``package.json`` file for the Video Processor Platform.
    - The script will create and use a ``node_modules`` directory in the same
      directory as this script.

    Exit codes:
    - 0: Success (platform started or stopped gracefully by user)
    - 1: Error (Node.js not found, package.json missing, dependency install
      failed, or platform start failed)

    Raises:
    - SystemExit: Always exits via sys.exit() with appropriate exit code.
    """
    script_dir = Path(__file__).parent.absolute()

    # Check if Node.js is available
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Node.js is not installed or not in PATH")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)

    # Check if npm is available (npm may not be installed even if Node.js is)
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: npm is not installed or not in PATH")
        print(
            "npm usually comes with Node.js. Please reinstall Node.js from https://nodejs.org/"
        )
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
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if isinstance(e, FileNotFoundError):
            print("ERROR: npm not found. This should have been caught earlier.")
        else:
            print(f"ERROR: Failed to start platform (exit code {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPlatform stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

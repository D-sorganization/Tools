#!/usr/bin/env python3
"""
Cross-platform launcher for Video Processor Platform.
Replaces launch_platform.bat for better portability.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Launch the Video Processor Platform."""
    script_dir = Path(__file__).parent.absolute()

    # Check if Node.js is available
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ERROR: Node.js is not installed or not in PATH")
        logger.info("Please install Node.js from https://nodejs.org/")
        sys.exit(1)

    # Check if package.json exists
    package_json = script_dir / "package.json"
    if not package_json.exists():
        logger.error("ERROR: package.json not found")
        logger.info(f"Expected at: {package_json}")
        sys.exit(1)

    # Install dependencies if node_modules doesn't exist
    node_modules = script_dir / "node_modules"
    if not node_modules.exists():
        logger.info("Installing dependencies...")
        try:
            subprocess.run(["npm", "install"], cwd=script_dir, check=True)
        except subprocess.CalledProcessError:
            logger.error("ERROR: Failed to install dependencies")
            sys.exit(1)

    # Launch the platform
    logger.info("Starting Video Processor Platform...")
    try:
        subprocess.run(["npm", "run", "dev"], cwd=script_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ERROR: Failed to start platform (exit code {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nPlatform stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

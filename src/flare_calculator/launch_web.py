#!/usr/bin/env python3
"""Launch the Flare Calculator React web application."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Entry point for the Flare Calculator web application."""
    web_dir = Path(__file__).parent / "web"

    if not web_dir.exists():
        print(f"Error: Web directory not found at {web_dir}")
        sys.exit(1)

    # Check if node_modules exists
    node_modules = web_dir / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=web_dir, check=True, shell=True)

    # Start the development server
    print("Starting Flare Calculator web application on http://localhost:5179")
    subprocess.run(
        ["npm", "run", "dev", "--", "--port", "5179"],
        cwd=web_dir,
        check=True,
        shell=True,
    )


if __name__ == "__main__":
    main()

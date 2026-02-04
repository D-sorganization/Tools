#!/usr/bin/env python3
"""Launch script for Acid Gas Dewpoint Calculator React Web GUI."""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from time import sleep


def main() -> int:
    """Launch the Acid Gas Dewpoint Calculator React application."""
    web_dir = Path(__file__).parent / "web"

    if not web_dir.exists():
        print(f"Error: Web directory not found at {web_dir}")
        return 1

    # Check for node_modules
    if not (web_dir / "node_modules").exists():
        print("Installing dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=web_dir,
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error installing dependencies: {result.stderr}")
            return 1

    # Set port
    port = int(os.environ.get("PORT", "5176"))

    # Open browser after delay
    def open_browser() -> None:
        sleep(2)
        webbrowser.open(f"http://localhost:{port}")

    import threading

    threading.Thread(target=open_browser, daemon=True).start()

    # Start dev server
    print(f"Starting Acid Gas Dewpoint Calculator on http://localhost:{port}")
    dev_result = subprocess.run(
        ["npm", "run", "dev", "--", "--port", str(port)],
        cwd=web_dir,
        shell=True,
    )
    return dev_result.returncode


if __name__ == "__main__":
    sys.exit(main())

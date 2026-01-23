#!/usr/bin/env python3
"""
Cross-platform launcher for Folder Tool.
Replaces Launch_FolderFix.bat for better portability.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch Folder Tool."""
    script_dir = Path(__file__).parent.absolute()
    tool_script = script_dir / "Folders_Tool_r0.py"

    if not tool_script.exists():
        print(f"ERROR: Tool script not found: {tool_script}")
        sys.exit(1)

    print("Starting Folder Tool...")
    try:
        subprocess.run([sys.executable, str(tool_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to launch tool (exit code {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTool stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()

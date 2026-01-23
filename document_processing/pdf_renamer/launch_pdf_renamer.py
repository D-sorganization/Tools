#!/usr/bin/env python3
"""
Cross-platform launcher for PDF Renamer GUI.
Replaces PDF_Renamer.bat for better portability.
"""

import subprocess
import sys
from pathlib import Path


def check_python() -> bool:
    """Check if Python is available."""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            print("ERROR: Python 3.10 or higher is required")
            print(f"Current version: {sys.version}")
            print("Please install Python 3.10+ from https://www.python.org/")
            return False
        return True
    except Exception:
        return False


def check_dependencies(script_dir: Path) -> bool:
    """Check and install dependencies if needed."""
    verify_script = script_dir / "verify_installation.py"
    if verify_script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(verify_script)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    # Try to install dependencies
    print("Some dependencies are missing. Installing...")
    requirements = script_dir / "requirements.txt"
    if requirements.exists():
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                check=True,
            )
            print("Dependencies installed successfully!")
            print()
            return True
        except subprocess.CalledProcessError:
            print("ERROR: Failed to install dependencies")
            print()
            print("Please try manually:")
            print(f"  pip install -r {requirements}")
            return False
    return True


def main() -> None:
    """Launch PDF Renamer GUI."""
    # Change to script directory
    script_dir = Path(__file__).parent.absolute()

    # Check Python version
    if not check_python():
        sys.exit(1)

    # Check dependencies
    if not check_dependencies(script_dir):
        sys.exit(1)

    # Launch the GUI
    print("Starting PDF Renamer...")
    launch_script = script_dir / "launch_gui.py"
    try:
        subprocess.run([sys.executable, str(launch_script)], check=True)
    except subprocess.CalledProcessError as e:
        print()
        print(f"An error occurred (exit code {e.returncode}).")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("Launch cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()

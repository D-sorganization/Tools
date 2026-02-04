#!/usr/bin/env python3
"""
PSA Package - Streamlit Web App Launcher
=========================================

Launch the PSA System Analysis as a Streamlit web application.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    missing = []

    try:
        import streamlit  # noqa: F401
    except ImportError:
        missing.append("streamlit")

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    try:
        import plotly  # noqa: F401
    except ImportError:
        missing.append("plotly")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False

    return True


def main() -> int:
    """Main entry point."""
    print("PSA Package - Streamlit Web Application")
    print("=" * 50)
    print()

    if not check_dependencies():
        return 1

    # Find the webapp file
    shared_dir = Path(__file__).parent.parent / "shared" / "python"
    webapp_path = (
        shared_dir
        / "upstream_drift_tools"
        / "process_calculators"
        / "psa_package"
        / "psa_webapp.py"
    )

    if not webapp_path.exists():
        print(f"Error: Web app not found at {webapp_path}")
        return 1

    print(f"Launching Streamlit app from: {webapp_path}")
    print("The app will open in your default browser.")
    print()

    try:
        # Launch streamlit
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(webapp_path)],
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

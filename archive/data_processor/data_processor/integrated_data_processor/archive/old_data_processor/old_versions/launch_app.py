#!/usr/bin/env python3
"""
Simple launcher script for the CSV Processor app.
"""

import os
import subprocess
import sys


def main():
    """Main launcher function."""
    print("Starting Advanced CSV Processor...")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, "Data_Processor_r0.py")

    if not os.path.exists(app_file):
        print(f"❌ Error: Could not find {app_file}")
        input("Press Enter to exit...")
        return

    try:
        print("🚀 Launching application...")
        # Launch the app
        subprocess.Popen([sys.executable, app_file])
        print("✅ Application launched successfully!")

    except Exception as e:
        print(f"❌ Error launching application: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()

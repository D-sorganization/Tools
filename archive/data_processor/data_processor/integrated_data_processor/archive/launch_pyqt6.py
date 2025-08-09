#!/usr/bin/env python3
# =============================================================================
# Launch script for PyQt6 version of the Advanced CSV Time Series Processor
# =============================================================================

import importlib
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if all required dependencies are installed."""
    package_modules = {
        "PyQt6": "PyQt6",
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "openpyxl": "openpyxl",
        "Pillow": "PIL",
        "simpledbf": "simpledbf",
        "pyarrow": "pyarrow",
    }

    missing_packages = []

    for package, module in package_modules.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def main():
    """Main entry point."""
    print("Starting PyQt6 Data Processor...")

    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed. Please install missing packages.")
        sys.exit(1)

    try:
        # Import and run the PyQt6 application
        from Data_Processor_PyQt6 import main as run_app

        run_app()
    except ImportError as e:
        print(f"Error importing PyQt6 application: {e}")
        print("Make sure Data_Processor_PyQt6.py exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running PyQt6 application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

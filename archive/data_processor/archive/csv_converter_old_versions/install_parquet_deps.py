#!/usr/bin/env python3
"""
Install script for Parquet support dependencies.
"""

import subprocess
import sys


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def main():
    """Install required packages for Parquet support."""
    print("Installing ML data format dependencies...")

    packages = ["pyarrow", "tables", "feather-format"]

    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1

    print(
        f"\nInstallation complete: {success_count}/{len(packages)} packages installed successfully."
    )

    if success_count == len(packages):
        print(
            "✅ All dependencies installed. You can now use all ML data formats (Parquet, HDF5, Feather, Pickle)."
        )
    else:
        print(
            "⚠️  Some dependencies failed to install. Some ML data format features may not work properly."
        )


if __name__ == "__main__":
    main()

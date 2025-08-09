#!/usr/bin/env python3
"""
Package the Folder Packer executable for distribution
Creates a clean deployment package with just the necessary files
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def create_deployment_package():
    """Create a deployment package"""

    current_dir = Path(__file__).parent
    dist_dir = current_dir / "dist"

    if not dist_dir.exists():
        print("❌ Error: dist folder not found! Please build the executable first.")
        return False

    exe_file = dist_dir / "FolderPacker.exe"
    if not exe_file.exists():
        print(
            "❌ Error: FolderPacker.exe not found! Please build the executable first."
        )
        return False

    # Create package directory
    package_name = f"FolderPacker_v1.1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_dir = current_dir / "packages" / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Creating deployment package: {package_name}")

    # Copy executable
    shutil.copy2(exe_file, package_dir / "FolderPacker.exe")
    print("✓ Copied executable")

    # Copy launcher
    launcher_file = dist_dir / "Launch_FolderPacker.bat"
    if launcher_file.exists():
        shutil.copy2(launcher_file, package_dir / "Launch_FolderPacker.bat")
        print("✓ Copied launcher script")

    # Copy README
    readme_file = current_dir / "README.md"
    if readme_file.exists():
        shutil.copy2(readme_file, package_dir / "README.md")
        print("✓ Copied README")

    # Create a simple user guide
    user_guide = package_dir / "Quick_Start_Guide.txt"
    with open(user_guide, "w") as f:
        f.write(
            """Folder Packer / Unpacker Tool - Quick Start Guide
================================================

GETTING STARTED:
1. Double-click "FolderPacker.exe" to run the application
   OR
   Double-click "Launch_FolderPacker.bat" for a friendlier startup

WHAT IT DOES:
- Packs multiple programming files into a single text file
- Preserves folder structure and hierarchy
- Can unpack the text file back to original folders
- Automatically excludes common backup/archive folders

HOW TO USE:

PACKING:
1. Click "Select Source Folder" and choose your project folder
2. (Optional) Configure exclusions in the "Folder Exclusions" tab
3. Click "Select Output File" and choose where to save the packed file
4. Click "Pack Folder" - done!

UNPACKING:
1. Click "Select Packed File" and choose a previously packed text file
2. Click "Select Destination Folder" for where to restore files
3. Click "Unpack File" - your folder structure is recreated!

FEATURES:
- Modern, clean interface
- Progress tracking for large operations
- Smart exclusion of common backup folders
- Supports 30+ programming file types
- Built-in help and instructions

SUPPORTED FILE TYPES:
.py .html .css .js .java .cpp .c .h .json .xml .txt .md
.yml .yaml .php .rb .go .rs .swift .kt .r .sql .sh .bat
...and more!

TROUBLESHOOTING:
- If the app doesn't start, make sure you have Windows 10 or later
- For very large projects, the packing may take a few minutes
- Check the "Instructions" tab in the app for detailed help

For more information, see README.md

Version: 1.1
Built: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        )
    print("✓ Created Quick Start Guide")

    # Create ZIP package
    zip_file = current_dir / "packages" / f"{package_name}.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zf.write(file_path, arcname)

    print(f"✅ Created ZIP package: {zip_file}")

    # Show summary
    print(f"\n📊 DEPLOYMENT PACKAGE SUMMARY")
    print(f"Package name: {package_name}")
    print(f"Package folder: {package_dir}")
    print(f"ZIP file: {zip_file}")
    print(f"ZIP size: {zip_file.stat().st_size / (1024*1024):.1f} MB")

    print(f"\n📋 PACKAGE CONTENTS:")
    for file_path in sorted(package_dir.rglob("*")):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {file_path.name} ({size_mb:.1f} MB)")

    print(f"\n🎉 Deployment package ready!")
    print(f"📁 You can distribute the ZIP file or the entire '{package_name}' folder")

    return True


if __name__ == "__main__":
    print("📦 Folder Packer - Deployment Packager")
    print("=" * 40)
    create_deployment_package()

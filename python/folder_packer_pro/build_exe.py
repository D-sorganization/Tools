"""Build script for Folder Packer Pro v2.0 executable."""

import subprocess
import sys
from pathlib import Path


def build_exe():
    """Build Windows executable using PyInstaller."""

    print("=" * 60)
    print("Building Folder Packer Pro v2.0 Executable")
    print("=" * 60)

    # Get script directory
    script_dir = Path(__file__).parent
    main_script = script_dir / "folder_packer_pro.py"

    if not main_script.exists():
        print(f"Error: Main script not found: {main_script}")
        sys.exit(1)

    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable
        "--windowed",  # No console window
        "--name=FolderPackerPro",  # Executable name
        ("--icon=paper_plane_icon.ico" if (script_dir / "paper_plane_icon.ico").exists() else ""),
        (
            "--add-data=paper_plane_icon.ico;."
            if (script_dir / "paper_plane_icon.ico").exists()
            else ""
        ),
        "--clean",  # Clean cache
        "--noconfirm",  # Overwrite without asking
        str(main_script),
    ]

    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]

    print("\nRunning PyInstaller...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, cwd=script_dir, check=True)

        print("\n" + "=" * 60)
        print("Build completed successfully!")
        print("=" * 60)
        print(f"\nExecutable location: {script_dir / 'dist' / 'FolderPackerPro.exe'}")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"\nError: Build failed with exit code {e.returncode}")
        return 1
    except FileNotFoundError:
        print("\nError: PyInstaller not found. Please install it:")
        print("  pip install pyinstaller")
        return 1


if __name__ == "__main__":
    sys.exit(build_exe())

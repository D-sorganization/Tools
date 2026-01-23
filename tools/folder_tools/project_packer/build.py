#!/usr/bin/env python3
"""
Cross-platform launcher for building Folder Packer executable.
Replaces build.bat for better portability.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Build Folder Packer executable."""
    print("Building Folder Packer Executable...")
    print()

    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    # Run the build script
    try:
        subprocess.run(
            [sys.executable, "build_exe.py"],
            check=True,
            capture_output=False,
        )
        print()
        print("Build process completed.")

        exe_path = script_dir / "dist" / "FolderPacker.exe"
        if exe_path.exists():
            print()
            print("✅ SUCCESS: Executable created at", exe_path)
            print()
            if sys.platform == "win32":
                response = input("Would you like to run the executable now? (y/n): ")
                if response.lower() == "y":
                    os.startfile(exe_path)  # type: ignore[attr-defined]
        else:
            print()
            print("❌ Build failed. Check the output above for errors.")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print()
        print(f"❌ Build failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("Build cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()

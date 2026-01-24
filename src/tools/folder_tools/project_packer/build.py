#!/usr/bin/env python3
"""
Cross-platform launcher for building Folder Packer executable.
Replaces build.bat for better portability.
"""

import os
import sys
from pathlib import Path

# Add utils to path
repo_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root / "src" / "python" / "src"))

try:
    from utils.subprocess_utils import run_python_script
    from utils.logging_utils import get_logger
except ImportError:
    # Fallback if shared utilities not available
    import subprocess

    def run_python_script(script_path: Path, **kwargs):
        return subprocess.run([sys.executable, str(script_path)], **kwargs)

    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger(__name__)


def main() -> None:
    """Build Folder Packer executable."""
    logger.info("Building Folder Packer Executable...")

    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    # Run the build script
    try:
        run_python_script(script_dir / "build_exe.py", cwd=script_dir, capture_output=False)
        logger.info("Build process completed.")

        exe_path = script_dir / "dist" / "FolderPacker.exe"
        if exe_path.exists():
            logger.info(f"✅ SUCCESS: Executable created at {exe_path}")
            if sys.platform == "win32":
                response = input("Would you like to run the executable now? (y/n): ")
                if response.lower() == "y":
                    os.startfile(exe_path)  # type: ignore[attr-defined]
        else:
            logger.error("❌ Build failed. Check the output above for errors.")
            sys.exit(1)

    except KeyboardInterrupt:
        print()
        print("Build cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()

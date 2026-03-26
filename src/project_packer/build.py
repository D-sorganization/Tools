#!/usr/bin/env python3
"""
Cross-platform launcher for building Folder Packer executable.
Replaces build.bat for better portability.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Use shared path utilities
from utils.path_helpers import ensure_utils_in_path  # noqa: E402

ensure_utils_in_path()

# Import shared utilities
from utils.logging_utils import get_logger  # noqa: E402
from utils.subprocess_utils import run_python_script  # noqa: E402

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
                    os.startfile(exe_path)
        else:
            logger.error("❌ Build failed. Check the output above for errors.")
            sys.exit(1)

    except KeyboardInterrupt:
        sys.stderr.write("\n")  # clean newline after ^C
        logger.info("Build cancelled by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()

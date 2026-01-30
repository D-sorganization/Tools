#!/usr/bin/env python3
"""
Cross-platform launcher for building Folder Packer executable.
Replaces build.bat for better portability.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add utils to path using shared utility

# Import path_setup utility
# Add utils to path using shared utility
try:
    from utils.path_helpers import ensure_utils_in_path

    ensure_utils_in_path()
except ImportError:
    # Fallback: try to add utils manually
    try:
        from utils.path_setup import add_utils_to_path

        add_utils_to_path()
    except ImportError:
        # Last resort fallback
        # We can't easily import from utils if utils isn't in path, but we can try to find repo root
        pass

try:
    from utils.logging_utils import get_logger
    from utils.subprocess_utils import run_python_script
except ImportError:
    # Fallback if shared utilities not available
    import subprocess

    def run_python_script(script_path: Path, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        return subprocess.run([sys.executable, str(script_path)], **kwargs)

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


try:
    from utils.path_helpers import ensure_utils_in_path
except ImportError:

    def ensure_utils_in_path() -> None:
        pass


logger = get_logger(__name__)


def main() -> None:
    """Build Folder Packer executable."""
    logger.info("Building Folder Packer Executable...")

    # Change to script directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    # Run the build script
    try:
        run_python_script(
            script_dir / "build_exe.py", cwd=script_dir, capture_output=False
        )
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

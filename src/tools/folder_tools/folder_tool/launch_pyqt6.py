#!/usr/bin/env python3
"""
Cross-platform launcher for Folder Tool.
Replaces Launch_FolderFix.bat for better portability.
"""

import sys
from pathlib import Path

# Add utils to path using shared utility
try:
    from utils.path_helpers import ensure_utils_in_path, get_project_root_from_file

    ensure_utils_in_path()
except ImportError:
    # Fallback: try to add utils manually
    try:
        from utils.path_setup import add_utils_to_path

        add_utils_to_path()
    except ImportError:
        # Last resort fallback
        repo_root = get_project_root_from_file(__file__)
        ensure_utils_in_path()

try:
    from utils.logging_utils import get_logger
    from utils.subprocess_utils import run_python_script
except ImportError:
    # Fallback if shared utilities not available
    import logging
    import subprocess

    def run_python_script(
        script_path: Path,
        args: list[str] | None = None,
        cwd: Path | str | None = None,
        timeout: int | None = None,
        check: bool = False,
    ):
        command = [sys.executable, str(script_path)]
        if args:
            command.extend(args)
        return subprocess.run(
            command, cwd=str(cwd) if cwd else None, timeout=timeout, check=check
        )

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


def main() -> None:
    """Launch Folder Tool."""
    script_dir = Path(__file__).parent.absolute()
    tool_script = script_dir / "Folders_Tool_r0.py"

    if not tool_script.exists():
        logger.error(f"Tool script not found: {tool_script}")
        sys.exit(1)

    logger.info("Starting Folder Tool...")
    try:
        run_python_script(tool_script, check=True)
    except KeyboardInterrupt:
        logger.info("Tool stopped by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to launch tool: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

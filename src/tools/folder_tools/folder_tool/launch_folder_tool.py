#!/usr/bin/env python3
"""
Cross-platform launcher for Folder Tool.
Replaces Launch_FolderFix.bat for better portability.
"""

import sys
from pathlib import Path

# Add utils to path using shared utility
try:
    repo_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
    utils_path = repo_root / "src" / "python" / "src"
    if utils_path.exists():
        sys.path.insert(0, str(utils_path))
    from utils.path_setup import add_utils_to_path
    add_utils_to_path()
except ImportError:
    # Fallback
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

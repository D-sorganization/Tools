#!/usr/bin/env python3
"""Remove broken fix scripts that are causing CI failures."""

import logging
import os
from pathlib import Path

try:
    from utils.file_utils import safe_read_text, safe_write_text
except ImportError:
    from pathlib import Path

    def safe_read_text(
        path: str | Path, encoding: str = "utf-8", default: str = ""
    ) -> str:
        try:
            return Path(path).read_text(encoding=encoding)
        except Exception:
            return default

    def safe_write_text(
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> None:
        p = Path(path)
        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def remove_known_broken_scripts(broken_scripts: list[str]) -> int:
    """Remove scripts from a known list of broken scripts.

    Args:
        broken_scripts: List of script filenames to remove.

    Returns:
        Number of scripts successfully removed.
    """
    removed_count = 0
    for script in broken_scripts:
        if Path(script).exists():
            try:
                os.remove(script)
                logger.info(f"✅ Removed broken script: {script}")
                removed_count += 1
            except PermissionError as e:
                logger.error(f"❌ Permission denied removing {script}: {e}")
            except OSError as e:
                logger.error(f"❌ OS error removing {script}: {e}")
        else:
            logger.info(f"⚠️  Script not found: {script}")
    return removed_count


def find_and_remove_syntax_error_scripts() -> int:
    """Find and remove fix scripts with syntax errors.

    Returns:
        Number of scripts successfully removed.
    """
    removed_count = 0
    for file_path in Path(".").glob("fix_*.py"):
        try:
            content = safe_read_text(file_path, default="")
            compile(content, str(file_path), "exec")
            logger.info(f"✅ Script is valid: {file_path}")
        except SyntaxError as e:
            logger.warning(f"🔥 Found broken script: {file_path} - {e}")
            try:
                os.remove(file_path)
                logger.info(f"✅ Removed broken script: {file_path}")
                removed_count += 1
            except PermissionError as remove_error:
                logger.error(
                    f"❌ Permission denied removing {file_path}: {remove_error}"
                )
            except OSError as remove_error:
                logger.error(f"❌ OS error removing {file_path}: {remove_error}")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"⚠️  Could not check {file_path}: {e}")
    return removed_count


def main() -> None:
    """Remove all broken fix scripts."""
    broken_scripts = [
        "comprehensive_fix.py",
        "comprehensive_syntax_fix.py",
        "fix_remaining_lines.py",
        "fix_remaining_long_lines.py",
        "fix_remaining_quality_issues.py",
        "fix_syntax_errors.py",
        "fix_targeted_issues.py",
        "fix_type_annotations.py",
        "final_data_processor_fix.py",
    ]

    removed_count = remove_known_broken_scripts(broken_scripts)
    logger.info(f"🎯 Removed {removed_count} known broken scripts")

    additional_removed = find_and_remove_syntax_error_scripts()
    removed_count += additional_removed

    logger.info(f"🎉 Total removed: {removed_count} broken scripts")


if __name__ == "__main__":
    main()

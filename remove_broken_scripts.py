#!/usr/bin/env python3
"""Remove broken fix scripts that are causing CI failures."""

import logging

# Use shared logging utility
try:
    from utils.logging_utils import init_default_logging
except ImportError:
    # Fallback
    def init_default_logging():
        init_default_logging()
import os
from pathlib import Path


try:
    from utils.file_utils import safe_read_text, safe_write_text
except ImportError:
    from pathlib import Path
    def safe_read_text(path, encoding='utf-8', default=''):
        try:
            return Path(path).read_text(encoding=encoding)
        except Exception:
            return default
    def safe_write_text(path, content, encoding='utf-8', create_parents=True):
        p = Path(path)
        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
# Configure logging
init_default_logging()s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Remove all broken fix scripts."""

    # List of broken scripts to remove
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

    removed_count = 0

    for script in broken_scripts:
        if Path(script).exists():
            try:
                os.remove(script)
                logger.info(f"✅ Removed broken script: {script}")
                removed_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to remove {script}: {e}")
        else:
            logger.info(f"⚠️  Script not found: {script}")

    logger.info(f"🎯 Removed {removed_count} broken scripts")

    # Also remove any other fix scripts that might be broken
    for file_path in Path(".").glob("fix_*.py"):
        try:
            # Try to compile the script
            content = safe_read_text(file_path, default='')

            compile(content, str(file_path), "exec")
            logger.info(f"✅ Script is valid: {file_path}")

        except SyntaxError as e:
            logger.warning(f"🔥 Found broken script: {file_path} - {e}")
            try:
                os.remove(file_path)
                logger.info(f"✅ Removed broken script: {file_path}")
                removed_count += 1
            except Exception as remove_error:
                logger.error(f"❌ Failed to remove {file_path}: {remove_error}")
        except Exception as e:
            logger.warning(f"⚠️  Could not check {file_path}: {e}")

    logger.info(f"🎉 Total removed: {removed_count} broken scripts")


if __name__ == "__main__":
    main()

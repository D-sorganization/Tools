#!/usr/bin/env python3
"""
Automated script to fix DRY violations across the codebase.

This script systematically replaces common patterns with shared utilities.
"""

import re
import sys
from pathlib import Path
from re import Match

try:
    from utils.path_helpers import ensure_utils_in_path
except ImportError:

    def ensure_utils_in_path():
        pass


try:
    from utils.file_utils import safe_read_text, safe_write_text
except ImportError:
    from pathlib import Path

    def safe_read_text(path, encoding="utf-8", default=""):
        try:
            return Path(path).read_text(encoding=encoding)
        except Exception:
            return default

    def safe_write_text(path, content, encoding="utf-8", create_parents=True):
        p = Path(path)
        if create_parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)


# Track fixes
fix_count = 0


def fix_sys_path_manipulations(content: str, file_path: Path) -> tuple[str, int]:
    """Replace sys.path manipulations with ensure_utils_in_path()."""
    global fix_count
    fixes = 0

    # Pattern 1: ensure_utils_in_path()
    pattern1 = r"sys\.path\.insert\s*\(\s*0\s*,\s*str\s*\([^)]+\)\s*\)"

    def replace1(match: Match[str]) -> str:
        nonlocal fixes
        fixes += 1
        return "ensure_utils_in_path()"

    content = re.sub(pattern1, replace1, content)

    # Pattern 2: ensure_utils_in_path()
    pattern2 = r"sys\.path\.append\s*\(\s*str\s*\([^)]+\)\s*\)"

    def replace2(match: Match[str]) -> str:
        nonlocal fixes
        fixes += 1
        return "ensure_utils_in_path()"

    content = re.sub(pattern2, replace2, content)

    # Add import if we made replacements
    if fixes > 0:
        # Check if import already exists
        if "from utils.path_helpers import ensure_utils_in_path" not in content:
            # Find where to insert import (after other imports)
            import_pattern = r"(import sys\n|from pathlib import Path\n)"
            match = re.search(import_pattern, content)
            if match:
                insert_pos = match.end()
                content = (
                    content[:insert_pos] + "\n# Use shared path utility\n"
                    "try:\n"
                    "    from utils.path_helpers import ensure_utils_in_path\n"
                    "except ImportError:\n"
                    "    # Fallback\n"
                    "    def ensure_utils_in_path() -> None:\n"
                    "        pass\n" + content[insert_pos:]
                )

    fix_count += fixes
    return content, fixes


def fix_os_path_join(content: str) -> tuple[str, int]:
    """Replace os.path.join with Path operations."""
    global fix_count
    fixes = 0

    # Pattern: Path(a) / b / c
    pattern = r"os\.path\.join\s*\(([^)]+)\)"

    def replace(match: Match[str]) -> str:
        nonlocal fixes
        fixes += 1
        args = match.group(1)
        # Convert to Path(...) / ... / ...
        parts = [p.strip().strip("\"'") for p in args.split(",")]
        if len(parts) == 1:
            return f"Path({parts[0]})"
        result = f"Path({parts[0]})"
        for part in parts[1:]:
            result += f" / {part}"
        return result

    content = re.sub(pattern, replace, content)
    fix_count += fixes
    return content, fixes


def process_file(file_path: Path) -> int:
    """Process a single file and fix DRY violations."""
    try:
        content = safe_read_text(file_path, default="")

        original_content = content
        total_fixes = 0

        # Apply fixes
        content, fixes = fix_sys_path_manipulations(content, file_path)
        total_fixes += fixes

        # Only write if changes were made
        if content != original_content:
            safe_write_text(file_path, content)
            return total_fixes

        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return 0


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path(".")

    # Find all Python files
    python_files = list(target.rglob("*.py"))

    print(f"Processing {len(python_files)} Python files...")

    total_fixes = 0
    for file_path in python_files:
        if "__pycache__" in str(file_path) or ".git" in str(file_path):
            continue

        fixes = process_file(file_path)
        if fixes > 0:
            print(f"Fixed {fixes} violations in {file_path}")
            total_fixes += fixes

    print(f"\nTotal fixes applied: {total_fixes}")
    return 0 if total_fixes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

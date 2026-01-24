#!/usr/bin/env python3
"""Quality check script to verify AI-generated code meets standards."""

import sys
from pathlib import Path

try:
    from tools.quality_utils import (
        check_banned_patterns,
        check_magic_numbers,
        check_ast_issues,
    )
except ImportError:
    # If tools package not found (e.g. running from different dir), try to add to path
    sys.path.append(str(Path(__file__).parent))
    from tools.quality_utils import (
        check_banned_patterns,
        check_magic_numbers,
        check_ast_issues,
    )


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a Python file for quality issues."""
    try:
        content = filepath.read_text(encoding="utf-8")
        lines = content.splitlines()

        issues = []
        issues.extend(check_banned_patterns(lines, filepath))
        issues.extend(check_magic_numbers(lines, filepath))
        issues.extend(check_ast_issues(content))
    except (OSError, UnicodeDecodeError) as e:
        return [(0, f"Error reading file: {e}", "")]
    else:
        return issues


def main() -> None:
    """Run quality checks on Python files."""
    # Start from current directory
    root_dir = Path(".")
    python_files = list(root_dir.rglob("*.py"))

    # Exclude certain directories
    exclude_dirs = {
        "archive",
        "legacy",
        "experimental",
        ".git",
        "__pycache__",
        ".ruff_cache",
        ".mypy_cache",
        "matlab",
        "output",
        ".ipynb_checkpoints",
        ".Trash",
    }

    # Filter files
    python_files = [
        f for f in python_files if not any(part in exclude_dirs for part in f.parts)
    ]

    all_issues = []
    for filepath in python_files:
        issues = check_file(filepath)
        if issues:
            all_issues.append((filepath, issues))

    # Report
    if all_issues:
        sys.stderr.write("❌ Quality check FAILED\n\n")
        for filepath, issues in all_issues:
            sys.stderr.write(f"\n{filepath}:\n")
            for line_num, message, code in issues:
                if line_num > 0:
                    sys.stderr.write(f"  Line {line_num}: {message}\n")
                    if code:
                        sys.stderr.write(f"    > {code}\n")
                else:
                    sys.stderr.write(f"  {message}\n")

        sys.stderr.write(
            f"\nTotal issues: {sum(len(issues) for _, issues in all_issues)}\n",
        )
        sys.exit(1)
    else:
        sys.stderr.write("✅ Quality check PASSED\n")
        sys.stderr.write(f"Checked {len(python_files)} Python files\n")
        sys.exit(0)


if __name__ == "__main__":
    main()

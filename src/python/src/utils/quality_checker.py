#!/usr/bin/env python3
"""Quality check script to verify AI-generated code meets standards."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to find the tools package
try:
    from tools.quality_utils import (
        Colors,
        check_file,
    )
except ImportError:
    # Walk up until we find the repo root or give up
    current = Path(__file__).resolve().parent
    repo_root = None
    for _ in range(5):
        if (current / "tools" / "quality_utils.py").exists():
            repo_root = current
            break
        current = current.parent

    if repo_root:
        sys.path.append(str(repo_root))
        from tools.quality_utils import (
            Colors,
            check_file,
        )
    else:
        # Fallback for when running from elsewhere
        logger.error("Error: Could not locate tools package.")
        sys.exit(1)


def main() -> None:
    """Run quality checks on Python files."""
    # Support direct file arguments from pre-commit
    if len(sys.argv) > 1:
        python_files = [Path(arg) for arg in sys.argv[1:]]
    else:
        python_files = list(Path().rglob("*.py"))

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

    # Filter if scanning directory
    if len(sys.argv) <= 1:
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
        sys.stderr.write(
            f"{Colors.FAIL}{Colors.BOLD}❌ Quality check FAILED{Colors.ENDC}\n\n"
        )
        for filepath, issues in all_issues:
            sys.stderr.write(f"\n{Colors.CYAN}{filepath}:{Colors.ENDC}\n")
            for line_num, message, code in issues:
                if line_num > 0:
                    sys.stderr.write(
                        f"  Line {Colors.BOLD}{line_num}{Colors.ENDC}: {message}\n"
                    )
                    if code:
                        sys.stderr.write(f"    > {Colors.WARNING}{code}{Colors.ENDC}\n")
                else:
                    sys.stderr.write(f"  {message}\n")

        total_issues = sum(len(issues) for _, issues in all_issues)
        sys.stderr.write(
            f"\n{Colors.FAIL}Total issues: {total_issues}{Colors.ENDC}\n",
        )
        sys.exit(1)
    else:
        # success silent for pre-commit usually, but ok to print
        sys.exit(0)


if __name__ == "__main__":
    main()

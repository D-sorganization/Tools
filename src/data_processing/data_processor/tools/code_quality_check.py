#!/usr/bin/env python3
"""Quality check script to verify AI-generated code meets standards.

This script uses the shared quality_checker utility module to ensure consistency
across the repository and follow DRY principles.
"""

import sys
from pathlib import Path

# Add utils to path for import
try:
    from utils.path_helpers import ensure_utils_in_path

    ensure_utils_in_path()
except ImportError:
    # Fallback
    from utils.path_helpers import get_project_root_from_file

    repo_root = get_project_root_from_file(__file__)
    sys.path.insert(0, str(repo_root / "src" / "python" / "src"))

from utils.quality_checker import (
    check_file,
    find_python_files,
    get_default_exclude_dirs,
    report_issues,
)


def main() -> None:
    """Run quality checks on Python files."""
    # Support direct file arguments from pre-commit
    file_args = sys.argv[1:] if len(sys.argv) > 1 else None

    python_files = find_python_files(
        root=Path(),
        exclude_dirs=get_default_exclude_dirs(),
        file_args=file_args,
    )

    all_issues = []
    for filepath in python_files:
        issues = check_file(filepath)
        if issues:
            all_issues.append((filepath, issues))

    # Report
    if all_issues:
        report_issues(all_issues)
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

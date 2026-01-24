#!/usr/bin/env python3
"""Wrapper for compiled MATLAB quality checks."""

import sys
from pathlib import Path

# Try to find the tools package
try:
    from tools.matlab_quality_utils import run_matlab_quality_checks_cli
except ImportError:
    # Walk up until we find the repo root or give up
    current = Path(__file__).resolve().parent
    repo_root = None
    for _ in range(5):
        if (current / "tools" / "matlab_quality_utils.py").exists():
            repo_root = current
            break
        current = current.parent

    if repo_root:
        sys.path.append(str(repo_root))
        from tools.matlab_quality_utils import run_matlab_quality_checks_cli
    else:
        # Fallback for when running from elsewhere
        print("Error: Could not locate tools package.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_matlab_quality_checks_cli()

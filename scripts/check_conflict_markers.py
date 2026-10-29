#!/usr/bin/env python3
"""Scan the repository for raw git merge-conflict markers (#2050, RM-6).

Mirrors the ``Reject Raw Merge Conflict Markers`` step in ``ci-standard.yml`` so
``python -m scripts.pre_pr``, staff bootstrap, and CI share one implementation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Built from parts so this file never matches its own pattern.
_CONFLICT_MARKER = "<" * 7 + " HEAD"


def scan_conflict_markers(root: Path) -> list[str]:
    """Return ``git grep`` hits for raw merge-conflict markers (excluding .github)."""
    proc = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            _CONFLICT_MARKER,
            "--",
            ".",
            ":(exclude).github",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 1:
        return []
    if proc.returncode != 0:
        message = (proc.stderr or proc.stdout or "git grep failed").strip()
        raise RuntimeError(message)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()

    try:
        hits = scan_conflict_markers(root)
    except RuntimeError as exc:
        sys.stderr.write(f"::error::{exc}\n")
        return 1

    if hits:
        sys.stderr.write("::error::Raw git merge conflict markers found:\n")
        for hit in hits:
            sys.stderr.write(f"  {hit}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

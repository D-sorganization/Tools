#!/usr/bin/env python3
"""Run pre-PR local checks mirroring CI gates (RM-6).

AGENTS.md's synced PR Lifecycle rule says to run ``python -m scripts.pre_pr``
before pushing. Tools had no such module, so staff bootstrap and Desk roles
failed with ``No module named scripts.pre_pr``.

This entry point composes:

- ``scripts.check_conflict_markers`` (raw merge-conflict marker scan, same as CI)
- ``scripts.pre_pr_delta_gates`` (changed-file ruff, ruff format, mypy, and
  ``check_changed_python_policy`` — the blocking delta checks from
  ``quality-gate`` in ``ci-standard.yml``)

Repository_Management's ``scripts/pre_pr.py`` additionally wires RM-only
``pr_cap_preflight`` / ``pr_closing_ref`` checks; Runner_Dashboard uses
``verify_merge_gates`` instead. This module is Tools-appropriate, not a
byte-for-byte copy of either.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scripts import check_conflict_markers, pre_pr_delta_gates

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Ref the pull request will merge into",
    )
    args = parser.parse_args(argv)

    root_argv = ["--repository-root", str(args.repository_root)]
    markers_status = check_conflict_markers.main(root_argv)
    delta_status = pre_pr_delta_gates.main(
        [*root_argv, "--base", args.base],
    )

    if markers_status or delta_status:
        return 1
    sys.stdout.write("pre_pr: all local gates passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

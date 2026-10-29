#!/usr/bin/env python3
"""Run pre-PR local checks mirroring CI gates (RM-6).

AGENTS.md's synced PR Lifecycle rule says to run ``python -m scripts.pre_pr``
before pushing. Tools had no such module, so staff bootstrap and Desk roles
failed with ``No module named scripts.pre_pr``.

Runs the raw merge-conflict marker scan and the blocking delta checks from
``quality-gate`` in ``ci-standard.yml`` (changed-file ruff, format, mypy, and
``check_changed_python_policy``).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Built from parts so this file never matches its own pattern.
_CONFLICT_MARKER = "<" * 7 + " HEAD"

RUFF_EXCLUDE = re.compile(
    r"^src/document_processing/|"
    r"^src/pendulum_simulator/|"
    r"^src/movement_optimizer/|"
    r"^src/scientific_modeling/|"
    r"^src/tools/folder_tools/|"
    r"^src/tools/code_quality_check\.py"
)

MYPY_LEGACY_EXCLUDE = re.compile(
    r"src/electrode_advisor/|"
    r"src/function_generator/|"
    r"src/data_processing/data_processor/|"
    r"src/glass_bath_fea/|"
    r"src/movement_optimizer/|"
    r"src/tools/folder_tools/|"
    r"simulation_golfer_gpu|"
    r"optimizer_gpu|"
    r"src/shared/python/codemap/"
)


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


def run_conflict_marker_check(root: Path) -> int:
    """Scan for raw merge-conflict markers."""
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


def merge_base_sha(base_ref: str, root: Path) -> str | None:
    """Return the merge base between ``base_ref`` and ``HEAD``."""
    proc = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def changed_python_files(base_sha: str, root: Path) -> tuple[list[str], list[str]]:
    """Return changed ``*.py`` paths and the mypy subset (non-test paths)."""
    proc = subprocess.run(
        [
            "git",
            "diff",
            "--diff-filter=ACMRT",
            "--name-only",
            base_sha,
            "HEAD",
            "--",
            "*.py",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return [], []
    changed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    mypy_candidates = [
        path for path in changed if not (path.startswith("tests/") or "/tests/" in path)
    ]
    return changed, mypy_candidates


def _filter_ruff(paths: Sequence[str]) -> list[str]:
    return [path for path in paths if not RUFF_EXCLUDE.search(path)]


def _filter_mypy(paths: Sequence[str]) -> list[str]:
    return [path for path in paths if not MYPY_LEGACY_EXCLUDE.search(path)]


def _run(cmd: list[str], root: Path, *, env: dict[str, str] | None = None) -> int:
    proc = subprocess.run(cmd, cwd=root, env=env, check=False)
    return proc.returncode


def _mypy_available() -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _run_policy(root: Path, base_ref: str) -> int:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.check_changed_python_policy",
            "--base-ref",
            base_ref,
        ],
        cwd=root,
        check=False,
    )
    return proc.returncode


def run_delta_gates(root: Path, base_ref: str) -> int:
    """Mirror blocking delta steps from ``ci-standard.yml`` ``quality-gate``."""
    base_sha = merge_base_sha(base_ref, root)
    if base_sha is None:
        sys.stderr.write(f"::error::Could not resolve merge base for {base_ref}\n")
        return 1

    changed, mypy_changed = changed_python_files(base_sha, root)
    status = _run_policy(root, base_ref)

    ruff_paths = _filter_ruff(changed)
    if ruff_paths:
        status |= _run(
            [sys.executable, "-m", "ruff", "check", *ruff_paths],
            root,
        )
        status |= _run(
            [sys.executable, "-m", "ruff", "format", "--check", *ruff_paths],
            root,
        )

    mypy_paths = _filter_mypy(mypy_changed)
    if mypy_paths:
        if not _mypy_available():
            sys.stderr.write(
                "::error::mypy is required for changed-file type checks; "
                "install mypy or run through the pre-commit hook environment.\n"
            )
            status |= 1
        else:
            env = os.environ.copy()
            env["MYPYPATH"] = "src:src/python/src"
            status |= _run(
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--ignore-missing-imports",
                    "--follow-imports=silent",
                    *mypy_paths,
                ],
                root,
                env=env,
            )

    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Ref the pull request will merge into",
    )
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()

    markers_status = run_conflict_marker_check(root)
    delta_status = run_delta_gates(root, args.base)

    if markers_status or delta_status:
        return 1
    sys.stdout.write("pre_pr: all local gates passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CI gate enforcing a per-file LOC budget for .py and .rs sources.

Fails when any source file under ``src/`` exceeds ``--max-loc`` lines and is
not grandfathered in the baseline file. Supports scanning either the full tree
or only files changed relative to ``origin/staging`` (via ``--changed-only``).

Usage:
    python3 scripts/check_file_size_budget.py \
        --max-loc 500 --baseline-file scripts/monolith_baseline.txt

    python3 scripts/check_file_size_budget.py \
        --max-loc 500 --changed-only \
        --baseline-file scripts/monolith_baseline.txt
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

DEFAULT_MAX_LOC = 500
DEFAULT_ROOT = Path("src")
TRACKED_SUFFIXES = frozenset({".py", ".rs"})
EXCLUDE_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "archive",
        "legacy",
        "__pycache__",
        "build",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
        "target",
    }
)

logger = logging.getLogger("check_file_size_budget")


def _normalize(path: Path) -> str:
    """Return a POSIX-style path string for portability."""
    return str(path).replace("\\", "/")


def count_loc(path: Path) -> int:
    """Count lines in ``path`` (tolerates bad encodings)."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def should_skip(path: Path) -> bool:
    """Return True if any path component is in the exclusion set."""
    return any(part in EXCLUDE_PARTS for part in path.parts)


def load_baseline(path: Path | None) -> set[str]:
    """Load a newline-separated baseline file into a set of paths."""
    if path is None or not path.exists():
        return set()
    entries: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            entries.add(stripped.replace("\\", "/"))
    return entries


def iter_tracked_files(root: Path) -> list[Path]:
    """Return all tracked .py / .rs files under ``root`` honoring exclusions."""
    if not root.exists():
        return []
    results: list[Path] = []
    for suffix in TRACKED_SUFFIXES:
        for candidate in root.rglob(f"*{suffix}"):
            if candidate.is_file() and not should_skip(candidate):
                results.append(candidate)
    return sorted(results)


def changed_files(base_ref: str, root: Path) -> list[Path]:
    """Return tracked files changed relative to ``base_ref`` via git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git diff failed (%s); scanning all files instead.", exc)
        return iter_tracked_files(root)

    candidates: list[Path] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        if path.suffix not in TRACKED_SUFFIXES:
            continue
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.exists() and not should_skip(path):
            candidates.append(path)
    return sorted(candidates)


def find_violations(
    files: list[Path], max_loc: int, baseline: set[str]
) -> list[tuple[str, int]]:
    """Return [(path, loc)] for files over ``max_loc`` and not in baseline."""
    violations: list[tuple[str, int]] = []
    for file_path in files:
        rel = _normalize(file_path)
        if rel in baseline:
            continue
        loc = count_loc(file_path)
        if loc > max_loc:
            violations.append((rel, loc))
    return sorted(violations, key=lambda item: (-item[1], item[0]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enforce a per-file LOC budget for Python and Rust sources."
    )
    parser.add_argument("--max-loc", type=int, default=DEFAULT_MAX_LOC)
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--changed-only", action="store_true")
    parser.add_argument("--base-ref", default="origin/staging")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    baseline = load_baseline(args.baseline_file)
    logger.info("Baseline entries loaded: %d", len(baseline))

    if args.changed_only:
        files = changed_files(args.base_ref, args.root)
        logger.info("Scanning %d changed files vs %s", len(files), args.base_ref)
    else:
        files = iter_tracked_files(args.root)
        logger.info("Scanning %d tracked files under %s", len(files), args.root)

    violations = find_violations(files, args.max_loc, baseline)

    if not violations:
        logger.info("File-size budget passed: 0 violations (max %d LOC).", args.max_loc)
        return 0

    # CLI report: use sys.stderr directly for structured failure output.
    sys.stderr.write(
        f"File-size budget FAILED: {len(violations)} violations "
        f"(max {args.max_loc} LOC).\n"
    )
    for rel, loc in violations:
        sys.stderr.write(f"  {rel}: {loc} LOC\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

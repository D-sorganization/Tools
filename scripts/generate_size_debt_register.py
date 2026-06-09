#!/usr/bin/env python3
"""Generate the large-file (size) debt register for issue #3261.

The repository standard caps source files at 400 lines and marks files above
800 lines as critical structural debt. This script enumerates every tracked
source file under ``src/`` that is at or above the 800-line threshold and emits
a ranked Markdown register, so the architecture debt has a single, reproducible
source of truth that refactor work can be tracked against.

Usage:
    python scripts/generate_size_debt_register.py            # print to stdout
    python scripts/generate_size_debt_register.py --check    # fail if doc stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
REGISTER_PATH = REPO_ROOT / "docs" / "development" / "SIZE_DEBT_REGISTER.md"

# Source-like extensions whose line count counts as code structure.
SOURCE_SUFFIXES = frozenset({".py", ".tsx", ".ts", ".jsx", ".js", ".m", ".rs"})
# Directory names never counted (dependencies, caches, build output).
SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "target",
        "dist",
        "build",
        ".gaai",
        "_build",
        "vendor",
        "site-packages",
    }
)

# Repository policy thresholds.
CRITICAL_LOC = 1000
TRACK_LOC = 800

_HEADER = """\
<!--
  GENERATED FILE - do not edit by hand.
  Regenerate with:
    python scripts/generate_size_debt_register.py --write
  Source of truth for issue #3261 (retire monoliths / structural debt).
-->

# Size debt register

Tracks every source file under `src/` at or above **{track} lines**. The
repository standard caps files at 400 lines; files at/above 800 lines are
structural debt and files at/above {critical} lines are **CRITICAL**.

This register is the ranked work queue for issue #3261. Refactor one file per
PR (responsibility-preserving extraction, behaviour pinned by characterization
tests), then regenerate this file so the count ratchets down. The register is
intentionally non-blocking: it informs prioritisation, it does not gate CI.

"""


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def collect(src_root: Path = SRC_ROOT) -> list[tuple[int, str]]:
    """Return ``(loc, posix_path)`` for every tracked file at/above TRACK_LOC.

    Paths are reported relative to ``src_root``'s parent, so the real tree yields
    ``src/...`` and a test fixture rooted elsewhere yields the same shape.
    """
    base = src_root.parent
    rows: list[tuple[int, str]] = []
    for path in src_root.rglob("*"):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        loc = _line_count(path)
        if loc >= TRACK_LOC:
            rows.append((loc, path.relative_to(base).as_posix()))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return rows


def render(rows: list[tuple[int, str]]) -> str:
    critical = sum(1 for loc, _ in rows if loc >= CRITICAL_LOC)
    out = [_HEADER.format(track=TRACK_LOC, critical=CRITICAL_LOC)]
    out.append(f"- Files at/above {TRACK_LOC} LOC: **{len(rows)}**")
    out.append(f"- CRITICAL (at/above {CRITICAL_LOC} LOC): **{critical}**")
    out.append("")
    out.append("| Rank | LOC | Class | File |")
    out.append("| ---- | --- | ----- | ---- |")
    for rank, (loc, rel) in enumerate(rows, 1):
        cls = "CRITICAL" if loc >= CRITICAL_LOC else "HIGH"
        out.append(f"| {rank} | {loc} | {cls} | `{rel}` |")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed register is out of date.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the register to docs/development/SIZE_DEBT_REGISTER.md (UTF-8).",
    )
    args = parser.parse_args(argv)

    content = render(collect())

    if args.write:
        REGISTER_PATH.write_text(content, encoding="utf-8")
        sys.stdout.write(f"Wrote {REGISTER_PATH.relative_to(REPO_ROOT).as_posix()}\n")
        return 0

    if args.check:
        existing = (
            REGISTER_PATH.read_text(encoding="utf-8") if REGISTER_PATH.exists() else ""
        )
        if existing != content:
            sys.stderr.write(
                "docs/development/SIZE_DEBT_REGISTER.md is stale; regenerate with "
                "`python scripts/generate_size_debt_register.py > "
                "docs/development/SIZE_DEBT_REGISTER.md`\n"
            )
            return 1
        sys.stdout.write("Size debt register is up to date.\n")
        return 0

    sys.stdout.write(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

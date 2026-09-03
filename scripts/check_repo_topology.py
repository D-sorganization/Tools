#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "src",
    "tests",
    "docs",
    "scripts",
    "config",
    "docs/architecture/CANONICAL_TOPOLOGY.md",
]


def _root_allowlist_offenders() -> tuple[str, ...]:
    """Tracked top-level entries missing from scripts/check_root_allowlist.py."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.check_root_allowlist import (
        disallowed_entries,
        tracked_top_level_entries,
    )

    return disallowed_entries(tracked_top_level_entries(ROOT))


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        sys.stderr.write("Repository topology check failed. Missing required paths:\n")
        for path in missing:
            sys.stderr.write(f"- {path}\n")
        return 1

    offenders = _root_allowlist_offenders()
    if offenders:
        sys.stderr.write(
            "Repository topology check failed. Tracked top-level entries not in "
            "scripts/check_root_allowlist.py ROOT_ALLOWLIST (#4917):\n"
        )
        for entry in offenders:
            sys.stderr.write(f"- {entry}\n")
        return 1

    sys.stdout.write("Repository topology check passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

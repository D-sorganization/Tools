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


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        sys.stderr.write("Repository topology check failed. Missing required paths:\n")
        for path in missing:
            sys.stderr.write(f"- {path}\n")
        return 1

    sys.stdout.write("Repository topology check passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

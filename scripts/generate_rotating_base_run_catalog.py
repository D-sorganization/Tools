#!/usr/bin/env python3
"""Generate the qualified rotating-base web trace catalog atomically."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src"
    / "shared"
    / "python"
    / "swing_sim"
    / "rotating_base"
    / "resources"
    / "rotating_base_registered_runs_v1.json"
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    from shared.python.swing_sim.rotating_base import (
        generate_registered_run_catalog,
        registered_run_catalog_json,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = generate_registered_run_catalog()
    content = registered_run_catalog_json(results)
    _atomic_write(args.output.resolve(), content)
    digest = sha256(content.encode("utf-8")).hexdigest()
    print(f"wrote {len(results)} registered runs to {args.output}")
    print(f"catalog_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

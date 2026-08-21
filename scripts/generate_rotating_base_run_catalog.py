#!/usr/bin/env python3
"""Generate the qualified rotating-base web trace catalog atomically."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from hashlib import sha256
from pathlib import Path

from shared.python.swing_sim.rotating_base import (
    generate_registered_run_catalog,
    registered_run_catalog_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = generate_registered_run_catalog()
    content = registered_run_catalog_json(results)
    _atomic_write(args.output.resolve(), content)
    digest = sha256(content.encode("utf-8")).hexdigest()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info("wrote %d registered runs to %s", len(results), args.output)
    LOGGER.info("catalog_sha256=%s", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

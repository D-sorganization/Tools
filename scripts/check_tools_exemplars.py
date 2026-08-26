#!/usr/bin/env python3
"""Verify TOOLS-D4 exemplar manuals and all governed traceability targets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools_exemplar_contract import (
    ExemplarContractError,
    verify_exemplar_repository,
)


def main() -> int:
    """Run the fail-closed D4 gate and emit one stable summary."""
    try:
        summary = verify_exemplar_repository(Path(__file__).resolve().parents[1])
    except (ExemplarContractError, OSError, json.JSONDecodeError) as error:
        sys.stderr.write(f"Tools exemplar contract failed: {error}\n")
        return 1
    sys.stdout.write(
        "Tools exemplar contract verified: "
        f"{summary.verified_exemplar_count} verified-unapproved, "
        f"{summary.blocked_exemplar_count} blocked.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

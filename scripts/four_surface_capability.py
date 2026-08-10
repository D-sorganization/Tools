"""Generate or validate the Rate of Closure four-surface capability contract."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from rate_of_closure.four_surface_capability import (
    DEFAULT_MANIFEST_PATH,
    canonical_manifest_json,
    load_four_surface_capability,
    render_json_schema,
    validate_freshness,
    validate_repository_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--schema", action="store_true", help="emit JSON Schema")
    parser.add_argument("--normalize", action="store_true", help="emit canonical JSON")
    parser.add_argument("--on-date", type=date.fromisoformat, default=date.today())
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate deterministic output or validate all local evidence gates."""
    args = _parser().parse_args(argv)
    if args.schema:
        sys.stdout.buffer.write(render_json_schema())
        return 0
    try:
        manifest = load_four_surface_capability(args.manifest)
        validate_repository_evidence(manifest, REPO_ROOT)
        validate_freshness(manifest, on_date=args.on_date)
    except (OSError, ValueError, ValidationError) as error:
        logging.error("four-surface capability validation failed: %s", error)
        return 1
    if args.normalize:
        sys.stdout.write(canonical_manifest_json(manifest))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regenerate the ``themes`` section of design_tokens.json from themes.json.

themes.json is the single source of theme colors (Tools #5917). The themes
table in design_tokens.json is kept only for readers of that file and is
generated, never hand-edited.

Usage:
    python scripts/sync_design_tokens.py          # rewrite design_tokens.json
    python scripts/sync_design_tokens.py --check  # exit 1 if it has drifted
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
THEMES_JSON = REPO_ROOT / "src" / "shared" / "theme-definitions" / "themes.json"
TOKENS_JSON = REPO_ROOT / "src" / "shared" / "design_tokens.json"

logger = logging.getLogger("sync_design_tokens")


def build_design_tokens(
    catalog: dict[str, Any], tokens: dict[str, Any]
) -> dict[str, Any]:
    """Return ``tokens`` with its themes table rebuilt from ``catalog``.

    Each theme's ``colors`` and ``semantic`` maps are flattened in catalog
    order; every other token section is kept unchanged and in place.

    Raises:
        ValueError: If either document lacks a ``themes`` mapping.
    """
    if not isinstance(catalog.get("themes"), dict):
        raise ValueError("themes.json has no 'themes' mapping")
    if not isinstance(tokens.get("themes"), dict):
        raise ValueError("design_tokens.json has no 'themes' mapping")
    tokens["themes"] = {
        theme_id: {**theme["colors"], **theme["semantic"]}
        for theme_id, theme in catalog["themes"].items()
    }
    return tokens


def _render(tokens: dict[str, Any]) -> str:
    return json.dumps(tokens, indent=2, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite or check design_tokens.json; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on drift")
    parser.add_argument("--themes", type=Path, default=THEMES_JSON)
    parser.add_argument("--tokens", type=Path, default=TOKENS_JSON)
    args = parser.parse_args(argv)

    catalog = json.loads(args.themes.read_text(encoding="utf-8"))
    current = args.tokens.read_text(encoding="utf-8")
    expected = _render(build_design_tokens(catalog, json.loads(current)))

    if expected == current:
        return 0
    if args.check:
        logger.error(
            "%s themes drifted from %s; run scripts/sync_design_tokens.py",
            args.tokens,
            args.themes,
        )
        return 1
    args.tokens.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())

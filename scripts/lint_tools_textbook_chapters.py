#!/usr/bin/env python3
"""Lint every registered Tools textbook chapter through the D3 contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools_formula_traceability_contract import (
    FormulaTraceabilityError,
    verify_formula_traceability,
)
from scripts.tools_textbook_chapter_contract import (
    TextbookChapterError,
)
from scripts.tools_textbook_chapter_lint import verify_textbook_chapters

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run the linter and emit one deterministic repository summary."""
    try:
        summary = verify_textbook_chapters(REPOSITORY_ROOT)
        formula_summary = verify_formula_traceability(REPOSITORY_ROOT)
    except (
        FormulaTraceabilityError,
        TextbookChapterError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        sys.stderr.write(f"Tools textbook chapter lint failed: {error}\n")
        return 1
    sys.stdout.write(
        "Tools textbook chapter contract verified: "
        f"{summary.chapter_count} registered chapters, "
        f"{formula_summary.family_count} derivation families, "
        f"{len(formula_summary.formula_ids)} bidirectional formulas, "
        f"release={summary.release_status}.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

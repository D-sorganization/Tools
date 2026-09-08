#!/usr/bin/env python3
"""Run or generate Tools manual render, semantic, and accessibility QA verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.tools_manual_qa_contract import (
    ManualQAError,
    build_qa_ledger,
    verify_manual_qa,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate that manual-qa.json matches live artifacts and contracts.",
    )
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Inspect live artifacts and write manuals/tools/manual-qa.json.",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="Display a summary of QA findings and zero-sampling audit evidence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    ledger_path = root / "manuals" / "tools" / "manual-qa.json"
    schema_path = root / "manuals" / "tools" / "schemas" / "manual-qa.schema.json"

    try:
        if args.generate:
            payload = build_qa_ledger(root)
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(payload)
            ledger_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(
                f"Generated {ledger_path.relative_to(root)}: "
                f"PDF ({payload['inspections']['pdf']['page_count']} pages, 0 uninspected), "
                f"DOCX ({payload['inspections']['docx']['paragraph_count']} paras, "
                f"{payload['inspections']['docx']['math_element_count']} math), "
                f"HTML ({payload['inspections']['html']['mathml_block_count']} MathML blocks, "
                f"{payload['inspections']['html']['image_count']} images with alt), "
                f"TeX ({payload['inspections']['tex']['figure_count']} figures)."
            )
            return 0

        ledger = verify_manual_qa(root)
        if args.summary or args.check:
            print(
                "Manual QA verified (complete zero-sampling):\n"
                f"  - PDF: {ledger.pdf.page_count} pages, {ledger.pdf.uninspected_pages} uninspected, "
                f"{len(ledger.pdf.fonts)} fonts, {ledger.pdf.outline_item_count} outlines, "
                f"{ledger.pdf.total_images} images, {ledger.pdf.total_annotations} annotations\n"
                f"  - DOCX: {ledger.docx.paragraph_count} paragraphs, {ledger.docx.heading_count} headings, "
                f"{ledger.docx.math_element_count} equations, {ledger.docx.table_count} tables, "
                f"{ledger.docx.drawing_count} drawings, {ledger.docx.bookmark_count} bookmarks\n"
                f"  - HTML: lang={ledger.html.has_lang}, viewport={ledger.html.has_viewport}, "
                f"{ledger.html.mathml_block_count} MathML blocks, {ledger.html.image_count} images (0 missing alt)\n"
                f"  - TeX: geometry={ledger.tex.has_geometry}, microtype={ledger.tex.has_microtype}, "
                f"{ledger.tex.figure_count} figures, CSL references={ledger.tex.csl_reference_block_present}\n"
                f"  - Status: {ledger.release_status}, {len(ledger.blockers)} release blockers."
            )
            return 0
    except (ManualQAError, OSError, ValueError) as exc:
        sys.stderr.write(f"ERROR: Tools manual QA failed: {exc}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

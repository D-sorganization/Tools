#!/usr/bin/env python3
"""Run or generate Tools public publication projection verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.tools_publication_projection_contract import (
    PublicationProjectionError,
    build_publication_projection,
    verify_publication_projection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate that publication-projection.json matches live authorities and artifacts.",
    )
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Inspect live authorities and write manuals/tools/publication-projection.json.",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="Display summary of publication projection evidence and authority bindings.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "manuals" / "tools" / "publication-projection.json"
    schema_path = (
        root / "manuals" / "tools" / "schemas" / "publication-projection.schema.json"
    )

    try:
        if args.generate:
            payload = build_publication_projection(root)
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(payload)
            manifest_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(
                f"Generated {manifest_path.relative_to(root)}: "
                f"commit={payload['evidence']['immutable_source_commit']}, "
                f"tree={payload['evidence']['source_tree_sha256']}, "
                f"status={payload['release_status']}."
            )
            return 0

        ledger = verify_publication_projection(root)

        if args.summary:
            print(
                "Tools Engineering Design Manual Public Publication Projection Summary:"
            )
            print(f"  Schema Version:       {ledger.schema_version}")
            print(f"  Authority Repo:       {ledger.authority_repository}")
            print(f"  Catalog Repo:         {ledger.catalog_repository}")
            print(f"  Release Status:       {ledger.release_status}")
            print(f"  Source Commit:        {ledger.evidence.immutable_source_commit}")
            print(f"  Source Tree:          {ledger.evidence.source_tree_sha256}")
            print(
                f"  Calc Registry SHA:    {ledger.evidence.calculation_registry_sha256}"
            )
            print(f"  Toolchain Lock SHA:   {ledger.evidence.toolchain_lock_sha256}")
            print(
                f"  PDF Pages Reviewed:   {ledger.evidence.pdf_page_review.page_count} (uninspected={ledger.evidence.pdf_page_review.uninspected_pages})"
            )
            print(
                f"  DOCX Headings/Math:   {ledger.evidence.docx_page_review.heading_count} / {ledger.evidence.docx_page_review.math_element_count}"
            )
            print(
                f"  HTML Accessibility:   Lang={ledger.evidence.accessibility_review.has_lang}, MissingAlt={ledger.evidence.accessibility_review.images_missing_alt}"
            )
            print(f"  Human Approval State: {ledger.evidence.human_approval.state}")
            print(f"  Blockers:             {len(ledger.blockers)} active")
            return 0

        # --check mode
        print(
            f"Publication projection verified: commit={ledger.evidence.immutable_source_commit}, "
            f"tree={ledger.evidence.source_tree_sha256}, "
            f"status={ledger.release_status}, "
            f"{len(ledger.blockers)} blockers active."
        )
        return 0

    except (PublicationProjectionError, json.JSONDecodeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

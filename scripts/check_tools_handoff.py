#!/usr/bin/env python3
"""Run or generate Tools handoff and maintenance contract verification."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.tools_handoff_contract import (
    HandoffMaintenanceError,
    build_handoff_manifest,
    check_diff_aware_freshness,
    verify_handoff_maintenance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate that manuals/tools/handoff-manifest.json matches live authorities and handoff files.",
    )
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Inspect live authorities and write manuals/tools/handoff-manifest.json.",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="Display summary of handoff maintenance contract and governed files.",
    )
    parser.add_argument(
        "--diff-check",
        action="store_true",
        help="Enforce diff-aware CI gate against base ref (default: origin/main).",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base git reference for diff-aware check.",
    )
    parser.add_argument(
        "--pr-url",
        default="https://github.com/D-sorganization/Tools/pull/5055",
        help="Candidate PR URL to record in manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "manuals" / "tools" / "handoff-manifest.json"
    schema_path = (
        root / "manuals" / "tools" / "schemas" / "handoff-maintenance.schema.json"
    )

    try:
        if args.generate:
            payload = build_handoff_manifest(root, pr_url=args.pr_url)
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(payload)
            manifest_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"Generated handoff manifest at {manifest_path.relative_to(root)}")
            return 0

        if args.check:
            manifest = verify_handoff_maintenance(root)
            if args.diff_check:
                check_diff_aware_freshness(root, base_ref=args.base_ref)
            print(
                f"Handoff maintenance contract verified: {len(manifest.governed_handoff_files)} governed handoffs, "
                f"release={manifest.release_status}, commit={manifest.commit_evidence.local_head_sha[:9]}."
            )
            return 0

        if args.summary:
            manifest = verify_handoff_maintenance(root)
            print(f"Schema Version:       {manifest.schema_version}")
            print(
                f"Program:              Epic {manifest.program['epic']}, Subepic {manifest.program['subepic']}"
            )
            print(f"Authority Repo:       {manifest.repository}")
            print(f"Release Status:       {manifest.release_status}")
            print(f"Local HEAD SHA:       {manifest.commit_evidence.local_head_sha}")
            print(f"Reviewed Tree SHA:    {manifest.commit_evidence.reviewed_tree_sha}")
            print(
                f"Branch:               {manifest.branch_and_worktree.branch} (clean: {manifest.branch_and_worktree.is_clean})"
            )
            print(f"Governed Handoffs:    {len(manifest.governed_handoff_files)}")
            for rel_path, rec in manifest.governed_handoff_files.items():
                print(
                    f"  - {rel_path} ({rec.line_count}/{rec.max_line_budget} lines, {rec.sha256_lf[:12]}...)"
                )
            print(f"Active Blockers:      {len(manifest.blockers)}")
            for b in manifest.blockers:
                print(f"  - [{b.id}] {b.owner}: {b.resolution}")
            print(f"Next Dependency:      {manifest.next_dependency}")
            return 0

    except (HandoffMaintenanceError, OSError, json.JSONDecodeError) as err:
        sys.stderr.write(f"ERROR: {err}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

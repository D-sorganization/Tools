#!/usr/bin/env python3
"""Verify a heavy-integration JUnit report against the required-case manifest.

The lane must not go green when it exercised nothing. This fails when the
report is missing or unparseable, contains zero cases, executed zero cases
(all skipped), or when any case in the manifest is absent, skipped, failed
or errored. Optional cases outside the manifest may skip.

Tracking: issue #5921.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_MANIFEST = Path("tests/heavy_integration/required_cases.txt")


def load_manifest(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")
    ]


def case_outcomes(report: Path) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        case_id = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("skipped") is not None:
            outcomes[case_id] = "skipped"
        elif case.find("failure") is not None or case.find("error") is not None:
            outcomes[case_id] = "failed"
        else:
            outcomes[case_id] = "passed"
    return outcomes


def check(report: Path, manifest: Path) -> list[str]:
    """Return a list of problems; empty means the lane exercised what it claims."""
    if not report.is_file():
        return [f"report not found: {report}"]
    try:
        outcomes = case_outcomes(report)
    except ET.ParseError as exc:
        return [f"report unparseable: {exc}"]
    if not outcomes:
        return ["report contains zero test cases"]
    problems: list[str] = []
    if not any(v != "skipped" for v in outcomes.values()):
        problems.append(f"all {len(outcomes)} selected cases skipped")
    required = load_manifest(manifest)
    if not required:
        problems.append(f"required-case manifest is empty: {manifest}")
    for case_id in required:
        outcome = outcomes.get(case_id, "missing")
        if outcome != "passed":
            problems.append(f"required case {outcome}: {case_id}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    problems = check(args.report, args.manifest)
    for problem in problems:
        print(f"::error::heavy lane: {problem}")
    if not problems:
        print(
            f"heavy lane: all {len(load_manifest(args.manifest))} required cases passed"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

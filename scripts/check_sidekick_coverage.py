#!/usr/bin/env python3
"""Script to enforce a per-file 50% minimum coverage on all sidekick modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import defusedxml.ElementTree as ET


def check_sidekick_coverage(
    coverage_file: Path, changed_files_path: Path | None = None
) -> int:
    if not coverage_file.exists():
        print(f"Error: Coverage file {coverage_file} does not exist.", file=sys.stderr)
        return 1

    try:
        root = ET.parse(coverage_file).getroot()
    except Exception as e:
        print(f"Error parsing coverage file {coverage_file}: {e}", file=sys.stderr)
        return 1

    target_files = {
        "src/shared/python/sidekick/latex_renderer.py",
        "src/shared/python/sidekick/notes_store.py",
        "src/shared/python/sidekick/notes_tab.py",
        "src/shared/python/sidekick/selected_tab_panel.py",
        "src/shared/python/sidekick/tab_context_menu.py",
        "src/shared/python/sidekick/symbolic_engine.py",
    }

    changed_files = set()
    if changed_files_path and changed_files_path.exists():
        with open(changed_files_path, encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip().replace("\\", "/")
                if line:
                    changed_files.add(line)

    # Sidekick production files that were changed in this PR. These MUST appear
    # in the coverage data; a changed Sidekick module missing from coverage is a
    # failure, not a vacuous pass (issue #3139).
    changed_sidekick_files = {
        c
        for c in changed_files
        if "/sidekick/" in c and "/tests/" not in c and c.endswith(".py")
    }
    seen_changed_sidekick: set[str] = set()

    # Extract source directories from XML
    sources = []
    for s in root.findall(".//sources/source"):
        if s.text:
            sources.append(Path(s.text).resolve())

    failed_files = []
    sidekick_files_found = 0

    print("Sidekick per-file coverage report:")
    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        # Resolve to absolute path using source directories
        abs_path = None
        for src in sources:
            possible_path = (src / filename).resolve()
            if possible_path.exists():
                abs_path = possible_path
                break
        if abs_path is None:
            abs_path = Path(filename).resolve()

        norm_path = abs_path.as_posix()

        # Match files under sidekick/ excluding tests/
        if "/sidekick/" in norm_path and "/tests/" not in norm_path:
            # We enforce coverage gate on:
            # 1. Any of the 6 target files we added tests for.
            # 2. Any other file under sidekick/ that is modified/changed in this PR.
            is_target = any(norm_path.endswith(t) for t in target_files)
            is_changed = any(norm_path.endswith(c) for c in changed_files)

            # If changed-files parameter is not provided, we fall back to checking target files.
            should_check = is_target or (changed_files_path is not None and is_changed)

            if is_changed:
                for c in changed_sidekick_files:
                    if norm_path.endswith(c):
                        seen_changed_sidekick.add(c)

            if not should_check:
                continue

            lines = cls.findall("./lines/line")
            valid = len(lines)
            if valid == 0:
                continue
            covered = sum(1 for ln in lines if int(ln.attrib.get("hits", "0")) > 0)
            rate = covered / valid
            pct = round(rate * 100, 2)
            sidekick_files_found += 1
            print(f"- {abs_path.name}: {pct}% ({covered}/{valid} lines) [Enforced]")
            if rate < 0.50:
                failed_files.append((filename, pct))

    print(f"\nTotal sidekick files checked: {sidekick_files_found}")

    # A changed Sidekick production file that never appears in the coverage XML
    # cannot be enforced and must fail the gate (issue #3139).
    missing_changed = sorted(changed_sidekick_files - seen_changed_sidekick)
    if missing_changed:
        print(
            "\n[FAIL] Changed sidekick files missing from coverage data "
            "(no coverage class entry):",
            file=sys.stderr,
        )
        for missing in missing_changed:
            print(f"  - {missing}", file=sys.stderr)
        return 1

    if failed_files:
        print("\n[FAIL] Following sidekick files have < 50% coverage:", file=sys.stderr)
        for failed_file, p in failed_files:
            print(f"  - {failed_file}: {p}% (minimum required: 50.0%)", file=sys.stderr)
        return 1

    # A run that checked zero Sidekick files is a vacuous pass: the gate exists
    # precisely to enforce per-file Sidekick coverage. When Sidekick files were
    # changed in this PR (an enforced run) but none were counted, the coverage
    # data is stale or missing and the gate must fail rather than pass
    # vacuously (issue #3139). When no Sidekick files were changed and a
    # changed-files manifest was supplied, there is legitimately nothing to
    # enforce for this PR, so zero is acceptable.
    enforced_run = bool(changed_sidekick_files) or changed_files_path is None
    if sidekick_files_found == 0 and enforced_run:
        print(
            "\n[FAIL] Sidekick coverage gate checked zero files. The coverage "
            "XML contains no enforced Sidekick classes; coverage data is stale "
            "or missing.",
            file=sys.stderr,
        )
        return 1

    print("\n[PASS] All checked sidekick files meet the 50.0% coverage threshold.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check per-file coverage for sidekick modules."
    )
    parser.add_argument(
        "--coverage-file", default="coverage.xml", help="Path to coverage.xml file"
    )
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Path to text file containing list of changed files",
    )
    args = parser.parse_args()
    changed_path = Path(args.changed_files) if args.changed_files else None
    return check_sidekick_coverage(Path(args.coverage_file), changed_path)


if __name__ == "__main__":
    raise SystemExit(main())

"""Reconcile debt-marker counts and reject new unlinked markers.

The count mode intentionally mirrors the command from issue #2360, including
its path scope and the historical ``grep -v test_`` filter.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

MARKERS = tuple(marker.upper() for marker in ("todo", "fixme", "xxx", "hack", "kludge"))
MARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(marker) for marker in MARKERS) + r")\b"
)
ISSUE_LINK_RE = re.compile(
    r"(#\d+|GH-\d+|https://github\.com/[^/\s]+/[^/\s]+/issues/\d+)"
)
CANONICAL_PATHS = ("src/**", "scripts/**", "*.py", "*.cpp", "*.h", "*.ts", "*.js")
SELF_PATH = "scripts/check_todo_issue_links.py"


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def authoritative_count() -> int:
    result = run_git(
        [
            "grep",
            "-nE",
            "|".join(MARKERS),
            "--",
            *CANONICAL_PATHS,
        ]
    )
    if result.returncode not in (0, 1):
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    return sum(1 for line in result.stdout.splitlines() if "test_" not in line)


def staged_added_marker_lines() -> list[tuple[str, int | None, str]]:
    result = run_git(
        [
            "diff",
            "--cached",
            "--unified=0",
            "--diff-filter=ACMRT",
            "--",
            *CANONICAL_PATHS,
        ]
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    findings: list[tuple[str, int | None, str]] = []
    current_path: str | None = None
    new_line: int | None = None

    for raw_line in result.stdout.splitlines():
        if raw_line.startswith("+++ b/"):
            current_path = raw_line.removeprefix("+++ b/")
            new_line = None
            continue

        if raw_line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)?", raw_line)
            new_line = int(match.group(1)) if match else None
            continue

        if not raw_line.startswith("+") or raw_line.startswith("+++"):
            continue

        line = raw_line[1:]
        path = current_path or "<unknown>"
        line_number = new_line
        if new_line is not None:
            new_line += 1

        if path == SELF_PATH:
            continue
        if MARKER_RE.search(line) and not ISSUE_LINK_RE.search(line):
            findings.append((path, line_number, line.strip()))

    return findings


def check_staged() -> int:
    findings = staged_added_marker_lines()
    if not findings:
        print("No new unlinked debt markers found.")
        return 0

    print("New debt markers must include a GitHub issue link.")
    for path, line_number, text in findings:
        location = f"{path}:{line_number}" if line_number else path
        print(f"- {location}: {text}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="print the authoritative issue #2360 count and exit",
    )
    parser.add_argument(
        "--check-staged",
        action="store_true",
        help="reject staged added debt markers without issue links",
    )
    args = parser.parse_args()

    if args.count_only:
        print(authoritative_count())
        return 0
    if args.check_staged:
        return check_staged()

    count = authoritative_count()
    print(f"Authoritative debt-marker density count: {count}")
    return check_staged()


if __name__ == "__main__":
    raise SystemExit(main())

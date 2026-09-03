#!/usr/bin/env python3
"""Resolve every ``ADR-NNNN`` cited under ``src/`` and keep the ADR index fresh.

Governed by ADR-0049 (fleet ADR home). Two checks, both fail-closed:

1. Every four-digit ADR number cited anywhere under ``src/`` must resolve to a
   file ``docs/adr/ADR-NNNN-*.md``.
2. The Records table in ``docs/adr/README.md`` (between the ``adr-index``
   markers) must equal the table generated from the ADR files present, so the
   index can never link to a record that does not exist (the ADR-008 dead link
   this script was written to prevent).

Usage::

    python scripts/check_adr_references.py           # check both, exit 1 on drift
    python scripts/check_adr_references.py --write   # regenerate the index table
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START_MARKER = "<!-- adr-index:start -->"
END_MARKER = "<!-- adr-index:end -->"
CITATION_PATTERN = re.compile(r"\bADR-(\d{4})(?!\d)")
FILE_PATTERN = re.compile(r"^ADR-(\d{3,4})-(.+)\.md$")
STATUS_PATTERN = re.compile(r"^(?:-\s*)?\**Status\**\s*:\s*(.+?)\s*$", re.MULTILINE)
TITLE_PATTERN = re.compile(r"^#\s+ADR[- ]\d{3,4}\s*:?\s*(.+?)\s*$", re.MULTILINE)
MIRROR_PATTERN = re.compile(r"^> \*\*Mirrored ADR", re.MULTILINE)
EXCLUDED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip"}


def _tracked_src_files(root: Path) -> list[Path]:
    """Return tracked files under ``src/`` (falls back to a walk without git)."""
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--", "src"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout:
        names = [name for name in proc.stdout.decode("utf-8").split("\0") if name]
        return [root / name for name in names]
    src = root / "src"
    if not src.is_dir():
        return []
    return [path for path in src.rglob("*") if path.is_file()]


def cited_adr_numbers(root: Path) -> dict[str, list[str]]:
    """Map each cited ``NNNN`` to the ``src/`` files (posix, repo-relative) citing it."""
    cited: dict[str, set[str]] = {}
    for path in _tracked_src_files(root):
        if path.suffix.lower() in EXCLUDED_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for number in CITATION_PATTERN.findall(text):
            cited.setdefault(number, set()).add(path.relative_to(root).as_posix())
    return {number: sorted(files) for number, files in sorted(cited.items())}


def adr_files(adr_dir: Path) -> dict[str, Path]:
    """Map each ADR number (as written in the filename) to its file.

    Raises ``SystemExit`` on a duplicate number, which is exactly the ADR-007
    collision this gate exists to keep out.
    """
    records: dict[str, Path] = {}
    for path in sorted(adr_dir.glob("ADR-*.md")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        number = match.group(1)
        if number in records:
            raise SystemExit(
                f"duplicate ADR number {number}: {records[number].name} and {path.name}"
            )
        records[number] = path
    return records


def unresolved_citations(
    cited: dict[str, list[str]], records: dict[str, Path]
) -> dict[str, list[str]]:
    """Return the cited numbers that have no ``docs/adr/ADR-NNNN-*.md``."""
    return {number: files for number, files in cited.items() if number not in records}


def _record_row(number: str, path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    title_match = TITLE_PATTERN.search(text)
    title = title_match.group(1) if title_match else path.stem
    status_match = STATUS_PATTERN.search(text)
    status = status_match.group(1) if status_match else "Unknown"
    status = status.split("|")[0].strip().rstrip(".")
    origin = "mirror (UpstreamDrift)" if MIRROR_PATTERN.search(text) else "Tools"
    return f"| [ADR-{number}]({path.name}) | {status} | {origin} | {title} |"


def render_index(records: dict[str, Path]) -> str:
    """Render the Records table from the ADR files present."""
    lines = [
        "| ADR | Status | Origin | Title |",
        "| --- | ------ | ------ | ----- |",
    ]
    for number in sorted(records, key=lambda item: (len(item), item)):
        lines.append(_record_row(number, records[number]))
    return "\n".join(lines) + "\n"


def _split_index(index_path: Path) -> tuple[str, str, str]:
    text = index_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise SystemExit(
            f"{index_path.as_posix()} is missing the {START_MARKER} / "
            f"{END_MARKER} markers"
        )
    head = text[: start + len(START_MARKER)] + "\n"
    body = text[start + len(START_MARKER) + 1 : end]
    tail = text[end:]
    return head, body, tail


def index_is_fresh(index_path: Path, records: dict[str, Path]) -> bool:
    """Return whether the index table equals the generated table."""
    _head, body, _tail = _split_index(index_path)
    return body == render_index(records)


def write_index(index_path: Path, records: dict[str, Path]) -> None:
    """Rewrite the index table between the markers."""
    head, _body, tail = _split_index(index_path)
    index_path.write_text(
        head + render_index(records) + tail, encoding="utf-8", newline="\n"
    )


def _report(lines: Iterable[str]) -> None:
    for line in lines:
        sys.stderr.write(line + "\n")


def run(root: Path, write: bool) -> int:
    """Run both checks against ``root``; return a process exit code."""
    adr_dir = root / "docs" / "adr"
    index_path = adr_dir / "README.md"
    records = adr_files(adr_dir)
    cited = cited_adr_numbers(root)
    missing = unresolved_citations(cited, records)
    if missing:
        _report(
            [
                "Unresolved ADR citations under src/ (ADR-0049 requires a local "
                "docs/adr/ADR-NNNN-*.md for each):",
                *(
                    f"- ADR-{number}: cited by {len(files)} file(s), e.g. {files[0]}"
                    for number, files in missing.items()
                ),
            ]
        )
        return 1
    if write:
        write_index(index_path, records)
        sys.stdout.write(f"wrote {index_path.relative_to(root).as_posix()}\n")
        return 0
    if not index_is_fresh(index_path, records):
        _report(
            [
                f"{index_path.relative_to(root).as_posix()} Records table is "
                "stale; run: python scripts/check_adr_references.py --write"
            ]
        )
        return 1
    sys.stdout.write(
        f"ADR references resolved: {len(cited)} cited number(s), "
        f"{len(records)} record(s) indexed\n"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check ADR citations under src/ and the docs/adr index."
    )
    parser.add_argument(
        "--write", action="store_true", help="regenerate the docs/adr/README.md table"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run(args.root.resolve(), args.write)


if __name__ == "__main__":
    raise SystemExit(main())

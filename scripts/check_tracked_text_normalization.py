"""Reject tracked blobs whose index endings contradict an LF attribute."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404 - fixed Git executable and argv only.
from collections.abc import Iterable
from pathlib import Path

_LF_ATTRIBUTE = "attr/text eol=lf"
_ALLOWED_INDEX_STATES = frozenset({"i/lf", "i/none"})


def non_lf_index_paths(records: Iterable[str]) -> tuple[str, ...]:
    """Return LF-governed paths whose committed blobs are CRLF or mixed."""

    violations: list[str] = []
    for record in records:
        metadata, separator, path = record.partition("\t")
        if not separator or _LF_ATTRIBUTE not in metadata:
            continue
        index_state = metadata.split(maxsplit=1)[0]
        if index_state not in _ALLOWED_INDEX_STATES:
            violations.append(path)
    return tuple(violations)


def tracked_eol_records(root: Path) -> tuple[str, ...]:
    """Read NUL-delimited tracked-file EOL metadata from Git."""

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to verify tracked text normalization")
    result = subprocess.run(  # nosec B603 - fixed Git argv, no shell.
        [git, "ls-files", "--eol", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(os.fsdecode(record) for record in result.stdout.split(b"\0") if record)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    violations = non_lf_index_paths(tracked_eol_records(root))
    if not violations:
        print("Tracked LF-governed blobs are normalized.")
        return 0
    print("Tracked blobs contradict their eol=lf attribute:")
    for path in violations:
        print(f"- {path}")
    print("Normalize the listed paths and commit the resulting blob changes.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

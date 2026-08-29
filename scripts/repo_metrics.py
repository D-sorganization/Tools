#!/usr/bin/env python3
"""Pure-Python counters for coarse repository metrics.

These replace shell pipelines of the form ``grep -rnw WORD dir | wc -l`` and
``find dir -name PATTERN | wc -l``. Running those through a shell meant a
``shell=True`` subprocess per metric, and -- worse -- a failing ``grep`` or
``find`` returned its error text where a count was expected, so a broken
invocation produced a plausible but wrong number instead of raising.

Counting in Python removes both problems and makes the metrics portable to
hosts without POSIX ``grep``/``find``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["count_files", "count_matching_lines", "list_directory_entries"]


def count_matching_lines(roots: Iterable[Path], word: str) -> int:
    """Count lines under *roots* containing *word* as a whole word.

    Equivalent to ``grep -rnw WORD ROOTS | wc -l``: a line with several
    occurrences counts once, and ``TODONT`` does not match ``TODO``.

    Preconditions:
        *word* is non-empty.

    Postconditions:
        The result is non-negative. Missing roots contribute zero rather than
        raising, matching the tolerance the shell pipeline had in practice.
    """
    if not word:
        raise ValueError("word must be non-empty to count whole-word matches")

    pattern = re.compile(rf"\b{re.escape(word)}\b")
    total = 0
    for path in _iter_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Binary or unreadable content; `grep` skips these too.
            continue
        total += sum(1 for line in text.splitlines() if pattern.search(line))

    assert total >= 0
    return total


def count_files(roots: Iterable[Path], patterns: Sequence[str]) -> int:
    """Count files under *roots* whose name matches any of *patterns*.

    Equivalent to ``find ROOTS -name P1 -o -name P2 | wc -l``. A file matching
    more than one pattern is counted once, and directories never count.

    Preconditions:
        *patterns* contains at least one glob.

    Postconditions:
        The result is non-negative.
    """
    if not patterns:
        raise ValueError("count_files needs at least one pattern")

    matched = {
        path
        for path in _iter_files(roots)
        if any(path.match(pattern) for pattern in patterns)
    }

    assert len(matched) >= 0
    return len(matched)


def list_directory_entries(directory: Path) -> list[str]:
    """Return the sorted entry names in *directory*, or ``[]`` if absent.

    Equivalent to ``ls DIRECTORY`` for the purpose of counting its contents.
    """
    if not directory.is_dir():
        return []
    return sorted(entry.name for entry in directory.iterdir())


def _iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield every regular file under each existing root, deduplicated."""
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path

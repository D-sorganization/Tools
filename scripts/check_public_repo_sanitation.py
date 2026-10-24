"""Fail when tracked text contains developer-specific or fleet-local paths.

Scans git-tracked files with generic home-path regexes. Operators may add
repo-specific literals (internal hostnames, personal email) via an untracked
``.local/public_sanitation_denylist.txt`` (one case-insensitive substring per
line; never commit real values into this script).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

LOCAL_DENYLIST = Path(".local") / "public_sanitation_denylist.txt"

# Tracked paths scanned on every CI run (expand deliberately as backlog is scrubbed).
SCAN_PREFIXES: tuple[str, ...] = (
    "src/",
    "scripts/",
    "config/",
    "manuals/",
    "docs/specs/",
    "docs/development/impact-acoustics/",
    "docs/ops/",
    "docs/putting_launch_monitor/",
    ".github/agents/",
    ".github/scripts/",
)

# Prefixes excluded from scanning (historical snapshots, fleet-local artefacts).
EXCLUDE_PREFIXES: tuple[str, ...] = (
    "docs/assessments/",
    "docs/changelog_archive/",
    "docs/agent_handoff_archive/",
    ".gaai/",
)

# Files that intentionally mention patterns in negative tests or guard docs.
EXCLUDE_FILES: frozenset[str] = frozenset(
    {
        "tests/scripts/test_no_hardcoded_user_paths.py",
        "tests/ops/test_ci_standard_web_dependencies.py",
        "scripts/check_public_repo_sanitation.py",
    }
)

BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".whl",
        ".so",
        ".dll",
        ".dylib",
        ".pyd",
        ".mat",
        ".h5",
        ".slx",
        ".mdl",
        ".woff",
        ".woff2",
    }
)

BUILTIN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "windows-user-home",
        re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+"),
    ),
    (
        "posix-user-home",
        re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    ),
    (
        "wsl-user-home",
        re.compile(r"/mnt/[a-z]/Users/[A-Za-z0-9_.-]+"),
    ),
)


@dataclass(frozen=True)
class Violation:
    """A single sanitation finding."""

    path: str
    line: int
    pattern_id: str

    def key(self) -> str:
        return f"{self.path}:{self.line}:{self.pattern_id}"


def _tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    raw = result.stdout.decode("utf-8", errors="replace")
    return [p for p in raw.split("\0") if p]


def _should_scan(rel: str) -> bool:
    if rel in EXCLUDE_FILES:
        return False
    if not any(rel.startswith(prefix) for prefix in SCAN_PREFIXES):
        return False
    if any(rel.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return False
    suffix = Path(rel).suffix.lower()
    if suffix in BINARY_SUFFIXES:
        return False
    return True


def _load_local_terms(root: Path) -> tuple[str, ...]:
    path = root / LOCAL_DENYLIST
    if not path.is_file():
        return ()
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        terms.append(stripped)
    return tuple(terms)


def scan(root: Path) -> list[Violation]:
    """Return every built-in and local-denylist violation under *root*."""
    local_terms = _load_local_terms(root)
    local_patterns: list[tuple[str, re.Pattern[str]]] = [
        (f"local-denylist-{idx}", re.compile(re.escape(term), re.IGNORECASE))
        for idx, term in enumerate(local_terms, start=1)
    ]
    patterns: Sequence[tuple[str, re.Pattern[str]]] = (
        *BUILTIN_PATTERNS,
        *local_patterns,
    )

    violations: list[Violation] = []
    for rel in _tracked_files(root):
        if not _should_scan(rel):
            continue
        path = root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern_id, pattern in patterns:
                if pattern.search(line):
                    violations.append(
                        Violation(path=rel, line=line_no, pattern_id=pattern_id)
                    )
    violations.sort(key=lambda v: v.key())
    return violations


def _format_report(violations: Sequence[Violation]) -> str:
    lines = [
        "Public repo sanitation check failed. Replace developer-specific paths "
        "with placeholders or env vars.",
        f"Local denylist (optional, untracked): {LOCAL_DENYLIST}",
        "",
    ]
    for item in violations[:40]:
        lines.append(f"  {item.path}:{item.line} [{item.pattern_id}]")
    if len(violations) > 40:
        lines.append(f"  ... and {len(violations) - 40} more")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()

    violations = scan(root)
    if violations:
        print(_format_report(violations), file=sys.stderr)
        return 1
    print(f"Public repo sanitation check passed ({root}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

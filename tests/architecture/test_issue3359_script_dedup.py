"""Regression coverage for issue #3359 script-dedup cleanup."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = (
    REPO_ROOT / ".github",
    REPO_ROOT / "scripts",
    REPO_ROOT / "Makefile",
    REPO_ROOT / "docs",
)
REMOVED_SCRIPT_REFERENCES = (
    "migrate_print_to_logging",
    "generate_assessments",
    "generate_fresh_assessments",
)
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _iter_text_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        if path.suffix in TEXT_SUFFIXES:
            yield path
        return

    if not path.exists():
        return

    for child in path.rglob("*"):
        if child.is_file() and child.suffix in TEXT_SUFFIXES:
            yield child


def test_removed_legacy_quality_scripts_are_not_referenced() -> None:
    """Deleted duplicate scripts must not remain in docs or automation paths."""
    violations: list[str] = []
    for root in SEARCH_ROOTS:
        for file_path in _iter_text_files(root):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            for removed_reference in REMOVED_SCRIPT_REFERENCES:
                if removed_reference in text:
                    rel_path = file_path.relative_to(REPO_ROOT)
                    violations.append(f"{rel_path}: {removed_reference}")

    assert not violations, "\n".join(violations)

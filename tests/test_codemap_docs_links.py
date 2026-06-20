"""Regression tests for codemap documentation links."""

from __future__ import annotations

import re
from pathlib import Path


def test_codemap_full_design_link_resolves() -> None:
    """The codemap design cross-reference should point at a real repo file."""
    doc_path = Path("docs/codemap.md")
    text = doc_path.read_text(encoding="utf-8")
    match = re.search(r"Full design: \[[^\]]+\]\(([^)]+)\)", text)

    assert match is not None
    target_without_fragment = match.group(1).split("#", maxsplit=1)[0]
    target_path = (doc_path.parent / target_without_fragment).resolve()

    assert target_path.exists()

"""Architecture fitness test for issue #3322 — no frameless main windows.

Background:
    24 standalone tool windows applied ``Qt.WindowType.FramelessWindowHint``
    without providing any replacement chrome (no custom title bar, no drag
    handling, no min/max/close buttons, no size grip). The result was windows
    that could not be moved, resized, minimized, or closed by mouse — a severe
    accessibility and usability regression that arrived fleet-wide in a single
    commit.

    The fix (Option A) removed every ``FramelessWindowHint`` so the OS draws
    normal window chrome. This guard ensures the regression cannot be
    reintroduced wholesale by another automated commit.

    If a future window *deliberately* needs frameless chrome, it must provide
    a real custom title bar; at that point add the specific file to
    ``_ALLOWED_FRAMELESS`` below with a justification comment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Files that are sanctioned to use FramelessWindowHint because they implement
# their own full custom chrome (title bar + drag + buttons + size grip).
# Empty by design — Option A removed all uses. Add entries here (relative to
# SRC_ROOT, using forward slashes) only with a justification.
_ALLOWED_FRAMELESS: frozenset[str] = frozenset()


def _iter_python_sources() -> list[Path]:
    return [p for p in SRC_ROOT.rglob("*.py") if p.is_file()]


@pytest.mark.unit
def test_no_frameless_window_hint_in_src() -> None:
    """No source file may use FramelessWindowHint without sanctioned chrome."""
    offenders: list[str] = []
    for path in _iter_python_sources():
        rel = path.relative_to(SRC_ROOT).as_posix()
        if rel in _ALLOWED_FRAMELESS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "FramelessWindowHint" in text:
            offenders.append(rel)

    assert not offenders, (
        "FramelessWindowHint found without custom window chrome (issue #3322) "
        "in:\n  " + "\n  ".join(sorted(offenders)) + "\n\n"
        "These windows cannot be moved, resized, minimized, or closed by mouse. "
        "Remove the flag (let the OS draw chrome) or implement a real custom "
        "title bar and add the file to _ALLOWED_FRAMELESS with justification."
    )

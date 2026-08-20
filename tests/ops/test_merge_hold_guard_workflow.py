"""Contracts for non-held pull requests in the merge-hold guard."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "Merge-Hold-Guard.yml"
)


def test_absent_hold_signals_do_not_trip_runner_errexit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'has_label "$hold" && add_reason' not in text
    assert '[ "$DRAFT" = "true" ] && add_reason' not in text
    assert 'if has_label "$hold"; then' in text
    assert 'if [ "$DRAFT" = "true" ]; then' in text
    assert "LAST_DISARM=\"$(grep -v" not in text
    assert "LAST_DISARM=\"$(awk -F'\\t'" in text

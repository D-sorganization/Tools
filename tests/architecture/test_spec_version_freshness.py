"""SPEC.md version-header freshness contract.

`SPEC.md`'s §1 Identity table carries a `**Spec Version**` field meant to
answer "what version is this spec"; §12 Change Log is a newest-first table of
dated, versioned entries. Nothing enforced the two staying in agreement:
`spec-check.yml` only requires a PR to *touch* SPEC.md, and the version-bump
instruction is prose in that job's failure message. With several PRs
regenerating SPEC.md concurrently, that has repeatedly let the header drift
behind (or even duplicate against) the table it is supposed to summarize.

See GH issue #4827.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "SPEC.md"

_VERSION = r"[0-9]+(?:\.[0-9]+){2}"
_HEADER_ROW = re.compile(
    r"^\|\s*\*\*Spec Version\*\*\s*\|\s*(" + _VERSION + r")\s*\|\s*$",
    re.MULTILINE,
)
_CHANGE_LOG_HEADING = re.compile(r"^##\s*12\.\s*Change Log\s*$", re.MULTILINE)
_CHANGE_LOG_ROW = re.compile(
    r"^\|\s*\d{4}-\d{2}-\d{2}\s*\|\s*(" + _VERSION + r")\s*\|", re.MULTILINE
)


def _spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def _header_versions(text: str) -> list[str]:
    return _HEADER_ROW.findall(text)


def _newest_change_log_version(text: str) -> str:
    heading = _CHANGE_LOG_HEADING.search(text)
    assert heading is not None, "SPEC.md is missing a '## 12. Change Log' heading"
    row = _CHANGE_LOG_ROW.search(text, heading.end())
    assert row is not None, "§12 Change Log has no dated, versioned rows"
    return row.group(1)


def test_spec_has_exactly_one_version_header() -> None:
    """A merge that duplicates the `**Spec Version**` row must fail loudly.

    Two concurrently merged PRs each regenerating SPEC.md have produced a
    literal duplicate row before (one stale, one current) rather than a
    textual conflict — Git's line-based merge doesn't consider that a clash.
    """
    versions = _header_versions(_spec_text())
    assert len(versions) == 1, (
        "SPEC.md §1 Identity table must contain exactly one '**Spec "
        f"Version**' row, found {len(versions)}: {versions}. This shape "
        "results from two PRs each adding their own row across a concurrent "
        "merge — keep the row matching the newest §12 Change Log entry and "
        "delete the rest."
    )


def test_spec_version_header_matches_newest_change_log_entry() -> None:
    """The Identity header must equal §12's newest (topmost) dated row.

    The table is newest-first, so "newest" means the first matching row, not
    the numeric maximum — a row inserted in the wrong position should also
    fail this check rather than silently averaging out.
    """
    text = _spec_text()
    versions = _header_versions(text)
    assert len(versions) == 1, "run alongside test_spec_has_exactly_one_version_header"
    header_version = versions[0]
    newest_logged_version = _newest_change_log_version(text)
    assert header_version == newest_logged_version, (
        "SPEC.md '**Spec Version**' header "
        f"({header_version}) does not match the newest §12 Change Log row "
        f"({newest_logged_version}). Bump the header to match the row you "
        "just added, or move your row to the top if it is meant to be "
        "newest."
    )

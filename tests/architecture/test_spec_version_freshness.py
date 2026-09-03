"""SPEC.md change-log contract: rows are keyed by pull request, not by serial.

History of this file
--------------------
It was added by #4827 to close a real hole: `SPEC.md`'s §1 Identity table
carries a `**Spec Version**` field, §12 Change Log was a newest-first table of
dated *versioned* entries, and nothing enforced the two staying in agreement,
so the header repeatedly drifted behind (or literally duplicated against) the
table it summarised.

The equality it enforced turned out to be the more expensive defect.  A serial
spec version and a header field that must match it are both **global
counters**, so two concurrent pull requests necessarily read the same "next"
value, write it into the same row position, and edit the same header line.
Every second merge conflicted, and the only available resolution was "renumber
my row above theirs" — a conflict carrying no information about either change.
Measured on 2026-09-03: twelve mechanical re-merges in one day across
UpstreamDrift, Tools, Repository_Management and AffineDrift.

So the key became the pull request, which is unique by construction::

    | Date       | PR    | Changes          |
    | ---------- | ----- | ---------------- |
    | 2026-09-03 | #4951 | one-line summary |

What survives from the old contract, and what does not:

* **kept** — a substantive pull request still has to add a change-log row
  (`shared_scripts/fleet_hooks.py spec-changelog` and `spec-check.yml`);
* **kept** — rows are still format-validated, with `#<number>` where the
  serial used to be;
* **kept** — exactly one `**Spec Version**` row in §1.  That ratchet is
  orthogonal to keys: it catches a concurrent merge duplicating the row rather
  than conflicting on it, which line-based merging does not consider a clash;
* **changed** — "unique serial" became "unique PR key", exempting rows dated
  before the `PR_KEYED_SINCE` cutover because several historical rows
  genuinely share one governing issue (one issue, several pull requests);
* **removed** — the header no longer has to equal the newest row.  The field
  is release-derived now; it is only required to *be* a semantic version;
* **added** — a serial version sitting in the key column is an error, since
  that is how a pre-migration row or a stale agent habit shows up.

The row grammar itself lives once, in the fleet-shared
`shared_scripts/spec_changelog.py`, and is exercised here rather than
re-expressed as a second set of regexes that could disagree with the hook.

See GH issue #4827 and Repository_Management#1520 (program #1505).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "SPEC.md"
CHECKER = ROOT / "shared_scripts" / "spec_changelog.py"

_VERSION = r"[0-9]+(?:\.[0-9]+){2}"
_HEADER_ROW = re.compile(
    r"^\|\s*\*\*Spec Version\*\*\s*\|\s*(" + _VERSION + r")\s*\|\s*$",
    re.MULTILINE,
)


def _load_checker() -> ModuleType:
    """Import the vendored change-log module by path.

    It ships next to `fleet_hooks.py` in `shared_scripts/`, which is not an
    importable package, so the hook and this test both load it by file path.
    """
    spec = importlib.util.spec_from_file_location("tools_spec_changelog", CHECKER)
    assert spec is not None and spec.loader is not None, f"cannot load {CHECKER}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, and dataclasses
    # resolves field types by looking the defining module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load_checker()


@pytest.fixture(scope="module")
def spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def _table(header_and_rows: str) -> str:
    return (
        "## 12. Change Log\n\n"
        "| Date       | PR         | Changes    |\n"
        "| ---------- | ---------- | ---------- |\n"
        f"{header_and_rows}"
    )


def _failures(checker: ModuleType, text: str) -> list[str]:
    # The module is loaded by path, so it is untyped from mypy's point of view.
    failures: list[str] = checker.validate(checker.parse_changelog(text))
    return failures


# ---------------------------------------------------------------------------
# The shipped file
# ---------------------------------------------------------------------------


def test_the_shipped_spec_changelog_validates(
    checker: ModuleType, spec_text: str
) -> None:
    """The change log in this repository's own SPEC.md must be clean.

    This is the gate a pull request actually trips: the same call
    `shared_scripts/fleet_hooks.py spec-changelog` makes, run in-process so a
    failure names the offending rows instead of an exit code.
    """
    failures = _failures(checker, spec_text)
    assert not failures, "SPEC.md §12 change log is invalid:\n  - " + "\n  - ".join(
        failures
    )


def test_the_shipped_table_header_is_pr_keyed(
    checker: ModuleType, spec_text: str
) -> None:
    changelog = checker.parse_changelog(spec_text)
    assert changelog.header[:3] == checker.CANONICAL_HEADER, (
        f"§12 table header is {changelog.header[:3]}; expected "
        f"{checker.CANONICAL_HEADER} (Repository_Management#1520)"
    )
    assert changelog.rows, "§12 Change Log has no rows"


def test_the_shipped_spec_is_fully_migrated(
    checker: ModuleType, spec_text: str
) -> None:
    """Migration is idempotent, so a migrated file rewrites zero rows.

    A non-zero count means a serial-keyed row got back in — most likely a
    rebase that resurrected a pre-#1520 row.
    """
    _, rewritten = checker.migrate_text(spec_text)
    assert rewritten == 0, (
        f"{rewritten} SPEC.md row(s) still carry a serial spec version in the "
        "key column. Run: python shared_scripts/spec_changelog.py migrate "
        "--spec SPEC.md --write"
    )


# ---------------------------------------------------------------------------
# §1 Identity header: release-derived, no longer coupled to the newest row
# ---------------------------------------------------------------------------


def test_spec_has_exactly_one_version_header(spec_text: str) -> None:
    """A merge that duplicates the `**Spec Version**` row must fail loudly.

    Retained verbatim in intent from #4827: two concurrently merged PRs each
    regenerating SPEC.md have produced a literal duplicate row before (one
    stale, one current) rather than a textual conflict — Git's line-based
    merge doesn't consider that a clash.  This is orthogonal to row keys and
    survives #1520 unchanged.
    """
    versions = _HEADER_ROW.findall(spec_text)
    assert len(versions) == 1, (
        "SPEC.md §1 Identity table must contain exactly one '**Spec "
        f"Version**' row, found {len(versions)}: {versions}. This shape "
        "results from two PRs each adding their own row across a concurrent "
        "merge — keep one and delete the rest."
    )


def test_spec_version_header_is_a_semantic_version(spec_text: str) -> None:
    """The header must be well formed — and nothing more.

    This deliberately replaces #4827's
    `test_spec_version_header_matches_newest_change_log_entry`.  Requiring the
    header to equal the newest row is what made every pull request edit a
    global counter; the field is bumped at release time now, so a pull request
    that adds a row does not touch it and CI only checks its shape.
    """
    match = _HEADER_ROW.search(spec_text)
    assert match is not None, (
        "SPEC.md §1 Identity table has no well-formed '**Spec Version**' row "
        "(expected a three-part semantic version)"
    )


def test_a_pull_request_row_does_not_have_to_touch_the_header(
    checker: ModuleType, spec_text: str
) -> None:
    """Adding a row must not invalidate the header. This is the whole point.

    Prepending a row to the real file and re-validating proves the two are
    decoupled: under the old contract this exact edit failed the gate until
    the author also bumped §1.
    """
    changelog = checker.parse_changelog(spec_text)
    row = checker.Row(date="2099-01-01", key="#999999", summary="probe row")
    probed = checker.replace_rows(spec_text, changelog, [row, *changelog.rows])

    assert not _failures(checker, probed)
    assert len(_HEADER_ROW.findall(probed)) == 1
    assert _HEADER_ROW.search(probed).group(1) == _HEADER_ROW.search(  # type: ignore[union-attr]
        spec_text
    ).group(1), "adding a change-log row must not require a Spec Version bump"


# ---------------------------------------------------------------------------
# Row grammar
# ---------------------------------------------------------------------------


def test_a_pr_keyed_row_is_accepted(checker: ModuleType) -> None:
    text = _table("| 2026-09-03 | #4951 | key SPEC.md rows by PR |\n")
    assert not _failures(checker, text)


def test_a_serial_version_in_the_key_column_is_rejected(checker: ModuleType) -> None:
    """The replaced contract must now be an error, with the fix in the message.

    A serial key is the single most likely mistake for a while: it is what
    every pre-#1520 row looks like and what an agent working from a stale
    AGENTS.md will write.
    """
    text = _table("| 2026-09-03 | 1.18.124 | a serial where the key belongs |\n")
    failures = _failures(checker, text)
    assert failures, "a serial spec version in the key column must be rejected"
    joined = "\n".join(failures)
    assert "serial spec version" in joined
    assert "#<pr or issue number>" in joined, (
        "the failure must name the fix, not just the fault:\n" + joined
    )


def test_a_duplicate_key_after_the_cutover_is_rejected(checker: ModuleType) -> None:
    """Unique-PR-key replaces unique-serial: one row per pull request."""
    text = _table(
        "| 2026-09-04 | #4951 | second row for the same pull request |\n"
        "| 2026-09-03 | #4951 | key SPEC.md rows by PR |\n"
    )
    failures = _failures(checker, text)
    assert any("duplicate change-log key #4951" in failure for failure in failures), (
        "a reused PR key on or after the cutover must be rejected:\n"
        + "\n".join(failures)
    )


def test_a_duplicate_key_before_the_cutover_is_tolerated(checker: ModuleType) -> None:
    """History that legitimately shares a governing issue must still pass.

    Tools' migrated log recovers keys from prose, and one issue routinely
    spans several pull requests (`#4130` appears under two dates), so
    uniqueness cannot be enforced backwards over rows nobody can renumber.
    """
    assert checker.PR_KEYED_SINCE == "2026-09-03"
    text = _table(
        "| 2026-08-31 | #4827 | one issue, two pull requests |\n"
        "| 2026-08-30 | #4827 | the other one |\n"
    )
    assert not _failures(checker, text)


def test_a_malformed_key_is_rejected(checker: ModuleType) -> None:
    text = _table("| 2026-09-03 | 4951 | missing the hash |\n")
    failures = _failures(checker, text)
    assert any("#<number>" in failure for failure in failures), failures


def test_an_empty_summary_is_rejected(checker: ModuleType) -> None:
    text = _table("| 2026-09-03 | #4951 |  |\n")
    assert any("empty summary" in failure for failure in _failures(checker, text))


def test_the_legacy_no_key_marker_is_tolerated(checker: ModuleType) -> None:
    """Migrated rows referencing nothing carry `n/a` rather than being dropped."""
    text = _table(
        f"| 2026-06-01 | {checker.NO_KEY} | a historical row with no reference "
        "(spec 1.1.1) |\n"
    )
    assert not _failures(checker, text)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migration_preserves_every_row_summary(checker: ModuleType) -> None:
    """Campaign invariant 2: no content is lost, and the serial stays traceable."""
    before = _table(
        "| 2026-09-03 | 1.18.124 | feat(#4951): key rows by PR |\n"
        "| 2026-09-02 | 1.18.123 | a row that references nothing |\n"
    )
    original = checker.parse_changelog(before).rows
    after, rewritten = checker.migrate_text(before)
    migrated = checker.parse_changelog(after).rows

    assert rewritten == 2
    assert len(migrated) == len(original)
    for old, new in zip(original, migrated, strict=True):
        assert old.summary.rstrip() in new.summary, (old.summary, new.summary)
        assert f"(spec {old.key})" in new.summary, new.summary
    assert [row.key for row in migrated] == ["#4951", checker.NO_KEY]
    assert not _failures(checker, after)


def test_migration_is_idempotent(checker: ModuleType) -> None:
    before = _table("| 2026-09-03 | 1.18.124 | feat(#4951): key rows by PR |\n")
    once, first = checker.migrate_text(before)
    twice, second = checker.migrate_text(once)
    assert first == 1
    assert second == 0
    assert twice == once


def test_migration_preserves_the_shipped_row_count(
    checker: ModuleType, spec_text: str
) -> None:
    """Re-migrating the real file must not add, drop or reorder a row.

    The migration renders rows unpadded, so its diff looks enormous — it
    strips the old column alignment.  Row count is the invariant that tells a
    reviewer the byte-size drop is whitespace, not data.
    """
    rows_before = checker.parse_changelog(spec_text).rows
    after, _ = checker.migrate_text(spec_text)
    rows_after = checker.parse_changelog(after).rows
    assert len(rows_after) == len(rows_before)
    assert [row.identity for row in rows_after] == [row.identity for row in rows_before]

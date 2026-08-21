"""Tests for the rate-of-closure-explorer mirror freshness check (TDD).

Covers the pure drift-assessment core of ``scripts/check_mirror_freshness.py``
with injected fixture data -- no network, no git subprocesses:

- timestamp signal: fresh when the mirror synced at/after the last canonical
  web/ change, drifted when canonical moved afterwards (the live 17-day
  drift that motivated issue #4624)
- recorded-sha signal takes precedence over timestamps, matching on short
  or full SHAs
- deep tree comparison: blob-SHA equality, missing files, mirror-only
  scaffolding ignored
- DbC preconditions: naive datetimes and malformed SHAs are rejected
- report shape: JSON round-trip and exit-code mapping (0 fresh / 1 drifted)

Acceptance criteria (issue #4624): drift between canonical
``src/rate_of_closure/web`` and the public mirror is detected and surfaced
as a machine-readable report plus a failing exit code.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from check_mirror_freshness import (  # noqa: E402
    EXIT_DRIFTED,
    EXIT_FRESH,
    CanonicalState,
    ContractViolation,
    FreshnessReport,
    MirrorState,
    assess_freshness,
    compare_trees,
    exit_code_for,
    parse_recorded_canonical_commit,
)

CANONICAL_SHA = "9b24fc6d22df8104a515e93706d0068c2b440f06"
MIRROR_SHA = "84a589a964aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

AUG_04 = datetime(2026, 8, 4, 4, 18, 53, tzinfo=UTC)
AUG_21 = datetime(2026, 8, 21, 21, 44, 54, tzinfo=UTC)


def canonical(last_change: datetime = AUG_21, **kwargs: object) -> CanonicalState:
    return CanonicalState(commit=CANONICAL_SHA, last_change=last_change, **kwargs)


def mirror(last_sync: datetime = AUG_04, **kwargs: object) -> MirrorState:
    return MirrorState(last_sync_commit=MIRROR_SHA, last_sync_date=last_sync, **kwargs)


# ---------------------------------------------------------------------------
# Timestamp signal (primary)
# ---------------------------------------------------------------------------


class TestTimestampSignal:
    def test_canonical_changed_after_sync_is_drifted(self) -> None:
        report = assess_freshness(canonical(AUG_21), mirror(AUG_04))
        assert report.fresh is False
        assert report.signal == "timestamp"

    def test_sync_after_canonical_change_is_fresh(self) -> None:
        report = assess_freshness(canonical(AUG_04), mirror(AUG_21))
        assert report.fresh is True
        assert report.signal == "timestamp"

    def test_sync_exactly_at_canonical_change_is_fresh(self) -> None:
        report = assess_freshness(canonical(AUG_21), mirror(AUG_21))
        assert report.fresh is True

    def test_report_carries_both_states(self) -> None:
        report = assess_freshness(canonical(), mirror())
        assert report.canonical_commit == CANONICAL_SHA
        assert report.mirror_last_sync == MIRROR_SHA
        assert report.canonical_last_change == AUG_21.isoformat()
        assert report.mirror_last_sync_date == AUG_04.isoformat()


# ---------------------------------------------------------------------------
# Recorded-sha signal (takes precedence over timestamps)
# ---------------------------------------------------------------------------


class TestRecordedShaSignal:
    def test_matching_recorded_sha_is_fresh_despite_newer_dates(self) -> None:
        report = assess_freshness(
            canonical(AUG_21),
            mirror(AUG_04, recorded_canonical_commit=CANONICAL_SHA),
        )
        assert report.fresh is True
        assert report.signal == "recorded-sha"

    def test_short_recorded_sha_matches_by_prefix(self) -> None:
        report = assess_freshness(
            canonical(),
            mirror(recorded_canonical_commit=CANONICAL_SHA[:10]),
        )
        assert report.fresh is True

    def test_mismatched_recorded_sha_is_drifted(self) -> None:
        report = assess_freshness(
            canonical(AUG_04),
            mirror(AUG_21, recorded_canonical_commit="deadbeef00"),
        )
        assert report.fresh is False
        assert report.signal == "recorded-sha"

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("sync: from Tools commit 9b24fc6d22", "9b24fc6d22"),
            ("sync web\n\nCanonical-SHA: 9b24fc6d22df", "9b24fc6d22df"),
            ("sync from tools@9b24fc6", "9b24fc6"),
            ("sync: moving-head default view, true arrowheads (#2)", None),
            ("", None),
        ],
    )
    def test_parse_recorded_canonical_commit(
        self, message: str, expected: str | None
    ) -> None:
        assert parse_recorded_canonical_commit(message) == expected


# ---------------------------------------------------------------------------
# Deep tree comparison
# ---------------------------------------------------------------------------


class TestDeepComparison:
    def test_identical_trees_are_fresh(self) -> None:
        files = {"index.html": "aa" * 20, "src/App.tsx": "bb" * 20}
        report = assess_freshness(
            canonical(files=dict(files)), mirror(files=dict(files))
        )
        assert report.fresh is True
        assert report.deep is True
        assert report.signal == "tree"
        assert report.drifted_files == []

    def test_differing_blob_is_drifted(self) -> None:
        report = assess_freshness(
            canonical(files={"index.html": "aa" * 20}),
            mirror(files={"index.html": "cc" * 20}),
        )
        assert report.fresh is False
        assert report.drifted_files == ["index.html"]

    def test_missing_canonical_file_is_drifted(self) -> None:
        report = assess_freshness(
            canonical(files={"index.html": "aa" * 20, "new.ts": "bb" * 20}),
            mirror(files={"index.html": "aa" * 20}),
        )
        assert report.drifted_files == ["new.ts"]

    def test_mirror_only_scaffolding_is_ignored(self) -> None:
        report = assess_freshness(
            canonical(files={"index.html": "aa" * 20}),
            mirror(
                files={
                    "index.html": "aa" * 20,
                    "LICENSE": "11" * 20,
                    "scripts/sync-from-tools.ps1": "22" * 20,
                }
            ),
        )
        assert report.fresh is True

    def test_deep_fresh_iff_no_drifted_files_postcondition(self) -> None:
        drifted = compare_trees({"a": "aa" * 20}, {"a": "bb" * 20})
        assert drifted == ["a"]
        assert compare_trees({"a": "aa" * 20}, {"a": "aa" * 20}) == []

    def test_deep_takes_precedence_over_recorded_sha(self) -> None:
        report = assess_freshness(
            canonical(files={"a": "aa" * 20}),
            mirror(
                recorded_canonical_commit="deadbeef00",
                files={"a": "aa" * 20},
            ),
        )
        assert report.signal == "tree"
        assert report.fresh is True


# ---------------------------------------------------------------------------
# DbC preconditions
# ---------------------------------------------------------------------------


class TestContracts:
    def test_naive_canonical_datetime_rejected(self) -> None:
        with pytest.raises(ContractViolation, match="timezone-aware"):
            CanonicalState(commit=CANONICAL_SHA, last_change=datetime(2026, 8, 21))

    def test_naive_mirror_datetime_rejected(self) -> None:
        with pytest.raises(ContractViolation, match="timezone-aware"):
            MirrorState(
                last_sync_commit=MIRROR_SHA, last_sync_date=datetime(2026, 8, 4)
            )

    @pytest.mark.parametrize("bad", ["", "xyz", "12345", "G" * 40, "9B24FC6D22"])
    def test_malformed_sha_rejected(self, bad: str) -> None:
        with pytest.raises(ContractViolation, match="hex sha"):
            CanonicalState(commit=bad, last_change=AUG_21)

    def test_malformed_recorded_sha_rejected(self) -> None:
        with pytest.raises(ContractViolation, match="hex sha"):
            mirror(recorded_canonical_commit="not-a-sha")


# ---------------------------------------------------------------------------
# Report shape and exit codes
# ---------------------------------------------------------------------------


class TestReport:
    def test_json_round_trip(self) -> None:
        report = assess_freshness(canonical(), mirror())
        data = json.loads(report.to_json())
        assert set(data) >= {
            "fresh",
            "canonical_commit",
            "mirror_last_sync",
            "drifted_files",
        }
        assert data["fresh"] is False
        assert data["drifted_files"] == []

    def test_summary_names_verdict(self) -> None:
        assert "DRIFTED" in assess_freshness(canonical(), mirror()).summary()
        assert "FRESH" in assess_freshness(canonical(AUG_04), mirror(AUG_21)).summary()

    def test_exit_codes(self) -> None:
        drifted = assess_freshness(canonical(), mirror())
        fresh = assess_freshness(canonical(AUG_04), mirror(AUG_21))
        assert exit_code_for(drifted) == EXIT_DRIFTED == 1
        assert exit_code_for(fresh) == EXIT_FRESH == 0

    def test_report_is_frozen(self) -> None:
        report = assess_freshness(canonical(), mirror())
        assert isinstance(report, FreshnessReport)
        with pytest.raises(AttributeError):
            report.fresh = True  # type: ignore[misc]

"""Tests for scripts/measure_workflow_runner_share.py (TOOLS-CI-011 / #5695).

Aggregation logic is driven from fixture JSON so CI does not call GitHub.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import measure_workflow_runner_share as mwrs  # noqa: E402

_FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "measure_workflow_runner_share"
    / "sample_day_slices.json"
)


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


class TestUtcDayWindows:
    def test_fourteen_day_window_length(self) -> None:
        end = date(2026, 10, 20)
        days = list(mwrs.iter_utc_day_windows(end=end, window_days=14))
        assert len(days) == 14
        assert days[0].day == date(2026, 10, 7)
        assert days[-1].day == date(2026, 10, 20)

    def test_created_filter_is_inclusive_single_day(self) -> None:
        assert mwrs.created_filter_for_day(date(2026, 10, 20)) == (
            "2026-10-20..2026-10-20"
        )


class TestAggregateFromFixture:
    def test_counts_by_event_and_top_workflows(self) -> None:
        payload = _load_fixture()
        report = mwrs.aggregate_from_day_slices(payload["slices"])
        assert report.total_runs == 6
        assert report.counts_by_event["pull_request"] == 1
        assert report.counts_by_event["schedule"] == 2
        assert report.counts_by_event["merge_group"] == 1
        assert report.counts_by_event["repository_dispatch"] == 1
        assert report.counts_by_event["workflow_dispatch"] == 1

        top_names = [entry.name for entry in report.ranked_workflows(limit=10)]
        assert top_names[0] == "CI Standard"
        assert top_names.count("Full Suite Nightly") == 1

    def test_flags_total_count_cap_at_one_thousand(self) -> None:
        payload = _load_fixture()
        report = mwrs.aggregate_from_day_slices(payload["slices"])
        assert any("2026-10-20" in warning for warning in report.cap_warnings)

    def test_lane_shares(self) -> None:
        payload = _load_fixture()
        report = mwrs.aggregate_from_day_slices(payload["slices"])
        assert report.pr_merge_runs == 2
        assert report.scheduled_utility_runs == 4
        assert report.pr_merge_share == pytest.approx(2 / 6)
        assert report.scheduled_utility_share == pytest.approx(4 / 6)


class TestFetchDayRuns:
    def test_paginates_until_short_page(self) -> None:
        page1_runs = [
            {
                "id": index,
                "event": "schedule",
                "name": "A",
                "path": "a.yml",
            }
            for index in range(1, 101)
        ]
        page1 = {"total_count": 101, "workflow_runs": page1_runs}
        page2 = {
            "total_count": 101,
            "workflow_runs": [
                {"id": 101, "event": "push", "name": "C", "path": "c.yml"},
            ],
        }

        def _page_number(path: str) -> int:
            match = re.search(r"[?&]page=(\d+)", path)
            assert match is not None
            return int(match.group(1))

        def fake_get(path: str, token: str) -> dict[str, Any]:
            page_number = _page_number(path)
            if page_number == 1:
                return page1
            if page_number == 2:
                return page2
            raise AssertionError(f"unexpected path {path}")

        with patch.object(mwrs, "github_get", side_effect=fake_get):
            runs, cap_hit = mwrs.fetch_runs_for_utc_day(
                owner="D-sorganization",
                repo="Tools",
                token="tok",
                day=date(2026, 10, 20),
            )
        assert len(runs) == 101
        assert cap_hit is False

    def test_detects_cap_when_total_count_is_one_thousand(self) -> None:
        response = {
            "total_count": 1000,
            "workflow_runs": [
                {"id": 1, "event": "schedule", "name": "Nightly", "path": "n.yml"},
            ],
        }

        with patch.object(mwrs, "github_get", return_value=response):
            _, cap_hit = mwrs.fetch_runs_for_utc_day(
                owner="D-sorganization",
                repo="Tools",
                token="tok",
                day=date(2026, 10, 20),
            )
        assert cap_hit is True


class TestPreconditions:
    def test_window_days_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="window_days"):
            list(mwrs.iter_utc_day_windows(end=date(2026, 10, 20), window_days=0))

    def test_aggregate_rejects_missing_workflow_runs(self) -> None:
        with pytest.raises(ValueError, match="workflow_runs"):
            mwrs.aggregate_from_day_slices(
                [{"day": "2026-10-20", "responses": [{"total_count": 0}]}]
            )


class TestReportSerialization:
    def test_report_round_trips_json(self) -> None:
        payload = _load_fixture()
        report = mwrs.aggregate_from_day_slices(payload["slices"])
        restored = mwrs.RunnerLoadReport.from_dict(report.to_dict())
        assert restored.total_runs == report.total_runs
        assert restored.counts_by_event == report.counts_by_event

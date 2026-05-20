"""Tests for scripts/runner_capacity_check.py (TDD — issue #2946).

Validates:
- get_queue_depth() returns correct int from mocked API response
- calculate_needed_runners() applies Little's Law correctly
- check_and_alert() combines both and produces correct advisory strings
- DbC preconditions reject invalid inputs
- Edge cases: zero queue, exact-capacity match, large backlog

Mocking strategy: patch ``runner_capacity_check._github_get`` to avoid
real network calls. All tests are deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Ensure scripts/ is on the path
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import runner_capacity_check as rcc  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _queued_runs_response(total_count: int) -> dict[str, Any]:
    """Build a fake GitHub /orgs/{org}/actions/runs response."""
    return {
        "total_count": total_count,
        "workflow_runs": [{"id": i, "status": "queued"} for i in range(total_count)],
    }


def _runners_response(online: int, offline: int, busy: int) -> dict[str, Any]:
    """Build a fake GitHub /orgs/{org}/actions/runners response."""
    runners: list[dict[str, Any]] = []
    for i in range(online):
        runners.append(
            {
                "id": i,
                "name": f"runner-{i}",
                "status": "online",
                "busy": i < busy,
            }
        )
    for i in range(offline):
        runners.append(
            {
                "id": online + i,
                "name": f"runner-offline-{i}",
                "status": "offline",
                "busy": False,
            }
        )
    return {"total_count": online + offline, "runners": runners}


# ---------------------------------------------------------------------------
# Tests: get_queue_depth
# ---------------------------------------------------------------------------


class TestGetQueueDepth:
    """Unit tests for get_queue_depth()."""

    def test_returns_total_count_from_api(self) -> None:
        """Happy path: API returns total_count=7 → function returns 7."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(7)):
            result = rcc.get_queue_depth(token="tok", org="myorg")
        assert result == 7

    def test_returns_zero_when_no_queued_runs(self) -> None:
        """Queue depth is 0 when no runs are queued."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(0)):
            result = rcc.get_queue_depth(token="tok", org="myorg")
        assert result == 0

    def test_returns_large_count(self) -> None:
        """Handles large queue depths (> 100) correctly."""
        with patch.object(rcc, "_github_get", return_value={"total_count": 250}):
            result = rcc.get_queue_depth(token="tok", org="myorg")
        assert result == 250

    def test_missing_total_count_defaults_to_zero(self) -> None:
        """If total_count is absent from API response, return 0."""
        with patch.object(rcc, "_github_get", return_value={}):
            result = rcc.get_queue_depth(token="tok", org="myorg")
        assert result == 0

    def test_passes_correct_path_to_api(self) -> None:
        """Verifies the correct API path is used (queued status filter)."""
        with patch.object(
            rcc, "_github_get", return_value=_queued_runs_response(1)
        ) as mock_get:
            rcc.get_queue_depth(token="mytoken", org="testorg")
        call_args = mock_get.call_args
        path_arg = call_args[0][0]
        assert "testorg" in path_arg
        assert "status=queued" in path_arg

    def test_passes_token_to_api(self) -> None:
        """Token is forwarded to the API helper."""
        with patch.object(
            rcc, "_github_get", return_value=_queued_runs_response(0)
        ) as mock_get:
            rcc.get_queue_depth(token="secret-token", org="org")
        call_args = mock_get.call_args
        token_arg = call_args[0][1]
        assert token_arg == "secret-token"

    # DbC tests
    def test_rejects_empty_token(self) -> None:
        """Empty token raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.get_queue_depth(token="", org="org")

    def test_rejects_empty_org(self) -> None:
        """Empty org raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.get_queue_depth(token="tok", org="")


# ---------------------------------------------------------------------------
# Tests: calculate_needed_runners
# ---------------------------------------------------------------------------


class TestCalculateNeededRunners:
    """Unit tests for calculate_needed_runners()."""

    def test_zero_queue_returns_no_change(self) -> None:
        """Zero queue depth → delta is 0, recommended == current."""
        rec = rcc.calculate_needed_runners(
            queue_depth=0, current_runners=4, target_wait_sec=300
        )
        assert rec.delta == 0
        assert rec.recommended_runners == 4
        assert rec.queue_depth == 0

    def test_small_queue_within_capacity_no_delta(self) -> None:
        """Queue well within runner capacity → no additional runners needed."""
        # 4 runners, 120s avg, 300s target → can drain 10 jobs in 300s
        # queue_depth=5 → needed=ceil(5*120/300)=2 → max(2,4)=4 → delta=0
        rec = rcc.calculate_needed_runners(
            queue_depth=5,
            current_runners=4,
            target_wait_sec=300,
            avg_job_sec=120,
        )
        assert rec.delta == 0
        assert rec.recommended_runners == 4

    def test_large_queue_recommends_more_runners(self) -> None:
        """Large queue backlog → positive delta recommended."""
        # queue_depth=60, avg_job_sec=120, target=300 → needed=ceil(60*120/300)=24
        rec = rcc.calculate_needed_runners(
            queue_depth=60,
            current_runners=4,
            target_wait_sec=300,
            avg_job_sec=120,
        )
        assert rec.recommended_runners == 24
        assert rec.delta == 20

    def test_recommendation_never_below_current_runners(self) -> None:
        """Recommended count is always at least current_runners."""
        rec = rcc.calculate_needed_runners(
            queue_depth=1, current_runners=10, target_wait_sec=300, avg_job_sec=120
        )
        assert rec.recommended_runners >= 10

    def test_rationale_mentions_delta(self) -> None:
        """Rationale string mentions the runner count change."""
        rec = rcc.calculate_needed_runners(
            queue_depth=30,
            current_runners=2,
            target_wait_sec=300,
            avg_job_sec=120,
        )
        assert rec.delta > 0
        assert str(rec.delta) in rec.rationale or "Add" in rec.rationale

    def test_rationale_for_zero_queue(self) -> None:
        """Rationale for empty queue says no change needed."""
        rec = rcc.calculate_needed_runners(
            queue_depth=0, current_runners=4, target_wait_sec=300
        )
        assert "empty" in rec.rationale.lower() or "no" in rec.rationale.lower()

    def test_exact_capacity_boundary(self) -> None:
        """Queue at exact capacity boundary → delta is 0."""
        # needed=ceil(10*120/300)=4 → max(4,4)=4 → delta=0
        rec = rcc.calculate_needed_runners(
            queue_depth=10, current_runners=4, target_wait_sec=300, avg_job_sec=120
        )
        assert rec.delta == 0

    def test_one_over_capacity_boundary(self) -> None:
        """Queue one job over capacity → recommend one additional runner."""
        # needed=ceil(11*120/300)=5 → max(5,4)=5 → delta=1
        rec = rcc.calculate_needed_runners(
            queue_depth=11, current_runners=4, target_wait_sec=300, avg_job_sec=120
        )
        assert rec.delta == 1

    def test_target_wait_sec_field_populated(self) -> None:
        """CapacityRecommendation stores target_wait_sec."""
        rec = rcc.calculate_needed_runners(
            queue_depth=0, current_runners=2, target_wait_sec=600
        )
        assert rec.target_wait_sec == 600

    # DbC tests
    def test_rejects_negative_queue_depth(self) -> None:
        """Negative queue_depth raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.calculate_needed_runners(queue_depth=-1, current_runners=4)

    def test_rejects_zero_current_runners(self) -> None:
        """Zero current_runners raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.calculate_needed_runners(queue_depth=5, current_runners=0)

    def test_rejects_zero_target_wait_sec(self) -> None:
        """Zero target_wait_sec raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.calculate_needed_runners(
                queue_depth=5, current_runners=4, target_wait_sec=0
            )

    def test_rejects_zero_avg_job_sec(self) -> None:
        """Zero avg_job_sec raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.calculate_needed_runners(
                queue_depth=5, current_runners=4, avg_job_sec=0
            )


# ---------------------------------------------------------------------------
# Tests: check_and_alert
# ---------------------------------------------------------------------------


class TestCheckAndAlert:
    """Unit tests for check_and_alert()."""

    def test_ok_when_queue_empty(self) -> None:
        """Returns 'OK' when no jobs are queued."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(0)):
            result = rcc.check_and_alert(token="tok", current_runners=4)
        assert result == "OK"

    def test_warn_below_threshold(self) -> None:
        """Returns WARN advisory when queue is non-empty but below threshold."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(3)):
            result = rcc.check_and_alert(
                token="tok", current_runners=4, alert_threshold=10
            )
        assert result.startswith("WARN")

    def test_alert_at_threshold(self) -> None:
        """Returns ALERT when queue depth equals the alert threshold."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(10)):
            result = rcc.check_and_alert(
                token="tok", current_runners=4, alert_threshold=10
            )
        assert result.startswith("ALERT")

    def test_alert_above_threshold(self) -> None:
        """Returns ALERT when queue depth exceeds the alert threshold."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(50)):
            result = rcc.check_and_alert(
                token="tok", current_runners=4, alert_threshold=10
            )
        assert result.startswith("ALERT")

    def test_alert_contains_queue_depth(self) -> None:
        """Alert message includes the current queue depth."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(25)):
            result = rcc.check_and_alert(
                token="tok", current_runners=4, alert_threshold=10
            )
        assert "25" in result

    def test_warn_contains_queue_depth(self) -> None:
        """Warn message includes the current queue depth."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(5)):
            result = rcc.check_and_alert(
                token="tok", current_runners=4, alert_threshold=10
            )
        assert "5" in result

    def test_return_type_is_string(self) -> None:
        """Return value is always a str regardless of queue state."""
        with patch.object(rcc, "_github_get", return_value=_queued_runs_response(0)):
            result = rcc.check_and_alert(token="tok", current_runners=2)
        assert isinstance(result, str)

    # DbC tests
    def test_rejects_empty_token(self) -> None:
        """Empty token raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.check_and_alert(token="", current_runners=4)

    def test_rejects_zero_current_runners(self) -> None:
        """Zero current_runners raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.check_and_alert(token="tok", current_runners=0)

    def test_rejects_zero_alert_threshold(self) -> None:
        """Zero alert_threshold raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.check_and_alert(token="tok", current_runners=4, alert_threshold=0)


# ---------------------------------------------------------------------------
# Tests: RunnerStats dataclass
# ---------------------------------------------------------------------------


class TestRunnerStats:
    """Validate RunnerStats dataclass invariants."""

    def test_total_registered_is_online_plus_offline(self) -> None:
        """total_registered == online + offline."""
        stats = rcc.RunnerStats(online=3, offline=2, busy=1, idle=2)
        assert stats.total_registered == 5

    def test_idle_is_online_minus_busy(self) -> None:
        """Caller sets idle = online - busy; dataclass stores it as given."""
        stats = rcc.RunnerStats(online=5, offline=1, busy=3, idle=2)
        assert stats.idle == 2


# ---------------------------------------------------------------------------
# Tests: get_runner_stats
# ---------------------------------------------------------------------------


class TestGetRunnerStats:
    """Unit tests for get_runner_stats()."""

    def test_counts_online_offline_runners(self) -> None:
        """Correctly counts online and offline runners."""
        mock_resp = _runners_response(online=3, offline=1, busy=0)
        with patch.object(rcc, "_github_get", return_value=mock_resp):
            stats = rcc.get_runner_stats(token="tok", org="org")
        assert stats.online == 3
        assert stats.offline == 1

    def test_counts_busy_runners(self) -> None:
        """Correctly identifies busy vs idle runners."""
        mock_resp = _runners_response(online=4, offline=0, busy=2)
        with patch.object(rcc, "_github_get", return_value=mock_resp):
            stats = rcc.get_runner_stats(token="tok", org="org")
        assert stats.busy == 2
        assert stats.idle == 2

    def test_all_idle_when_busy_zero(self) -> None:
        """All online runners are idle when none are busy."""
        mock_resp = _runners_response(online=3, offline=0, busy=0)
        with patch.object(rcc, "_github_get", return_value=mock_resp):
            stats = rcc.get_runner_stats(token="tok", org="org")
        assert stats.idle == 3
        assert stats.busy == 0

    # DbC tests
    def test_rejects_empty_token(self) -> None:
        """Empty token raises AssertionError (DbC)."""
        with pytest.raises(AssertionError):
            rcc.get_runner_stats(token="", org="org")

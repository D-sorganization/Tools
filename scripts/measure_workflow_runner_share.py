#!/usr/bin/env python
"""Measure workflow runner load by trigger event (RM-CI-000 / TOOLS-CI-011).

Fetches GitHub Actions workflow runs for a repository using one UTC calendar
day per API query so each slice stays under the 1k ``total_count`` cap, then
aggregates run counts by ``event`` and workflow name.

Usage:
    python scripts/measure_workflow_runner_share.py \\
        --owner D-sorganization --repo Tools --token "$GITHUB_TOKEN"

See: docs/development/ci-tools-runner-load-measurement.md
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from runner_capacity_check import _github_get as github_get

logger = logging.getLogger(__name__)

METHODOLOGY_ID = "RM-CI-000"
GITHUB_TOTAL_COUNT_CAP = 1000
DEFAULT_WINDOW_DAYS = 14
PER_PAGE = 100
MAX_PAGES_PER_DAY = 12

PR_MERGE_EVENTS = frozenset({"pull_request", "merge_group"})
SCHEDULED_UTILITY_EVENTS = frozenset(
    {"schedule", "repository_dispatch", "workflow_dispatch"}
)


@dataclass(frozen=True)
class DayWindow:
    """Inclusive UTC calendar day used for one API slice."""

    day: date


@dataclass(frozen=True)
class WorkflowRunSummary:
    """Minimal workflow-run fields used for aggregation."""

    run_id: int
    event: str
    workflow_name: str
    workflow_path: str


@dataclass(frozen=True)
class WorkflowCount:
    """Run count for a single workflow name."""

    name: str
    count: int


@dataclass
class RunnerLoadReport:
    """Aggregated runner-load statistics for a measurement window."""

    methodology: str
    owner: str
    repo: str
    window_days: int
    window_end_utc: str
    total_runs: int
    counts_by_event: dict[str, int]
    top_workflows: list[WorkflowCount]
    pr_merge_runs: int
    scheduled_utility_runs: int
    cap_warnings: list[str] = field(default_factory=list)

    def ranked_workflows(self, limit: int = 10) -> list[WorkflowCount]:
        return self.top_workflows[:limit]

    @property
    def pr_merge_share(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.pr_merge_runs / self.total_runs

    @property
    def scheduled_utility_share(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.scheduled_utility_runs / self.total_runs

    def to_dict(self) -> dict[str, Any]:
        return {
            "methodology": self.methodology,
            "owner": self.owner,
            "repo": self.repo,
            "window_days": self.window_days,
            "window_end_utc": self.window_end_utc,
            "total_runs": self.total_runs,
            "counts_by_event": self.counts_by_event,
            "top_workflows": [
                {"name": entry.name, "count": entry.count}
                for entry in self.top_workflows
            ],
            "pr_merge_runs": self.pr_merge_runs,
            "scheduled_utility_runs": self.scheduled_utility_runs,
            "pr_merge_share": self.pr_merge_share,
            "scheduled_utility_share": self.scheduled_utility_share,
            "cap_warnings": self.cap_warnings,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunnerLoadReport:
        top = [
            WorkflowCount(name=item["name"], count=int(item["count"]))
            for item in payload.get("top_workflows", [])
        ]
        return cls(
            methodology=str(payload.get("methodology", METHODOLOGY_ID)),
            owner=str(payload["owner"]),
            repo=str(payload["repo"]),
            window_days=int(payload["window_days"]),
            window_end_utc=str(payload["window_end_utc"]),
            total_runs=int(payload["total_runs"]),
            counts_by_event={
                str(k): int(v) for k, v in payload["counts_by_event"].items()
            },
            top_workflows=top,
            pr_merge_runs=int(payload["pr_merge_runs"]),
            scheduled_utility_runs=int(payload["scheduled_utility_runs"]),
            cap_warnings=list(payload.get("cap_warnings", [])),
        )


def iter_utc_day_windows(end: date, window_days: int) -> Iterator[DayWindow]:
    """Yield ``window_days`` inclusive UTC days ending at ``end``.

    Precondition:
        ``window_days`` must be a positive integer.
    """
    if not isinstance(window_days, int) or window_days <= 0:
        raise ValueError(f"window_days must be a positive int, got {window_days!r}")
    start = end - timedelta(days=window_days - 1)
    day = start
    while day <= end:
        yield DayWindow(day=day)
        day += timedelta(days=1)


def created_filter_for_day(day: date) -> str:
    """Return the GitHub ``created`` query value for a single UTC day."""
    if not isinstance(day, date):
        raise TypeError(f"day must be a date, got {type(day)!r}")
    iso = day.isoformat()
    return f"{iso}..{iso}"


def _parse_run(raw: dict[str, Any]) -> WorkflowRunSummary:
    return WorkflowRunSummary(
        run_id=int(raw["id"]),
        event=str(raw.get("event") or "unknown"),
        workflow_name=str(raw.get("name") or "unknown"),
        workflow_path=str(raw.get("path") or ""),
    )


def fetch_runs_for_utc_day(
    owner: str,
    repo: str,
    token: str,
    day: date,
) -> tuple[list[WorkflowRunSummary], bool]:
    """Fetch all workflow runs created on ``day`` (UTC).

    Returns:
        Tuple of run summaries and whether the API ``total_count`` hit the 1k cap.
    """
    assert owner and repo and token
    created = urllib.parse.quote(created_filter_for_day(day), safe="")
    cap_hit = False
    runs: list[WorkflowRunSummary] = []
    page = 1
    while True:
        if page > MAX_PAGES_PER_DAY:
            cap_hit = True
            logger.warning(
                "Stopped pagination for %s at page %s (safety limit)",
                day.isoformat(),
                MAX_PAGES_PER_DAY,
            )
            break
        path = (
            f"/repos/{owner}/{repo}/actions/runs"
            f"?created={created}&per_page={PER_PAGE}&page={page}"
        )
        data = github_get(path, token)
        total_count = int(data.get("total_count", 0))
        if total_count >= GITHUB_TOTAL_COUNT_CAP:
            cap_hit = True
        batch = data.get("workflow_runs") or []
        if not isinstance(batch, list):
            raise ValueError("workflow_runs must be a list when present")
        runs.extend(_parse_run(item) for item in batch)
        if len(batch) < PER_PAGE:
            break
        page += 1
    return runs, cap_hit


def _aggregate_runs(
    runs: list[WorkflowRunSummary],
    cap_warnings: list[str],
    *,
    methodology: str,
    owner: str,
    repo: str,
    window_days: int,
    window_end_utc: str,
) -> RunnerLoadReport:
    event_counts: Counter[str] = Counter()
    workflow_counts: Counter[str] = Counter()
    for summary in runs:
        event_counts[summary.event] += 1
        workflow_counts[summary.workflow_name] += 1

    top = [
        WorkflowCount(name=name, count=count)
        for name, count in workflow_counts.most_common(10)
    ]
    pr_merge = sum(event_counts[event] for event in PR_MERGE_EVENTS)
    utility = sum(event_counts[event] for event in SCHEDULED_UTILITY_EVENTS)

    return RunnerLoadReport(
        methodology=methodology,
        owner=owner,
        repo=repo,
        window_days=window_days,
        window_end_utc=window_end_utc,
        total_runs=len(runs),
        counts_by_event=dict(event_counts),
        top_workflows=top,
        pr_merge_runs=pr_merge,
        scheduled_utility_runs=utility,
        cap_warnings=cap_warnings,
    )


def aggregate_from_day_slices(slices: list[dict[str, Any]]) -> RunnerLoadReport:
    """Aggregate run counts from fixture-shaped per-day API slices."""
    cap_warnings: list[str] = []
    runs: list[WorkflowRunSummary] = []

    for slice_entry in slices:
        day_label = str(slice_entry.get("day", "unknown"))
        for response in slice_entry.get("responses", []):
            total_count = int(response.get("total_count", 0))
            if total_count >= GITHUB_TOTAL_COUNT_CAP:
                cap_warnings.append(
                    f"{day_label}: total_count={total_count} hit GitHub cap "
                    f"({GITHUB_TOTAL_COUNT_CAP}); slice may be incomplete"
                )
            batch = response.get("workflow_runs")
            if batch is None:
                raise ValueError("each response must include workflow_runs")
            runs.extend(_parse_run(raw) for raw in batch)

    return _aggregate_runs(
        runs,
        cap_warnings,
        methodology=METHODOLOGY_ID,
        owner="",
        repo="",
        window_days=0,
        window_end_utc="",
    )


def measure_repo(
    owner: str,
    repo: str,
    token: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    end: date | None = None,
) -> RunnerLoadReport:
    """Live measurement using per-day GitHub API slices."""
    if not owner or not repo or not token:
        raise ValueError("owner, repo, and token are required")
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    end_day = end or datetime.now(UTC).date()
    cap_warnings: list[str] = []
    all_runs: list[WorkflowRunSummary] = []

    for window in iter_utc_day_windows(end=end_day, window_days=window_days):
        day_runs, cap_hit = fetch_runs_for_utc_day(
            owner=owner,
            repo=repo,
            token=token,
            day=window.day,
        )
        all_runs.extend(day_runs)
        if cap_hit:
            cap_warnings.append(
                f"{window.day.isoformat()}: total_count>={GITHUB_TOTAL_COUNT_CAP}; "
                "slice may be incomplete"
            )

    return _aggregate_runs(
        all_runs,
        cap_warnings,
        methodology=METHODOLOGY_ID,
        owner=owner,
        repo=repo,
        window_days=window_days,
        window_end_utc=end_day.isoformat(),
    )


def format_markdown_summary(report: RunnerLoadReport) -> str:
    """Render a human-readable summary for PRs and docs."""
    lines = [
        f"# Tools runner load ({report.window_days}d UTC ending {report.window_end_utc})",
        "",
        f"- Methodology: {report.methodology}",
        f"- Total workflow runs: {report.total_runs}",
        (f"- PR / merge_group: {report.pr_merge_runs} ({report.pr_merge_share:.1%})"),
        (
            f"- schedule / repository_dispatch / workflow_dispatch: "
            f"{report.scheduled_utility_runs} "
            f"({report.scheduled_utility_share:.1%})"
        ),
        "",
        "## Counts by event",
        "",
    ]
    for event, count in sorted(
        report.counts_by_event.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- `{event}`: {count}")
    lines.extend(["", "## Top workflows", ""])
    for entry in report.top_workflows:
        lines.append(f"- {entry.name}: {entry.count}")
    if report.cap_warnings:
        lines.extend(["", "## Cap warnings", ""])
        lines.extend(f"- {warning}" for warning in report.cap_warnings)
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure workflow runner load by trigger event.",
    )
    parser.add_argument("--owner", default="D-sorganization")
    parser.add_argument("--repo", default="Tools")
    parser.add_argument("--token", help="GitHub token with actions:read")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument(
        "--end-date",
        help="UTC end date (YYYY-MM-DD). Defaults to today (UTC).",
    )
    parser.add_argument(
        "--from-fixture",
        help="Aggregate from fixture JSON instead of calling GitHub.",
    )
    parser.add_argument("--output-json", help="Write JSON report to this path")
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print markdown summary to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    end = (
        date.fromisoformat(args.end_date) if args.end_date else datetime.now(UTC).date()
    )

    if args.from_fixture:
        payload = json.loads(Path(args.from_fixture).read_text(encoding="utf-8"))
        partial = aggregate_from_day_slices(payload["slices"])
        report = RunnerLoadReport(
            methodology=str(payload.get("methodology", METHODOLOGY_ID)),
            owner=str(payload.get("owner", args.owner)),
            repo=str(payload.get("repo", args.repo)),
            window_days=int(payload.get("window_days", args.window_days)),
            window_end_utc=end.isoformat(),
            total_runs=partial.total_runs,
            counts_by_event=partial.counts_by_event,
            top_workflows=partial.top_workflows,
            pr_merge_runs=partial.pr_merge_runs,
            scheduled_utility_runs=partial.scheduled_utility_runs,
            cap_warnings=partial.cap_warnings,
        )
    else:
        if not args.token:
            raise SystemExit("--token is required unless --from-fixture is set")
        report = measure_repo(
            owner=args.owner,
            repo=args.repo,
            token=args.token,
            window_days=args.window_days,
            end=end,
        )

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    if args.print_markdown or not args.output_json:
        print(format_markdown_summary(report))


if __name__ == "__main__":
    main()

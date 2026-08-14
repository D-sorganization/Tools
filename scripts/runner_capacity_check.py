#!/usr/bin/env python
"""Fleet runner capacity planning and alerting tool.

Queries the GitHub Actions API to measure current queue depth and recommend
runner count adjustments to keep queued workflow runs bounded.

Usage:
    python scripts/runner_capacity_check.py --token $GITHUB_TOKEN \
        --org D-sorganization --current-runners 4

DbC preconditions are enforced on all public functions via assert statements.

See also: docs/ops/runners.md
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_TARGET_WAIT_SEC = 300  # 5 minutes queue-wait budget
DEFAULT_SECONDS_PER_JOB = 120  # average job runtime assumption
DEFAULT_ALERT_THRESHOLD_JOBS = 10  # alert when queue exceeds this
ALERT_SUSTAINED_MINUTES = 5  # alert when threshold exceeded > N minutes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunnerStats:
    """Current state of the self-hosted runner pool.

    Attributes:
        online: Number of runners currently online and accepting jobs.
        offline: Number of runners registered but offline.
        busy: Number of runners currently executing a job.
        idle: Number of runners online and waiting for work.
    """

    online: int
    offline: int
    busy: int
    idle: int
    total_registered: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_registered = self.online + self.offline


@dataclass
class QueueReport:
    """Summary of current workflow queue state.

    Attributes:
        queued_runs: Total workflow runs in ``queued`` status across the org.
        in_progress_runs: Workflow runs currently executing.
        org: GitHub organisation queried.
    """

    queued_runs: int
    in_progress_runs: int
    org: str


@dataclass
class CapacityRecommendation:
    """Output of :func:`calculate_needed_runners`.

    Attributes:
        current_runners: Runners available at the time of measurement.
        recommended_runners: Recommended total runner count.
        delta: ``recommended_runners - current_runners`` (positive means add).
        queue_depth: The queue depth used for the calculation.
        target_wait_sec: The wait-time SLO the recommendation targets.
        rationale: Human-readable explanation.
    """

    current_runners: int
    recommended_runners: int
    delta: int
    queue_depth: int
    target_wait_sec: int
    rationale: str


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def _github_get(path: str, token: str) -> Any:
    """Make an authenticated GET request to the GitHub API.

    Precondition:
        ``path`` must be a non-empty string starting with ``/``.
        ``token`` must be a non-empty string.

    Args:
        path: API path relative to ``GITHUB_API_BASE`` (must start with ``/``).
        token: GitHub personal access token or Actions token.

    Returns:
        Parsed JSON response body.

    Raises:
        urllib.error.HTTPError: On non-2xx responses.
        ValueError: If the response body is not valid JSON.
    """
    assert isinstance(path, str) and path, "path must be a non-empty string"
    assert path.startswith("/"), f"path must start with '/': {path!r}"
    assert isinstance(token, str) and token, "token must be a non-empty string"

    url = GITHUB_API_BASE + path
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    logger.debug("GET %s", url)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_queue_depth(token: str, org: str = "D-sorganization") -> int:
    """Return the number of workflow runs currently in ``queued`` status.

    Queries all repositories in *org* and sums workflow runs with status
    ``queued``.  Uses the ``/orgs/{org}/actions/runs`` endpoint with a
    ``status=queued`` filter.

    Precondition:
        ``token`` must be a non-empty string.
        ``org`` must be a non-empty string.

    Args:
        token: GitHub personal access token with ``actions:read`` scope.
        org: GitHub organisation name.

    Returns:
        Total count of queued workflow runs across the organisation.

    Raises:
        urllib.error.HTTPError: On API error.
    """
    assert isinstance(token, str) and token, "token must be a non-empty string"
    assert isinstance(org, str) and org, "org must be a non-empty string"

    data = _github_get(
        f"/orgs/{org}/actions/runs?status=queued&per_page=100",
        token,
    )
    total = int(data.get("total_count", 0))
    logger.info("Queue depth for org %s: %d", org, total)
    return total


def get_runner_stats(token: str, org: str = "D-sorganization") -> RunnerStats:
    """Return statistics about self-hosted runners registered to *org*.

    Precondition:
        ``token`` must be a non-empty string.
        ``org`` must be a non-empty string.

    Args:
        token: GitHub personal access token with ``admin:org`` scope.
        org: GitHub organisation name.

    Returns:
        :class:`RunnerStats` populated from the GitHub API.

    Raises:
        urllib.error.HTTPError: On API error.
    """
    assert isinstance(token, str) and token, "token must be a non-empty string"
    assert isinstance(org, str) and org, "org must be a non-empty string"

    data = _github_get(f"/orgs/{org}/actions/runners?per_page=100", token)
    runners: list[dict[str, Any]] = data.get("runners", [])

    online = sum(1 for r in runners if r.get("status") == "online")
    offline = sum(1 for r in runners if r.get("status") == "offline")
    busy = sum(
        1 for r in runners if r.get("busy") is True and r.get("status") == "online"
    )
    idle = online - busy

    stats = RunnerStats(online=online, offline=offline, busy=busy, idle=idle)
    logger.info(
        "Runner stats — online=%d offline=%d busy=%d idle=%d",
        online,
        offline,
        busy,
        idle,
    )
    return stats


def calculate_needed_runners(
    queue_depth: int,
    current_runners: int,
    target_wait_sec: int = DEFAULT_TARGET_WAIT_SEC,
    avg_job_sec: int = DEFAULT_SECONDS_PER_JOB,
) -> CapacityRecommendation:
    """Calculate the recommended runner count to keep queue wait bounded.

    Uses a simple Little's-Law approximation:

        needed = ceil(queue_depth / (target_wait_sec / avg_job_sec))

    The recommendation is never less than ``current_runners`` (we do not
    recommend removing runners unless the queue is empty and ``current_runners``
    is significantly over-provisioned — that case is reported as ``delta == 0``).

    Precondition:
        ``queue_depth`` must be a non-negative integer.
        ``current_runners`` must be a positive integer.
        ``target_wait_sec`` must be a positive integer.
        ``avg_job_sec`` must be a positive integer.

    Args:
        queue_depth: Number of workflow runs currently queued.
        current_runners: Number of runners currently available.
        target_wait_sec: Maximum acceptable queue wait time in seconds.
        avg_job_sec: Average job runtime used for throughput estimation.

    Returns:
        :class:`CapacityRecommendation` with suggested runner count.
    """
    assert isinstance(queue_depth, int) and queue_depth >= 0, (
        f"queue_depth must be a non-negative int, got {queue_depth!r}"
    )
    assert isinstance(current_runners, int) and current_runners > 0, (
        f"current_runners must be a positive int, got {current_runners!r}"
    )
    assert isinstance(target_wait_sec, int) and target_wait_sec > 0, (
        f"target_wait_sec must be a positive int, got {target_wait_sec!r}"
    )
    assert isinstance(avg_job_sec, int) and avg_job_sec > 0, (
        f"avg_job_sec must be a positive int, got {avg_job_sec!r}"
    )

    if queue_depth == 0:
        return CapacityRecommendation(
            current_runners=current_runners,
            recommended_runners=current_runners,
            delta=0,
            queue_depth=0,
            target_wait_sec=target_wait_sec,
            rationale="Queue is empty — no capacity change needed.",
        )

    # Throughput = runners * (1 / avg_job_sec) jobs/sec
    # To drain queue_depth jobs in target_wait_sec seconds:
    #   needed_runners * (target_wait_sec / avg_job_sec) >= queue_depth
    #   needed_runners >= queue_depth / (target_wait_sec / avg_job_sec)
    #   needed_runners >= queue_depth * avg_job_sec / target_wait_sec
    import math

    needed = math.ceil(queue_depth * avg_job_sec / target_wait_sec)
    recommended = max(needed, current_runners)
    delta = recommended - current_runners

    if delta > 0:
        rationale = (
            f"Queue depth {queue_depth} exceeds capacity. "
            f"Add {delta} runner(s) to drain queue within "
            f"{target_wait_sec}s (assuming {avg_job_sec}s avg job time)."
        )
    else:
        rationale = (
            f"Queue depth {queue_depth} is within capacity "
            f"({current_runners} runners, {avg_job_sec}s avg job time, "
            f"{target_wait_sec}s target wait). No change needed."
        )

    logger.info(
        "Capacity recommendation: current=%d recommended=%d delta=%+d",
        current_runners,
        recommended,
        delta,
    )
    return CapacityRecommendation(
        current_runners=current_runners,
        recommended_runners=recommended,
        delta=delta,
        queue_depth=queue_depth,
        target_wait_sec=target_wait_sec,
        rationale=rationale,
    )


def check_and_alert(
    token: str,
    current_runners: int,
    org: str = "D-sorganization",
    alert_threshold: int = DEFAULT_ALERT_THRESHOLD_JOBS,
    target_wait_sec: int = DEFAULT_TARGET_WAIT_SEC,
) -> str:
    """Check queue depth and return an advisory message.

    Combines :func:`get_queue_depth` and :func:`calculate_needed_runners`
    into a single convenience call suitable for alerting pipelines.

    Precondition:
        ``token`` must be a non-empty string.
        ``current_runners`` must be a positive integer.
        ``org`` must be a non-empty string.
        ``alert_threshold`` must be a positive integer.
        ``target_wait_sec`` must be a positive integer.

    Args:
        token: GitHub personal access token with ``actions:read`` scope.
        current_runners: Number of self-hosted runners currently provisioned.
        org: GitHub organisation to query.
        alert_threshold: Queue depth above which an ALERT is emitted.
        target_wait_sec: Queue-wait SLO in seconds for capacity calculation.

    Returns:
        Advisory string: one of ``"OK"``, ``"WARN: ..."``, or ``"ALERT: ..."``.
    """
    assert isinstance(token, str) and token, "token must be a non-empty string"
    assert isinstance(current_runners, int) and current_runners > 0, (
        f"current_runners must be a positive int, got {current_runners!r}"
    )
    assert isinstance(org, str) and org, "org must be a non-empty string"
    assert isinstance(alert_threshold, int) and alert_threshold > 0, (
        f"alert_threshold must be a positive int, got {alert_threshold!r}"
    )
    assert isinstance(target_wait_sec, int) and target_wait_sec > 0, (
        f"target_wait_sec must be a positive int, got {target_wait_sec!r}"
    )

    queue_depth = get_queue_depth(token=token, org=org)
    rec = calculate_needed_runners(
        queue_depth=queue_depth,
        current_runners=current_runners,
        target_wait_sec=target_wait_sec,
    )

    if queue_depth == 0:
        return "OK"
    if queue_depth >= alert_threshold:
        return (
            f"ALERT: queue depth {queue_depth} exceeds threshold {alert_threshold}. "
            f"{rec.rationale}"
        )
    return f"WARN: queue depth {queue_depth}. {rec.rationale}"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> Any:
    import argparse

    p = argparse.ArgumentParser(
        description="Fleet runner capacity check and alerting tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--token", required=True, help="GitHub PAT with actions:read scope")
    p.add_argument("--org", default="D-sorganization", help="GitHub organisation")
    p.add_argument(
        "--current-runners",
        type=int,
        required=True,
        help="Number of self-hosted runners currently provisioned",
    )
    p.add_argument(
        "--alert-threshold",
        type=int,
        default=DEFAULT_ALERT_THRESHOLD_JOBS,
        help="Queue depth that triggers ALERT level",
    )
    p.add_argument(
        "--target-wait-sec",
        type=int,
        default=DEFAULT_TARGET_WAIT_SEC,
        help="Maximum acceptable queue wait in seconds",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit JSON output suitable for machine consumption",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the runner capacity checker."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    message = check_and_alert(
        token=args.token,
        current_runners=args.current_runners,
        org=args.org,
        alert_threshold=args.alert_threshold,
        target_wait_sec=args.target_wait_sec,
    )
    queue_depth = get_queue_depth(token=args.token, org=args.org)
    rec = calculate_needed_runners(
        queue_depth=queue_depth,
        current_runners=args.current_runners,
        target_wait_sec=args.target_wait_sec,
    )

    if args.json:
        output = {
            "status": message.split(":")[0],
            "message": message,
            "queue_depth": rec.queue_depth,
            "current_runners": rec.current_runners,
            "recommended_runners": rec.recommended_runners,
            "delta": rec.delta,
            "rationale": rec.rationale,
        }
        print(json.dumps(output, indent=2))
    else:
        print(message)
        if rec.delta > 0:
            print(f"  Recommendation: add {rec.delta} runner(s)")
            print(f"  Rationale: {rec.rationale}")


if __name__ == "__main__":
    main()

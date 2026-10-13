"""Alarm performance KPIs conforming to ISA-18.2 / EEMUA 191 benchmarks.

Architecture: Level 3 Plant Historian & SCADA (Tools #4046, H7 #5546).

Standards benchmarks (ISA-18.2 / EEMUA 191 §4.3):
- Average Alarm Rate: <= 1 alarm per 10 minutes is manageable under normal operation.
- Peak Alarm Rate: <= 10 alarms per 10 minutes during upset conditions.
- Alarm Flood: An episode where the alarm rate exceeds 10 alarms per 10 minutes.
- Chattering Alarms: An alarm that repeatedly triggers (> 3 transitions in 1 minute).
- Stale / Standing Alarms: An alarm continuously active for > 24 hours.
- Top 10 Contributors: The top 10 most frequent alarms, targeted at < 5% of all alarms.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

UTC = UTC

__all__ = [
    "ISA182_CHATTER_TRANSITIONS_THRESHOLD",
    "ISA182_CHATTER_WINDOW_SECONDS",
    "ISA182_FLOOD_THRESHOLD_PER_10MIN",
    "ISA182_MANAGEABLE_RATE_PER_10MIN",
    "ISA182_RATE_PERIOD_MINUTES",
    "ISA182_STALE_DURATION_SECONDS",
    "KNOWN_KPI_KEYS",
    "TOP_CONTRIBUTORS_LIMIT",
    "AlarmEventRecord",
    "AlarmKpiSummary",
    "compute_alarm_kpis",
]

#: Standard period for alarm rate metric: 10 minutes (ISA-18.2 §4.3, EEMUA 191).
ISA182_RATE_PERIOD_MINUTES: int = 10

#: Target manageable alarm rate: <= 1.0 alarm per 10 minutes (ISA-18.2 §4.3).
ISA182_MANAGEABLE_RATE_PER_10MIN: float = 1.0

#: Flood threshold: > 10 alarms per 10-minute window (ISA-18.2 §4.3, EEMUA 191 §4.3.4).
ISA182_FLOOD_THRESHOLD_PER_10MIN: int = 10

#: Chattering threshold: > 3 state transitions within 60 seconds (ISA-18.2 §4.3).
ISA182_CHATTER_TRANSITIONS_THRESHOLD: int = 3

#: Window duration for chattering detection: 60 seconds (ISA-18.2 §4.3).
ISA182_CHATTER_WINDOW_SECONDS: int = 60

#: Standing/stale duration threshold: 24h (86400s) (ISA-18.2 §4.3, EEMUA 191 §4.3.5).
ISA182_STALE_DURATION_SECONDS: int = 86400

#: Number of top contributors to report: 10 (ISA-18.2 §4.3).
TOP_CONTRIBUTORS_LIMIT: int = 10

#: Documented KPI keys expected by reporting dashboards.
KNOWN_KPI_KEYS: frozenset[str] = frozenset(
    {
        "alarm_rate_10min",
        "flood_episodes",
        "chattering_alarms",
        "stale_alarms",
        "top_contributors",
        "priority_distribution",
    }
)


@dataclass(frozen=True)
class AlarmEventRecord:
    """Normalized representation of a single alarm transition event."""

    timestamp: datetime
    tag: str
    state: str
    severity: int = 1


@dataclass(frozen=True)
class AlarmKpiSummary:
    """Aggregated ISA-18.2 / EEMUA 191 alarm performance indicators."""

    total_alarms: int
    mean_alarm_rate_10min: float
    peak_alarm_rate_10min: int
    flood_episodes: int
    chattering_alarms_count: int
    stale_alarms_count: int
    top_10_contributors: list[dict[str, Any]] = field(default_factory=list)
    priority_distribution: dict[int, int] = field(default_factory=dict)


def _to_utc(ts: Any) -> datetime:
    """Normalize arbitrary timestamp to UTC datetime."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            pass
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=UTC)
    return datetime.now(UTC)


def _tag_from_event_description(desc: Any) -> str | None:
    """Parse a tag name from free-text event descriptions when explicit tags are absent."""
    if not isinstance(desc, str) or "tag " not in desc.lower():
        return None
    parts = desc.split()
    tag_word_indexes = [i for i, part in enumerate(parts) if part.lower() == "tag"]
    if not tag_word_indexes:
        return None
    tag_index = tag_word_indexes[0]
    if tag_index + 1 >= len(parts):
        return None
    return parts[tag_index + 1]


def _tag_from_object(raw: Any) -> str:
    """Resolve tag from duck-typed alarm/event records (e.g. EventLog)."""
    tag_raw = getattr(raw, "tag", getattr(raw, "tag_name", getattr(raw, "tag_id", None)))
    if tag_raw is not None:
        return str(tag_raw)
    from_description = _tag_from_event_description(getattr(raw, "description", ""))
    if from_description is not None:
        return from_description
    return "UNKNOWN"


def _extract_record(raw: Any) -> AlarmEventRecord:
    """Extract an AlarmEventRecord from a dict, ORM model, or record object."""
    if isinstance(raw, AlarmEventRecord):
        return raw

    if isinstance(raw, dict):
        ts = _to_utc(raw.get("timestamp") or raw.get("ts"))
        tag = str(
            raw.get("tag")
            or raw.get("tag_name")
            or raw.get("tag_id")
            or raw.get("description", "UNKNOWN")
        )
        state = str(raw.get("state") or raw.get("event_type") or "Alarm")
        severity = int(raw.get("severity") or 1)
        return AlarmEventRecord(timestamp=ts, tag=tag, state=state, severity=severity)

    # Duck-typed object (e.g. EventLog)
    ts = _to_utc(getattr(raw, "timestamp", getattr(raw, "ts", None)))
    tag = _tag_from_object(raw)
    state = getattr(raw, "state", getattr(raw, "event_type", "Alarm"))
    severity = getattr(raw, "severity", 1)
    return AlarmEventRecord(
        timestamp=ts,
        tag=tag,
        state=str(state),
        severity=int(severity),
    )


def compute_alarm_kpis(events: Iterable[Any]) -> AlarmKpiSummary:
    """Compute ISA-18.2 / EEMUA 191 alarm KPIs from an event stream.

    Args:
        events: Sequence of alarm events (AlarmEventRecord, dict, or EventLog).

    Returns:
        AlarmKpiSummary populated with exact KPI values.
    """
    records: list[AlarmEventRecord] = [_extract_record(e) for e in events]
    if not records:
        return AlarmKpiSummary(
            total_alarms=0,
            mean_alarm_rate_10min=0.0,
            peak_alarm_rate_10min=0,
            flood_episodes=0,
            chattering_alarms_count=0,
            stale_alarms_count=0,
            top_10_contributors=[],
            priority_distribution={1: 0, 2: 0},
        )

    # Sort events in chronological order
    records.sort(key=lambda r: r.timestamp)

    # Filter alarm activation events (non-Normal state)
    activations = [r for r in records if r.state.lower() != "normal"]
    total_alarms = len(activations)

    # 1. Alarm Rate per 10 minutes & Flood Episodes
    if not activations:
        mean_rate = 0.0
        peak_rate = 0
        flood_episodes = 0
    else:
        min_ts = activations[0].timestamp
        max_ts = activations[-1].timestamp
        span_s = max(0.0, (max_ts - min_ts).total_seconds())

        # Bucket width in seconds (10 min = 600s)
        bucket_s = ISA182_RATE_PERIOD_MINUTES * 60
        num_buckets = max(1, int(math.ceil(span_s / bucket_s))) if span_s > 0 else 1

        bucket_counts: Counter[int] = Counter()
        for r in activations:
            offset = max(0.0, (r.timestamp - min_ts).total_seconds())
            b_idx = int(offset // bucket_s)
            bucket_counts[b_idx] += 1

        peak_rate = max(bucket_counts.values()) if bucket_counts else 0
        mean_rate = round(total_alarms / num_buckets, 2)

        # Count flood episodes: contiguous runs of 10-minute buckets > 10 alarms
        flood_episodes = 0
        in_flood = False
        max_bucket_idx = max(bucket_counts.keys()) if bucket_counts else 0
        for b_idx in range(max_bucket_idx + 1):
            count = bucket_counts.get(b_idx, 0)
            if count > ISA182_FLOOD_THRESHOLD_PER_10MIN:
                if not in_flood:
                    flood_episodes += 1
                    in_flood = True
            else:
                in_flood = False

    # 2. Chattering Alarms (> 3 transitions in 60 seconds per tag)
    tag_events: dict[str, list[datetime]] = {}
    for r in records:
        tag_events.setdefault(r.tag, []).append(r.timestamp)

    chattering_tags: set[str] = set()
    for tag, timestamps in tag_events.items():
        if len(timestamps) <= ISA182_CHATTER_TRANSITIONS_THRESHOLD:
            continue
        left = 0
        for right in range(len(timestamps)):
            while (
                timestamps[right] - timestamps[left]
            ).total_seconds() > ISA182_CHATTER_WINDOW_SECONDS:
                left += 1
            if (right - left + 1) > ISA182_CHATTER_TRANSITIONS_THRESHOLD:
                chattering_tags.add(tag)
                break

    # 3. Stale / Standing Alarms (> 24 hours continuously active)
    stale_tags: set[str] = set()
    tag_state_history: dict[str, list[AlarmEventRecord]] = {}
    for r in records:
        tag_state_history.setdefault(r.tag, []).append(r)

    dataset_end = records[-1].timestamp
    for tag, hist in tag_state_history.items():
        active_start: datetime | None = None
        for r in hist:
            is_active = r.state.lower() != "normal"
            if is_active and active_start is None:
                active_start = r.timestamp
            elif not is_active and active_start is not None:
                duration_s = (r.timestamp - active_start).total_seconds()
                if duration_s >= ISA182_STALE_DURATION_SECONDS:
                    stale_tags.add(tag)
                active_start = None

        if active_start is not None:
            duration_s = (dataset_end - active_start).total_seconds()
            if duration_s >= ISA182_STALE_DURATION_SECONDS:
                stale_tags.add(tag)

    # 4. Top 10 Contributors
    tag_counter = Counter(r.tag for r in activations)
    top_10: list[dict[str, Any]] = []
    for tag, count in tag_counter.most_common(TOP_CONTRIBUTORS_LIMIT):
        pct = round((count / total_alarms) * 100, 2) if total_alarms > 0 else 0.0
        top_10.append({"tag": tag, "count": count, "percentage": pct})

    # 5. Priority Distribution
    priority_counter = Counter(r.severity for r in activations)
    priority_dist: dict[int, int] = {
        1: priority_counter.get(1, 0),
        2: priority_counter.get(2, 0),
    }
    for sev, cnt in priority_counter.items():
        priority_dist[sev] = cnt

    return AlarmKpiSummary(
        total_alarms=total_alarms,
        mean_alarm_rate_10min=mean_rate,
        peak_alarm_rate_10min=peak_rate,
        flood_episodes=flood_episodes,
        chattering_alarms_count=len(chattering_tags),
        stale_alarms_count=len(stale_tags),
        top_10_contributors=top_10,
        priority_distribution=priority_dist,
    )

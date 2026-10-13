"""Unit and contract tests for ISA-18.2 / EEMUA 191 alarm performance KPIs.

H7 / #5546:
1. Synthetic event streams calculate exact alarm rate, flood episodes,
   chattering alarms, stale/standing counts, and top-10 contributors.
2. Empty input returns zeros, not errors.
3. Named constants cite ISA-18.2 / EEMUA 191 benchmarks.
4. Grafana alarm-performance dashboard JSON parses and every panel maps
   to a documented KPI name.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alarm_kpis import (  # noqa: E402
    ISA182_CHATTER_TRANSITIONS_THRESHOLD,
    ISA182_CHATTER_WINDOW_SECONDS,
    ISA182_FLOOD_THRESHOLD_PER_10MIN,
    ISA182_RATE_PERIOD_MINUTES,
    ISA182_STALE_DURATION_SECONDS,
    KNOWN_KPI_KEYS,
    AlarmEventRecord,
    compute_alarm_kpis,
)

# Repository root: src/p1am_control_system/backend/tests/ -> repo root
ROOT = Path(__file__).resolve().parents[4]
DASHBOARD_FILE = (
    ROOT / "deploy" / "historian" / "grafana" / "dashboards" / "alarm-performance.json"
)

UTC = UTC


def test_isa182_constants_match_standards() -> None:
    """Named constants must cite and reflect ISA-18.2 / EEMUA 191 thresholds."""
    assert ISA182_RATE_PERIOD_MINUTES == 10
    assert ISA182_FLOOD_THRESHOLD_PER_10MIN == 10
    assert ISA182_CHATTER_TRANSITIONS_THRESHOLD == 3
    assert ISA182_CHATTER_WINDOW_SECONDS == 60
    assert ISA182_STALE_DURATION_SECONDS == 86400


def test_empty_events_returns_zeros() -> None:
    """Empty event stream must return zeros and empty structures without error."""
    summary = compute_alarm_kpis([])

    assert summary.total_alarms == 0
    assert summary.mean_alarm_rate_10min == 0.0
    assert summary.peak_alarm_rate_10min == 0
    assert summary.flood_episodes == 0
    assert summary.chattering_alarms_count == 0
    assert summary.stale_alarms_count == 0
    assert summary.top_10_contributors == []
    assert summary.priority_distribution == {1: 0, 2: 0}


def test_alarm_rate_calculation() -> None:
    """Alarm rate per 10 minutes calculates average and peak across intervals."""
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    # 6 alarms across 3 10-minute buckets:
    # Bucket 0 [12:00 - 12:10): 3 alarms
    # Bucket 1 [12:10 - 12:20): 1 alarm
    # Bucket 2 [12:20 - 12:30): 2 alarms
    events = [
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=1), tag="TI_101", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=3), tag="TI_102", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=8), tag="PI_101", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=15), tag="TI_101", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=22), tag="TI_101", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=29), tag="FI_101", state="High"
        ),
    ]

    summary = compute_alarm_kpis(events)

    assert summary.total_alarms == 6
    assert summary.peak_alarm_rate_10min == 3
    # 6 alarms / 3 buckets = 2.0 alarms / 10 min
    assert summary.mean_alarm_rate_10min == 2.0


def test_flood_episode_detection() -> None:
    """Alarm flood is detected when rate exceeds 10 alarms per 10 minutes."""
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    # 11 alarms in Bucket 0 -> Flood episode 1
    flood_events = [
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=i % 9),
            tag=f"TAG_{i}",
            state="High",
        )
        for i in range(11)
    ]
    # Quiet period
    quiet_events = [
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=30),
            tag="TAG_QUIET",
            state="High",
        )
    ]
    # 12 alarms in Bucket 6 (60-70 min) -> Flood episode 2
    flood_events_2 = [
        AlarmEventRecord(
            timestamp=base + timedelta(minutes=60 + (i % 8)),
            tag=f"TAG_B_{i}",
            state="High",
        )
        for i in range(12)
    ]

    all_events = flood_events + quiet_events + flood_events_2
    summary = compute_alarm_kpis(all_events)

    assert summary.flood_episodes == 2


def test_chattering_alarm_detection() -> None:
    """Alarms transitioning >3 times in 1 minute are flagged as chattering."""
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    # TAG_CHATTER transitions 4 times in 40 seconds (> 3 transitions in 60s)
    chattering_events = [
        AlarmEventRecord(
            timestamp=base + timedelta(seconds=5), tag="TAG_CHATTER", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(seconds=15), tag="TAG_CHATTER", state="Normal"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(seconds=25), tag="TAG_CHATTER", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(seconds=35), tag="TAG_CHATTER", state="Normal"
        ),
    ]
    # TAG_STABLE transitions only twice in 60s
    stable_events = [
        AlarmEventRecord(
            timestamp=base + timedelta(seconds=10), tag="TAG_STABLE", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(seconds=40), tag="TAG_STABLE", state="Normal"
        ),
    ]

    summary = compute_alarm_kpis(chattering_events + stable_events)

    assert summary.chattering_alarms_count == 1


def test_stale_alarm_detection() -> None:
    """Alarms continuously active for >24 hours are flagged as stale/standing."""
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)

    events = [
        # TAG_STALE activates at t=0 and clears 26 hours later
        AlarmEventRecord(timestamp=base, tag="TAG_STALE", state="High"),
        AlarmEventRecord(
            timestamp=base + timedelta(hours=26), tag="TAG_STALE", state="Normal"
        ),
        # TAG_BRIEF activates at t=1h and clears at t=3h
        AlarmEventRecord(
            timestamp=base + timedelta(hours=1), tag="TAG_BRIEF", state="High"
        ),
        AlarmEventRecord(
            timestamp=base + timedelta(hours=3), tag="TAG_BRIEF", state="Normal"
        ),
    ]

    summary = compute_alarm_kpis(events)

    assert summary.stale_alarms_count == 1


def test_top_10_contributors() -> None:
    """Top 10 contributors ranks tags by alarm count with percentage of total."""
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    events: list[AlarmEventRecord] = []

    # Generate 12 tags with counts 1..12
    for tag_idx in range(1, 13):
        tag_name = f"TAG_{tag_idx:02d}"
        for c in range(tag_idx):
            events.append(
                AlarmEventRecord(
                    timestamp=base + timedelta(minutes=c),
                    tag=tag_name,
                    state="High",
                )
            )

    summary = compute_alarm_kpis(events)

    assert len(summary.top_10_contributors) == 10
    # TAG_12 had 12 alarms, should be #1
    assert summary.top_10_contributors[0]["tag"] == "TAG_12"
    assert summary.top_10_contributors[0]["count"] == 12
    total = sum(range(1, 13))  # 78
    expected_pct = round((12 / total) * 100, 2)
    assert summary.top_10_contributors[0]["percentage"] == expected_pct


def test_priority_distribution() -> None:
    """Alarm priority distribution counts events by severity tier."""
    base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
    events = [
        AlarmEventRecord(timestamp=base, tag="TAG_1", state="Low", severity=1),
        AlarmEventRecord(timestamp=base, tag="TAG_2", state="High", severity=1),
        AlarmEventRecord(timestamp=base, tag="TAG_3", state="LoLo", severity=2),
        AlarmEventRecord(timestamp=base, tag="TAG_4", state="HiHi", severity=2),
        AlarmEventRecord(timestamp=base, tag="TAG_5", state="BadQuality", severity=2),
    ]

    summary = compute_alarm_kpis(events)

    assert summary.priority_distribution[1] == 2
    assert summary.priority_distribution[2] == 3


def test_dashboard_json_exists_and_maps_panels_to_kpi_names() -> None:
    """Grafana dashboard JSON must exist and map panels to KPI names."""
    assert DASHBOARD_FILE.is_file(), f"Dashboard JSON missing: {DASHBOARD_FILE}"

    with open(DASHBOARD_FILE, encoding="utf-8") as f:
        dashboard = json.load(f)

    assert dashboard.get("uid") == "alarm-performance"
    assert "Alarm Performance" in dashboard.get("title", "")

    panels = dashboard.get("panels", [])
    assert len(panels) >= 6, f"Expected at least 6 KPI panels, got {len(panels)}"

    mapped_keys: set[str] = set()
    for panel in panels:
        kpi_key = panel.get("custom", {}).get("kpi_key") or panel.get("description", "")
        # Check against KNOWN_KPI_KEYS
        matched = next((k for k in KNOWN_KPI_KEYS if k in kpi_key), None)
        assert matched is not None, (
            f"Panel '{panel.get('title')}' does not map to any known KPI in "
            f"{KNOWN_KPI_KEYS}"
        )
        mapped_keys.add(matched)

    assert KNOWN_KPI_KEYS.issubset(mapped_keys), (
        f"Missing dashboard panels for KPIs: {KNOWN_KPI_KEYS - mapped_keys}"
    )

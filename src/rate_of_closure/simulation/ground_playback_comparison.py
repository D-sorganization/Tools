"""Synchronized, observation-bounded comparison of two ground results."""

from __future__ import annotations

import bisect
import csv
import io
from dataclasses import dataclass

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json

from .ground_playback import GroundPlaybackFrame, GroundPlaybackTimeline

GROUND_PLAYBACK_COMPARISON_SCHEMA = "rate-of-closure-ground-playback-comparison/v1"


@dataclass(frozen=True)
class GroundComparisonFrame:
    """Two result frames sampled on one absolute-time clock."""

    time_s: float
    primary: GroundPlaybackFrame
    comparison: GroundPlaybackFrame
    primary_state: str
    comparison_state: str


@dataclass(frozen=True)
class GroundComparisonMetric:
    """One directly observed scalar and comparison-minus-primary delta."""

    metric_id: str
    label: str
    unit: str
    primary: float
    comparison: float

    @property
    def delta(self) -> float:
        return self.comparison - self.primary


@dataclass(frozen=True)
class GroundComparisonProvenance:
    """One paired identity, status, or provenance field."""

    field: str
    primary: str
    comparison: str


class GroundPlaybackComparison:
    """Pair two strict timelines without executing or extrapolating physics."""

    def __init__(
        self,
        primary: GroundPlaybackTimeline,
        comparison: GroundPlaybackTimeline,
    ) -> None:
        if type(primary) is not GroundPlaybackTimeline:
            raise TypeError("primary must use the exact GroundPlaybackTimeline type")
        if type(comparison) is not GroundPlaybackTimeline:
            raise TypeError("comparison must use the exact GroundPlaybackTimeline type")
        if primary.result.unit_system != comparison.result.unit_system:
            raise ValueError("comparison requires matching unit systems")
        if primary.result.frame is not comparison.result.frame:
            raise ValueError("comparison requires matching coordinate frames")
        self.primary = primary
        self.comparison = comparison

    @property
    def start_time_s(self) -> float:
        return min(self.primary.start_time_s, self.comparison.start_time_s)

    @property
    def end_time_s(self) -> float:
        return max(self.primary.end_time_s, self.comparison.end_time_s)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def frame_at(self, time_s: float) -> GroundComparisonFrame:
        """Sample both results; clamped markers are explicitly state-labelled."""
        primary = self.primary.frame_at(time_s)
        comparison = self.comparison.frame_at(time_s)
        return GroundComparisonFrame(
            time_s=min(max(float(time_s), self.start_time_s), self.end_time_s),
            primary=primary,
            comparison=comparison,
            primary_state=self._state(self.primary, time_s),
            comparison_state=self._state(self.comparison, time_s),
        )

    def step_time(self, current_time_s: float, direction: int) -> float:
        """Return the adjacent exact sample time across both result ledgers."""
        if direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        self.frame_at(current_time_s)  # validates finite input
        times = tuple(
            sorted(
                {
                    *(point.time_s for point in self.primary.result.trajectory),
                    *(point.time_s for point in self.comparison.result.trajectory),
                }
            )
        )
        if direction > 0:
            index = bisect.bisect_right(times, current_time_s + 1e-12)
            return float(times[min(index, len(times) - 1)])
        index = bisect.bisect_left(times, current_time_s - 1e-12) - 1
        return float(times[max(index, 0)])

    @staticmethod
    def _state(timeline: GroundPlaybackTimeline, time_s: float) -> str:
        if time_s < timeline.start_time_s:
            return "waiting for first contact"
        if time_s > timeline.end_time_s:
            return f"held at {timeline.end_label.lower()}"
        return "active"

    @property
    def metric_rows(self) -> tuple[GroundComparisonMetric, ...]:
        """Return the complete direct scalar comparison table."""
        left = self.primary.result.summary
        right = self.comparison.result.summary
        if left is None or right is None:  # timelines guarantee summaries
            raise RuntimeError("playable comparison requires both summaries")
        definitions = (
            ("carry_distance_m", "Carry distance", "m"),
            ("bounce_air_distance_m", "Bounce air distance", "m"),
            ("skid_distance_m", "Skid distance", "m"),
            ("roll_distance_m", "Roll distance", "m"),
            ("surface_path_distance_m", "Surface path distance", "m"),
            ("total_distance_m", "Total distance", "m"),
            ("final_downrange_m", "Final downrange", "m"),
            ("final_offline_m", "Final offline", "m"),
            ("bounce_count", "Bounce count", "count"),
        )
        rows = [
            GroundComparisonMetric(
                field,
                label,
                unit,
                float(getattr(left, field)),
                float(getattr(right, field)),
            )
            for field, label, unit in definitions
        ]
        rows.extend(
            (
                GroundComparisonMetric(
                    "start_time_s",
                    "First contact time",
                    "s",
                    self.primary.start_time_s,
                    self.comparison.start_time_s,
                ),
                GroundComparisonMetric(
                    "end_time_s",
                    "Observed end time",
                    "s",
                    self.primary.end_time_s,
                    self.comparison.end_time_s,
                ),
                GroundComparisonMetric(
                    "duration_s",
                    "Observed duration",
                    "s",
                    self.primary.duration_s,
                    self.comparison.duration_s,
                ),
                GroundComparisonMetric(
                    "event_count",
                    "Event count",
                    "count",
                    float(len(self.primary.result.events)),
                    float(len(self.comparison.result.events)),
                ),
                GroundComparisonMetric(
                    "trajectory_sample_count",
                    "Trajectory samples",
                    "count",
                    float(len(self.primary.result.trajectory)),
                    float(len(self.comparison.result.trajectory)),
                ),
            )
        )
        return tuple(rows)

    @property
    def provenance_rows(self) -> tuple[GroundComparisonProvenance, ...]:
        """Return paired result identity and provenance without inference."""
        left = self.primary.result
        right = self.comparison.result
        values = (
            ("Request ID", left.request_id, right.request_id),
            ("Status", left.status.value, right.status.value),
            ("Surface ID", left.surface_id, right.surface_id),
            (
                "Model",
                f"{left.model_id} {left.model_version}",
                f"{right.model_id} {right.model_version}",
            ),
            (
                "Termination",
                left.termination.reason.value,
                right.termination.reason.value,
            ),
            (
                "Producer",
                f"{left.provenance.producer} {left.provenance.producer_version}",
                f"{right.provenance.producer} {right.provenance.producer_version}",
            ),
            (
                "Source revision",
                left.provenance.source_revision,
                right.provenance.source_revision,
            ),
            (
                "Input SHA-256",
                left.provenance.input_sha256,
                right.provenance.input_sha256,
            ),
            (
                "Calibration ID",
                left.calibration.calibration_id,
                right.calibration.calibration_id,
            ),
        )
        return tuple(GroundComparisonProvenance(*value) for value in values)


def ground_comparison_json(comparison: GroundPlaybackComparison) -> str:
    """Export exact inputs plus direct deltas as deterministic JSON."""
    if type(comparison) is not GroundPlaybackComparison:
        raise TypeError("comparison must use the exact GroundPlaybackComparison type")
    return str(
        canonical_numeric_json(
            {
                "schema_version": GROUND_PLAYBACK_COMPARISON_SCHEMA,
                "delta_definition": "comparison_minus_primary",
                "primary": comparison.primary.result.to_dict(),
                "comparison": comparison.comparison.result.to_dict(),
                "metrics": [
                    {
                        "metric_id": row.metric_id,
                        "label": row.label,
                        "unit": row.unit,
                        "primary": row.primary,
                        "comparison": row.comparison,
                        "comparison_minus_primary": row.delta,
                    }
                    for row in comparison.metric_rows
                ],
            }
        )
    )


def ground_comparison_csv(comparison: GroundPlaybackComparison) -> str:
    """Export the complete direct scalar table as deterministic RFC 4180 CSV."""
    if type(comparison) is not GroundPlaybackComparison:
        raise TypeError("comparison must use the exact GroundPlaybackComparison type")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "metric_id",
            "label",
            "unit",
            "primary",
            "comparison",
            "comparison_minus_primary",
        )
    )
    for row in comparison.metric_rows:
        writer.writerow(
            (row.metric_id, row.label, row.unit, row.primary, row.comparison, row.delta)
        )
    return output.getvalue()


__all__ = [
    "GROUND_PLAYBACK_COMPARISON_SCHEMA",
    "GroundComparisonFrame",
    "GroundComparisonMetric",
    "GroundComparisonProvenance",
    "GroundPlaybackComparison",
    "ground_comparison_csv",
    "ground_comparison_json",
]

"""Synchronization analysis and sequence tracking for multi-camera streams."""

from __future__ import annotations

import collections
import statistics
from dataclasses import dataclass
from enum import StrEnum

from ._validation import (
    require_finite,
    require_nonnegative_integer,
    require_text,
)
from .acquisition import FramePacket


class SyncQuality(StrEnum):
    """Declared or measured synchronization confidence between streams."""

    HARDWARE_LOCKED = "hardware-locked"
    CROSS_TRIGGERED = "cross-triggered"
    SOFTWARE_MONOTONIC_ALIGNED = "software-monotonic-aligned"
    UNALIGNED_SKEW_EXCEEDED = "unaligned-skew-exceeded"


class SyncAnomalyType(StrEnum):
    """Type of anomaly detected in a camera frame sequence."""

    DROPPED = "dropped"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out-of-order"


@dataclass(frozen=True, slots=True)
class SyncAnomaly:
    """Evidence record for a detected sequence or timing anomaly."""

    stream_id: str
    anomaly_type: SyncAnomalyType
    expected_seq: int
    actual_seq: int
    dropped_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", require_text(self.stream_id, "stream_id"))
        if not isinstance(self.anomaly_type, SyncAnomalyType):
            raise TypeError("anomaly_type must be a SyncAnomalyType")
        object.__setattr__(
            self,
            "expected_seq",
            require_nonnegative_integer(self.expected_seq, "expected_seq"),
        )
        object.__setattr__(
            self,
            "actual_seq",
            require_nonnegative_integer(self.actual_seq, "actual_seq"),
        )
        object.__setattr__(
            self,
            "dropped_count",
            require_nonnegative_integer(self.dropped_count, "dropped_count"),
        )


@dataclass(frozen=True, slots=True)
class ClockSkewEstimate:
    """Estimated clock offset, drift rate, and jitter bounds between two sources."""

    source_a: str
    source_b: str
    offset_ns: int
    drift_rate_ppm: float
    max_jitter_ns: int
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_a", require_text(self.source_a, "source_a"))
        object.__setattr__(self, "source_b", require_text(self.source_b, "source_b"))
        if self.source_a == self.source_b:
            raise ValueError("cannot estimate skew between a source and itself")
        object.__setattr__(
            self,
            "drift_rate_ppm",
            require_finite(self.drift_rate_ppm, "drift_rate_ppm"),
        )
        object.__setattr__(
            self,
            "max_jitter_ns",
            require_nonnegative_integer(self.max_jitter_ns, "max_jitter_ns"),
        )
        samples = require_nonnegative_integer(self.sample_count, "sample_count")
        if samples == 0:
            raise ValueError("sample_count must be positive")
        object.__setattr__(self, "sample_count", samples)


class SyncMonitor:
    """Monitors inter-camera timestamp offsets, drift, and sequence anomalies."""

    def __init__(self, max_allowable_skew_ns: int = 5_000_000) -> None:
        self._max_allowable_skew_ns: int = require_nonnegative_integer(
            max_allowable_skew_ns, "max_allowable_skew_ns"
        )
        self._last_sequences: dict[str, int] = {}
        # Stores recent (timestamp_ns, host_monotonic_ns) per source_id
        self._recent_timestamps: dict[str, collections.deque[tuple[int, int]]] = {}
        # Pairing history by sequence number: {seq: {source_id: timestamp_ns}}
        self._seq_pairs: dict[int, dict[str, int]] = {}

    def record_frame(self, packet: FramePacket) -> list[SyncAnomaly]:
        """Record an incoming frame packet and return any detected anomalies."""
        if not isinstance(packet, FramePacket):
            raise TypeError("packet must be a FramePacket")

        s_id = packet.source_id
        seq = packet.sequence_number
        anomalies: list[SyncAnomaly] = []

        if s_id not in self._last_sequences:
            self._last_sequences[s_id] = seq
            self._recent_timestamps[s_id] = collections.deque(maxlen=100)
            self._recent_timestamps[s_id].append(
                (packet.timestamp_ns, packet.host_monotonic_ns)
            )
        else:
            prev_seq = self._last_sequences[s_id]
            if seq == prev_seq:
                anomalies.append(
                    SyncAnomaly(
                        stream_id=s_id,
                        anomaly_type=SyncAnomalyType.DUPLICATE,
                        expected_seq=prev_seq + 1,
                        actual_seq=seq,
                    )
                )
            elif seq < prev_seq:
                anomalies.append(
                    SyncAnomaly(
                        stream_id=s_id,
                        anomaly_type=SyncAnomalyType.OUT_OF_ORDER,
                        expected_seq=prev_seq + 1,
                        actual_seq=seq,
                    )
                )
            elif seq > prev_seq + 1:
                dropped = seq - (prev_seq + 1)
                anomalies.append(
                    SyncAnomaly(
                        stream_id=s_id,
                        anomaly_type=SyncAnomalyType.DROPPED,
                        expected_seq=prev_seq + 1,
                        actual_seq=seq,
                        dropped_count=dropped,
                    )
                )
                self._last_sequences[s_id] = seq
            else:
                self._last_sequences[s_id] = seq

            self._recent_timestamps[s_id].append(
                (packet.timestamp_ns, packet.host_monotonic_ns)
            )

        if seq not in self._seq_pairs:
            self._seq_pairs[seq] = {}
        self._seq_pairs[seq][s_id] = packet.timestamp_ns

        # Keep seq_pairs bounded
        if len(self._seq_pairs) > 500:
            oldest_keys = sorted(self._seq_pairs.keys())[:100]
            for old_key in oldest_keys:
                del self._seq_pairs[old_key]

        return anomalies

    def estimate_skew(self, source_a: str, source_b: str) -> ClockSkewEstimate | None:
        """Estimate clock skew, drift, and jitter between two paired sources."""
        require_text(source_a, "source_a")
        require_text(source_b, "source_b")
        if source_a == source_b:
            raise ValueError("cannot estimate skew between a source and itself")

        offsets: list[int] = []
        for pair in self._seq_pairs.values():
            if source_a in pair and source_b in pair:
                offsets.append(pair[source_b] - pair[source_a])

        if not offsets:
            return None

        mean_offset = int(statistics.mean(offsets))
        jitter = (
            int(max(abs(off - mean_offset) for off in offsets))
            if len(offsets) > 1
            else 0
        )
        drift_ppm = 0.0
        if len(offsets) >= 2:
            # Simple drift rate over observed samples
            drift_ns = offsets[-1] - offsets[0]
            total_duration_ns = max(abs(offsets[-1]), abs(offsets[0]), 1_000_000)
            drift_ppm = (drift_ns / total_duration_ns) * 1_000_000.0

        return ClockSkewEstimate(
            source_a=source_a,
            source_b=source_b,
            offset_ns=mean_offset,
            drift_rate_ppm=drift_ppm,
            max_jitter_ns=jitter,
            sample_count=len(offsets),
        )

    def evaluate_quality(self, source_a: str, source_b: str) -> SyncQuality:
        """Evaluate sync quality, failing closed if skew exceeds threshold."""
        skew = self.estimate_skew(source_a, source_b)
        if skew is None:
            return SyncQuality.UNALIGNED_SKEW_EXCEEDED

        if (
            abs(skew.offset_ns) > self._max_allowable_skew_ns
            or skew.max_jitter_ns > self._max_allowable_skew_ns
        ):
            return SyncQuality.UNALIGNED_SKEW_EXCEEDED

        if abs(skew.offset_ns) < 100_000 and skew.max_jitter_ns < 50_000:
            return SyncQuality.HARDWARE_LOCKED

        return SyncQuality.SOFTWARE_MONOTONIC_ALIGNED


__all__ = [
    "ClockSkewEstimate",
    "SyncAnomaly",
    "SyncAnomalyType",
    "SyncMonitor",
    "SyncQuality",
]

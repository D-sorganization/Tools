"""Bounded scalar analysis over checksum-verified durable ensemble chunks."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from shared.python.contracts import require

from .durable_ensemble_chunks import (
    DurableEnsembleArchive,
    DurableEnsembleChunkSink,
)
from .ensemble_chunks import SimulationResultChunk
from .ensemble_source import SimulationEnsembleSource
from .plot_labels import OUTPUT_UNITS
from .simulation_types import (
    ALL_OUTPUT_NAMES,
    APP_FRAME_ID,
    TrialEvaluationStatus,
)

_STATUS_NAMES = frozenset(item.value for item in TrialEvaluationStatus)


@dataclass(frozen=True, slots=True)
class DurableEnsembleLayout:
    """Public trace layout retained without any archive filesystem identity."""

    sample_count: int
    point_ids: tuple[str, ...]
    coordinate_frame: str

    def __post_init__(self) -> None:
        require(self.sample_count >= 1, "sample_count must be positive")
        require(bool(self.point_ids), "point_ids must be nonempty")
        require(
            len(set(self.point_ids)) == len(self.point_ids),
            "point_ids must be unique",
        )
        require(
            self.coordinate_frame == APP_FRAME_ID,
            "coordinate frame is not supported",
        )


@dataclass(frozen=True, slots=True)
class StreamingOutputMoments:
    """Availability and first two sample moments for one canonical output."""

    name: str
    unit: str
    available_count: int
    mean: float | None
    sample_std: float | None

    def __post_init__(self) -> None:
        require(self.name in ALL_OUTPUT_NAMES, "unknown ensemble output", self.name)
        require(self.unit == OUTPUT_UNITS[self.name], "output unit is not canonical")
        require(self.available_count >= 0, "available_count must be non-negative")
        require(
            self.mean is None or math.isfinite(self.mean),
            "available output mean must be finite",
        )
        require(
            self.sample_std is None
            or (math.isfinite(self.sample_std) and self.sample_std >= 0.0),
            "available output sample_std must be finite and non-negative",
        )
        require(
            (self.mean is None) == (self.available_count == 0),
            "mean availability must match available_count",
        )
        require(
            (self.sample_std is None) == (self.available_count < 2),
            "sample_std requires at least two available values",
        )


@dataclass(frozen=True, slots=True)
class DurableEnsembleSummary:
    """Incremental scalar evidence for one verified durable archive prefix."""

    archive: DurableEnsembleArchive
    layout: DurableEnsembleLayout
    analyzed_trial_count: int
    status_counts: Mapping[str, int]
    failure_type_counts: Mapping[str, int]
    output_moments: tuple[StreamingOutputMoments, ...]

    def __post_init__(self) -> None:
        status_counts = dict(self.status_counts)
        failure_counts = dict(self.failure_type_counts)
        require(set(status_counts) == _STATUS_NAMES, "status count keys are invalid")
        require(
            all(type(value) is int and value >= 0 for value in status_counts.values()),
            "status counts must be non-negative integers",
        )
        require(
            self.analyzed_trial_count == self.archive.next_index,
            "analysis count must match verified archive prefix",
        )
        require(
            sum(status_counts.values()) == self.analyzed_trial_count,
            "status counts must cover the analyzed prefix",
        )
        require(
            all(
                name and type(count) is int and count > 0
                for name, count in failure_counts.items()
            ),
            "failure counts must contain named positive integers",
        )
        require(
            sum(failure_counts.values()) == status_counts["numerical_failure"],
            "failure types must cover numerical failures",
        )
        require(
            tuple(item.name for item in self.output_moments) == ALL_OUTPUT_NAMES,
            "output moments must use canonical order",
        )
        object.__setattr__(self, "status_counts", MappingProxyType(status_counts))
        object.__setattr__(
            self, "failure_type_counts", MappingProxyType(failure_counts)
        )


@dataclass(slots=True)
class _Moments:
    count: int = 0
    mean: float = 0.0
    centered_sum: float = 0.0

    def add(self, value: float) -> None:
        """Update stable first and second moments with one finite value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.centered_sum += delta * (value - self.mean)

    def freeze(self, name: str) -> StreamingOutputMoments:
        """Return the immutable public summary for one output."""
        sample_std = None
        if self.count >= 2:
            sample_std = math.sqrt(max(0.0, self.centered_sum / (self.count - 1)))
        return StreamingOutputMoments(
            name=name,
            unit=OUTPUT_UNITS[name],
            available_count=self.count,
            mean=None if self.count == 0 else self.mean,
            sample_std=sample_std,
        )


class _SummaryAccumulator:
    """Coordinator-owned bounded accumulator for canonical scalar outcomes."""

    def __init__(self) -> None:
        self.status_counts = Counter({name: 0 for name in _STATUS_NAMES})
        self.failure_counts: Counter[str] = Counter()
        self.moments = {name: _Moments() for name in ALL_OUTPUT_NAMES}

    def accept(self, chunk: SimulationResultChunk) -> None:
        """Accumulate one already verified chunk without retaining it."""
        for outcome in chunk.outcomes:
            self.status_counts[outcome.status.value] += 1
            if outcome.failure_type is not None:
                self.failure_counts[outcome.failure_type] += 1
            for name, accumulator in self.moments.items():
                value = outcome.value(name)
                if value is not None:
                    accumulator.add(value)


def analyze_durable_ensemble(
    request: SimulationEnsembleSource, directory: str | Path
) -> DurableEnsembleSummary:
    """Scan one strict archive into bounded typed counts and scalar moments."""
    require(
        isinstance(request, SimulationEnsembleSource),
        "request must be a SimulationEnsembleSource",
    )
    accumulator = _SummaryAccumulator()
    archive, header = DurableEnsembleChunkSink(directory).scan_with_header(
        request, accumulator.accept
    )
    moments = tuple(accumulator.moments[name].freeze(name) for name in ALL_OUTPUT_NAMES)
    return DurableEnsembleSummary(
        archive=archive,
        layout=DurableEnsembleLayout(
            sample_count=header.sample_times_s.size,
            point_ids=header.point_ids,
            coordinate_frame=header.coordinate_frame,
        ),
        analyzed_trial_count=archive.next_index,
        status_counts=accumulator.status_counts,
        failure_type_counts=accumulator.failure_counts,
        output_moments=moments,
    )


__all__ = [
    "DurableEnsembleLayout",
    "DurableEnsembleSummary",
    "StreamingOutputMoments",
    "analyze_durable_ensemble",
]

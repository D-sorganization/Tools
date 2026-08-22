"""Bounded geometric analysis over verified durable ensemble chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rate_of_closure.variation.durable_ensemble_chunks import (
    DurableEnsembleArchive,
    DurableEnsembleChunkSink,
)
from rate_of_closure.variation.ensemble_chunks import SimulationResultChunk
from rate_of_closure.variation.ensemble_source import SimulationEnsembleSource
from shared.python.contracts import require
from shared.python.swing_sim.variation import (
    PositionDispersion,
    PositionDispersionAccumulator,
)


@dataclass(frozen=True, slots=True)
class DurableEnsembleGeometry:
    """Incremental position dispersion bound to one verified archive prefix."""

    archive: DurableEnsembleArchive
    analyzed_trial_count: int
    dispersion: PositionDispersion

    def __post_init__(self) -> None:
        require(
            self.analyzed_trial_count == self.archive.next_index,
            "geometry count must match the verified archive prefix",
        )


def analyze_durable_ensemble_geometry(
    request: SimulationEnsembleSource, directory: str | Path
) -> DurableEnsembleGeometry:
    """Scan strict chunks into bounded per-sample position moments."""
    require(
        isinstance(request, SimulationEnsembleSource),
        "request must be a SimulationEnsembleSource",
    )
    accumulator: PositionDispersionAccumulator | None = None

    def accept(chunk: SimulationResultChunk) -> None:
        nonlocal accumulator
        if accumulator is None:
            accumulator = PositionDispersionAccumulator(
                chunk.positions_m.shape[1], chunk.positions_m.shape[2]
            )
        accumulator.accept(chunk.positions_m, chunk.sample_valid)

    archive, header = DurableEnsembleChunkSink(directory).scan_with_header(
        request, accept
    )
    if accumulator is None:
        accumulator = PositionDispersionAccumulator(
            header.sample_times_s.size, len(header.point_ids)
        )
    dispersion = accumulator.freeze(
        header.sample_times_s, header.coordinate_frame, header.point_ids
    )
    return DurableEnsembleGeometry(archive, archive.next_index, dispersion)


__all__ = ["DurableEnsembleGeometry", "analyze_durable_ensemble_geometry"]

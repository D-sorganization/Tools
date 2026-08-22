"""Atomic, checksum-verified persistence for complete ensemble chunks."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from shared.python.contracts import require

from ._durable_ensemble_io import (
    MANIFEST_NAME,
)
from ._durable_ensemble_io import (
    chunk_record as _chunk_record,
)
from ._durable_ensemble_io import (
    file_sha256 as _file_sha256,
)
from ._durable_ensemble_io import (
    header_document as _header_document,
)
from ._durable_ensemble_io import (
    json_sha256 as _json_sha256,
)
from ._durable_ensemble_io import (
    new_manifest as _new_manifest,
)
from ._durable_ensemble_io import (
    read_chunk as _read_chunk,
)
from ._durable_ensemble_io import (
    read_manifest as _read_manifest,
)
from ._durable_ensemble_io import (
    verify_header as _verify_header,
)
from ._durable_ensemble_io import (
    write_chunk_atomic as _write_chunk_atomic,
)
from ._durable_ensemble_io import (
    write_json_atomic as _write_json_atomic,
)
from .ensemble_chunks import (
    EnsembleResumeState,
    EnsembleStreamHeader,
    SimulationResultChunk,
    require_chunk_matches_header,
)
from .ensemble_source import SimulationEnsembleSource
from .simulation_types import NUMERICAL_FAILURE


@dataclass(frozen=True, slots=True)
class DurableEnsembleArchive:
    """Lightweight inspection record for a durable ensemble stream."""

    directory: Path
    header_sha256: str
    status: Literal["in_progress", "complete"]
    trial_count: int
    next_index: int
    failed_count: int
    chunk_count: int
    elapsed_s: float | None

    def __post_init__(self) -> None:
        require(self.directory.is_absolute(), "archive directory must be absolute")
        require(len(self.header_sha256) == 64, "invalid archive header checksum")
        require(self.status in {"in_progress", "complete"}, "invalid archive status")
        require(0 <= self.next_index <= self.trial_count, "invalid archive prefix")
        require(0 <= self.failed_count <= self.next_index, "invalid failure count")
        require(self.chunk_count >= 0, "invalid archive chunk count")
        require(
            self.elapsed_s is None
            or (math.isfinite(self.elapsed_s) and self.elapsed_s >= 0.0),
            "invalid archive elapsed time",
        )


class DurableEnsembleChunkSink:
    """Persist a contiguous chunk stream and retain valid work on abort."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory).resolve()
        self._manifest_path = self._directory / MANIFEST_NAME
        self._header: EnsembleStreamHeader | None = None
        self._manifest: dict[str, Any] | None = None
        self._active = False

    def begin(self, header: EnsembleStreamHeader) -> None:
        """Create or verify an archive before any evaluation may resume."""
        self._open(header)
        self._verify_prefix(cast(dict[str, Any], self._manifest))

    def scan(
        self,
        request: SimulationEnsembleSource,
        visitor: Callable[[SimulationResultChunk], None],
    ) -> DurableEnsembleArchive:
        """Visit each verified prefix chunk without retaining prior chunks."""
        archive, _header = self.scan_with_header(request, visitor)
        return archive

    def scan_with_header(
        self,
        request: SimulationEnsembleSource,
        visitor: Callable[[SimulationResultChunk], None],
    ) -> tuple[DurableEnsembleArchive, EnsembleStreamHeader]:
        """Scan once and return the exact verified layout used by the visitor."""
        from .simulation_adapter import build_ensemble_stream_header

        require(callable(visitor), "archive visitor must be callable")
        header = build_ensemble_stream_header(request)
        self._open(header)
        try:
            manifest = self._require_active()
            for chunk in self._verified_prefix(manifest):
                visitor(chunk)
            return self._archive(), header
        finally:
            self.abort()

    def _open(self, header: EnsembleStreamHeader) -> None:
        """Open one exact archive lifecycle without scanning its prefix twice."""
        require(not self._active, "sink lifecycle has already begun")
        self._directory.mkdir(parents=True, exist_ok=True)
        header_document = _header_document(header)
        header_sha256 = _json_sha256(header_document)
        if self._manifest_path.exists():
            manifest = _read_manifest(self._manifest_path)
        else:
            manifest = _new_manifest(header_document, header_sha256)
            _write_json_atomic(self._manifest_path, manifest)
        _verify_header(manifest, header_document, header_sha256)
        self._header = header
        self._manifest = manifest
        self._active = True

    def resume_state(self) -> EnsembleResumeState:
        """Return the prefix already proven against this run's header."""
        manifest = self._require_active()
        return EnsembleResumeState(
            cast(int, manifest["next_index"]), cast(int, manifest["failed_count"])
        )

    def accept(self, chunk: SimulationResultChunk) -> None:
        """Atomically add one verified contiguous chunk to the durable prefix."""
        manifest = self._require_active()
        require(manifest["status"] == "in_progress", "archive is already complete")
        header = cast(EnsembleStreamHeader, self._header)
        next_index = cast(int, manifest["next_index"])
        require_chunk_matches_header(header, chunk, next_index)
        record = self._persist_chunk(chunk)
        updated = dict(manifest)
        updated["chunks"] = [*cast(list[object], manifest["chunks"]), record]
        updated["next_index"] = record["stop_index"]
        updated["failed_count"] = cast(int, manifest["failed_count"]) + cast(
            int, record["failed_count"]
        )
        _write_json_atomic(self._manifest_path, updated)
        self._manifest = updated

    def commit(self, elapsed_s: float) -> DurableEnsembleArchive:
        """Mark a complete exact-once prefix and return its lightweight record."""
        manifest = self._require_active()
        require(math.isfinite(elapsed_s) and elapsed_s >= 0.0, "invalid elapsed_s")
        header = cast(EnsembleStreamHeader, self._header)
        require(
            manifest["next_index"] == header.plan.n_runs,
            "cannot commit partial durable stream",
        )
        if manifest["status"] == "in_progress":
            updated = dict(manifest)
            updated["status"] = "complete"
            updated["elapsed_s"] = float(elapsed_s)
            _write_json_atomic(self._manifest_path, updated)
            self._manifest = updated
        self._active = False
        return self._archive()

    def abort(self) -> None:
        """End this lifecycle without deleting the last valid prefix."""
        self._active = False

    def inspect(self, request: SimulationEnsembleSource) -> DurableEnsembleArchive:
        """Verify and summarize an existing archive against one exact request."""
        from .simulation_adapter import build_ensemble_stream_header

        require(self._manifest_path.is_file(), "durable archive does not exist")
        self.begin(build_ensemble_stream_header(request))
        result = self._archive()
        self.abort()
        return result

    def _persist_chunk(self, chunk: SimulationResultChunk) -> dict[str, object]:
        stop_index = chunk.start_index + len(chunk.outcomes)
        filename = f"chunk-{chunk.start_index:08d}-{stop_index:08d}.npz"
        destination = self._directory / filename
        _write_chunk_atomic(destination, chunk)
        return {
            "file": filename,
            "start_index": chunk.start_index,
            "stop_index": stop_index,
            "sha256": _file_sha256(destination),
            "failed_count": sum(
                outcome.status is NUMERICAL_FAILURE for outcome in chunk.outcomes
            ),
        }

    def _verify_prefix(self, manifest: dict[str, Any]) -> None:
        for _ in self._verified_prefix(manifest):
            pass

    def _verified_prefix(
        self, manifest: dict[str, Any]
    ) -> Iterator[SimulationResultChunk]:
        """Yield a bounded prefix and validate its aggregate manifest identity."""
        header = cast(EnsembleStreamHeader, self._header)
        next_index = 0
        failed_count = 0
        for raw_record in cast(list[object], manifest["chunks"]):
            record = _chunk_record(raw_record, next_index, header.plan.n_runs)
            chunk = _read_chunk(self._directory, record)
            require_chunk_matches_header(header, chunk, next_index)
            actual_failed = sum(
                outcome.status is NUMERICAL_FAILURE for outcome in chunk.outcomes
            )
            require(
                actual_failed == record["failed_count"],
                "chunk failure count does not match content",
            )
            failed_count += actual_failed
            next_index = cast(int, record["stop_index"])
            yield chunk
        require(next_index == manifest["next_index"], "manifest prefix is inconsistent")
        require(
            failed_count == manifest["failed_count"],
            "manifest failure count is inconsistent",
        )
        if manifest["status"] == "complete":
            require(next_index == header.plan.n_runs, "complete archive is partial")

    def _require_active(self) -> dict[str, Any]:
        require(self._active and self._manifest is not None, "sink is not active")
        return cast(dict[str, Any], self._manifest)

    def _archive(self) -> DurableEnsembleArchive:
        manifest = cast(dict[str, Any], self._manifest)
        header = cast(EnsembleStreamHeader, self._header)
        return DurableEnsembleArchive(
            directory=self._directory,
            header_sha256=cast(str, manifest["header_sha256"]),
            status=cast(Literal["in_progress", "complete"], manifest["status"]),
            trial_count=header.plan.n_runs,
            next_index=cast(int, manifest["next_index"]),
            failed_count=cast(int, manifest["failed_count"]),
            chunk_count=len(cast(list[object], manifest["chunks"])),
            elapsed_s=cast(float | None, manifest["elapsed_s"]),
        )


__all__ = ["DurableEnsembleArchive", "DurableEnsembleChunkSink"]

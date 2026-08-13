"""Crash-resumable non-materializing storage for complete ensemble chunks."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from shared.python.contracts import require

from ._ensemble_limits import MAX_ARCHIVE_CHUNKS
from .ensemble_archive_codec import read_chunk_file, write_chunk_file
from .ensemble_archive_contracts import (
    ARCHIVE_SCHEMA_ID,
    ARCHIVE_SCHEMA_VERSION,
    ZERO_SHA256,
    CommittedEnsembleArchive,
    EnsembleResumeCursor,
    canonical_json_bytes,
)
from .ensemble_archive_storage import (
    COMMIT_NAME,
    HEADER_NAME,
    atomic_bytes,
    chunk_paths,
    initialize_archive,
    load_commit,
    load_header,
    require_same_header,
)
from .ensemble_chunks import (
    CollectingEnsembleSink,
    EnsembleStreamHeader,
    SimulationResultChunk,
    require_chunk_matches_header,
)
from .simulation_types import SimulationEnsembleResult


class DurableEnsembleArchiveSink:
    """Coordinator-owned provisional archive sink with verified resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._header: EnsembleStreamHeader | None = None
        self._archive_sha256: str | None = None
        self._next_index = 0
        self._failure_count = 0
        self._previous_sha256 = ZERO_SHA256
        self._chunk_count = 0
        self._finished = False

    def begin(self, header: EnsembleStreamHeader) -> EnsembleResumeCursor:
        require(
            self._header is None and not self._finished, "sink lifecycle already began"
        )
        require(
            header.authority_layout is not None, "durable archive requires authority"
        )
        require(
            header.request_identity_sha256 is not None,
            "durable archive requires request identity",
        )
        if not self.path.exists() or not (self.path / HEADER_NAME).exists():
            archive_sha = initialize_archive(self.path, header)
            stored = header
        else:
            require(
                not (self.path / COMMIT_NAME).exists(),
                "committed archive cannot resume",
            )
            stored, archive_sha = load_header(self.path)
            require_same_header(stored, header)
        self._header = stored
        self._archive_sha256 = archive_sha
        for chunk_path in chunk_paths(self.path):
            chunk, chunk_sha = read_chunk_file(
                chunk_path, stored, archive_sha, self._previous_sha256, self._next_index
            )
            self._next_index += len(chunk.outcomes)
            self._failure_count += sum(
                item.failure_type is not None for item in chunk.outcomes
            )
            self._previous_sha256 = chunk_sha
            self._chunk_count += 1
        return EnsembleResumeCursor(
            self._next_index,
            self._failure_count,
            self._previous_sha256,
            self._chunk_count,
        )

    def accept(self, chunk: SimulationResultChunk) -> None:
        header = self._active()
        require_chunk_matches_header(header, chunk, self._next_index)
        require(
            self._chunk_count < MAX_ARCHIVE_CHUNKS, "archive chunk-count limit exceeded"
        )
        stop = chunk.start_index + len(chunk.outcomes)
        target = self.path / "chunks" / f"{chunk.start_index:012d}-{stop:012d}.roc"
        require(not target.exists(), "chunk file already exists")
        temporary = target.with_name(f"{target.name}.partial")
        if temporary.exists():
            temporary.unlink()
        assert self._archive_sha256 is not None
        chunk_sha = write_chunk_file(
            temporary, chunk, self._archive_sha256, self._previous_sha256
        )
        os.replace(temporary, target)
        self._next_index = stop
        self._failure_count += sum(
            item.failure_type is not None for item in chunk.outcomes
        )
        self._previous_sha256 = chunk_sha
        self._chunk_count += 1

    def commit(self, elapsed_s: float) -> CommittedEnsembleArchive:
        header = self._active()
        require(self._next_index == header.plan.n_runs, "cannot commit partial archive")
        require(math.isfinite(elapsed_s) and elapsed_s >= 0.0, "invalid elapsed_s")
        assert self._archive_sha256 is not None
        document = {
            "schema_id": ARCHIVE_SCHEMA_ID,
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "archive_sha256": self._archive_sha256,
            "scientific_root_sha256": self._previous_sha256,
            "trial_count": self._next_index,
            "chunk_count": self._chunk_count,
            "elapsed_s": float(elapsed_s),
        }
        atomic_bytes(self.path / COMMIT_NAME, canonical_json_bytes(document))
        self._finished = True
        return CommittedEnsembleArchive(
            self.path,
            self._previous_sha256,
            self._next_index,
            self._chunk_count,
            float(elapsed_s),
        )

    def abort(self) -> None:
        """Close this process lifecycle while preserving verified provisional files."""
        self._finished = True

    def _active(self) -> EnsembleStreamHeader:
        require(self._header is not None and not self._finished, "sink is not active")
        return cast(EnsembleStreamHeader, self._header)


class DurableEnsembleChunkSource:
    """Lazy verified source exposing at most one owned chunk per iteration."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.header, self._archive_sha256 = load_header(self.path)
        self.commit = load_commit(self.path, self._archive_sha256)
        self._verify_committed_prefix()

    def _verify_committed_prefix(self) -> None:
        next_index = 0
        previous = ZERO_SHA256
        count = 0
        for path in chunk_paths(self.path):
            chunk, previous = read_chunk_file(
                path, self.header, self._archive_sha256, previous, next_index
            )
            next_index += len(chunk.outcomes)
            count += 1
        require(next_index == self.commit.trial_count, "archive trial count mismatch")
        require(count == self.commit.chunk_count, "archive chunk count mismatch")
        require(previous == self.commit.scientific_root_sha256, "archive root mismatch")

    def __iter__(self) -> Iterator[SimulationResultChunk]:
        next_index = 0
        previous = ZERO_SHA256
        count = 0
        for path in chunk_paths(self.path):
            chunk, previous = read_chunk_file(
                path, self.header, self._archive_sha256, previous, next_index
            )
            next_index += len(chunk.outcomes)
            count += 1
            yield chunk
        require(next_index == self.commit.trial_count, "archive trial count mismatch")
        require(count == self.commit.chunk_count, "archive chunk count mismatch")
        require(previous == self.commit.scientific_root_sha256, "archive root mismatch")

    def materialize_compatibility(self) -> SimulationEnsembleResult:
        """Explicitly reconstruct the legacy aggregate under existing global caps."""
        collector = CollectingEnsembleSink()
        collector.begin(self.header)
        for chunk in self:
            collector.accept(chunk)
        return collector.commit(self.commit.elapsed_s)


__all__ = ["DurableEnsembleArchiveSink", "DurableEnsembleChunkSource"]

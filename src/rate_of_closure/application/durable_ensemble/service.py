"""Single numerical authority for resumable durable simulation ensembles."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from rate_of_closure.variation import (
    AnalyzingDurableEnsembleSink,
    DurableEnsembleEvidence,
    analyze_durable_ensemble,
    durable_ensemble_evidence,
    run_simulation_ensemble_chunks,
)

from .contracts import DurableEnsembleAuthorityRequest

EvidenceSink = Callable[[DurableEnsembleEvidence], None]
SimulationExecutor = Callable[[SimulationConfig], SimulationRun]


class RateDurableEnsembleService:
    """Execute, inspect, and exactly resume archives below one owned root."""

    def __init__(
        self,
        archive_root: str | Path,
        executor: SimulationExecutor = run_simulation,
    ) -> None:
        root = Path(archive_root).resolve()
        if not callable(executor):
            raise TypeError("executor must be callable")
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ValueError("archive_root must be a directory")
        self._root = root
        self._executor = executor

    def execute(
        self,
        request: DurableEnsembleAuthorityRequest,
        cancel: threading.Event,
        progress: EvidenceSink,
    ) -> DurableEnsembleEvidence:
        """Run bounded chunks and emit evidence after every durable prefix."""
        self._validate_controls(request, cancel, progress)
        source = request.source()
        sink = AnalyzingDurableEnsembleSink(self._archive_path(request.archive_id))

        def report(_update: object) -> None:
            progress(durable_ensemble_evidence(sink.snapshot()))

        summary = run_simulation_ensemble_chunks(
            source,
            sink,
            chunk_size=request.chunk_size,
            executor=self._executor,
            progress_cb=report,
            cancel_event=cancel,
        )
        result = durable_ensemble_evidence(summary)
        progress(result)
        return result

    def inspect(
        self, request: DurableEnsembleAuthorityRequest
    ) -> DurableEnsembleEvidence:
        """Verify and summarize the retained prefix without executing trials."""
        if not isinstance(request, DurableEnsembleAuthorityRequest):
            raise TypeError("request must be a DurableEnsembleAuthorityRequest")
        summary = analyze_durable_ensemble(
            request.source(), self._archive_path(request.archive_id)
        )
        return durable_ensemble_evidence(summary)

    def _archive_path(self, archive_id: str) -> Path:
        candidate = self._root / archive_id
        resolved = candidate.resolve()
        if resolved.parent != self._root:
            raise ValueError("archive identity escapes the authority root")
        return candidate

    @staticmethod
    def _validate_controls(
        request: DurableEnsembleAuthorityRequest,
        cancel: threading.Event,
        progress: EvidenceSink,
    ) -> None:
        if not isinstance(request, DurableEnsembleAuthorityRequest):
            raise TypeError("request must be a DurableEnsembleAuthorityRequest")
        if not isinstance(cancel, threading.Event) or not callable(progress):
            raise TypeError("cancel and progress controls are required")


__all__ = ["EvidenceSink", "RateDurableEnsembleService"]

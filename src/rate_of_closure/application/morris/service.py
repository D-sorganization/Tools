"""UI-neutral execution service for Rate fixed-ball Morris studies."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.variation.morris_rate_adapter import (
    RATE_MORRIS_OUTPUTS,
    RateMorrisEvaluator,
)
from shared.python.swing_sim.solver.solve import ProgressReport
from shared.python.swing_sim.variation import (
    MorrisDesign,
    MorrisEvaluator,
    MorrisExecutionOptions,
    MorrisObservationArchive,
    analyze_morris,
    evaluate_morris_design,
    make_morris_observation_archive,
)

from .contracts import MorrisAuthorityRequest

ProgressSink = Callable[[int, int], None]
EvaluatorFactory = Callable[[MorrisDesign, SimulationConfig], MorrisEvaluator]


@dataclass(frozen=True)
class MorrisServiceResult:
    """One unchanged report plus its separately versioned raw authority."""

    report: dict[str, Any]
    observations: MorrisObservationArchive

    def __post_init__(self) -> None:
        if not isinstance(self.report, dict) or not isinstance(
            self.observations, MorrisObservationArchive
        ):
            raise TypeError("report dictionary and observation archive are required")


def morris_request_sha256(request: MorrisAuthorityRequest) -> str:
    """Return the canonical digest used to bind observations to a request."""
    if not isinstance(request, MorrisAuthorityRequest):
        raise TypeError("request must be a MorrisAuthorityRequest")
    request_bytes = json.dumps(
        request.to_json_dict(),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(request_bytes).hexdigest()


class MorrisExecutionService(Protocol):
    """Minimal dependency injected into a job registry."""

    def execute(
        self,
        request: MorrisAuthorityRequest,
        cancel: threading.Event,
        progress: ProgressSink,
    ) -> dict[str, Any]:
        """Execute one validated request or raise."""
        ...


def _rate_evaluator(design: MorrisDesign, config: SimulationConfig) -> MorrisEvaluator:
    return RateMorrisEvaluator(design, config)


class RateMorrisService:
    """Build, execute, analyze, and serialize one deterministic study."""

    def __init__(self, evaluator_factory: EvaluatorFactory = _rate_evaluator) -> None:
        if not callable(evaluator_factory):
            raise TypeError("evaluator_factory must be callable")
        self._evaluator_factory = evaluator_factory

    def execute(
        self,
        request: MorrisAuthorityRequest,
        cancel: threading.Event,
        progress: ProgressSink,
    ) -> dict[str, Any]:
        """Return the unchanged shared Morris report-v1 document."""
        return self.execute_with_observations(request, cancel, progress).report

    def execute_with_observations(
        self,
        request: MorrisAuthorityRequest,
        cancel: threading.Event,
        progress: ProgressSink,
    ) -> MorrisServiceResult:
        """Return report-v1 plus its separately versioned scalar observations."""
        if not isinstance(request, MorrisAuthorityRequest):
            raise TypeError("request must be a MorrisAuthorityRequest")
        if not isinstance(cancel, threading.Event) or not callable(progress):
            raise TypeError("cancel and progress controls are required")
        design = request.design()
        total = request.total_samples

        def report(update: ProgressReport) -> None:
            progress(update.iteration, total)

        options = MorrisExecutionOptions(request.worker_count, report, cancel)
        evaluator = self._evaluator_factory(design, request.base_config())
        observations = evaluate_morris_design(
            design, RATE_MORRIS_OUTPUTS, evaluator, options
        )
        report_document = cast(
            dict[str, Any],
            analyze_morris(observations, request.minimum_effects).to_json_dict(),
        )
        archive = make_morris_observation_archive(
            observations,
            study_id=request.request_id,
            provenance={
                "producer": "rate-of-closure-morris-authority",
                "request_schema": "rate-of-closure/morris-request@1",
                "request_sha256": morris_request_sha256(request),
                "report_schema": "swing-sim/morris-global-sensitivity-report@1",
            },
        )
        return MorrisServiceResult(report_document, archive)


__all__ = [
    "MorrisExecutionService",
    "MorrisServiceResult",
    "morris_request_sha256",
    "ProgressSink",
    "RateMorrisService",
]

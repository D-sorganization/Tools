"""UI-neutral execution service for Rate fixed-ball Morris studies."""

from __future__ import annotations

import threading
from collections.abc import Callable
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
    analyze_morris,
    evaluate_morris_design,
)

from .contracts import MorrisAuthorityRequest

ProgressSink = Callable[[int, int], None]
EvaluatorFactory = Callable[[MorrisDesign, SimulationConfig], MorrisEvaluator]


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
        return cast(
            dict[str, Any],
            analyze_morris(observations, request.minimum_effects).to_json_dict(),
        )


__all__ = ["MorrisExecutionService", "ProgressSink", "RateMorrisService"]

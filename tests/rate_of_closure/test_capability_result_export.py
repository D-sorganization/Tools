"""Capability-result exports retain every engineering diagnostic."""

from __future__ import annotations

import csv
import io
import json

from rate_of_closure.application.capability_result_export import (
    capability_alternatives_csv,
    capability_result_export_json,
)
from rate_of_closure.application.capability_workflow import (
    CapabilityWorkflowInputs,
    build_capability_workflow,
)
from rate_of_closure.variation.capability_observation_adapter import (
    CapabilityObservationEnsembleBuilder,
)
from rate_of_closure.variation.scalar_ensemble_contract import ScalarEnsembleDataset
from shared.python.swing_sim.flight.capability_flight_evaluator import (
    make_capability_flight_evaluator,
)
from shared.python.swing_sim.flight.capability_observation import (
    CapabilityOptimizationHooks,
)
from shared.python.swing_sim.flight.capability_optimizer import optimize_capability
from shared.python.swing_sim.flight.capability_result import OptimizationResult


def _output() -> tuple[OptimizationResult, ScalarEnsembleDataset]:
    document = build_capability_workflow(
        CapabilityWorkflowInputs(
            candidate_budget=1, ensemble_size=2, alternatives_count=1
        )
    )
    total = document.request.candidate_budget * document.request.ensemble_size
    builder = CapabilityObservationEnsembleBuilder(
        document.request.target, total, document.profile.provenance
    )
    evaluator = make_capability_flight_evaluator(
        document.profile, document.request, document.evaluator_config
    )
    result = optimize_capability(
        document.profile,
        document.request,
        evaluator,
        hooks=CapabilityOptimizationHooks(observation_sink=builder),
    )
    return result, builder.build()


def test_capability_alternatives_csv_retains_diagnostics_and_units() -> None:
    result, dataset = _output()

    source = capability_alternatives_csv(result, dataset)
    rows = list(csv.DictReader(io.StringIO(source)))

    assert set(rows[0]) == {
        "rank",
        "club_id",
        "parameters",
        "score",
        "mean_carry_m",
        "expected_miss_m",
        "dispersion_rms_m",
        "target_hold_probability",
        "cvar_miss_m",
        "downside_carry_m",
        "sample_count",
        "successful_count",
        "no_impact_count",
        "failed_count",
        "failure_fraction",
        "confidence",
        "extrapolated",
        "pareto_efficient",
        "limiting_constraints",
    }
    assert "ball_speed=" in rows[0]["parameters"]
    assert "m/s" in rows[0]["parameters"]


def test_capability_result_json_uses_versioned_export_envelope() -> None:
    result, dataset = _output()

    payload = json.loads(capability_result_export_json(result, dataset))

    assert payload["schema_version"] == "capability-result-export/v1"
    alternative = payload["result"]["alternatives"][0]
    assert {
        "cvar_miss_m",
        "downside_carry_m",
        "sample_count",
        "successful_count",
        "no_impact_count",
        "failed_count",
        "failure_fraction",
        "extrapolated",
        "pareto_efficient",
    } <= alternative.keys()
    assert {"parameter_id": "ball_speed", "unit": "m/s"} in payload["parameter_units"]

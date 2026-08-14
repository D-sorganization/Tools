"""Cross-runtime contracts for the persisted variation-study specification."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.workspace_variation_session import (
    LegacyVariationMigrationRequired,
    VariationAnalysisExecution,
    migrate_legacy_variation_fallback,
    variation_workspace_from_payload,
    variation_workspace_to_payload,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan

_FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/workspace_variation_parity.json"
)


def _raw_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _plan(raw: dict[str, object]) -> VariationPlan:
    value = raw["plan"]
    assert isinstance(value, dict)
    return VariationPlan.from_json_dict(value)


def test_cross_runtime_fixture_round_trips_the_authored_specification() -> None:
    raw = _raw_fixture()

    state = variation_workspace_from_payload(raw["selection"], _plan(raw))

    assert state.analysis_execution is VariationAnalysisExecution.BOTH
    assert state.selected_output_metrics == ("carry_m", "lateral_m", "apex_m")
    assert state.plan.n_runs == 300
    assert state.plan.seed == 42
    assert variation_workspace_to_payload(state) == raw["selection"]
    assert state.plan.to_json_dict() == raw["plan"]


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("schema_version",), 2, "unsupported"),
        (("data", "analysis_execution"), "parallel", "execution"),
        (("data", "selected_output_metrics"), ["unknown_metric"], "metric"),
        (("data", "selected_output_metrics"), ["carry_m", "carry_m"], "unique"),
    ],
)
def test_selection_rejects_schema_execution_and_metric_corruption(
    path: tuple[str, ...], value: object, match: str
) -> None:
    raw = _raw_fixture()
    selection = deepcopy(raw["selection"])
    target = selection
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError), match=match):
        variation_workspace_from_payload(selection, _plan(raw))


def test_legacy_migration_requires_the_same_root_plan_or_no_root_plan() -> None:
    raw = _raw_fixture()
    state = variation_workspace_from_payload(raw["selection"], _plan(raw))

    assert migrate_legacy_variation_fallback(state, None) == state
    assert migrate_legacy_variation_fallback(state, state.plan) == state

    conflicting = replace(state.plan, seed=state.plan.seed + 1)
    with pytest.raises(LegacyVariationMigrationRequired, match="conflicts"):
        migrate_legacy_variation_fallback(state, conflicting)


def test_state_rejects_output_metrics_from_another_pipeline() -> None:
    raw = _raw_fixture()
    launch = _plan(raw)
    delivery = VariationPlan(
        mode="delivery",
        noise=(NoiseSpec("swing_sim.impact.delivery.face_angle_deg", scale=1.0),),
    )
    state = variation_workspace_from_payload(raw["selection"], launch)

    with pytest.raises(ValueError, match="metric"):
        replace(state, plan=delivery, selected_output_metrics=("candidate_time_s",))

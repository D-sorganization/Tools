from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.variation_visual_state import (
    VariationVisualEvent,
    parse_variation_visual_state_matrix,
    simulation_authority_identity,
    variation_visual_state,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan
from shared.python.swing_sim.variation import registry as variation_registry

_FIXTURE = Path(
    "src/rate_of_closure/web/src/model/__fixtures__/"
    "variation_visual_state_matrix_v1.json"
)


def _plan() -> VariationPlan:
    return VariationPlan(
        mode="launch",
        noise=(NoiseSpec("swing_sim.flight.launch.ball_speed_mph", "normal", 1.0),),
        n_runs=3,
        seed=7,
    )


def test_python_consumes_every_strict_shared_visual_transition() -> None:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    matrix = parse_variation_visual_state_matrix(document)
    assert len(matrix) == len(VariationVisualEvent)
    for row in document["states"]:
        state = variation_visual_state(VariationVisualEvent(row["event"]))
        assert state.phase.value == row["phase"]
        assert state.visual_origin.value == row["visual_origin"]
        assert state.announcement_role.value == row["announcement_role"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.__setitem__("schema_version", True),
        lambda value: value["states"][0].update(phase="busy"),
        lambda value: value["states"].append(value["states"][0]),
    ],
)
def test_visual_state_matrix_rejects_unknown_coercive_and_duplicate_data(
    mutation,
) -> None:  # type: ignore[no-untyped-def]
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    mutation(document)
    with pytest.raises((TypeError, ValueError)):
        parse_variation_visual_state_matrix(document)


def test_complete_simulation_authority_identity_changes_for_nested_fields() -> None:
    plan = _plan()
    config = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=100.0),
        club=get_club("Driver 10.5°"),
        source_kind="double_pendulum",
    )
    baseline = simulation_authority_identity(plan, config, "both")
    assert baseline == simulation_authority_identity(plan, config, "both")
    assert baseline != simulation_authority_identity(
        replace(plan, seed=8), config, "both"
    )
    assert baseline != simulation_authority_identity(plan, config, "all_together")
    assert baseline != simulation_authority_identity(
        plan,
        replace(config, scenario=replace(config.scenario, clubhead_speed_mph=101.0)),
        "both",
    )
    assert baseline != simulation_authority_identity(
        plan,
        replace(
            config,
            swing_run_config=replace(
                config.swing_run_config,
                joint_locks=replace(
                    config.swing_run_config.joint_locks,
                    locked_joint_ids=("joint.shoulder",),
                ),
            ),
        ),
        "both",
    )


def test_execution_identity_snapshots_resolved_registry_defaults(monkeypatch) -> None:
    plan = _plan()
    config = SimulationConfig(
        scenario=ImpactScenario(clubhead_speed_mph=100.0),
        club=get_club("Driver 10.5°"),
    )
    baseline = simulation_authority_identity(plan, config, "all_together")
    key = plan.noise[0].variable_key
    monkeypatch.setitem(
        variation_registry._REGISTRY,
        key,
        replace(variation_registry._REGISTRY[key], default=101.0),
    )

    assert simulation_authority_identity(plan, config, "all_together") != baseline

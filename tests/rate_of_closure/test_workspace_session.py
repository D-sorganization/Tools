"""Live explorer-state adapters for whole-workspace files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rate_of_closure.application.capability_workflow import (
    CapabilityWorkflowInputs,
    build_capability_workflow,
)
from rate_of_closure.application.workspace_session import (
    CANONICAL_MODULE_IDS,
    ExplorerWorkspaceState,
    LegacyCapabilityMigrationRequired,
    WorkspaceSessionMetadata,
    document_from_state,
    state_from_document,
)
from rate_of_closure.application.workspace_simulation_session import (
    LegacySimulationMigrationRequired,
    SimulationWorkspaceState,
)
from rate_of_closure.application.workspace_torque_session import (
    LegacyTorqueMigrationRequired,
    TorqueWorkspaceState,
)
from rate_of_closure.application.workspace_variation_session import (
    LegacyVariationMigrationRequired,
    VariationAnalysisExecution,
    VariationWorkspaceState,
)
from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.view_workspace import ViewWorkspace
from shared.python.swing_sim.ball_setup import (
    DEFAULT_DRIVER_TEE_HEIGHT_M,
    BallSetup,
    BallSupportMode,
)
from shared.python.swing_sim.run_config import DoublePendulumRunConfig
from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    TargetPoint,
)
from shared.python.swing_sim.torque_profiles import PrescribedTorqueProfile
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan


def _state() -> ExplorerWorkspaceState:
    fixture_path = (
        Path(__file__).parents[2]
        / "src/rate_of_closure/web/src/model/__fixtures__/torque_profile_parity.json"
    )
    profile = PrescribedTorqueProfile.loads(fixture_path.read_text(encoding="utf-8"))
    return ExplorerWorkspaceState(
        scenario=ImpactScenario(clubhead_speed_mph=111.0, omega_shaft_dps=-900.0),
        club=get_club("Driver 10.5°"),
        units={
            "speed": "mph",
            "rotation": "deg/s",
            "length": "mm",
            "distance": "yd",
        },
        simulation=SimulationWorkspaceState(
            ball_setup=BallSetup(
                BallSupportMode.TEE,
                DEFAULT_DRIVER_TEE_HEIGHT_M,
            ),
            ball_setup_user_overridden=False,
            spatial_target=SpatialTarget(
                label="Apex gate",
                kind="aerial_waypoint",
                point=TargetPoint.from_frame((137.5, 3.25, 24.25), "flight"),
                tolerance=BoxTolerance((4.5, 2.5, 3.5)),
                elevation_source="absolute",
            ),
        ),
        torque=TorqueWorkspaceState(
            profiles=(profile,),
            active_profile_id=profile.profile_id,
            run_config=DoublePendulumRunConfig.prescribed(profile.profile_id),
        ),
        variation=VariationWorkspaceState(
            plan=VariationPlan(
                mode="launch",
                noise=(
                    NoiseSpec(
                        "swing_sim.flight.launch.ball_speed_mph",
                        scale=1.0,
                    ),
                ),
                n_runs=24,
                seed=7,
            ),
            analysis_execution=VariationAnalysisExecution.BOTH,
            selected_output_metrics=("carry_m", "lateral_m"),
        ),
        capability=build_capability_workflow(
            CapabilityWorkflowInputs(
                profile_id="workspace-profile",
                objective="minimize_expected_miss",
                target_distance_m=241.0,
                target_lateral_m=-4.0,
                spin_axis_tilt_deg=-3.5,
            )
        ),
        # Track the canonical registry rather than a copy of it: the module set
        # grows whenever both clients gain a view (Ground Surfaces and Ground
        # Playback arrived with the ground families).
        module_order=CANONICAL_MODULE_IDS,
        visible_module_ids=("explorer", "simulation"),
        active_module_id="simulation",
        view_workspace=ViewWorkspace.default(),
    )


def _metadata() -> WorkspaceSessionMetadata:
    return WorkspaceSessionMetadata(
        document_id="workspace.session.test",
        title="Session test",
        created_at_utc="2026-08-10T12:00:00Z",
        modified_at_utc="2026-08-10T12:01:00Z",
        app_version="1.14.30",
    )


def test_live_state_round_trips_through_strict_whole_workspace_document() -> None:
    state = _state()

    restored = state_from_document(document_from_state(state, _metadata()))

    assert restored == state

    payload = document_from_state(state, _metadata()).model_session
    assert payload.schema_version == 5
    setup = payload.to_json_dict()["data"]["simulation_setup"]
    assert setup["schema"] == "rate_of_closure.simulation_setup"
    assert setup["data"]["ball_setup"]["provenance"] == {
        "kind": "club_default",
        "club_name": "Driver 10.5°",
    }
    assert setup["data"]["spatial_target"]["source_frame"] == "flight"
    assert setup["data"]["spatial_target"]["tolerance"] == {
        "kind": "box",
        "half_extents_m": {"x": 4.5, "elevation": 2.5, "right": 3.5},
    }
    torque = payload.to_json_dict()["data"]["torque_selection"]
    assert torque["schema"] == "rate_of_closure.torque_workspace_selection"
    assert torque["data"]["active_profile_id"] == "profile.web_parity.v1"
    assert document_from_state(state, _metadata()).prescribed_torque_profiles == (
        state.torque.profiles
    )
    variation = payload.to_json_dict()["data"]["variation_study"]
    assert variation["schema"] == "rate_of_closure.variation_workspace_selection"
    assert variation["data"]["analysis_execution"] == "both"
    assert (
        document_from_state(state, _metadata()).variation_plan == state.variation.plan
    )
    capability = payload.to_json_dict()["data"]["capability_request"]
    assert capability["schema_version"] == "capability-optimization-workflow/v1"
    assert capability["request"]["objective"] == "minimize_expected_miss"
    assert capability["request"]["target"]["lateral_m"] == -4.0
    assert "result" not in capability


def test_legacy_v4_requires_explicit_capability_request_fallback() -> None:
    from rate_of_closure.application.workspace_document import VersionedPayload

    current = document_from_state(_state(), _metadata())
    data = current.model_session.to_json_dict()["data"]
    legacy = replace(
        current,
        model_session=VersionedPayload(
            current.model_session.schema,
            4,
            {key: value for key, value in data.items() if key != "capability_request"},
        ),
    )

    with pytest.raises(LegacyCapabilityMigrationRequired, match="explicit"):
        state_from_document(legacy)

    restored = state_from_document(
        legacy,
        legacy_capability_fallback=_state().capability,
    )
    assert restored.capability == _state().capability


def test_workspace_rejects_corrupt_capability_before_returning_state() -> None:
    from rate_of_closure.application.workspace_document import WorkspaceDocument

    raw = document_from_state(_state(), _metadata()).to_json_dict()
    raw["model_session"]["data"]["capability_request"]["computed_result"] = {}

    with pytest.raises(ValueError, match="capability workflow"):
        state_from_document(WorkspaceDocument.from_json_dict(raw))


def test_club_default_provenance_must_match_persisted_club_and_geometry() -> None:
    raw = document_from_state(_state(), _metadata()).to_json_dict()
    ball_setup = raw["model_session"]["data"]["simulation_setup"]["data"]["ball_setup"]
    ball_setup["setup"]["tee_height_m"] = 0.05
    ball_setup["setup"]["ball_center_m"][1] = 0.05 + 0.04267 / 2.0

    from rate_of_closure.application.workspace_document import WorkspaceDocument

    with pytest.raises(ValueError, match="club-default ball setup"):
        state_from_document(WorkspaceDocument.from_json_dict(raw))


def test_tee_height_variation_requires_tee_support_context() -> None:
    state = _state()
    plan = VariationPlan(
        mode="swing",
        noise=(NoiseSpec("swing_sim.ball_setup.tee_height_m", scale=0.002),),
        n_runs=24,
        seed=7,
    )
    variation = VariationWorkspaceState(
        plan=plan,
        analysis_execution=VariationAnalysisExecution.BOTH,
        selected_output_metrics=("carry_m",),
    )

    with pytest.raises(ValueError, match="tee-height variation requires Tee"):
        replace(
            state,
            simulation=replace(
                state.simulation,
                ball_setup=BallSetup(BallSupportMode.GROUND, 0.0),
                ball_setup_user_overridden=True,
            ),
            variation=variation,
        )


def test_legacy_v1_session_requires_and_uses_an_explicit_simulation_fallback() -> None:
    from rate_of_closure.application.workspace_document import VersionedPayload

    current = document_from_state(_state(), _metadata())
    current_data = current.model_session.to_json_dict()["data"]
    legacy = replace(
        current,
        model_session=VersionedPayload(
            current.model_session.schema,
            1,
            {"scenario": current_data["scenario"], "units": current_data["units"]},
        ),
    )

    with pytest.raises(LegacySimulationMigrationRequired, match="explicit"):
        state_from_document(legacy)

    migrated = state_from_document(
        legacy,
        legacy_simulation_fallback=_state().simulation,
        legacy_torque_fallback=_state().torque,
        legacy_variation_fallback=_state().variation,
        legacy_capability_fallback=_state().capability,
    )
    assert migrated.simulation == _state().simulation


def test_legacy_cross_club_fallback_preserves_geometry_as_an_override() -> None:
    from rate_of_closure.application.workspace_document import VersionedPayload

    iron_state = replace(
        _state(),
        club=get_club("7-Iron"),
        simulation=replace(
            _state().simulation,
            ball_setup=BallSetup(BallSupportMode.GROUND, 0.0),
        ),
    )
    current = document_from_state(iron_state, _metadata())
    current_data = current.model_session.to_json_dict()["data"]
    legacy = replace(
        current,
        model_session=VersionedPayload(
            current.model_session.schema,
            1,
            {"scenario": current_data["scenario"], "units": current_data["units"]},
        ),
    )

    migrated = state_from_document(
        legacy,
        legacy_simulation_fallback=_state().simulation,
        legacy_torque_fallback=_state().torque,
        legacy_variation_fallback=_state().variation,
        legacy_capability_fallback=_state().capability,
    )
    assert migrated.simulation.ball_setup == _state().simulation.ball_setup
    assert migrated.simulation.ball_setup_user_overridden


def test_legacy_v2_requires_explicit_torque_fallback_without_inventing_profiles() -> (
    None
):
    from rate_of_closure.application.workspace_document import VersionedPayload

    current = document_from_state(_state(), _metadata())
    data = current.model_session.to_json_dict()["data"]
    legacy = replace(
        current,
        model_session=VersionedPayload(
            current.model_session.schema,
            2,
            {
                "scenario": data["scenario"],
                "units": data["units"],
                "simulation_setup": data["simulation_setup"],
            },
        ),
        prescribed_torque_profiles=(),
    )

    with pytest.raises(LegacyTorqueMigrationRequired, match="explicit"):
        state_from_document(legacy)

    restored = state_from_document(
        legacy,
        legacy_torque_fallback=_state().torque,
        legacy_variation_fallback=_state().variation,
        legacy_capability_fallback=_state().capability,
    )
    assert restored.torque == _state().torque


@pytest.mark.parametrize(
    ("field", "value"),
    [("torque_unit", "lbf*ft"), ("coefficient_order", "descending")],
)
def test_workspace_rejects_noncanonical_torque_units_and_coefficient_order(
    field: str, value: str
) -> None:
    from rate_of_closure.application.workspace_document import WorkspaceDocument

    raw = document_from_state(_state(), _metadata()).to_json_dict()
    raw["prescribed_torque_profiles"][0][field] = value

    with pytest.raises(ValueError, match=field):
        WorkspaceDocument.from_json_dict(raw)


def test_legacy_v3_variation_migration_requires_a_nonconflicting_fallback() -> None:
    from rate_of_closure.application.workspace_document import VersionedPayload

    current = document_from_state(_state(), _metadata())
    data = current.model_session.to_json_dict()["data"]
    legacy = replace(
        current,
        model_session=VersionedPayload(
            current.model_session.schema,
            3,
            {
                "scenario": data["scenario"],
                "units": data["units"],
                "simulation_setup": data["simulation_setup"],
                "torque_selection": data["torque_selection"],
            },
        ),
    )

    with pytest.raises(LegacyVariationMigrationRequired, match="explicit"):
        state_from_document(legacy)

    assert (
        state_from_document(
            legacy,
            legacy_variation_fallback=_state().variation,
            legacy_capability_fallback=_state().capability,
        ).variation
        == _state().variation
    )

    conflicting = replace(
        _state().variation, plan=replace(_state().variation.plan, seed=8)
    )
    with pytest.raises(LegacyVariationMigrationRequired, match="conflicts"):
        state_from_document(
            legacy,
            legacy_variation_fallback=conflicting,
            legacy_capability_fallback=_state().capability,
        )


def test_state_contract_rejects_invalid_units_and_incomplete_module_registry() -> None:
    with pytest.raises(ValueError, match="units"):
        replace(_state(), units={**_state().units, "speed": "furlong/fortnight"})
    with pytest.raises(ValueError, match="module_order"):
        replace(_state(), module_order=("explorer",))

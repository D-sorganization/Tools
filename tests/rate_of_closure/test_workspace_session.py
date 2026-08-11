"""Live explorer-state adapters for whole-workspace files."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.application.workspace_session import (
    ExplorerWorkspaceState,
    WorkspaceSessionMetadata,
    document_from_state,
    state_from_document,
)
from rate_of_closure.application.workspace_simulation_session import (
    LegacySimulationMigrationRequired,
    SimulationWorkspaceState,
)
from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.view_workspace import ViewWorkspace
from shared.python.swing_sim.ball_setup import (
    DEFAULT_DRIVER_TEE_HEIGHT_M,
    BallSetup,
    BallSupportMode,
)
from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    TargetPoint,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan


def _state() -> ExplorerWorkspaceState:
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
        module_order=(
            "explorer",
            "calculation",
            "simulation",
            "plots",
            "flight",
            "launch-monitor-analytics",
            "capability-optimization",
            "variation",
            "putting",
            "glossary",
        ),
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
    assert payload.schema_version == 2
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


def test_club_default_provenance_must_match_persisted_club_and_geometry() -> None:
    raw = document_from_state(_state(), _metadata()).to_json_dict()
    ball_setup = raw["model_session"]["data"]["simulation_setup"]["data"]["ball_setup"]
    ball_setup["setup"]["tee_height_m"] = 0.05
    ball_setup["setup"]["ball_center_m"][1] = 0.05 + 0.04267 / 2.0

    from rate_of_closure.application.workspace_document import WorkspaceDocument

    with pytest.raises(ValueError, match="club-default ball setup"):
        state_from_document(WorkspaceDocument.from_json_dict(raw))


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
    )
    assert migrated.simulation.ball_setup == _state().simulation.ball_setup
    assert migrated.simulation.ball_setup_user_overridden


def test_unsupported_domain_state_is_rejected_before_any_ui_mutation() -> None:
    document = document_from_state(_state(), _metadata())
    document = replace(
        document,
        variation_plan=VariationPlan(
            mode="delivery",
            noise=(NoiseSpec("swing_sim.impact.delivery.face_angle_deg", scale=1.0),),
        ),
    )

    with pytest.raises(TypeError, match="variation"):
        state_from_document(document)


def test_state_contract_rejects_invalid_units_and_incomplete_module_registry() -> None:
    with pytest.raises(ValueError, match="units"):
        replace(_state(), units={**_state().units, "speed": "furlong/fortnight"})
    with pytest.raises(ValueError, match="module_order"):
        replace(_state(), module_order=("explorer",))

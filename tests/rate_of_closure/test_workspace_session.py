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
from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.view_workspace import ViewWorkspace
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

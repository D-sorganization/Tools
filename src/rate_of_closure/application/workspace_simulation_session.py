"""Versioned persistence for ball support and the spatial target."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rate_of_closure.club import ClubSpec, ClubType
from shared.python.swing_sim.ball_setup import (
    DEFAULT_DRIVER_TEE_HEIGHT_M,
    BallSetup,
    BallSupportMode,
)
from shared.python.swing_sim.solver import (
    SpatialTarget,
    spatial_target_from_json_dict,
    spatial_target_to_json_dict,
)

SIMULATION_SETUP_SCHEMA = "rate_of_closure.simulation_setup"
SIMULATION_SETUP_SCHEMA_VERSION = 1
BALL_SETUP_SELECTION_SCHEMA = "swing_sim.ball_setup_selection"
BALL_SETUP_SELECTION_SCHEMA_VERSION = 1

_SIMULATION_ENVELOPE_FIELDS = frozenset({"schema", "schema_version", "data"})
_SIMULATION_DATA_FIELDS = frozenset({"ball_setup", "spatial_target"})
_BALL_SELECTION_FIELDS = frozenset({"schema", "schema_version", "setup", "provenance"})
_BALL_SETUP_FIELDS = frozenset(
    {"support_mode", "tee_height_m", "height_reference", "ball_center_m"}
)
_PROVENANCE_FIELDS = frozenset({"kind", "club_name"})


class LegacySimulationMigrationRequired(ValueError):
    """A legacy explorer session needs an explicit live-state fallback."""


@dataclass(frozen=True)
class SimulationWorkspaceState:
    """Complete simulation state currently owned by both application clients."""

    ball_setup: BallSetup
    ball_setup_user_overridden: bool
    spatial_target: SpatialTarget

    def __post_init__(self) -> None:
        """Reject incomplete state before it reaches a UI adapter."""
        if not isinstance(self.ball_setup, BallSetup):
            raise TypeError("ball_setup must be a BallSetup")
        if not isinstance(self.ball_setup_user_overridden, bool):
            raise TypeError("ball_setup_user_overridden must be a bool")
        if not isinstance(self.spatial_target, SpatialTarget):
            raise TypeError("spatial_target must be a SpatialTarget")


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TypeError(f"{context} has invalid fields")
    return value


def _club_default_ball_setup(club: ClubSpec) -> BallSetup:
    if club.club_type is ClubType.DRIVER:
        return BallSetup(BallSupportMode.TEE, DEFAULT_DRIVER_TEE_HEIGHT_M)
    return BallSetup(BallSupportMode.GROUND, 0.0)


def validate_simulation_workspace(
    state: SimulationWorkspaceState,
    club: ClubSpec,
) -> None:
    """Validate cross-field provenance against the persisted club identity."""
    if not isinstance(state, SimulationWorkspaceState):
        raise TypeError("simulation must be a SimulationWorkspaceState")
    if not isinstance(club, ClubSpec):
        raise TypeError("club must be a ClubSpec")
    if (
        not state.ball_setup_user_overridden
        and state.ball_setup != _club_default_ball_setup(club)
    ):
        raise ValueError("club-default ball setup does not match the persisted club")


def migrate_legacy_simulation_fallback(
    state: SimulationWorkspaceState,
    club: ClubSpec,
) -> SimulationWorkspaceState:
    """Preserve legacy live values, making mismatched defaults explicit.

    A v1 file has no ball/target authority.  The caller supplies the current
    live state deliberately.  If that setup was a default for a different
    current club, its unchanged geometry becomes an explicit override instead
    of being relabelled as the newly loaded club's default.
    """
    if not isinstance(state, SimulationWorkspaceState):
        raise TypeError("legacy simulation fallback must be complete")
    migrated = state
    if (
        not state.ball_setup_user_overridden
        and state.ball_setup != _club_default_ball_setup(club)
    ):
        migrated = SimulationWorkspaceState(
            state.ball_setup,
            True,
            state.spatial_target,
        )
    validate_simulation_workspace(migrated, club)
    return migrated


def _ball_setup_document(
    state: SimulationWorkspaceState,
    club: ClubSpec,
) -> dict[str, object]:
    validate_simulation_workspace(state, club)
    overridden = state.ball_setup_user_overridden
    return {
        "schema": BALL_SETUP_SELECTION_SCHEMA,
        "schema_version": BALL_SETUP_SELECTION_SCHEMA_VERSION,
        "setup": state.ball_setup.to_json_dict(),
        "provenance": {
            "kind": "explicit_override" if overridden else "club_default",
            "club_name": None if overridden else club.name,
        },
    }


def simulation_workspace_to_payload(
    state: SimulationWorkspaceState,
    club: ClubSpec,
) -> dict[str, object]:
    """Serialize a strict simulation subpayload with source provenance."""
    return {
        "schema": SIMULATION_SETUP_SCHEMA,
        "schema_version": SIMULATION_SETUP_SCHEMA_VERSION,
        "data": {
            "ball_setup": _ball_setup_document(state, club),
            "spatial_target": spatial_target_to_json_dict(state.spatial_target),
        },
    }


def _ball_setup_from_document(
    value: object,
    club: ClubSpec,
) -> tuple[BallSetup, bool]:
    selection = _exact_mapping(value, _BALL_SELECTION_FIELDS, "ball_setup")
    if (
        selection["schema"] != BALL_SETUP_SELECTION_SCHEMA
        or selection["schema_version"] != BALL_SETUP_SELECTION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported ball setup selection payload")
    setup_data = _exact_mapping(
        selection["setup"], _BALL_SETUP_FIELDS, "ball_setup.setup"
    )
    setup = BallSetup.from_json_dict(setup_data)
    provenance = _exact_mapping(
        selection["provenance"], _PROVENANCE_FIELDS, "ball_setup.provenance"
    )
    kind = provenance["kind"]
    club_name = provenance["club_name"]
    if kind == "club_default":
        if club_name != club.name or setup != _club_default_ball_setup(club):
            raise ValueError(
                "club-default ball setup does not match the persisted club"
            )
        return setup, False
    if kind == "explicit_override":
        if club_name is not None:
            raise ValueError(
                "explicit ball setup provenance cannot name a default club"
            )
        return setup, True
    raise ValueError(f"unknown ball setup provenance {kind!r}")


def simulation_workspace_from_payload(
    value: object,
    club: ClubSpec,
) -> SimulationWorkspaceState:
    """Parse and validate the complete versioned simulation subpayload."""
    envelope = _exact_mapping(value, _SIMULATION_ENVELOPE_FIELDS, "simulation_setup")
    if (
        envelope["schema"] != SIMULATION_SETUP_SCHEMA
        or envelope["schema_version"] != SIMULATION_SETUP_SCHEMA_VERSION
    ):
        raise ValueError("unsupported simulation setup payload")
    data = _exact_mapping(
        envelope["data"], _SIMULATION_DATA_FIELDS, "simulation_setup.data"
    )
    ball_setup, overridden = _ball_setup_from_document(data["ball_setup"], club)
    target = spatial_target_from_json_dict(data["spatial_target"])
    state = SimulationWorkspaceState(ball_setup, overridden, target)
    validate_simulation_workspace(state, club)
    return state


__all__ = [
    "BALL_SETUP_SELECTION_SCHEMA",
    "BALL_SETUP_SELECTION_SCHEMA_VERSION",
    "LegacySimulationMigrationRequired",
    "SIMULATION_SETUP_SCHEMA",
    "SIMULATION_SETUP_SCHEMA_VERSION",
    "SimulationWorkspaceState",
    "simulation_workspace_from_payload",
    "simulation_workspace_to_payload",
    "migrate_legacy_simulation_fallback",
    "validate_simulation_workspace",
]

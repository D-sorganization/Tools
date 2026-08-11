"""Strict adapters between live explorer state and workspace documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType
from typing import cast

from rate_of_closure.club import ClubSpec, ClubType, HeadStyle
from rate_of_closure.model import ImpactScenario
from rate_of_closure.units import QUANTITY_UNITS
from rate_of_closure.view_workspace import (
    FORMAT,
    FORMAT_V1,
    ViewWorkspace,
    workspace_from_document,
    workspace_to_document,
)
from shared.python.swing_sim.ball_setup import BallSupportMode

from ._workspace_validation import FrozenJsonValue
from .workspace_document import (
    VersionedPayload,
    WorkspaceDocument,
    WorkspaceLayout,
    WorkspaceMetadata,
)
from .workspace_simulation_session import (
    LegacySimulationMigrationRequired,
    SimulationWorkspaceState,
    migrate_legacy_simulation_fallback,
    simulation_workspace_from_payload,
    simulation_workspace_to_payload,
    validate_simulation_workspace,
)
from .workspace_torque_session import (
    LegacyTorqueMigrationRequired,
    TorqueWorkspaceState,
    migrate_legacy_torque_fallback,
    torque_workspace_from_payload,
    torque_workspace_to_payload,
)
from .workspace_variation_session import (
    LegacyVariationMigrationRequired,
    VariationWorkspaceState,
    migrate_legacy_variation_fallback,
    variation_workspace_from_payload,
    variation_workspace_to_payload,
)

_TEE_HEIGHT_VARIATION_KEY = "swing_sim.ball_setup.tee_height_m"

EXPLORER_SESSION_SCHEMA = "rate_of_closure.explorer_session"
CLUB_CONFIGURATION_SCHEMA = "rate_of_closure.club_configuration"
SESSION_SCHEMA_VERSION = 4
CLUB_CONFIGURATION_SCHEMA_VERSION = 1
CANONICAL_MODULE_IDS = (
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
)
_SCENARIO_FIELDS = frozenset(field.name for field in fields(ImpactScenario))
_CLUB_FIELDS = frozenset(
    {
        "name",
        "club_type",
        "length_m",
        "head_mass_kg",
        "loft_deg",
        "lie_deg",
        "moi_about_shaft_kg_m2",
        "cg_depth_m",
        "cg_height_m",
        "face_bulge_radius_m",
        "face_roll_radius_m",
        "head_style",
    }
)
_SESSION_V1_FIELDS = frozenset({"scenario", "units"})
_SESSION_V2_FIELDS = _SESSION_V1_FIELDS | {"simulation_setup"}
_SESSION_V3_FIELDS = _SESSION_V2_FIELDS | {"torque_selection"}
_SESSION_FIELDS = _SESSION_V3_FIELDS | {"variation_study"}


@dataclass(frozen=True)
class WorkspaceSessionMetadata:
    """Mutable-lifecycle metadata supplied by a file-session controller."""

    document_id: str
    title: str
    created_at_utc: str
    modified_at_utc: str
    app_version: str


@dataclass(frozen=True)
class ExplorerWorkspaceState:
    """The complete state currently supported by live File operations."""

    scenario: ImpactScenario
    club: ClubSpec
    units: Mapping[str, str]
    simulation: SimulationWorkspaceState
    torque: TorqueWorkspaceState
    variation: VariationWorkspaceState
    module_order: tuple[str, ...]
    visible_module_ids: tuple[str, ...]
    active_module_id: str
    view_workspace: ViewWorkspace

    def __post_init__(self) -> None:
        """Validate all values before a controller can mutate live widgets."""
        if not isinstance(self.scenario, ImpactScenario):
            raise TypeError("scenario must be an ImpactScenario")
        if not isinstance(self.club, ClubSpec):
            raise TypeError("club must be a ClubSpec")
        validate_simulation_workspace(self.simulation, self.club)
        if not isinstance(self.torque, TorqueWorkspaceState):
            raise TypeError("torque must be a TorqueWorkspaceState")
        if not isinstance(self.variation, VariationWorkspaceState):
            raise TypeError("variation must be a VariationWorkspaceState")
        varies_tee_height = any(
            spec.variable_key == _TEE_HEIGHT_VARIATION_KEY
            for spec in self.variation.plan.noise
        )
        if varies_tee_height and (
            self.simulation.ball_setup.support_mode is not BallSupportMode.TEE
        ):
            raise ValueError("tee-height variation requires Tee ball support")
        units = dict(self.units)
        if set(units) != set(QUANTITY_UNITS) or any(
            value not in QUANTITY_UNITS[key] for key, value in units.items()
        ):
            raise ValueError("units must select one declared unit per quantity")
        order = tuple(self.module_order)
        visible = tuple(self.visible_module_ids)
        if set(order) != set(CANONICAL_MODULE_IDS) or len(order) != len(
            CANONICAL_MODULE_IDS
        ):
            raise ValueError("module_order must contain every canonical module once")
        if (
            not visible
            or len(set(visible)) != len(visible)
            or not set(visible).issubset(order)
        ):
            raise ValueError("visible_module_ids must be a non-empty module subset")
        if "explorer" not in visible or self.active_module_id not in visible:
            raise ValueError("the explorer and active module must be visible")
        if not isinstance(self.view_workspace, ViewWorkspace):
            raise TypeError("view_workspace must be a ViewWorkspace")
        self.view_workspace.validate()
        object.__setattr__(self, "units", MappingProxyType(units))
        object.__setattr__(self, "module_order", order)
        object.__setattr__(self, "visible_module_ids", visible)


def _exact_mapping(
    value: object, fields: frozenset[str], context: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TypeError(f"{context} has invalid fields")
    return value


def _scenario_data(scenario: ImpactScenario) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(scenario).items()}


def _club_data(club: ClubSpec) -> dict[str, FrozenJsonValue]:
    data: dict[str, FrozenJsonValue] = asdict(club)
    data["club_type"] = club.club_type.value
    data["head_style"] = club.head_style.value
    return data


def _club_from_data(value: object) -> ClubSpec:
    data = dict(_exact_mapping(value, _CLUB_FIELDS, "club_configuration.data"))
    data["club_type"] = ClubType(data["club_type"])
    data["head_style"] = HeadStyle(data["head_style"])
    return ClubSpec(**data)


def document_from_state(
    state: ExplorerWorkspaceState, metadata: WorkspaceSessionMetadata
) -> WorkspaceDocument:
    """Compose one current strict document from fully validated live state."""
    if not isinstance(state, ExplorerWorkspaceState):
        raise TypeError("state must be an ExplorerWorkspaceState")
    if not isinstance(metadata, WorkspaceSessionMetadata):
        raise TypeError("metadata must be WorkspaceSessionMetadata")
    return WorkspaceDocument(
        metadata=WorkspaceMetadata(
            document_id=metadata.document_id,
            title=metadata.title,
            created_at_utc=metadata.created_at_utc,
            modified_at_utc=metadata.modified_at_utc,
            app_version=metadata.app_version,
            provenance={"surface": "rate-of-closure-file-adapter/v1"},
        ),
        model_session=VersionedPayload(
            EXPLORER_SESSION_SCHEMA,
            SESSION_SCHEMA_VERSION,
            {
                "scenario": _scenario_data(state.scenario),
                "units": dict(state.units),
                "simulation_setup": cast(
                    FrozenJsonValue,
                    simulation_workspace_to_payload(state.simulation, state.club),
                ),
                "torque_selection": cast(
                    FrozenJsonValue,
                    torque_workspace_to_payload(state.torque),
                ),
                "variation_study": cast(
                    FrozenJsonValue,
                    variation_workspace_to_payload(state.variation),
                ),
            },
        ),
        prescribed_torque_profiles=state.torque.profiles,
        club_configuration=VersionedPayload(
            CLUB_CONFIGURATION_SCHEMA,
            CLUB_CONFIGURATION_SCHEMA_VERSION,
            _club_data(state.club),
        ),
        variation_plan=state.variation.plan,
        layout=WorkspaceLayout(
            module_order=state.module_order,
            visible_module_ids=state.visible_module_ids,
            active_module_id=state.active_module_id,
            view_workspace=VersionedPayload(
                "rate_of_closure.view_workspace",
                2,
                workspace_to_document(state.view_workspace),
            ),
        ),
    )


def state_from_document(
    document: WorkspaceDocument,
    *,
    legacy_simulation_fallback: SimulationWorkspaceState | None = None,
    legacy_torque_fallback: TorqueWorkspaceState | None = None,
    legacy_variation_fallback: VariationWorkspaceState | None = None,
) -> ExplorerWorkspaceState:
    """Validate a supported whole document before returning applicable state."""
    if not isinstance(document, WorkspaceDocument):
        raise TypeError("document must be a WorkspaceDocument")
    session = document.model_session
    club = document.club_configuration
    if session.schema != EXPLORER_SESSION_SCHEMA or session.schema_version not in (
        1,
        2,
        3,
        SESSION_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported explorer session payload")
    if (club.schema, club.schema_version) != (
        CLUB_CONFIGURATION_SCHEMA,
        CLUB_CONFIGURATION_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported club configuration payload")
    parsed_club = _club_from_data(club.data)
    session_fields = {
        1: _SESSION_V1_FIELDS,
        2: _SESSION_V2_FIELDS,
        3: _SESSION_V3_FIELDS,
        SESSION_SCHEMA_VERSION: _SESSION_FIELDS,
    }[session.schema_version]
    session_data = _exact_mapping(session.data, session_fields, "model_session.data")
    scenario_data = _exact_mapping(
        session_data["scenario"], _SCENARIO_FIELDS, "model_session.scenario"
    )
    units = session_data["units"]
    if not isinstance(units, Mapping):
        raise TypeError("model_session.units must be an object")
    view = document.layout.view_workspace
    if (
        view is None
        or view.schema != "rate_of_closure.view_workspace"
        or (
            view.schema_version,
            view.data.get("format"),
        )
        not in {(1, FORMAT_V1), (2, FORMAT)}
    ):
        raise ValueError("workspace requires a supported compositor payload")
    if session.schema_version == 1:
        if legacy_simulation_fallback is None:
            raise LegacySimulationMigrationRequired(
                "model_session v1 omitted ball setup and spatial target; "
                "an explicit simulation migration fallback is required"
            )
        simulation = migrate_legacy_simulation_fallback(
            legacy_simulation_fallback, parsed_club
        )
    else:
        simulation = simulation_workspace_from_payload(
            session_data["simulation_setup"], parsed_club
        )
    if session.schema_version < 3:
        if legacy_torque_fallback is None:
            raise LegacyTorqueMigrationRequired(
                "legacy model_session omitted torque selection; "
                "an explicit torque migration fallback is required"
            )
        torque = migrate_legacy_torque_fallback(
            legacy_torque_fallback,
            document.prescribed_torque_profiles,
        )
    else:
        torque = torque_workspace_from_payload(
            session_data["torque_selection"],
            document.prescribed_torque_profiles,
        )
    if session.schema_version < SESSION_SCHEMA_VERSION:
        if legacy_variation_fallback is None:
            raise LegacyVariationMigrationRequired(
                "legacy model_session omitted variation selection; "
                "an explicit variation migration fallback is required"
            )
        variation = migrate_legacy_variation_fallback(
            legacy_variation_fallback,
            document.variation_plan,
        )
    else:
        if document.variation_plan is None:
            raise ValueError("current workspace requires a canonical variation plan")
        variation = variation_workspace_from_payload(
            session_data["variation_study"],
            document.variation_plan,
        )
    return ExplorerWorkspaceState(
        scenario=ImpactScenario(**scenario_data),
        club=parsed_club,
        units=units,
        simulation=simulation,
        torque=torque,
        variation=variation,
        module_order=document.layout.module_order,
        visible_module_ids=document.layout.visible_module_ids,
        active_module_id=document.layout.active_module_id,
        view_workspace=workspace_from_document(view.to_json_dict()["data"]),
    )


__all__ = [
    "CANONICAL_MODULE_IDS",
    "CLUB_CONFIGURATION_SCHEMA",
    "CLUB_CONFIGURATION_SCHEMA_VERSION",
    "EXPLORER_SESSION_SCHEMA",
    "ExplorerWorkspaceState",
    "LegacySimulationMigrationRequired",
    "LegacyTorqueMigrationRequired",
    "LegacyVariationMigrationRequired",
    "SimulationWorkspaceState",
    "TorqueWorkspaceState",
    "VariationWorkspaceState",
    "WorkspaceSessionMetadata",
    "document_from_state",
    "state_from_document",
]

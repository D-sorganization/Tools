"""Strict adapters between live explorer state and workspace documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

from rate_of_closure.club import ClubSpec, ClubType, HeadStyle
from rate_of_closure.model import ImpactScenario
from rate_of_closure.units import QUANTITY_UNITS
from rate_of_closure.view_workspace import (
    ViewWorkspace,
    workspace_from_document,
    workspace_to_document,
)

from ._workspace_validation import FrozenJsonValue
from .workspace_document import (
    VersionedPayload,
    WorkspaceDocument,
    WorkspaceLayout,
    WorkspaceMetadata,
)

EXPLORER_SESSION_SCHEMA = "rate_of_closure.explorer_session"
CLUB_CONFIGURATION_SCHEMA = "rate_of_closure.club_configuration"
SESSION_SCHEMA_VERSION = 1
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
_SESSION_FIELDS = frozenset({"scenario", "units"})


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
            {"scenario": _scenario_data(state.scenario), "units": dict(state.units)},
        ),
        prescribed_torque_profiles=(),
        club_configuration=VersionedPayload(
            CLUB_CONFIGURATION_SCHEMA,
            SESSION_SCHEMA_VERSION,
            _club_data(state.club),
        ),
        variation_plan=None,
        layout=WorkspaceLayout(
            module_order=state.module_order,
            visible_module_ids=state.visible_module_ids,
            active_module_id=state.active_module_id,
            view_workspace=VersionedPayload(
                "rate_of_closure.view_workspace",
                1,
                workspace_to_document(state.view_workspace),
            ),
        ),
    )


def state_from_document(document: WorkspaceDocument) -> ExplorerWorkspaceState:
    """Validate a supported whole document before returning applicable state."""
    if not isinstance(document, WorkspaceDocument):
        raise TypeError("document must be a WorkspaceDocument")
    if document.prescribed_torque_profiles:
        raise ValueError("prescribed torque profiles are not supported by this adapter")
    if document.variation_plan is not None:
        raise TypeError("variation plans are not supported by this adapter")
    session = document.model_session
    club = document.club_configuration
    if (session.schema, session.schema_version) != (
        EXPLORER_SESSION_SCHEMA,
        SESSION_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported explorer session payload")
    if (club.schema, club.schema_version) != (
        CLUB_CONFIGURATION_SCHEMA,
        SESSION_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported club configuration payload")
    session_data = _exact_mapping(session.data, _SESSION_FIELDS, "model_session.data")
    scenario_data = _exact_mapping(
        session_data["scenario"], _SCENARIO_FIELDS, "model_session.scenario"
    )
    units = session_data["units"]
    if not isinstance(units, Mapping):
        raise TypeError("model_session.units must be an object")
    view = document.layout.view_workspace
    if view is None or (view.schema, view.schema_version) != (
        "rate_of_closure.view_workspace",
        1,
    ):
        raise ValueError("workspace requires a supported compositor payload")
    return ExplorerWorkspaceState(
        scenario=ImpactScenario(**scenario_data),
        club=_club_from_data(club.data),
        units=units,
        module_order=document.layout.module_order,
        visible_module_ids=document.layout.visible_module_ids,
        active_module_id=document.layout.active_module_id,
        view_workspace=workspace_from_document(view.to_json_dict()["data"]),
    )


__all__ = [
    "CANONICAL_MODULE_IDS",
    "CLUB_CONFIGURATION_SCHEMA",
    "EXPLORER_SESSION_SCHEMA",
    "ExplorerWorkspaceState",
    "WorkspaceSessionMetadata",
    "document_from_state",
    "state_from_document",
]

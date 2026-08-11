"""UI-neutral application contracts for Rate of Closure clients."""

from .commands import (
    APP_COMMAND_IDS,
    AppCommandId,
    CommandAvailability,
    CommandUnavailableError,
)
from .workspace_document import (
    WORKSPACE_SCHEMA,
    WORKSPACE_SCHEMA_VERSION,
    VersionedPayload,
    WorkspaceDocument,
    WorkspaceLayout,
    WorkspaceMetadata,
    workspace_from_json,
    workspace_to_json,
)
from .workspace_files import read_workspace, write_text_atomic, write_workspace_atomic
from .workspace_session import (
    ExplorerWorkspaceState,
    WorkspaceSessionMetadata,
    document_from_state,
    state_from_document,
)
from .workspace_simulation_session import (
    LegacySimulationMigrationRequired,
    SimulationWorkspaceState,
)
from .workspace_torque_session import (
    LegacyTorqueMigrationRequired,
    TorqueWorkspaceState,
)

__all__ = [
    "APP_COMMAND_IDS",
    "WORKSPACE_SCHEMA",
    "WORKSPACE_SCHEMA_VERSION",
    "AppCommandId",
    "CommandAvailability",
    "CommandUnavailableError",
    "ExplorerWorkspaceState",
    "LegacySimulationMigrationRequired",
    "LegacyTorqueMigrationRequired",
    "SimulationWorkspaceState",
    "TorqueWorkspaceState",
    "VersionedPayload",
    "WorkspaceDocument",
    "WorkspaceLayout",
    "WorkspaceMetadata",
    "WorkspaceSessionMetadata",
    "document_from_state",
    "read_workspace",
    "workspace_from_json",
    "workspace_to_json",
    "state_from_document",
    "write_text_atomic",
    "write_workspace_atomic",
]

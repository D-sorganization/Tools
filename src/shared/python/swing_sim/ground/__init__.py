"""Strict reusable flight-to-ground transfer and result contracts."""

from .bounce_kinematics import interpolate_first_contact
from .bounce_simulation import simulate_repeated_bounce
from .bounce_types import (
    BounceAirSegment,
    BounceModelSettings,
    BounceTermination,
    BounceTerminationReason,
    RepeatedBounceResult,
)
from .contract_records import GroundSimulationRequest, GroundSimulationResult
from .contract_types import (
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    UNIT_SYSTEM_SI,
    CalibrationKind,
    GroundCalibration,
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundFrame,
    GroundPhase,
    GroundProvenance,
    GroundResultStatus,
    GroundSurfaceProfile,
    GroundTerminationReason,
    GroundTrajectoryPoint,
    GroundWarningSeverity,
)
from .contract_wire import request_from_json, result_from_json
from .impact_impulse import resolve_sphere_plane_impact
from .impact_types import (
    ImpactEnergyLedger,
    ImpactImpulseResult,
    ImpactRegime,
    ImpactRejectionReason,
    ImpactStateError,
    SphereProperties,
)
from .json_schema import (
    JSON_SCHEMA_DIALECT,
    request_json_schema,
    result_json_schema,
    schema_json,
)
from .migration import migrate_request_to_current, migrate_result_to_current
from .result_adapter import to_ground_model_result
from .result_types import GroundSummary, GroundTermination, GroundWarning
from .unavailable_types import (
    GroundUnavailableField,
    GroundUnavailableFieldId,
    GroundUnavailableReason,
)

__all__ = [
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UNIT_SYSTEM_SI",
    "JSON_SCHEMA_DIALECT",
    "CalibrationKind",
    "BounceModelSettings",
    "BounceAirSegment",
    "BounceTermination",
    "BounceTerminationReason",
    "GroundCalibration",
    "GroundContactState",
    "GroundEvent",
    "GroundEventType",
    "GroundFrame",
    "GroundPhase",
    "GroundProvenance",
    "GroundResultStatus",
    "GroundSimulationRequest",
    "GroundSimulationResult",
    "GroundSummary",
    "GroundSurfaceProfile",
    "GroundTermination",
    "GroundTerminationReason",
    "GroundTrajectoryPoint",
    "GroundUnavailableField",
    "GroundUnavailableFieldId",
    "GroundUnavailableReason",
    "GroundWarning",
    "GroundWarningSeverity",
    "ImpactEnergyLedger",
    "ImpactImpulseResult",
    "ImpactRegime",
    "ImpactRejectionReason",
    "ImpactStateError",
    "SphereProperties",
    "RepeatedBounceResult",
    "interpolate_first_contact",
    "request_from_json",
    "request_json_schema",
    "result_from_json",
    "result_json_schema",
    "schema_json",
    "migrate_request_to_current",
    "migrate_result_to_current",
    "to_ground_model_result",
    "resolve_sphere_plane_impact",
    "simulate_repeated_bounce",
]

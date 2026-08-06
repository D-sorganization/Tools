"""Canonical UI-independent domain contracts for modular golf clubs.

All positions and lengths are metres, masses kilograms, and inertia tensors
kilogram-metres squared. Every transform declares both frames and maps local
component coordinates into the assembly frame.
"""

from .assembly import ClubAssembly, assemble_mass_properties
from .serialization import (
    CURRENT_FORMAT,
    LEGACY_FORMAT,
    assembly_from_json,
    assembly_from_json_dict,
    assembly_to_json,
    assembly_to_json_dict,
)
from .shaft_assembly import shaft_component_mass_properties
from .shaft_dynamics import (
    ShaftModalResponse,
    ShaftModalSettings,
    solve_shaft_bending_modes,
)
from .shaft_profile import (
    ExtrapolationPolicy,
    ShaftProfile,
    ShaftProfileProvenance,
    ShaftStation,
)
from .shaft_scaling import ShaftProfileScaling, scale_shaft_profile
from .shaft_serialization import (
    SHAFT_PROFILE_FORMAT,
    shaft_profile_from_csv,
    shaft_profile_from_json,
    shaft_profile_from_json_dict,
    shaft_profile_to_csv,
    shaft_profile_to_json,
    shaft_profile_to_json_dict,
)
from .shaft_statics import (
    ShaftTipLoad,
    ShaftTipResponse,
    solve_cantilever_tip_response,
)
from .types import (
    AssembledMassProperties,
    ClubComponent,
    ClubLengthConvention,
    ClubLengthMeasurement,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
)
from .wedge_cad import WedgeMeasuredMetrics, WedgeSolidResult, build_wedge_solid
from .wedge_export import (
    WEDGE_EXPORT_FORMAT,
    WedgeExportArtifact,
    WedgeExportFormat,
    WedgeExportRequest,
    WedgeExportResult,
    export_wedge_artifacts,
)
from .wedge_parameters import (
    Handedness,
    WedgeGeometryProvenance,
    WedgeHeadParameters,
    WedgePreset,
    wedge_preset,
)
from .wedge_serialization import (
    WEDGE_PARAMETERS_FORMAT,
    wedge_parameters_from_json,
    wedge_parameters_to_json,
)

__all__ = [
    "CURRENT_FORMAT",
    "LEGACY_FORMAT",
    "AssembledMassProperties",
    "ClubAssembly",
    "ClubComponent",
    "ClubLengthConvention",
    "ClubLengthMeasurement",
    "ComponentMassProperties",
    "ComponentRole",
    "ExtrapolationPolicy",
    "Handedness",
    "RigidTransform",
    "SHAFT_PROFILE_FORMAT",
    "ShaftProfile",
    "ShaftProfileProvenance",
    "ShaftProfileScaling",
    "ShaftStation",
    "ShaftModalResponse",
    "ShaftModalSettings",
    "ShaftTipLoad",
    "ShaftTipResponse",
    "WedgeHeadParameters",
    "WedgeGeometryProvenance",
    "WEDGE_EXPORT_FORMAT",
    "WedgeExportArtifact",
    "WedgeExportFormat",
    "WedgeExportRequest",
    "WedgeExportResult",
    "WedgeMeasuredMetrics",
    "WedgePreset",
    "WedgeSolidResult",
    "WEDGE_PARAMETERS_FORMAT",
    "assemble_mass_properties",
    "assembly_from_json",
    "assembly_from_json_dict",
    "assembly_to_json",
    "assembly_to_json_dict",
    "build_wedge_solid",
    "export_wedge_artifacts",
    "scale_shaft_profile",
    "shaft_component_mass_properties",
    "shaft_profile_from_csv",
    "shaft_profile_from_json",
    "shaft_profile_from_json_dict",
    "shaft_profile_to_csv",
    "shaft_profile_to_json",
    "shaft_profile_to_json_dict",
    "solve_shaft_bending_modes",
    "solve_cantilever_tip_response",
    "wedge_preset",
    "wedge_parameters_from_json",
    "wedge_parameters_to_json",
]

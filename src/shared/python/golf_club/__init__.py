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
from .types import (
    AssembledMassProperties,
    ClubComponent,
    ClubLengthConvention,
    ClubLengthMeasurement,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
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
    "RigidTransform",
    "assemble_mass_properties",
    "assembly_from_json",
    "assembly_from_json_dict",
    "assembly_to_json",
    "assembly_to_json_dict",
]

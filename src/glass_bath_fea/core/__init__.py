"""Core modules for Glass Bath FEA.

This package contains the configuration, geometry generation,
mesh generation, and material property modules.
"""

from .config import (
    INCHES_TO_METERS,
    GlassBathFEAConfig,
    GlassComposition,
    MeshConfig,
)
from .geometry_generator import GeometryGenerator
from .material_properties import (
    DEFAULT_METAL_CONDUCTIVITY,
    GAS_CONSTANT,
    GlassMaterialModel,
    export_material_data,
    get_metal_conductivity,
)
from .mesh_generator import MeshGenerator

__all__ = [
    "GlassBathFEAConfig",
    "GlassComposition",
    "MeshConfig",
    "INCHES_TO_METERS",
    "GeometryGenerator",
    "MeshGenerator",
    "GlassMaterialModel",
    "export_material_data",
    "get_metal_conductivity",
    "DEFAULT_METAL_CONDUCTIVITY",
    "GAS_CONSTANT",
]

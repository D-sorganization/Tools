"""Mesh processing utilities for model generation.

Re-exports mesh processing components from humanoid_character_builder.mesh.
"""

from shared.python.humanoid_character_builder.mesh import (
    CollisionGeometry,
    CollisionGeometryGenerator,
    InertiaMode,
    InertiaResult,
    LODGenerationResult,
    LODGenerator,
    LODLevel,
    MeshInertiaCalculator,
    MeshProcessor,
    MeshSegmentResult,
    PrimitiveInertiaCalculator,
    PrimitiveShape,
)

# Not re-exported by ``humanoid_character_builder.mesh.__all__`` — import from
# the defining module rather than widening that package's public surface.
from shared.python.humanoid_character_builder.mesh.mesh_processor import (
    MeshExportConfig,
    PrimitiveMeshGenerator,
)

__all__: list[str] = [
    "CollisionGeometry",
    "CollisionGeometryGenerator",
    "InertiaMode",
    "InertiaResult",
    "LODGenerationResult",
    "LODGenerator",
    "LODLevel",
    "MeshExportConfig",
    "MeshInertiaCalculator",
    "MeshProcessor",
    "MeshSegmentResult",
    "PrimitiveInertiaCalculator",
    "PrimitiveMeshGenerator",
    "PrimitiveShape",
]

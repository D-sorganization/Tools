"""Mesh processing utilities for model generation.

Re-exports mesh processing components from humanoid_character_builder.mesh.
"""

try:
    from humanoid_character_builder.mesh import (
        CollisionGeometry,
        CollisionGeometryGenerator,
        LODGenerationResult,
        LODGenerator,
        LODLevel,
        MeshExportConfig,
        MeshInertiaCalculator,
        MeshProcessor,
        MeshSegmentResult,
        PrimitiveInertiaCalculator,
        PrimitiveMeshGenerator,
        PrimitiveShape,
    )
    from humanoid_character_builder.mesh.lod import (
        LODGenerationResult,
        LODGenerator,
        LODLevel,
    )
    from humanoid_character_builder.mesh.mesh_inertia import (
        InertiaMode,
        InertiaResult,
        MeshInertiaCalculator,
        PrimitiveInertiaCalculator,
        PrimitiveShape,
    )
except ImportError:  # pragma: no cover
    pass

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

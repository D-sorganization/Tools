# ARCHITECTURE_DEBT — tracked as GitHub issue #1937
# This file is 1,173 lines and contains 4 distinct mesh generator backends
# (Primitive, MakeHuman, SMPLX) plus the public MeshGenerator facade.
# Recommended split:
#   mesh_generator_primitive.py  — PrimitiveMeshGenerator
#   mesh_generator_makehuman.py  — MakeHumanMeshGenerator
#   mesh_generator_smplx.py      — SMPLXMeshGenerator
#   mesh_generator.py            — MeshGenerator facade + MeshGeneratorBackend enum
# Risk: low-medium — backends are independent; facade is the only public API.
# Prerequisite: parametrize existing tests over all backends before splitting.

"""
Mesh generation interfaces for humanoid character builder.

This module was refactored from a single 1155-line file into focused submodules
to comply with the line budget:

    _mesh_types          — MeshGeneratorBackend, GeneratedMeshResult,
                           MeshGeneratorInterface
    _primitive_generator — PrimitiveMeshGenerator
    _makehuman_generator — MakeHumanMeshGenerator
    _smplx_generator     — SMPLXMeshGenerator

All public symbols remain importable from this module.
"""

from __future__ import annotations  # noqa: E402, F404

<<<<<<< HEAD
import logging
from typing import Any

from humanoid_character_builder.core.body_parameters import BodyParameters  # noqa: F401

# Re-export sub-module symbols (public API unchanged)
from ._makehuman_generator import MakeHumanMeshGenerator  # noqa: F401
from ._mesh_types import (  # noqa: F401
    GeneratedMeshResult,
    MeshGeneratorBackend,
    MeshGeneratorInterface,
)
from ._primitive_generator import PrimitiveMeshGenerator  # noqa: F401
from ._smplx_generator import SMPLXMeshGenerator  # noqa: F401
=======
import logging  # noqa: E402
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, cast  # noqa: E402

from humanoid_character_builder.core.body_parameters import BodyParameters  # noqa: E402

from .mesh_generator_makehuman import PrimitiveMeshGenerator as MakeHumanMeshGenerator
from .mesh_generator_primitive import PrimitiveMeshGenerator
from .mesh_generator_smplx import PrimitiveMeshGenerator as SMPLXMeshGenerator
>>>>>>> origin/main

logger = logging.getLogger(__name__)


<<<<<<< HEAD
=======
class MeshGeneratorBackend(Enum):
    """Available mesh generation backends."""

    PRIMITIVE = "primitive"  # Generate primitive shapes (built-in)
    MAKEHUMAN = "makehuman"  # MakeHuman integration
    SMPLX = "smplx"  # SMPL-X body model
    CUSTOM = "custom"  # Custom mesh provider


@dataclass
class GeneratedMeshResult:
    """Result of mesh generation."""

    # Whether generation was successful
    success: bool

    # Path to generated mesh files (segment name -> path)
    mesh_paths: dict[str, Path] = field(default_factory=dict)

    # Path to collision mesh files
    collision_paths: dict[str, Path] = field(default_factory=dict)

    # Path to texture files
    texture_paths: dict[str, Path] = field(default_factory=dict)

    # Vertex group mapping (for segmentation)
    vertex_groups: dict[str, list[int]] = field(default_factory=dict)

    # Error message if failed
    error_message: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class MeshGeneratorInterface(ABC):
    """
    Abstract interface for mesh generation backends.

    Implement this interface to add new mesh generation sources
    (MakeHuman, SMPL, etc.).
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available (installed, configured)."""
        ...

    @abstractmethod
    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """
        Generate meshes for the given body parameters.

        Args:
            params: Body parameters
            output_dir: Directory to write mesh files
            **kwargs: Backend-specific options

        Returns:
            GeneratedMeshResult with paths to generated files
        """
        ...

    @abstractmethod
    def get_supported_segments(self) -> list[str]:
        """Return list of segment names this backend can generate."""
        ...


>>>>>>> origin/main
class MeshGenerator:
    """
    Factory class for creating mesh generators.

    Provides a unified interface to multiple mesh generation backends.
    """

    _generators: dict[MeshGeneratorBackend, type[Any]] = {
        MeshGeneratorBackend.PRIMITIVE: PrimitiveMeshGenerator,
        MeshGeneratorBackend.MAKEHUMAN: MakeHumanMeshGenerator,
        MeshGeneratorBackend.SMPLX: SMPLXMeshGenerator,
    }

    @classmethod
    def create(
        cls,
        backend: MeshGeneratorBackend | str = MeshGeneratorBackend.PRIMITIVE,
        **kwargs: Any,
    ) -> MeshGeneratorInterface:
        """
        Create a mesh generator for the specified backend.

        Args:
            backend: Backend to use
            **kwargs: Backend-specific initialization options

        Returns:
            MeshGeneratorInterface instance
        """
        if isinstance(backend, str):
            backend = MeshGeneratorBackend(backend.lower())

        generator_class = cls._generators.get(backend)
        if generator_class is None:
            raise ValueError(f"Unknown backend: {backend}")

        return cast(MeshGeneratorInterface, generator_class(**kwargs))

    @classmethod
    def get_available_backends(cls) -> list[MeshGeneratorBackend]:
        """Return list of available backends."""
        available = []
        for backend, generator_class in cls._generators.items():
            try:
                generator = generator_class()
                if generator.is_available:
                    available.append(backend)
            except (ImportError, RuntimeError, OSError) as e:
                logger.debug("Backend %s not available: %s", backend.value, e)
        return available

    @classmethod
    def get_best_available(cls) -> MeshGeneratorInterface:
        """
        Get the best available mesh generator.

        Preference order: MakeHuman > SMPL-X > Primitive
        """
        preference = [
            MeshGeneratorBackend.MAKEHUMAN,
            MeshGeneratorBackend.SMPLX,
            MeshGeneratorBackend.PRIMITIVE,
        ]

        for backend in preference:
            try:
                generator = cls.create(backend)
                if generator.is_available:
                    return generator
            except (ImportError, RuntimeError, OSError) as e:
                logger.debug("Backend %s not available: %s", backend.value, e)
                continue

<<<<<<< HEAD
        return PrimitiveMeshGenerator()
=======
        # Final fallback
        return cast(MeshGeneratorInterface, PrimitiveMeshGenerator())
>>>>>>> origin/main

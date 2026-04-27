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

This module defines interfaces for mesh generation backends
(MakeHuman, SMPL, etc.) and provides a factory for creating
mesh generators.
"""

from __future__ import annotations  # noqa: E402, F404

import logging  # noqa: E402
from abc import ABC, abstractmethod  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from enum import Enum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from humanoid_character_builder.core.body_parameters import BodyParameters  # noqa: E402

logger = logging.getLogger(__name__)


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


class PrimitiveMeshGenerator(MeshGeneratorInterface):
    """
    Generate meshes using MakeHuman.

    This is a placeholder for future MakeHuman integration.
    MakeHuman provides high-quality, customizable human meshes
    with proper vertex groups for segmentation.
    """

    def __init__(self, makehuman_path: Path | str | None = None):
        """
        Initialize MakeHuman generator.

        Args:
            makehuman_path: Path to MakeHuman installation
        """
        self.makehuman_path = Path(makehuman_path) if makehuman_path else None

    @property
    def backend_name(self) -> str:
        return "makehuman"

    @property
    def is_available(self) -> bool:
        # Check if MakeHuman is installed
        if self.makehuman_path and self.makehuman_path.exists():
            return True

        # Try to find MakeHuman in common locations
        common_paths = [
            Path("/usr/share/makehuman"),
            Path.home() / "makehuman",
            Path.home() / ".makehuman",
        ]
        for path in common_paths:
            if path.exists():
                self.makehuman_path = path
                return True

        return False

    def generate(
        self,
        params: BodyParameters,
        output_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using MakeHuman.

        Uses MakeHuman's Python API when available, or falls back to
        loading pre-made MakeHuman exports with vertex group segmentation.
        """
        if not (params is not None):
            raise ValueError("params must be provided")
        if not self.is_available:
            return GeneratedMeshResult(
                success=False,
                error_message="MakeHuman not found. Please install MakeHuman or provide path.",
            )

        output_dir = Path(output_dir)
        visual_dir = output_dir / "visual"
        collision_dir = output_dir / "collision"
        visual_dir.mkdir(parents=True, exist_ok=True)
        collision_dir.mkdir(parents=True, exist_ok=True)

        # Convert body parameters to MakeHuman modifiers
        modifiers = self._convert_params_to_makehuman(params)

        # Try to use MakeHuman scripting API
        try:
            return self._generate_via_api(
                params, modifiers, visual_dir, collision_dir, **kwargs
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"MakeHuman API generation failed: {e}")

        # Fallback: Try to load pre-exported MakeHuman mesh
        try:
            return self._generate_from_presets(
                params, visual_dir, collision_dir, **kwargs
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.warning(f"MakeHuman preset loading failed: {e}")

        # Final fallback to primitive generator
        logger.warning("Falling back to primitive mesh generation")
        primitive_gen = PrimitiveMeshGenerator()
        return primitive_gen.generate(params, output_dir, **kwargs)

    def _generate_via_api(
        self,
        params: BodyParameters,
        modifiers: dict[str, float],
        visual_dir: Path,
        collision_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Generate meshes using MakeHuman Python API.

        Requires MakeHuman to be installed and accessible via Python.
        """
        if not (params is not None):
            raise ValueError("params must be provided")
        import subprocess
        import tempfile

        # Create MakeHuman script
        script_content = self._create_makehuman_script(modifiers, visual_dir)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(script_content)
            script_path = script_file.name

        try:
            # Run MakeHuman in scripted mode
            if self.makehuman_path is None:
                raise RuntimeError("MakeHuman path not configured")
            mh_executable = self.makehuman_path / "makehuman.py"
            if not mh_executable.exists():
                mh_executable = self.makehuman_path / "makehuman"

            result = subprocess.run(
                ["python", str(mh_executable), "--nogui", "--script", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                raise RuntimeError(f"MakeHuman failed: {result.stderr}")

            # Load generated mesh and segment it
            return self._segment_mesh(visual_dir, collision_dir)

        finally:
            Path(script_path).unlink(missing_ok=True)

    def _create_makehuman_script(
        self, modifiers: dict[str, float], output_dir: Path
    ) -> str:
        """Create a MakeHuman Python script for mesh generation."""
        if not (modifiers is not None):
            raise ValueError("modifiers must be provided")
        script = f"""
import mh
import human
import export

def generate_human():
    # Get human object
    h = human.human

    # Apply modifiers
    modifiers = {repr(modifiers)}
    for key, value in modifiers.items():
        try:
            h.setDetail(key, value)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to set modifier {{key}}={{value}}: {{exc}}")

    # Export as OBJ with vertex groups
    export_path = "{output_dir}/humanoid.obj"
    export.exportObj(h, export_path, config={{
        'exportGroups': True,
        'helper': False,
        'scale': 1.0,
    }})

generate_human()
"""
        return script

    def _generate_from_presets(
        self,
        params: BodyParameters,
        visual_dir: Path,
        collision_dir: Path,
        **kwargs: Any,
    ) -> GeneratedMeshResult:
        """Load pre-exported MakeHuman mesh based on parameters."""
        try:
            import trimesh
        except ImportError as err:
            raise RuntimeError("trimesh required for mesh processing") from err

        # Look for pre-exported mesh files in MakeHuman data directory
        if self.makehuman_path is None:
            raise RuntimeError("MakeHuman path not configured")
        presets_dir = self.makehuman_path / "data" / "exports"
        if not presets_dir.exists():
            presets_dir = self.makehuman_path / "exports"

        # Select preset based on build type
        preset_name = params.build_type.value
        gender = "male" if params.get_effective_gender_factor() > 0.5 else "female"
        preset_file = presets_dir / f"{gender}_{preset_name}.obj"

        if not preset_file.exists():
            # Try default
            preset_file = presets_dir / f"{gender}_average.obj"

        if not preset_file.exists():
            raise FileNotFoundError(f"No MakeHuman preset found: {preset_file}")

        # Load and segment the mesh
        mesh = trimesh.load(str(preset_file))

        # Scale to target height
        current_height = mesh.bounds[1][2] - mesh.bounds[0][2]
        scale_factor = params.height_m / current_height
        mesh.apply_scale(scale_factor)

        return self._segment_mesh_from_groups(mesh, visual_dir, collision_dir, params)

    def _segment_mesh(
        self, visual_dir: Path, collision_dir: Path
    ) -> GeneratedMeshResult:
        """Segment a generated mesh by vertex groups."""
        try:
            import trimesh
        except ImportError as err:
            raise RuntimeError("trimesh required for mesh segmentation") from err

        obj_file = visual_dir / "humanoid.obj"
        if not obj_file.exists():
            raise FileNotFoundError(f"Generated mesh not found: {obj_file}")

        mesh = trimesh.load(str(obj_file))

        # Get vertex groups from OBJ file
        vertex_groups = self._parse_obj_vertex_groups(obj_file)

        return self._segment_mesh_from_groups(
            mesh, visual_dir, collision_dir, vertex_groups=vertex_groups
        )

    def _segment_mesh_from_groups(
        self,
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        params: BodyParameters | None = None,
        vertex_groups: dict[str, list[int]] | None = None,
    ) -> GeneratedMeshResult:
        """Segment mesh into body parts using vertex groups or geometry."""
        if not (visual_dir is not None):
            raise ValueError("visual_dir must be provided")
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        # Map MakeHuman vertex groups to our segment names
        group_mapping = {
            "head": "head",
            "neck": "neck",
            "torso": "torso",
            "upper_torso": "torso",
            "lower_torso": "pelvis",
            "pelvis": "pelvis",
            "left_upper_arm": "left_upper_arm",
            "right_upper_arm": "right_upper_arm",
            "left_forearm": "left_forearm",
            "right_forearm": "right_forearm",
            "left_hand": "left_hand",
            "right_hand": "right_hand",
            "left_thigh": "left_thigh",
            "right_thigh": "right_thigh",
            "left_shin": "left_shin",
            "right_shin": "right_shin",
            "left_foot": "left_foot",
            "right_foot": "right_foot",
        }

        if vertex_groups:
            mesh_paths, collision_paths = self._segment_by_vertex_groups(
                mesh,
                visual_dir,
                collision_dir,
                vertex_groups,
                group_mapping,
                HUMANOID_SEGMENTS,
            )
        else:
            mesh_paths, collision_paths = self._segment_by_geometry(
                mesh,
                visual_dir,
                collision_dir,
                HUMANOID_SEGMENTS,
            )

        return GeneratedMeshResult(
            success=len(mesh_paths) > 0,
            mesh_paths=mesh_paths,
            collision_paths=collision_paths,
            vertex_groups=vertex_groups or {},
            metadata={"backend": "makehuman"},
        )

    @staticmethod
    def _segment_by_vertex_groups(
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        vertex_groups: dict[str, list[int]],
        group_mapping: dict[str, str],
        valid_segments: Any,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """Segment mesh using vertex group indices."""
        if not (visual_dir is not None):
            raise ValueError("visual_dir must be provided")
        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}

        for group_name, vertex_indices in vertex_groups.items():
            segment_name = group_mapping.get(group_name.lower())
            if segment_name and segment_name in valid_segments:
                try:
                    face_mask = mesh.faces_sparse.rows[vertex_indices].indices
                    submesh = mesh.submesh([face_mask], append=True)

                    visual_path = visual_dir / f"{segment_name}.stl"
                    submesh.export(str(visual_path))
                    mesh_paths[segment_name] = visual_path

                    collision_mesh = submesh.convex_hull
                    collision_path = collision_dir / f"{segment_name}.stl"
                    collision_mesh.export(str(collision_path))
                    collision_paths[segment_name] = collision_path
                except (
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                    TypeError,
                ) as e:
                    logger.warning(f"Failed to extract {segment_name}: {e}")

        return mesh_paths, collision_paths

    @staticmethod
    def _segment_by_geometry(
        mesh: Any,
        visual_dir: Path,
        collision_dir: Path,
        valid_segments: Any,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """Segment mesh using bounding-box z-range slicing."""
        if not (visual_dir is not None):
            raise ValueError("visual_dir must be provided")
        mesh_paths: dict[str, Path] = {}
        collision_paths: dict[str, Path] = {}

        bounds = mesh.bounds
        height = bounds[1][2] - bounds[0][2]

        segment_z_ranges = {
            "head": (0.90, 1.0),
            "neck": (0.85, 0.90),
            "torso": (0.55, 0.85),
            "pelvis": (0.45, 0.55),
            "left_thigh": (0.25, 0.45),
            "right_thigh": (0.25, 0.45),
            "left_shin": (0.08, 0.25),
            "right_shin": (0.08, 0.25),
            "left_foot": (0.0, 0.08),
            "right_foot": (0.0, 0.08),
        }

        for segment_name, (z_low, _z_high) in segment_z_ranges.items():
            if segment_name in valid_segments:
                z_min = bounds[0][2] + z_low * height

                try:
                    plane_origin = [0, 0, z_min]
                    plane_normal = [0, 0, 1]
                    submesh = mesh.slice_plane(plane_origin, plane_normal)

                    if submesh and len(submesh.vertices) > 0:
                        visual_path = visual_dir / f"{segment_name}.stl"
                        submesh.export(str(visual_path))
                        mesh_paths[segment_name] = visual_path

                        collision_path = collision_dir / f"{segment_name}.stl"
                        submesh.convex_hull.export(str(collision_path))
                        collision_paths[segment_name] = collision_path
                except (
                    ValueError,
                    ZeroDivisionError,
                    OverflowError,
                    TypeError,
                ) as e:
                    logger.warning(f"Failed to slice {segment_name}: {e}")

        return mesh_paths, collision_paths

    def _parse_obj_vertex_groups(self, obj_file: Path) -> dict[str, list[int]]:
        """Parse vertex groups from OBJ file."""
        if not (obj_file is not None):
            raise ValueError("obj_file must be provided")
        groups: dict[str, list[int]] = {}
        current_group = "default"
        vertex_index = 0

        with open(obj_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("g "):
                    current_group = line[2:].strip()
                    if current_group not in groups:
                        groups[current_group] = []
                elif line.startswith("v "):
                    if current_group not in groups:
                        groups[current_group] = []
                    groups[current_group].append(vertex_index)
                    vertex_index += 1

        return groups

    def get_supported_segments(self) -> list[str]:
        # MakeHuman supports all standard humanoid segments
        from humanoid_character_builder.core.segment_definitions import (
            HUMANOID_SEGMENTS,
        )

        return list(HUMANOID_SEGMENTS.keys())

    def _convert_params_to_makehuman(self, params: BodyParameters) -> dict[str, float]:
        """Convert BodyParameters to MakeHuman modifier values."""
        # MakeHuman uses modifiers in range [-1, 1] or [0, 1]
        if not (params is not None):
            raise ValueError("params must be provided")
        modifiers = {}

        # Height is handled by overall scale
        # MakeHuman default is ~1.68m, adjust proportionally
        # height_scale = params.height_m / 1.68

        # Gender (MakeHuman: 0 = female, 1 = male)
        modifiers["macrodetails/Gender"] = params.get_effective_gender_factor()

        # Age (MakeHuman: range depends on modifier)
        modifiers["macrodetails/Age"] = min(
            1.0, max(0.0, params.appearance.age_years / 80.0)
        )

        # Muscularity (MakeHuman: muscle definition)
        modifiers["macrodetails-universal/Muscle"] = params.muscularity

        # Weight/body fat
        modifiers["macrodetails-universal/Weight"] = params.body_fat_factor

        # Proportions
        modifiers["macrodetails-proportions/BodyProportions"] = (
            params.torso_length_factor - 1.0
        )

        return modifiers

"""
Collision geometry generation for humanoid character builder.

This module provides advanced algorithms for generating simplified
collision geometry from visual meshes, including convex decomposition,
primitive fitting, and mesh decimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


@dataclass
class CollisionGeometry:
    """Result of collision geometry generation."""

    meshes: list[trimesh.Trimesh]
    method: str
    quality_score: float = 0.0
    volume_preservation: float = 0.0
    vertex_count: int = 0
    face_count: int = 0
    processing_time: float = 0.0

    def export(self, path: str) -> None:
        """Export collision geometry to file."""
        if not self.meshes:
            return

        import trimesh

        if len(self.meshes) == 1:
            self.meshes[0].export(path)
        else:
            # Combine or export as scene/multi-part
            combined = trimesh.util.concatenate(self.meshes)
            combined.export(path)


class CollisionGeometryGenerator:
    """
    Generate optimized collision geometry from visual meshes.

    Supports multiple strategies:
    - Convex Hull Decomposition (VHACD)
    - Primitive Fitting (Box, Sphere, Cylinder, Capsule)
    - Mesh Decimation
    - Hybrid/Auto selection
    """

    def __init__(self):
        """Initialize the generator."""
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check availability of required libraries."""
        try:
            import networkx  # noqa: F401
            import scipy  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError as e:
            logger.warning(f"Missing dependency for collision generation: {e}")

    def generate(
        self,
        visual_mesh: Any,
        method: str = "auto",  # "primitives", "vhacd", "decimation", "auto"
        target_complexity: str = "balanced",  # "minimal", "balanced", "accurate"
        max_primitives: int = 16,
        max_triangles: int = 500,
    ) -> CollisionGeometry:
        """
        Generate optimized collision geometry from visual mesh.

        Args:
            visual_mesh: Input trimesh object
            method: Simplification method to use
            target_complexity: Quality/Performance tradeoff preset
            max_primitives: Maximum number of hulls/primitives
            max_triangles: Target triangle count for decimation

        Returns:
            CollisionGeometry object containing simplified meshes
        """
        import time
        start_time = time.time()

        # Configure parameters based on complexity preset
        if target_complexity == "minimal":
            max_primitives = min(max_primitives, 4)
            max_triangles = min(max_triangles, 100)
        elif target_complexity == "accurate":
            max_primitives = max(max_primitives, 32)
            max_triangles = max(max_triangles, 2000)

        result_meshes = []

        # Dispatch to specific method
        if method == "vhacd":
            result_meshes = self._generate_vhacd(visual_mesh, max_primitives)
        elif method == "primitives":
            result_meshes = self._generate_primitives(visual_mesh)
        elif method == "decimation":
            result_meshes = self._generate_decimation(visual_mesh, max_triangles)
        elif method == "auto":
            result_meshes = self._generate_auto(visual_mesh, max_primitives, max_triangles)
        else:
            # Fallback to convex hull
            result_meshes = [visual_mesh.convex_hull]

        # Compute metrics
        processing_time = time.time() - start_time
        metrics = self.compute_quality_metrics(visual_mesh, result_meshes)

        total_verts = sum(len(m.vertices) for m in result_meshes)
        total_faces = sum(len(m.faces) for m in result_meshes)

        return CollisionGeometry(
            meshes=result_meshes,
            method=method,
            quality_score=metrics["quality_score"],
            volume_preservation=metrics["volume_preservation"],
            vertex_count=total_verts,
            face_count=total_faces,
            processing_time=processing_time
        )

    def _generate_vhacd(self, mesh: Any, max_hulls: int) -> list[Any]:
        """
        Generate collision geometry using VHACD.
        Falls back to convex hull if VHACD is not available.
        """
        import trimesh

        try:
            # Check if VHACD is available/callable
            # trimesh.decomposition.convex_decomposition uses testVHACD or vhacd binary
            # This might fail if binary is missing.
            components = trimesh.decomposition.convex_decomposition(
                mesh,
                maxhulls=max_hulls,
                resolution=100000,
                depth=20
            )

            if not isinstance(components, list):
                components = [components]

            return components

        except Exception as e:
            logger.warning(f"VHACD failed, falling back to convex hull: {e}")
            return [mesh.convex_hull]

    def _generate_primitives(self, mesh: Any) -> list[Any]:
        """
        Fit best primitive(s) to the mesh.
        Currently fits a single best primitive (Box, Sphere, Cylinder, or Capsule).
        """
        import trimesh

        # Calculate bounding primitives
        candidates = []

        # 1. Oriented Bounding Box
        try:
            obb = mesh.bounding_box_oriented
            candidates.append(('box', obb, obb.volume))
        except Exception:
            pass

        # 2. Minimum Sphere
        try:
            center, radius = trimesh.nsphere.minimum_nsphere(mesh.vertices)
            sphere = trimesh.creation.icosphere(radius=radius, subdivisions=3)
            # Center the sphere
            sphere.apply_translation(center)
            candidates.append(('sphere', sphere, sphere.volume))
        except Exception:
            pass

        # 3. Cylinder & Capsule fitting
        try:
            # Simple approximation using OBB
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            transform = np.linalg.inv(to_origin)

            # Assume longest axis is height
            axis_idx = np.argmax(extents)
            height = extents[axis_idx]
            # Radius is max of other two / 2
            other_axes = [i for i in range(3) if i != axis_idx]
            radius = max(extents[other_axes]) / 2.0

            # Align cylinder/capsule to the correct local axis (default is Z)
            align_transform = np.eye(4)
            if axis_idx == 0:  # Align Z to X
                align_transform = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
            elif axis_idx == 1:  # Align Z to Y
                align_transform = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])

            # Cylinder
            cyl = trimesh.creation.cylinder(radius=radius, height=height)
            cyl.apply_transform(align_transform)
            cyl.apply_transform(transform)
            candidates.append(('cylinder', cyl, cyl.volume))

            # Capsule
            # trimesh capsule height is length of cylinder segment
            cyl_len = max(0.0, height - 2 * radius)
            cap = trimesh.creation.capsule(radius=radius, height=cyl_len)
            cap.apply_transform(align_transform)
            cap.apply_transform(transform)
            candidates.append(('capsule', cap, cap.volume))

        except Exception:
            pass

        # Select candidate with volume closest to original mesh volume
        # This helps avoid picking smaller inscribed primitives that don't cover the shape
        target_volume = mesh.volume

        if not candidates:
            return [mesh.convex_hull]

        best_primitive = min(candidates, key=lambda x: abs(x[2] - target_volume))
        return [best_primitive[1]]

    def _generate_decimation(self, mesh: Any, max_triangles: int) -> list[Any]:
        """Simplify mesh using quadric decimation or fallback."""
        import trimesh

        try:
            simplified = mesh.simplify_quadric_decimation(face_count=max_triangles)
            return [simplified]
        except Exception:
            # Fallback: vertex clustering / voxelization
            try:
                current_faces = len(mesh.faces)
                if current_faces <= max_triangles:
                    return [mesh]

                # Heuristic for pitch based on target face count
                # Approx: Surface area / target_faces -> area per face
                # pitch ~ sqrt(area per face)
                try:
                    area = mesh.area
                    pitch = np.sqrt(area / max_triangles) * 1.5
                except Exception:
                    pitch = mesh.extents.max() / 10.0

                if pitch <= 1e-6:
                    return [mesh]

                simplified = trimesh.voxel.ops.points_to_marching_cubes(
                    mesh.vertices, pitch=pitch
                )
                return [simplified]
            except Exception as e:
                logger.warning(f"Decimation failed: {e}")
                return [mesh]

    def _generate_auto(self, mesh: Any, max_primitives: int, max_triangles: int) -> list[Any]:
        """Automatically select best method."""
        # Simple heuristic:
        # If mesh is convex-ish (high volume/convex_volume ratio), use convex hull or primitives
        # If mesh is complex (many components), try VHACD or decimation

        if mesh.is_convex:
            return [mesh] # Already convex

        convex_ratio = mesh.volume / mesh.convex_hull.volume

        if convex_ratio > 0.8:
            # Nearly convex, use hull or primitive
            # If fairly simple shape, try primitive
            return self._generate_primitives(mesh)

        if len(mesh.faces) > max_triangles * 2:
             # Very high detail, decimate first?
             # For collision, we usually want convex hulls or simple mesh.
             # If complex shape, VHACD is best for physics.
             return self._generate_vhacd(mesh, max_primitives)

        return self._generate_decimation(mesh, max_triangles)

    def compute_quality_metrics(self, original: Any, generated: list[Any]) -> dict[str, float]:
        """Compute quality metrics comparing generated collision geometry to original."""
        import trimesh

        if not generated:
            return {"quality_score": 0.0, "volume_preservation": 0.0}

        combined_gen = trimesh.util.concatenate(generated)

        # Volume preservation
        try:
            vol_orig = original.volume
            vol_gen = combined_gen.volume
            if vol_orig > 1e-6:
                vol_preservation = min(vol_gen / vol_orig, vol_orig / vol_gen) if vol_gen > 0 else 0
            else:
                vol_preservation = 1.0
        except Exception:
            vol_preservation = 0.0

        # Hausdorff distance (approximate)
        try:
            # Sample points on both meshes
            samples_orig, _ = trimesh.sample.sample_surface(original, 1000)
            samples_gen, _ = trimesh.sample.sample_surface(combined_gen, 1000)

            # Distance from orig to gen
            _, dist_o_g, _ = trimesh.proximity.closest_point(combined_gen, samples_orig)
            # Distance from gen to orig
            _, dist_g_o, _ = trimesh.proximity.closest_point(original, samples_gen)

            hausdorff = max(np.max(dist_o_g), np.max(dist_g_o))

            # Normalize score (0-1), assume 1.0 is good.
            # Hausdorff of 0 is perfect.
            # Scale relative to bounding box diagonal
            scale = original.scale
            normalized_hausdorff = hausdorff / scale if scale > 0 else hausdorff

            quality_score = max(0.0, 1.0 - normalized_hausdorff * 10) # Arbitrary scaling
            quality_score = (quality_score + vol_preservation) / 2.0

        except Exception as e:
            logger.warning(f"Quality metrics calculation failed: {e}")
            quality_score = 0.0

        return {
            "quality_score": quality_score,
            "volume_preservation": vol_preservation
        }

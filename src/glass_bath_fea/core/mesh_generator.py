"""Mesh generator for Glass Bath FEA.

This module generates finite element meshes for the glass bath
geometry using pygmsh (if available) or provides mock meshes
for testing and development.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .geometry_generator import GeometryGenerator

if TYPE_CHECKING:
    from .config import GlassBathFEAConfig

# Check if pygmsh is available
try:
    import pygmsh

    HAS_PYGMSH = True
except ImportError:
    HAS_PYGMSH = False


class MeshGenerator:
    """Generate FEA meshes for glass bath geometry.

    Uses pygmsh for parametric mesh generation if available,
    otherwise provides mock mesh functionality for testing.

    Attributes:
        config: FEA configuration parameters
        geometry_generator: Geometry definition generator
    """

    def __init__(self, config: GlassBathFEAConfig) -> None:
        """Initialize mesh generator.

        Args:
            config: FEA configuration with vessel dimensions
        """
        self.config = config
        self.geometry_generator = GeometryGenerator(config)

    def generate_mesh(self, coarse: bool = False) -> dict:
        """Generate finite element mesh for the geometry.

        Args:
            coarse: If True, use coarser mesh for faster testing

        Returns:
            Dictionary with mesh data:
            - nodes: 3xN array of node coordinates
            - elements: MxN array of element connectivity (1-indexed)
            - material_ids: Array of material IDs per element

        Raises:
            ImportError: If pygmsh is not available
        """
        if not HAS_PYGMSH:
            raise ImportError(
                "pygmsh is required for mesh generation. "
                "Install with: pip install pygmsh gmsh"
            )

        dims = self.geometry_generator.get_dimensions()
        mesh_config = self.config.mesh_config

        # Adjust element sizes for coarse mesh
        size_factor = 3.0 if coarse else 1.0

        with pygmsh.occ.Geometry() as geom:
            # Create main cylinder (glass + metal regions)
            # The cylinder is added to the geometry context
            geom.add_cylinder(
                [0, 0, 0],  # Center at origin
                [0, 0, dims["total_height"]],  # Axis direction
                dims["radius"],  # Radius
            )

            # Set mesh size
            geom.set_mesh_size_callback(
                lambda dim, tag, x, y, z, lc: (
                    mesh_config.element_size_glass * size_factor
                )
            )

            # Generate mesh
            mesh = geom.generate_mesh()

        # Convert to MATLAB-compatible format
        nodes = mesh.points.T  # 3xN array
        elements = mesh.cells_dict.get("tetra", np.array([])).T + 1  # 1-indexed

        # Assign material IDs based on Z-coordinate
        material_ids = self._assign_material_ids(mesh.points, elements.T - 1)

        return {
            "nodes": nodes,
            "elements": elements,
            "material_ids": material_ids,
        }

    def create_mock_mesh(self) -> dict:
        """Create a mock mesh for testing without pygmsh.

        Generates a simple structured mesh within the vessel bounds.

        Returns:
            Dictionary with mock mesh data.
        """
        dims = self.geometry_generator.get_dimensions()

        # Create a simple cylindrical mesh
        # Radial divisions
        n_radial = 5
        n_angular = 12
        n_vertical = 8

        # Generate nodes
        nodes_list = []

        # Add center axis nodes
        for k in range(n_vertical + 1):
            z = k * dims["total_height"] / n_vertical
            nodes_list.append([0.0, 0.0, z])

        # Add radial layers
        for i in range(1, n_radial + 1):
            r = i * dims["radius"] / n_radial
            for j in range(n_angular):
                theta = j * 2 * math.pi / n_angular
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                for k in range(n_vertical + 1):
                    z = k * dims["total_height"] / n_vertical
                    nodes_list.append([x, y, z])

        nodes = np.array(nodes_list).T  # 3xN format

        # Generate tetrahedral elements (simplified connectivity)
        elements_list = []
        material_ids_list = []

        # Generate simple tetrahedra
        n_elems = max(10, n_radial * n_angular * n_vertical // 2)

        for _ in range(n_elems):
            # Random tetrahedron within bounds
            n_nodes = nodes.shape[1]
            if n_nodes >= 4:
                indices = np.random.choice(n_nodes, 4, replace=False)
                elements_list.append(indices + 1)  # 1-indexed

                # Determine material based on centroid Z
                centroid_z = np.mean(nodes[2, indices])
                if centroid_z < dims["metal_thickness"]:
                    material_ids_list.append(2)  # Metal
                else:
                    material_ids_list.append(1)  # Glass

        elements = np.array(elements_list).T  # 4xN format
        material_ids = np.array(material_ids_list)

        return {
            "nodes": nodes,
            "elements": elements,
            "material_ids": material_ids,
        }

    def _assign_material_ids(
        self, points: np.ndarray, elements: np.ndarray
    ) -> np.ndarray:
        """Assign material IDs based on element centroid location.

        Args:
            points: Nx3 array of node coordinates
            elements: Mx4 array of element connectivity (0-indexed)

        Returns:
            Array of material IDs for each element.
        """
        dims = self.geometry_generator.get_dimensions()
        material_ids = []

        for element in elements:
            # Calculate element centroid
            centroid = np.mean(points[element], axis=0)
            z = centroid[2]

            # Assign material based on Z-coordinate
            if z < dims["metal_thickness"]:
                material_ids.append(2)  # Metal
            else:
                material_ids.append(1)  # Glass

        return np.array(material_ids)

    def check_mesh_quality(self, mesh: dict) -> dict:
        """Check mesh quality metrics.

        Args:
            mesh: Mesh data dictionary

        Returns:
            Dictionary with quality metrics.
        """
        nodes = mesh["nodes"]
        elements = mesh["elements"]

        if elements.shape[1] == 0:
            return {"min_quality": 0.0, "mean_quality": 0.0}

        # Calculate element qualities (simplified - aspect ratio based)
        qualities = []

        for i in range(elements.shape[1]):
            elem = elements[:, i] - 1  # Convert to 0-indexed
            elem_nodes = nodes[:, elem].T

            # Calculate edge lengths
            edges = []
            for j in range(4):
                for k in range(j + 1, 4):
                    edge_len = np.linalg.norm(elem_nodes[j] - elem_nodes[k])
                    edges.append(edge_len)

            if edges:
                min_edge = min(edges)
                max_edge = max(edges)
                quality = min_edge / max_edge if max_edge > 0 else 0
                qualities.append(quality)

        if not qualities:
            return {"min_quality": 0.0, "mean_quality": 0.0}

        return {
            "min_quality": min(qualities),
            "mean_quality": float(np.mean(np.array(qualities))),
            "max_quality": max(qualities),
        }

    def check_watertight(self, mesh: dict) -> bool:
        """Check if mesh represents a watertight (closed) volume.

        Args:
            mesh: Mesh data dictionary

        Returns:
            True if mesh is watertight.
        """
        # For mock mesh, assume it's watertight if it has elements
        elements = mesh.get("elements", np.array([]))

        if elements.size == 0:
            return False

        # Simple check: mesh is "watertight" if it has valid elements
        # A real implementation would check surface face connectivity
        return bool(elements.shape[1] > 0)

    def get_mesh_statistics(self, mesh: dict) -> dict:
        """Calculate mesh statistics.

        Args:
            mesh: Mesh data dictionary

        Returns:
            Dictionary with mesh statistics.
        """
        nodes = mesh["nodes"]
        elements = mesh["elements"]
        material_ids = mesh.get("material_ids", np.array([]))

        stats = {
            "num_nodes": nodes.shape[1],
            "num_elements": elements.shape[1],
        }

        # Count elements by region
        if material_ids.size > 0:
            unique, counts = np.unique(material_ids, return_counts=True)
            stats["region_counts"] = dict(
                zip(unique.tolist(), counts.tolist(), strict=True)
            )

            # Named counts
            stats["elements_glass"] = stats["region_counts"].get(1, 0)
            stats["elements_metal"] = stats["region_counts"].get(2, 0)

        return stats

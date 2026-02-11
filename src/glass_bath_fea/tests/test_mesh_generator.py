"""Tests for Glass Bath FEA mesh generator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]

from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

# Check if pygmsh is available
import importlib.util

HAS_PYGMSH = importlib.util.find_spec("pygmsh") is not None

if TYPE_CHECKING:
    from glass_bath_fea.core.config import GlassBathFEAConfig


class TestMeshGeneratorCreation:
    """Tests for mesh generator initialization."""

    def test_create_mesh_generator(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test creating mesh generator instance."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)

        assert gen is not None
        assert gen.config is not None

    def test_mesh_generator_has_geometry(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test that mesh generator has access to geometry."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)

        # Should have geometry data
        assert gen.geometry_generator is not None


@pytest.mark.skipif(not HAS_PYGMSH, reason="pygmsh not installed")
class TestMeshGeneration:
    """Tests for actual mesh generation (requires pygmsh)."""

    def test_generate_simple_mesh(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test generating a simple mesh."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)

        # Generate coarse mesh for testing
        mesh = gen.generate_mesh(coarse=True)

        assert mesh is not None
        assert "nodes" in mesh
        assert "elements" in mesh

    def test_mesh_has_nodes(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test that generated mesh has node coordinates."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        mesh = gen.generate_mesh(coarse=True)

        nodes = mesh["nodes"]

        # Should be 3xN array (MATLAB format)
        assert nodes.ndim == 2
        assert nodes.shape[0] == 3  # x, y, z coordinates

    def test_mesh_has_elements(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test that generated mesh has element connectivity."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        mesh = gen.generate_mesh(coarse=True)

        elements = mesh["elements"]

        # Should have element connectivity
        assert elements.ndim == 2
        assert elements.shape[0] >= 4  # At least tetrahedral (4 nodes)

    def test_mesh_has_material_ids(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test that mesh elements have material IDs."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        mesh = gen.generate_mesh(coarse=True)

        # Should have material IDs for each element
        assert "material_ids" in mesh
        assert len(mesh["material_ids"]) == mesh["elements"].shape[1]


class TestMeshDataStructure:
    """Tests for mesh data structure (without full generation)."""

    def test_create_mock_mesh(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test creating a mock mesh data structure."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)

        # Create mock mesh for testing exporters
        mock_mesh = gen.create_mock_mesh()

        assert "nodes" in mock_mesh
        assert "elements" in mock_mesh
        assert "material_ids" in mock_mesh

    def test_mock_mesh_geometry(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test that mock mesh respects geometry bounds."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        dims = gen.geometry_generator.get_dimensions()

        mock_mesh = gen.create_mock_mesh()
        nodes = mock_mesh["nodes"]

        # Check that all nodes are within vessel bounds
        radii = np.sqrt(nodes[0] ** 2 + nodes[1] ** 2)
        assert np.all(radii <= dims["radius"] * 1.01)  # Small tolerance

        heights = nodes[2]
        assert np.all(heights >= -0.001)  # Small tolerance
        assert np.all(heights <= dims["total_height"] * 1.01)


class TestMeshQuality:
    """Tests for mesh quality metrics."""

    def test_mesh_quality_check(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test mesh quality validation."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        mock_mesh = gen.create_mock_mesh()

        quality = gen.check_mesh_quality(mock_mesh)

        # Should return quality metrics
        assert "min_quality" in quality
        assert "mean_quality" in quality
        assert quality["min_quality"] >= 0
        assert quality["mean_quality"] <= 1.0

    def test_mesh_is_watertight(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test that mesh is watertight (closed surface)."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        mock_mesh = gen.create_mock_mesh()

        is_watertight = gen.check_watertight(mock_mesh)

        # Mock mesh should be watertight
        assert is_watertight


class TestMeshStatistics:
    """Tests for mesh statistics calculation."""

    def test_mesh_statistics(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test calculating mesh statistics."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        mock_mesh = gen.create_mock_mesh()

        stats = gen.get_mesh_statistics(mock_mesh)

        # Should have basic statistics
        assert "num_nodes" in stats
        assert "num_elements" in stats
        assert stats["num_nodes"] > 0
        assert stats["num_elements"] > 0

    def test_element_count_by_region(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test counting elements by material region."""
        from glass_bath_fea.core.mesh_generator import MeshGenerator

        gen = MeshGenerator(default_fea_config)
        mock_mesh = gen.create_mock_mesh()

        stats = gen.get_mesh_statistics(mock_mesh)

        # Should have counts per region
        assert "elements_glass" in stats or "region_counts" in stats

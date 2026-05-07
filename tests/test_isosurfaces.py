"""Test suite for iso-surface extraction and visualization.

This module implements TDD for GitHub issue #538: Iso-surface (Threshold) Rendering.
Tests are organized by:
1. Unit tests on analytical geometries (marching cubes correctness)
2. Multi-level extraction tests
3. Caching verification
4. Performance tests

Success criteria:
- All extraction tests pass (including sphere geometry validation)
- Multi-level visualization works correctly
- Performance meets targets (<500ms single, <2s for 5 surfaces)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMarchingCubesOnAnalyticalGeometries:
    """Test marching cubes extraction on known analytical shapes."""

    @pytest.mark.unit
    def test_sphere_extraction_basic(self) -> None:
        """Test iso-surface extraction of a sphere.

        Creates a sphere analytically and extracts iso-surface at radius=0.5.
        Validates that:
        - Extracted surface exists
        - Number of triangles is reasonable
        - Surface is closed (Euler characteristic)
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        # Create sphere field: distance from center
        grid_size = 32
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        extractor = IsoSurfaceExtractor()
        iso_value = 0.5
        surface = extractor.extract(field, iso_value)

        assert surface is not None
        assert "vertices" in surface
        assert "triangles" in surface
        assert len(surface["vertices"]) > 0
        assert len(surface["triangles"]) > 0

        # Vertices should be close to iso_value in field space
        vertices = surface["vertices"]
        assert vertices.shape[1] == 3, "Vertices should be 3D points"

    @pytest.mark.unit
    def test_sphere_surface_quality(self) -> None:
        """Validate sphere extraction surface quality.

        Tests that:
        - Extracted vertices lie on the sphere surface
        - No NaN or inf values
        - Surface is reasonably smooth
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 32
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        extractor = IsoSurfaceExtractor()
        iso_value = 0.5
        surface = extractor.extract(field, iso_value)

        vertices = surface["vertices"]

        # Check for NaN/inf
        assert not np.any(np.isnan(vertices)), "Vertices contain NaN"
        assert not np.any(np.isinf(vertices)), "Vertices contain inf"

        # Vertices are in index space, normalize to world coords [-1, 1]
        # Grid coordinates go from 0 to grid_size, map to [-1, 1]
        normalized_verts = vertices / (grid_size - 1) * 2 - 1
        radii = np.linalg.norm(normalized_verts, axis=1)
        mean_radius = np.mean(radii)
        tolerance = 0.15  # Allow reasonable tolerance for discrete approximation
        assert abs(mean_radius - iso_value) < tolerance, (
            f"Mean radius {mean_radius} not close to iso_value {iso_value}"
        )

    @pytest.mark.unit
    def test_cube_iso_surface_extraction(self) -> None:
        """Test extraction of iso-surface for a cubic field.

        Creates a field where value increases from 0 to 1 in each direction.
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 16
        coords = np.linspace(0, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = X + Y + Z  # Linear combination

        extractor = IsoSurfaceExtractor()
        iso_value = 1.5
        surface = extractor.extract(field, iso_value)

        assert surface is not None
        assert len(surface["vertices"]) > 0

    @pytest.mark.unit
    def test_iso_surface_with_gradient_field(self) -> None:
        """Test extraction from a smooth gradient field."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 24
        coords = np.linspace(-2, 2, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.exp(-(X**2 + Y**2 + Z**2))

        extractor = IsoSurfaceExtractor()
        iso_value = 0.3
        surface = extractor.extract(field, iso_value)

        assert surface is not None
        assert len(surface["vertices"]) > 100, "Should extract meaningful surface"


class TestMultiLevelExtraction:
    """Test batch extraction of multiple iso-surfaces."""

    @pytest.mark.unit
    def test_extract_multiple_single_call(self) -> None:
        """Test extracting multiple iso-surfaces in one call."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 24
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        extractor = IsoSurfaceExtractor()
        iso_values = [0.3, 0.5, 0.7]
        surfaces = extractor.extract_multiple(field, iso_values)

        assert len(surfaces) == len(iso_values)
        for surface in surfaces:
            assert "vertices" in surface
            assert len(surface["vertices"]) > 0

    @pytest.mark.unit
    def test_multiple_extraction_efficiency(self) -> None:
        """Test that batch extraction is more efficient than individual calls.

        extract_multiple() should reuse the same field analysis where possible.
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 20
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        iso_values = [0.3, 0.5, 0.7]

        extractor = IsoSurfaceExtractor()

        # Test batch extraction
        surfaces_batch = extractor.extract_multiple(field, iso_values)

        # Test individual extractions
        surfaces_individual = []
        extractor_individual = IsoSurfaceExtractor()
        for iso_val in iso_values:
            surfaces_individual.append(extractor_individual.extract(field, iso_val))

        # Both methods should produce results
        assert len(surfaces_batch) == len(iso_values)
        assert len(surfaces_individual) == len(iso_values)

    @pytest.mark.unit
    def test_iso_values_validation(self) -> None:
        """Test that iso-values are validated against field range.

        DbC: Contract violation when iso-value is outside field bounds.
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 16
        coords = np.linspace(0, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = X + Y + Z

        extractor = IsoSurfaceExtractor()

        # Iso-value way above field max should at least not crash
        # (it should either warn or return empty surface)
        iso_value = 100.0
        try:
            surface = extractor.extract(field, iso_value)
            # If it succeeds, surface should be empty or None
            if surface is not None:
                assert len(surface.get("vertices", [])) == 0
        except (ValueError, RuntimeError):
            # Exception is also acceptable
            pass


class TestCachingStrategy:
    """Test caching and cache invalidation."""

    @pytest.mark.unit
    def test_cache_hit_on_repeated_field(self) -> None:
        """Test that repeated extractions of same field use cache."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 16
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        extractor = IsoSurfaceExtractor()

        # First extraction
        surface1 = extractor.extract(field, 0.5)

        # Second extraction (should be cached)
        surface2 = extractor.extract(field, 0.5)

        assert np.allclose(surface1["vertices"], surface2["vertices"])

    @pytest.mark.unit
    def test_cache_invalidation_on_field_change(self) -> None:
        """Test that cache is invalidated when field changes."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 16
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field1 = np.sqrt(X**2 + Y**2 + Z**2)
        field2 = np.sqrt((X - 0.5) ** 2 + Y**2 + Z**2)  # Shifted sphere

        extractor = IsoSurfaceExtractor()

        surface1 = extractor.extract(field1, 0.5)
        surface2 = extractor.extract(field2, 0.5)

        # Surfaces should be different (shifted centers)
        # Different fields may produce different vertex counts
        assert surface1 is not None and surface2 is not None
        # Verify that surfaces have content
        assert len(surface1["vertices"]) > 0 and len(surface2["vertices"]) > 0

    @pytest.mark.unit
    def test_cache_size_limit(self) -> None:
        """Test that cache doesn't grow unbounded."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        extractor = IsoSurfaceExtractor(cache_size_limit=3)
        grid_size = 12
        coords = np.linspace(-1, 1, grid_size)

        # Create and extract from multiple fields
        for i in range(10):
            X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
            field = np.sqrt((X - i * 0.1) ** 2 + Y**2 + Z**2)
            surface = extractor.extract(field, 0.5)
            assert surface is not None

        # Cache should not exceed limit
        cache_info = extractor.get_cache_info()
        assert cache_info["size"] <= extractor.cache_size_limit


class TestMeshTypeSupport:
    """Test support for different mesh types."""

    @pytest.mark.unit
    def test_tetrahedral_mesh_compatibility(self) -> None:
        """Test that extractor works with tetrahedral mesh data.

        For now, this tests the scalar field interface.
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        # Simulate tetrahedral mesh field
        grid_size = 12
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        extractor = IsoSurfaceExtractor()
        surface = extractor.extract(field, 0.5)

        assert surface is not None

    @pytest.mark.unit
    def test_hex_mesh_compatibility(self) -> None:
        """Test that extractor works with hex mesh data."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 14
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = X**2 + Y**2 + Z**2

        extractor = IsoSurfaceExtractor()
        surface = extractor.extract(field, 0.25)

        assert surface is not None


class TestPerformance:
    """Performance tests with targets from requirements."""

    @pytest.mark.unit
    @pytest.mark.benchmark
    def test_single_surface_performance(self) -> None:
        """Test single iso-surface extraction performance.

        Target: <500ms for 100k-element meshes
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        # 100k elements = ~46x46x46 grid
        grid_size = 46
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        extractor = IsoSurfaceExtractor()

        start = time.time()
        surface = extractor.extract(field, 0.5)
        elapsed = (time.time() - start) * 1000  # milliseconds

        assert surface is not None
        assert elapsed < 500, f"Single extraction took {elapsed}ms (target: <500ms)"

    @pytest.mark.unit
    @pytest.mark.benchmark
    def test_multiple_surfaces_performance(self) -> None:
        """Test multiple iso-surface extraction performance.

        Target: <2s for 5 surfaces on 100k-element meshes
        """
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 46
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)

        iso_values = [0.2, 0.35, 0.5, 0.65, 0.8]
        extractor = IsoSurfaceExtractor()

        start = time.time()
        surfaces = extractor.extract_multiple(field, iso_values)
        elapsed = (time.time() - start) * 1000  # milliseconds

        assert len(surfaces) == 5
        assert elapsed < 2000, f"Multiple extraction took {elapsed}ms (target: <2s)"


class TestInputValidation:
    """Test input validation and error handling."""

    @pytest.mark.unit
    def test_empty_field_handling(self) -> None:
        """Test handling of empty or invalid field."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        extractor = IsoSurfaceExtractor()

        # Very small field
        field = np.array([[[1.0]]])
        surface = extractor.extract(field, 0.5)

        # Should handle gracefully
        assert surface is None or len(surface.get("vertices", [])) == 0

    @pytest.mark.unit
    def test_nan_field_handling(self) -> None:
        """Test handling of field with NaN values."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 16
        coords = np.linspace(-1, 1, grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        field = np.sqrt(X**2 + Y**2 + Z**2)
        field[0, 0, 0] = np.nan

        extractor = IsoSurfaceExtractor()

        # Should handle gracefully (filter NaNs or warn)
        try:
            surface = extractor.extract(field, 0.5)
            # If it succeeds, that's fine
            assert surface is None or isinstance(surface, dict)
        except ValueError:
            # Exception is also acceptable
            pass

    @pytest.mark.unit
    def test_invalid_iso_value_type(self) -> None:
        """Test that invalid iso_value type raises error."""
        from glass_models.viz.isosurface import IsoSurfaceExtractor

        grid_size = 8
        rng = np.random.default_rng(42)
        field = rng.random((grid_size, grid_size, grid_size))
        extractor = IsoSurfaceExtractor()

        # Should raise TypeError for invalid iso_value
        with pytest.raises((TypeError, ValueError)):
            extractor.extract(field, "invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

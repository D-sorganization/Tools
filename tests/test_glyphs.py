"""Test suite for glyph density control and vector field visualization.

Tests for GlyphDensityController covering:
- Subsampling indices calculation based on density
- Glyph style management (arrows, cones, spheres)
- Auto-scaling based on field magnitude
- Secondary field colormapping
- Performance requirements (<200ms updates)

See GitHub issue #547: Vector Field Density & Glyph Subsampling Control
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from glass_models.viz.glyphs import GlyphDensityController, GlyphStyle

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_mesh_3d() -> dict[str, int | np.ndarray]:
    """Create a sample 3D mesh with cell centers and field data."""
    # Create a simple 5x5x5 grid = 125 cells
    n_cells = 125
    cell_centers = np.random.RandomState(42).randn(n_cells, 3) * 10

    # Vector field (velocity-like)
    vector_field = np.random.RandomState(42).randn(n_cells, 3)

    # Scalar field for secondary colormapping
    scalar_field = np.random.RandomState(42).rand(n_cells) * 100

    return {
        "cell_centers": cell_centers,
        "vector_field": vector_field,
        "scalar_field": scalar_field,
        "n_cells": n_cells,
    }


@pytest.fixture
def sample_mesh_large() -> dict[str, int | np.ndarray]:
    """Create a larger mesh for performance testing."""
    n_cells = 10000
    cell_centers = np.random.RandomState(123).randn(n_cells, 3) * 50
    vector_field = np.random.RandomState(123).randn(n_cells, 3)
    scalar_field = np.random.RandomState(123).rand(n_cells) * 100

    return {
        "cell_centers": cell_centers,
        "vector_field": vector_field,
        "scalar_field": scalar_field,
        "n_cells": n_cells,
    }


@pytest.fixture
def controller(
    sample_mesh_3d: dict[str, int | np.ndarray],
) -> GlyphDensityController:
    """Create a GlyphDensityController with sample data."""
    return GlyphDensityController(
        cell_centers=sample_mesh_3d["cell_centers"],
        vector_field=sample_mesh_3d["vector_field"],
        scalar_field=sample_mesh_3d["scalar_field"],
    )


# =============================================================================
# Unit Tests: Density and Subsampling
# =============================================================================


class TestGlyphDensityController:
    """Test GlyphDensityController core functionality."""

    def test_initialization(self, sample_mesh_3d: dict[str, int | np.ndarray]) -> None:
        """Test controller initialization with valid data."""
        controller = GlyphDensityController(
            cell_centers=sample_mesh_3d["cell_centers"],
            vector_field=sample_mesh_3d["vector_field"],
            scalar_field=sample_mesh_3d["scalar_field"],
        )

        assert controller is not None
        assert controller.n_cells == 125
        assert controller.density == 1.0  # Default density
        assert controller.style == GlyphStyle.ARROWS  # Default style

    def test_initialization_defaults(
        self, sample_mesh_3d: dict[str, int | np.ndarray]
    ) -> None:
        """Test controller with minimal required parameters."""
        controller = GlyphDensityController(
            cell_centers=sample_mesh_3d["cell_centers"],
            vector_field=sample_mesh_3d["vector_field"],
        )

        assert controller is not None
        assert controller.scalar_field is None

    def test_density_validation(
        self, sample_mesh_3d: dict[str, int | np.ndarray]
    ) -> None:
        """Test that density must be between 0 and 1."""
        with pytest.raises(ValueError):
            GlyphDensityController(
                cell_centers=sample_mesh_3d["cell_centers"],
                vector_field=sample_mesh_3d["vector_field"],
                density=-0.1,
            )

        with pytest.raises(ValueError):
            GlyphDensityController(
                cell_centers=sample_mesh_3d["cell_centers"],
                vector_field=sample_mesh_3d["vector_field"],
                density=1.5,
            )

    def test_style_validation(
        self, sample_mesh_3d: dict[str, int | np.ndarray]
    ) -> None:
        """Test that style must be valid GlyphStyle enum."""
        with pytest.raises((ValueError, TypeError)):
            GlyphDensityController(
                cell_centers=sample_mesh_3d["cell_centers"],
                vector_field=sample_mesh_3d["vector_field"],
                style="invalid_style",  # type: ignore
            )

    def test_input_shape_validation(self) -> None:
        """Test that input arrays have consistent shapes."""
        cell_centers = np.random.randn(100, 3)
        vector_field = np.random.randn(50, 3)  # Wrong size

        with pytest.raises(ValueError):
            GlyphDensityController(
                cell_centers=cell_centers,
                vector_field=vector_field,
            )


class TestSubsamplingIndices:
    """Test subsample_indices() method with various density levels."""

    def test_subsample_density_100_percent(
        self, controller: GlyphDensityController
    ) -> None:
        """At 100% density, all cells should be included."""
        controller.density = 1.0
        indices = controller.get_subsample_indices()

        assert len(indices) == controller.n_cells
        assert np.all(np.isin(indices, np.arange(controller.n_cells)))

    def test_subsample_density_50_percent(
        self, controller: GlyphDensityController
    ) -> None:
        """At 50% density, approximately half should be selected."""
        controller.density = 0.5
        indices = controller.get_subsample_indices()

        expected = int(controller.n_cells * 0.5)
        actual = len(indices)

        # Allow ±5% tolerance due to random selection
        assert abs(actual - expected) <= int(controller.n_cells * 0.05)

    def test_subsample_density_10_percent(
        self, controller: GlyphDensityController
    ) -> None:
        """At 10% density, approximately 1/10 should be selected."""
        controller.density = 0.1
        indices = controller.get_subsample_indices()

        expected = max(1, int(controller.n_cells * 0.1))
        actual = len(indices)

        # Minimum 1 glyph, otherwise allow ±5% tolerance
        assert actual >= 1
        if expected > 1:
            assert abs(actual - expected) <= int(controller.n_cells * 0.05)

    def test_subsample_density_1_percent(
        self, controller: GlyphDensityController
    ) -> None:
        """At 1% density, at least 1 glyph should be selected."""
        controller.density = 0.01
        indices = controller.get_subsample_indices()

        assert len(indices) >= 1
        assert np.all(np.isin(indices, np.arange(controller.n_cells)))

    def test_subsample_reproducibility(
        self, controller: GlyphDensityController
    ) -> None:
        """Same density should produce same indices (deterministic)."""
        controller.density = 0.3
        indices1 = controller.get_subsample_indices()
        indices2 = controller.get_subsample_indices()

        np.testing.assert_array_equal(indices1, indices2)

    def test_subsample_indices_valid_range(
        self, controller: GlyphDensityController
    ) -> None:
        """All subsample indices must be valid cell indices."""
        controller.density = 0.5
        indices = controller.get_subsample_indices()

        assert np.all(indices >= 0)
        assert np.all(indices < controller.n_cells)
        assert len(np.unique(indices)) == len(indices)  # No duplicates


class TestDensityUpdates:
    """Test updating density and re-rendering."""

    def test_update_density_valid(self, controller: GlyphDensityController) -> None:
        """Test updating density to valid values."""
        for density in [0.1, 0.25, 0.5, 0.75, 1.0]:
            controller.set_density(density)
            assert controller.density == density

    def test_update_density_invalid(self, controller: GlyphDensityController) -> None:
        """Test that invalid densities are rejected."""
        with pytest.raises(ValueError):
            controller.set_density(-0.1)

        with pytest.raises(ValueError):
            controller.set_density(1.5)

    def test_density_update_invalidates_cache(
        self, controller: GlyphDensityController
    ) -> None:
        """Test that changing density invalidates cached indices."""
        controller.set_density(0.5)
        indices1 = controller.get_subsample_indices()

        controller.set_density(0.3)
        indices2 = controller.get_subsample_indices()

        # Different densities should (usually) produce different indices
        # (might occasionally match by random chance, but very unlikely)
        assert len(indices1) != len(indices2)


# =============================================================================
# Unit Tests: Glyph Styles
# =============================================================================


class TestGlyphStyles:
    """Test glyph style management and changes."""

    def test_all_styles_valid(self) -> None:
        """Test that all GlyphStyle enum members are accessible."""
        styles = [
            GlyphStyle.ARROWS,
            GlyphStyle.CONES,
            GlyphStyle.SPHERES,
        ]
        assert len(styles) == 3

    def test_style_initialization(
        self, sample_mesh_3d: dict[str, int | np.ndarray]
    ) -> None:
        """Test creating controller with each style."""
        for style in [GlyphStyle.ARROWS, GlyphStyle.CONES, GlyphStyle.SPHERES]:
            controller = GlyphDensityController(
                cell_centers=sample_mesh_3d["cell_centers"],
                vector_field=sample_mesh_3d["vector_field"],
                style=style,
            )
            assert controller.style == style

    def test_change_style(self, controller: GlyphDensityController) -> None:
        """Test changing glyph style after initialization."""
        assert controller.style == GlyphStyle.ARROWS

        controller.set_style(GlyphStyle.CONES)
        assert controller.style == GlyphStyle.CONES

        controller.set_style(GlyphStyle.SPHERES)
        assert controller.style == GlyphStyle.SPHERES

    def test_invalid_style_change(self, controller: GlyphDensityController) -> None:
        """Test that invalid styles are rejected."""
        with pytest.raises((ValueError, TypeError)):
            controller.set_style("invalid_style")  # type: ignore


# =============================================================================
# Unit Tests: Auto-Scaling and Field Magnitude
# =============================================================================


class TestAutoScaling:
    """Test auto-scaling based on vector field magnitude."""

    def test_scale_calculation_from_magnitude(
        self, controller: GlyphDensityController
    ) -> None:
        """Test that scale factors are derived from field magnitude."""
        scale_factors = controller.get_scale_factors()

        assert scale_factors.shape == (controller.n_cells,)
        assert np.all(scale_factors > 0)

    def test_scale_normalization(self, controller: GlyphDensityController) -> None:
        """Test that scale factors are normalized to [0, 1] range."""
        scale_factors = controller.get_scale_factors()

        assert np.all(scale_factors >= 0)
        assert np.all(scale_factors <= 1)
        assert np.max(scale_factors) > 0  # At least some non-zero

    def test_scale_magnitude_correlation(
        self,
        sample_mesh_3d: dict[str, int | np.ndarray],
    ) -> None:
        """Test that larger magnitude vectors get larger scale factors."""
        # Create controlled vector field: one strong, one weak
        vector_field = np.zeros((2, 3))
        vector_field[0] = [1.0, 0.0, 0.0]  # Magnitude 1
        vector_field[1] = [10.0, 0.0, 0.0]  # Magnitude 10

        controller = GlyphDensityController(
            cell_centers=np.array([[0, 0, 0], [1, 0, 0]]),
            vector_field=vector_field,
        )

        scale_factors = controller.get_scale_factors()
        assert scale_factors[1] > scale_factors[0]

    def test_custom_scale_factor(self, controller: GlyphDensityController) -> None:
        """Test setting a custom global scale factor."""
        controller.set_scale_factor(2.0)
        assert controller.scale_factor == 2.0


# =============================================================================
# Unit Tests: Colormapping
# =============================================================================


class TestColormapping:
    """Test secondary field colormapping."""

    def test_colormap_from_scalar_field(
        self, controller: GlyphDensityController
    ) -> None:
        """Test that colors are computed from scalar field."""
        colors = controller.get_colors()

        assert colors.shape == (controller.n_cells, 4)  # RGBA
        assert np.all(colors >= 0)
        assert np.all(colors <= 1)

    def test_colormap_type(self, controller: GlyphDensityController) -> None:
        """Test setting and retrieving colormap."""
        controller.set_colormap("viridis")
        assert controller.colormap == "viridis"

    def test_colormap_default(self, controller: GlyphDensityController) -> None:
        """Test default colormap."""
        assert controller.colormap in ["viridis", "jet", "coolwarm"]

    def test_no_colormap_without_scalar_field(
        self, sample_mesh_3d: dict[str, int | np.ndarray]
    ) -> None:
        """Test behavior when no scalar field is provided."""
        controller = GlyphDensityController(
            cell_centers=sample_mesh_3d["cell_centers"],
            vector_field=sample_mesh_3d["vector_field"],
        )

        # Should not raise; might return default colors
        colors = controller.get_colors()
        assert colors.shape == (controller.n_cells, 4)


# =============================================================================
# Integration Tests: Glyph Rendering
# =============================================================================


class TestGlyphRendering:
    """Test complete glyph rendering pipeline."""

    def test_get_glyph_data(self, controller: GlyphDensityController) -> None:
        """Test retrieving all glyph rendering data at once."""
        controller.set_density(0.5)

        glyph_data = controller.get_glyph_data()

        assert "positions" in glyph_data
        assert "vectors" in glyph_data
        assert "scale_factors" in glyph_data
        assert "colors" in glyph_data
        assert "style" in glyph_data

    def test_glyph_data_consistency(self, controller: GlyphDensityController) -> None:
        """Test that glyph data arrays have consistent sizes."""
        controller.set_density(0.5)
        glyph_data = controller.get_glyph_data()

        n_glyphs = len(glyph_data["positions"])
        assert len(glyph_data["vectors"]) == n_glyphs
        assert len(glyph_data["scale_factors"]) == n_glyphs
        assert len(glyph_data["colors"]) == n_glyphs

    def test_glyph_data_with_different_densities(
        self, controller: GlyphDensityController
    ) -> None:
        """Test glyph data at different density levels."""
        densities = [0.1, 0.3, 0.5, 1.0]

        for density in densities:
            controller.set_density(density)
            glyph_data = controller.get_glyph_data()

            expected_count = max(1, int(controller.n_cells * density))
            actual_count = len(glyph_data["positions"])

            # Allow some tolerance
            assert abs(actual_count - expected_count) <= 2


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests: updates must complete in <200ms."""

    def test_update_glyphs_performance_medium(
        self, sample_mesh_large: dict[str, int | np.ndarray]
    ) -> None:
        """Test update performance on medium mesh (10k cells)."""
        controller = GlyphDensityController(
            cell_centers=sample_mesh_large["cell_centers"],
            vector_field=sample_mesh_large["vector_field"],
            scalar_field=sample_mesh_large["scalar_field"],
        )

        start_time = time.perf_counter()
        controller.set_density(0.5)
        glyph_data = controller.get_glyph_data()
        elapsed = (time.perf_counter() - start_time) * 1000  # ms

        assert elapsed < 200, f"Update took {elapsed:.1f}ms, expected <200ms"
        assert len(glyph_data["positions"]) > 0

    def test_density_slider_rapid_updates(
        self, sample_mesh_large: dict[str, int | np.ndarray]
    ) -> None:
        """Test rapid density slider updates."""
        controller = GlyphDensityController(
            cell_centers=sample_mesh_large["cell_centers"],
            vector_field=sample_mesh_large["vector_field"],
            scalar_field=sample_mesh_large["scalar_field"],
        )

        densities = np.linspace(0.1, 1.0, 10)

        start_time = time.perf_counter()
        for density in densities:
            controller.set_density(density)
            _ = controller.get_glyph_data()
        elapsed = (time.perf_counter() - start_time) * 1000  # ms

        per_update = elapsed / len(densities)
        assert per_update < 200, (
            f"Average update took {per_update:.1f}ms, expected <200ms"
        )

    def test_style_change_performance(
        self, sample_mesh_large: dict[str, int | np.ndarray]
    ) -> None:
        """Test style change performance."""
        controller = GlyphDensityController(
            cell_centers=sample_mesh_large["cell_centers"],
            vector_field=sample_mesh_large["vector_field"],
        )

        styles = [GlyphStyle.ARROWS, GlyphStyle.CONES, GlyphStyle.SPHERES]

        start_time = time.perf_counter()
        for style in styles:
            controller.set_style(style)
            _ = controller.get_glyph_data()
        elapsed = (time.perf_counter() - start_time) * 1000  # ms

        per_update = elapsed / len(styles)
        assert per_update < 200, (
            f"Average style change took {per_update:.1f}ms, expected <200ms"
        )

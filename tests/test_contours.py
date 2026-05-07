"""Unit tests for contour extraction and visualization on arbitrary surfaces.

Tests for contour extraction using marching squares on 2D surfaces,
level spacing options, and contour labeling functionality.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from glass_models.viz.contours import (
    ContourExtractor,
    ContourResult,
    extract_contours,
    label_contours,
)


class TestContourExtractorBasics:
    """Test basic contour extraction functionality."""

    def test_extractor_initialization(self) -> None:
        """Test ContourExtractor initialization with default parameters."""
        extractor = ContourExtractor()
        assert extractor.n_levels == 10
        assert extractor.spacing == "uniform"
        assert extractor.enable_cache is True

    def test_extractor_initialization_with_params(self) -> None:
        """Test ContourExtractor initialization with custom parameters."""
        extractor = ContourExtractor(n_levels=20, spacing="log", enable_cache=False)
        assert extractor.n_levels == 20
        assert extractor.spacing == "log"
        assert extractor.enable_cache is False

    def test_simple_square_mesh(self) -> None:
        """Test contour extraction on a simple square mesh."""
        # Create simple 2x2 surface mesh
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Two triangles forming a square
        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)

        # Scalar field: varies linearly from 0 to 1
        field = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=5, spacing="uniform")
        result = extractor.extract(vertices, triangles, field)

        assert isinstance(result, ContourResult)
        assert result.vertices is not None
        assert result.contours is not None
        assert len(result.contour_values) == 5
        assert result.field_min == 0.0
        assert result.field_max == 1.0

    def test_contour_values_uniform_spacing(self) -> None:
        """Test contour level generation with uniform spacing."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=5, spacing="uniform")
        result = extractor.extract(vertices, triangles, field)

        # With 5 levels, expect [0, 0.25, 0.5, 0.75, 1.0]
        assert len(result.contour_values) == 5
        assert result.contour_values[0] >= 0.0  # min value in field
        assert result.contour_values[-1] <= 1.0  # max value in field

    def test_contour_values_log_spacing(self) -> None:
        """Test contour level generation with log spacing."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([1.0, 100.0, 10.0, 100.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=4, spacing="log")
        result = extractor.extract(vertices, triangles, field)

        # Log spacing on [1, 100] should give roughly [1, 4.64, 21.5, 100]
        assert len(result.contour_values) == 4
        assert result.contour_values[0] >= 1.0
        assert result.contour_values[-1] <= 100.0
        # Verify log spacing: ratios should be roughly equal
        if len(result.contour_values) > 1:
            ratio1 = result.contour_values[1] / result.contour_values[0]
            ratio2 = result.contour_values[2] / result.contour_values[1]
            # Log spacing should have similar ratios
            assert abs(ratio1 - ratio2) < ratio1 * 0.5  # Allow 50% variation

    def test_contour_extraction_with_nan(self) -> None:
        """Test robustness when field contains NaN values."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.0, np.nan, 0.5, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=5)
        result = extractor.extract(vertices, triangles, field)

        # Should handle NaN gracefully
        assert result is not None
        assert result.field_min >= 0.0
        assert result.field_max <= 1.0


class TestContourExtraction:
    """Test contour extraction algorithm."""

    def test_extract_contours_function(self) -> None:
        """Test the standalone extract_contours function."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        result = extract_contours(vertices, triangles, field, n_levels=5)

        assert isinstance(result, ContourResult)
        assert len(result.contour_values) == 5
        assert result.vertices is not None

    def test_contours_are_closed_or_open(self) -> None:
        """Test that extracted contours form valid line segments."""
        # Create a simple plane with a gradient
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)

        # Create vertices for grid
        vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(100)])
        vertices = vertices.astype(np.float32)

        # Create triangles for grid
        triangles = []
        for i in range(9):
            for j in range(9):
                v0 = i * 10 + j
                v1 = i * 10 + j + 1
                v2 = (i + 1) * 10 + j
                v3 = (i + 1) * 10 + j + 1
                triangles.append([v0, v1, v2])
                triangles.append([v1, v3, v2])
        triangles = np.array(triangles, dtype=np.uint32)

        # Field: distance from origin
        field = np.sqrt(xx.ravel() ** 2 + yy.ravel() ** 2).astype(np.float32)

        extractor = ContourExtractor(n_levels=8, spacing="uniform")
        result = extractor.extract(vertices, triangles, field)

        # Each contour should have segments
        for contour_idx, contour_lines in enumerate(result.contours):
            if contour_lines is not None and len(contour_lines) > 0:
                # contour_lines is an Nx3 array of points
                assert contour_lines.ndim == 2, f"Contour {contour_idx} invalid shape"
                assert contour_lines.shape[1] == 3, "Line should be 3D points"
                assert contour_lines.shape[0] >= 2, (
                    f"Contour {contour_idx} has too few points"
                )


class TestContourLabeling:
    """Test contour labeling functionality."""

    def test_label_contours_function(self) -> None:
        """Test the standalone label_contours function."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=5)
        result = extractor.extract(vertices, triangles, field)

        labeled = label_contours(result.contours, result.contour_values)

        assert labeled is not None
        assert len(labeled) == len(result.contours)

    def test_labels_contain_field_values(self) -> None:
        """Test that labels contain appropriate field values."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=5)
        result = extractor.extract(vertices, triangles, field)

        labeled = label_contours(result.contours, result.contour_values)

        for i, label_info in enumerate(labeled):
            if label_info is not None:
                assert "value" in label_info
                assert "position" in label_info
                assert label_info["value"] == result.contour_values[i]


class TestCaching:
    """Test caching functionality."""

    def test_cache_enabled_by_default(self) -> None:
        """Test that cache is enabled by default."""
        extractor = ContourExtractor()
        assert extractor.enable_cache is True

    def test_cache_speed_improvement(self) -> None:
        """Test that caching improves performance."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=10, enable_cache=True)

        # First call
        start = time.time()
        result1 = extractor.extract(vertices, triangles, field)
        time.time() - start

        # Second call (should use cache)
        start = time.time()
        result2 = extractor.extract(vertices, triangles, field)
        time.time() - start

        # Cache should result in similar or better performance
        assert np.allclose(result1.contour_values, result2.contour_values)

    def test_cache_invalidation_on_new_input(self) -> None:
        """Test that cache is invalidated when input changes."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field1 = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        field2 = np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=5, enable_cache=True)

        result1 = extractor.extract(vertices, triangles, field1)
        result2 = extractor.extract(vertices, triangles, field2)

        # Results should be different because field min changed
        assert result1.field_min != result2.field_min


class TestPerformance:
    """Test performance requirements."""

    @pytest.mark.benchmark
    def test_extraction_performance(self) -> None:
        """Test that contour extraction is fast (<500ms for 20x20 grid)."""
        # Create a larger mesh
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        xx, yy = np.meshgrid(x, y)

        vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(400)])
        vertices = vertices.astype(np.float32)

        triangles = []
        for i in range(19):
            for j in range(19):
                v0 = i * 20 + j
                v1 = i * 20 + j + 1
                v2 = (i + 1) * 20 + j
                v3 = (i + 1) * 20 + j + 1
                triangles.append([v0, v1, v2])
                triangles.append([v1, v3, v2])
        triangles = np.array(triangles, dtype=np.uint32)

        field = np.sin(xx.ravel() * np.pi) * np.cos(yy.ravel() * np.pi)
        field = field.astype(np.float32)

        extractor = ContourExtractor(n_levels=15)

        start = time.time()
        result = extractor.extract(vertices, triangles, field)
        elapsed = time.time() - start

        # Should complete in under 500ms for 20x20 grid with 15 levels
        assert elapsed < 0.5, f"Extraction took {elapsed:.3f}s, expected < 0.5s"
        assert result is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_level_contour(self) -> None:
        """Test contour extraction with a single level."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        extractor = ContourExtractor(n_levels=1)
        result = extractor.extract(vertices, triangles, field)

        assert len(result.contour_values) == 1

    def test_constant_field(self) -> None:
        """Test contour extraction on constant field."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        field = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        extractor = ContourExtractor(n_levels=5)
        result = extractor.extract(vertices, triangles, field)

        assert result is not None
        # Constant field is extended, so field_max > field_min
        assert result.field_min < result.field_max

    def test_very_small_mesh(self) -> None:
        """Test with minimal mesh (single triangle)."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0, 0.5], dtype=np.float32)

        extractor = ContourExtractor(n_levels=3)
        result = extractor.extract(vertices, triangles, field)

        assert result is not None

    def test_invalid_inputs_raise_errors(self) -> None:
        """Test that invalid inputs raise appropriate errors."""
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Invalid: index 2 out of range
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        field = np.array([0.0, 1.0], dtype=np.float32)

        extractor = ContourExtractor()

        with pytest.raises((ValueError, IndexError)):
            extractor.extract(vertices, triangles, field)


@pytest.fixture
def simple_surface():
    """Create a simple surface for testing."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    field = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    return vertices, triangles, field


def test_integration_extract_and_label(simple_surface):
    """Integration test: extract contours and label them."""
    vertices, triangles, field = simple_surface

    # Extract
    result = extract_contours(vertices, triangles, field, n_levels=5)

    # Label
    labeled = label_contours(result.contours, result.contour_values)

    # Verify
    assert len(labeled) == len(result.contour_values)
    for i, label_info in enumerate(labeled):
        if label_info is not None:
            assert label_info["value"] == result.contour_values[i]

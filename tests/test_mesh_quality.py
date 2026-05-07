"""Test suite for mesh quality and element metrics visualization.

This module implements TDD for GitHub issue #549: Mesh Quality & Element Metrics.
Tests are organized by:
1. Unit tests on analytical meshes (metric calculations)
2. Statistics computation validation
3. Problematic element identification
4. Performance tests

Success criteria:
- All metric calculation tests pass
- Statistics match analytical values
- Problematic elements correctly identified
- Performance meets <500ms target
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMeshQualityAnalyzerBasics:
    """Test MeshQualityAnalyzer class initialization and basic operations."""

    @pytest.mark.unit
    def test_analyzer_initialization(self) -> None:
        """Test MeshQualityAnalyzer can be created."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        analyzer = MeshQualityAnalyzer()
        assert analyzer is not None

    @pytest.mark.unit
    def test_analyzer_with_simple_tetrahedron(self) -> None:
        """Test analyzer with a single tetrahedron element.

        Creates a regular tetrahedron and verifies it can be analyzed.
        Regular tetrahedron has skewness = 0 (perfect).
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Regular tetrahedron vertices
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, np.sqrt(2.0 / 3.0)],
            ],
            dtype=np.float64,
        )

        # Single tetrahedron element
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)

        assert analyzer.vertices is not None
        assert analyzer.elements is not None
        assert len(analyzer.vertices) == 4
        assert len(analyzer.elements) == 1


class TestAspectRatioComputation:
    """Test aspect ratio calculations per element."""

    @pytest.mark.unit
    def test_aspect_ratio_unit_cube(self) -> None:
        """Test aspect ratio of a perfect cube hexahedron.

        A cube has edges (length 1) and diagonals (length sqrt(2) and sqrt(3)).
        AR = max_edge / min_edge = sqrt(3) / 1 = 1.732.
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Unit cube vertices
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Hex element: vertices in order
        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        aspect_ratios = analyzer.compute_aspect_ratios()

        assert aspect_ratios is not None
        assert len(aspect_ratios) == 1
        # Unit cube AR = sqrt(3) / 1 = 1.732
        assert 1.7 < aspect_ratios[0] < 1.8, f"AR={aspect_ratios[0]}"

    @pytest.mark.unit
    def test_aspect_ratio_stretched_element(self) -> None:
        """Test aspect ratio of a stretched (elongated) element.

        A 10x1x1 box should have AR = 10 (max/min dimension ratio).
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Stretched box: 10 units in x, 1 in y, 1 in z
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [10.0, 0.0, 1.0],
                [10.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        aspect_ratios = analyzer.compute_aspect_ratios()

        assert aspect_ratios is not None
        assert len(aspect_ratios) == 1
        # Stretched element should have AR ~ 10
        assert 9.0 < aspect_ratios[0] < 11.0, f"AR={aspect_ratios[0]}"

    @pytest.mark.unit
    def test_aspect_ratio_multiple_elements(self) -> None:
        """Test aspect ratio computation for multiple elements."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create 2 unit cubes
        vertices = np.array(
            [
                # Cube 1
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                # Cube 2
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [3.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        elements = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
            dtype=np.int32,
        )

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        aspect_ratios = analyzer.compute_aspect_ratios()

        assert aspect_ratios is not None
        assert len(aspect_ratios) == 2
        # Both should have AR = sqrt(3) / 1 = 1.732
        for ar in aspect_ratios:
            assert 1.7 < ar < 1.8


class TestSkewnessComputation:
    """Test skewness metric calculations (0=perfect, 1=degenerate)."""

    @pytest.mark.unit
    def test_skewness_regular_tetrahedron(self) -> None:
        """Test skewness of a regular tetrahedron.

        Regular tetrahedron should have skewness near 0 (perfect).
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Regular tetrahedron
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3) / 2, 0.0],
                [0.5, np.sqrt(3) / 6, np.sqrt(2.0 / 3.0)],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        skewness = analyzer.compute_skewness()

        assert skewness is not None
        assert len(skewness) == 1
        # Regular tetrahedron should have low skewness
        assert 0.0 <= skewness[0] < 0.2, f"Skewness={skewness[0]}"

    @pytest.mark.unit
    def test_skewness_degenerate_element(self) -> None:
        """Test skewness of a degenerate (flat) element.

        A flattened tetrahedron (all points in one plane) should have
        skewness approaching 1.0.
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Degenerate tetrahedron: all points in XY plane
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.289, 0.0],  # All at z=0
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        skewness = analyzer.compute_skewness()

        assert skewness is not None
        assert len(skewness) == 1
        # Degenerate element should have high skewness
        assert skewness[0] > 0.8, f"Skewness={skewness[0]}"

    @pytest.mark.unit
    def test_skewness_range(self) -> None:
        """Test that skewness values are in valid range [0, 1]."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create mixed quality elements
        vertices = np.array(
            [
                # Good tetrahedron
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.289, 0.866],
                # Poor tetrahedron
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.5, 0.1, 0.0],
                [2.5, 0.05, 0.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        skewness = analyzer.compute_skewness()

        assert skewness is not None
        for s in skewness:
            assert 0.0 <= s <= 1.0, f"Skewness out of range: {s}"


class TestJacobianComputation:
    """Test Jacobian determinant calculations per element."""

    @pytest.mark.unit
    def test_jacobian_unit_tetrahedron(self) -> None:
        """Test Jacobian determinant of a unit tetrahedron.

        Unit tetrahedron should have positive Jacobian.
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        jacobians = analyzer.compute_jacobian()

        assert jacobians is not None
        assert len(jacobians) == 1
        # Unit tetrahedron has Jacobian determinant = 1/6 (for element volume)
        assert jacobians[0] > 0, f"Jacobian should be positive: {jacobians[0]}"

    @pytest.mark.unit
    def test_jacobian_inverted_element(self) -> None:
        """Test Jacobian of an inverted (negative volume) element.

        Inverted element should have negative Jacobian.
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Inverted tetrahedron: vertices in reverse order
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        jacobians = analyzer.compute_jacobian()

        assert jacobians is not None
        # Inverted element should have negative Jacobian
        assert jacobians[0] < 0, f"Jacobian should be negative: {jacobians[0]}"

    @pytest.mark.unit
    def test_jacobian_multiple_elements(self) -> None:
        """Test Jacobian computation for multiple elements."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        vertices = np.array(
            [
                # Element 1
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                # Element 2
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        jacobians = analyzer.compute_jacobian()

        assert jacobians is not None
        assert len(jacobians) == 2


class TestStatisticsComputation:
    """Test statistics computation (min, max, mean, std)."""

    @pytest.mark.unit
    def test_statistics_aspect_ratio(self) -> None:
        """Test statistics calculation for aspect ratios."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create 3 cubes with different sizes
        vertices = np.array(
            [
                # 1x1x1 cube
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                # 2x2x2 cube
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
                [4.0, 0.0, 2.0],
                [4.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                # 3x3x3 cube
                [5.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [8.0, 3.0, 0.0],
                [5.0, 3.0, 0.0],
                [5.0, 0.0, 3.0],
                [8.0, 0.0, 3.0],
                [8.0, 3.0, 3.0],
                [5.0, 3.0, 3.0],
            ],
            dtype=np.float64,
        )

        elements = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
            ],
            dtype=np.int32,
        )

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        stats = analyzer.get_statistics("aspect_ratio")

        assert stats is not None
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["min"] > 0
        assert stats["max"] >= stats["min"]
        assert stats["mean"] >= stats["min"]
        assert stats["mean"] <= stats["max"]

    @pytest.mark.unit
    def test_statistics_skewness(self) -> None:
        """Test statistics for skewness metric."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create mixed quality tetrahedra
        vertices = np.array(
            [
                # Good tet
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.289, 0.866],
                # Poor tet
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.5, 0.1, 0.0],
                [2.5, 0.05, 0.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        stats = analyzer.get_statistics("skewness")

        assert stats is not None
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert 0 <= stats["min"] <= 1
        assert 0 <= stats["max"] <= 1

    @pytest.mark.unit
    def test_statistics_jacobian(self) -> None:
        """Test statistics for Jacobian metric."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        stats = analyzer.get_statistics("jacobian")

        assert stats is not None
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats


class TestProblematicElementDetection:
    """Test identification of problematic elements (>10% skewness)."""

    @pytest.mark.unit
    def test_flag_problematic_elements_skewness_threshold(self) -> None:
        """Test that elements with skewness > 0.1 are flagged."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create 2 elements: one good, one poor
        vertices = np.array(
            [
                # Good tet: regular tetrahedron
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.289, 0.866],
                # Poor tet: nearly degenerate
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.5, 0.05, 0.0],
                [2.5, 0.02, 0.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        problematic = analyzer.get_problematic_elements(threshold=0.1)

        assert problematic is not None
        # Should find at least element 1 as problematic
        assert 1 in problematic, f"Poor element not flagged: {problematic}"

    @pytest.mark.unit
    def test_problematic_elements_count(self) -> None:
        """Test counting problematic elements."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create 5 elements: 2 good, 3 poor
        vertices = np.array(
            [
                # Good tets
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.289, 0.866],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.5, 0.866, 0.0],
                [1.5, 0.289, 0.866],
                # Poor tets
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [3.5, 0.05, 0.0],
                [3.5, 0.02, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.5, 0.05, 0.0],
                [5.5, 0.02, 0.0],
                [7.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [7.5, 0.04, 0.0],
                [7.5, 0.01, 0.0],
            ],
            dtype=np.float64,
        )

        elements = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19],
            ],
            dtype=np.int32,
        )

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        problematic = analyzer.get_problematic_elements(threshold=0.1)

        assert len(problematic) >= 2, f"Should find at least 2 poor elements: {problematic}"

    @pytest.mark.unit
    def test_problematic_elements_empty(self) -> None:
        """Test that good mesh has no problematic elements."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create 3 good tetrahedra
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.289, 0.866],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.5, 0.866, 0.0],
                [1.5, 0.289, 0.866],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.5, 0.866, 0.0],
                [2.5, 0.289, 0.866],
            ],
            dtype=np.float64,
        )

        elements = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=np.int32
        )

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)
        problematic = analyzer.get_problematic_elements(threshold=0.1)

        assert isinstance(problematic, list)
        # Should have few or no problematic elements
        assert len(problematic) < len(elements)


class TestPerformance:
    """Test performance requirements (<500ms for computation)."""

    @pytest.mark.performance
    @pytest.mark.unit
    def test_performance_large_mesh(self) -> None:
        """Test performance on larger mesh (1000+ elements).

        Should compute all metrics in <500ms.
        """
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Create a grid of tetrahedra: 10x10x10 = 1000 elements
        grid_size = 10
        vertices_list = []
        elements_list = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    base_idx = len(vertices_list)
                    # Add 4 vertices for a tetrahedron
                    x, y, z = float(i), float(j), float(k)
                    vertices_list.extend(
                        [
                            [x, y, z],
                            [x + 1.0, y, z],
                            [x + 0.5, y + 0.866, z],
                            [x + 0.5, y + 0.289, z + 0.866],
                        ]
                    )
                    elements_list.append([base_idx, base_idx + 1, base_idx + 2, base_idx + 3])

        vertices = np.array(vertices_list, dtype=np.float64)
        elements = np.array(elements_list, dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)

        # Time all metric computations
        start = time.time()
        aspect_ratios = analyzer.compute_aspect_ratios()
        skewness = analyzer.compute_skewness()
        jacobians = analyzer.compute_jacobian()
        stats_ar = analyzer.get_statistics("aspect_ratio")
        stats_sk = analyzer.get_statistics("skewness")
        stats_jc = analyzer.get_statistics("jacobian")
        elapsed = time.time() - start

        assert aspect_ratios is not None
        assert skewness is not None
        assert jacobians is not None
        assert stats_ar is not None
        assert stats_sk is not None
        assert stats_jc is not None
        assert elapsed < 0.5, f"Performance too slow: {elapsed:.3f}s"


class TestEdgeCases:
    """Test edge case handling."""

    @pytest.mark.unit
    def test_single_element_mesh(self) -> None:
        """Test mesh with single element."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)

        ar = analyzer.compute_aspect_ratios()
        sk = analyzer.compute_skewness()
        jc = analyzer.compute_jacobian()

        assert len(ar) == 1
        assert len(sk) == 1
        assert len(jc) == 1

    @pytest.mark.unit
    def test_empty_mesh(self) -> None:
        """Test handling of empty mesh."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        analyzer = MeshQualityAnalyzer()
        vertices = np.array([], dtype=np.float64).reshape(0, 3)
        elements = np.array([], dtype=np.int32).reshape(0, 4)

        analyzer.set_mesh(vertices, elements)

        ar = analyzer.compute_aspect_ratios()
        assert len(ar) == 0

    @pytest.mark.unit
    def test_nan_handling(self) -> None:
        """Test that invalid inputs produce valid results or raise."""
        from glass_models.viz.mesh_quality import MeshQualityAnalyzer

        # Vertices with some very small coordinates
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-10, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        analyzer = MeshQualityAnalyzer()
        analyzer.set_mesh(vertices, elements)

        ar = analyzer.compute_aspect_ratios()
        assert not np.any(np.isnan(ar))
        assert not np.any(np.isinf(ar))

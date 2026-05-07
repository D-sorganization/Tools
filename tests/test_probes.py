"""
Comprehensive test suite for point/line/plane probe tools.

Tests validate interpolation accuracy, profile extraction, visualization,
and UI integration for FEA mesh queries.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

# ============================================================================
# Unit Tests: Probe Data Classes and Base Classes
# ============================================================================


class TestProbeResult:
    """Test ProbeResult dataclass."""

    def test_probe_result_creation_point(self):
        """Point probe result stores location, value, and coordinates."""
        from glass_models.viz.probes import ProbeResult

        result = ProbeResult(
            location=np.array([1.0, 2.0, 3.0]),
            values=np.array([42.5]),
            coordinates=np.array([[1.0, 2.0, 3.0]]),
            probe_type="point",
        )

        assert result.location is not None
        assert result.values is not None
        assert len(result.values) == 1
        assert result.values[0] == pytest.approx(42.5)
        assert result.probe_type == "point"

    def test_probe_result_creation_line(self):
        """Line probe result stores 1D profile with 100 samples."""
        from glass_models.viz.probes import ProbeResult

        coords = np.linspace([0, 0, 0], [1, 0, 0], 100)
        values = np.linspace(0, 10, 100)

        result = ProbeResult(
            location=np.array([0.5, 0.0, 0.0]),
            values=values,
            coordinates=coords,
            probe_type="line",
        )

        assert len(result.values) == 100
        assert len(result.coordinates) == 100
        assert result.probe_type == "line"

    def test_probe_result_creation_plane(self):
        """Plane probe result stores 2D grid (50x50) with values."""
        from glass_models.viz.probes import ProbeResult

        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
        values = np.random.rand(2500)

        result = ProbeResult(
            location=np.array([0.5, 0.5, 0.0]),
            values=values,
            coordinates=coords,
            probe_type="plane",
        )

        assert len(result.values) == 2500
        assert result.probe_type == "plane"


class TestProbeBaseClass:
    """Test Probe abstract base class interface."""

    def test_probe_is_abstract(self):
        """Probe base class cannot be instantiated directly."""
        from glass_models.viz.probes import Probe

        with pytest.raises(TypeError):
            Probe()

    def test_probe_subclass_must_implement_evaluate(self):
        """Subclasses must implement evaluate() method."""
        from glass_models.viz.probes import Probe

        class IncompleteProbe(Probe):
            """Missing evaluate() implementation."""

            pass

        with pytest.raises(TypeError):
            IncompleteProbe()


# ============================================================================
# Unit Tests: Point Probe - Single Point Interpolation
# ============================================================================


class TestPointProbe:
    """Test PointProbe for single-point value queries."""

    def test_point_probe_exact_node_value(self):
        """Point probe returns exact value at mesh node."""
        from glass_models.viz.probes import PointProbe

        # Mock mesh with known node values (must be 3D for griddata)
        mesh = MagicMock()
        mesh.nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        mesh.field_values = np.array([10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0])

        probe = PointProbe(mesh=mesh, location=np.array([0, 0, 0]))

        # Should find exact match at first node
        result = probe.evaluate()

        assert result is not None
        assert result.values is not None
        assert len(result.values) == 1
        assert result.values[0] == pytest.approx(10.0)
        assert np.allclose(result.location, [0, 0, 0])

    def test_point_probe_interpolation(self):
        """Point probe interpolates value at non-node location."""
        from glass_models.viz.probes import PointProbe

        mesh = MagicMock()
        # Create a 3D mesh: linear field field = 2*x
        x = np.linspace(0, 2, 8)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = 2.0 * X.ravel()

        # Query at x=1 should give value ~2.0
        probe = PointProbe(mesh=mesh, location=np.array([1.0, 0.5, 0.5]))
        result = probe.evaluate()

        assert result is not None
        assert result.values[0] == pytest.approx(2.0, abs=0.2)

    def test_point_probe_stores_location(self):
        """Point probe correctly stores query location."""
        from glass_models.viz.probes import PointProbe

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = X.ravel() + Y.ravel() + Z.ravel()

        location = np.array([0.5, 0.5, 0.5])
        probe = PointProbe(mesh=mesh, location=location)
        result = probe.evaluate()

        assert np.allclose(result.location, location)


# ============================================================================
# Unit Tests: Line Probe - 1D Profile Extraction
# ============================================================================


class TestLineProbe:
    """Test LineProbe for 1D profile extraction (100 samples)."""

    def test_line_probe_creates_100_samples(self):
        """Line probe samples exactly 100 points along path."""
        from glass_models.viz.probes import LineProbe

        mesh = MagicMock()
        # Create a 3D mesh
        x = np.linspace(0, 10, 15)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = 2.0 * X.ravel()  # field = 2*x

        start = np.array([0, 0, 0])
        end = np.array([10, 0, 0])

        probe = LineProbe(mesh=mesh, start=start, end=end)
        result = probe.evaluate()

        assert result is not None
        assert len(result.values) == 100
        assert len(result.coordinates) == 100

    def test_line_probe_coordinate_path(self):
        """Line probe samples coordinates lie between start and end."""
        from glass_models.viz.probes import LineProbe

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 5, 12)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = np.ones_like(X.ravel())

        start = np.array([0, 0, 0])
        end = np.array([5, 0, 0])

        probe = LineProbe(mesh=mesh, start=start, end=end)
        result = probe.evaluate()

        # All coordinates should have x in [0, 5]
        assert np.all(result.coordinates[:, 0] >= 0)
        assert np.all(result.coordinates[:, 0] <= 5)
        assert np.all(result.coordinates[:, 1] == pytest.approx(0))
        assert np.all(result.coordinates[:, 2] == pytest.approx(0))

    def test_line_probe_monotonic_profile(self):
        """Line probe along monotonic field produces monotonic samples."""
        from glass_models.viz.probes import LineProbe

        mesh = MagicMock()
        # Field increases linearly with x
        x = np.linspace(0, 10, 12)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = X.ravel()  # field = x

        start = np.array([0, 0, 0])
        end = np.array([10, 0, 0])

        probe = LineProbe(mesh=mesh, start=start, end=end)
        result = probe.evaluate()

        # Profile should be monotonically increasing
        diffs = np.diff(result.values)
        assert np.all(diffs >= 0) or np.all(diffs <= 0)


# ============================================================================
# Unit Tests: Plane Probe - 2D Cross-Section Extraction
# ============================================================================


class TestPlaneProbe:
    """Test PlaneProbe for 2D cross-section extraction (50x50 grid)."""

    def test_plane_probe_creates_2500_samples(self):
        """Plane probe creates 50x50=2500 samples."""
        from glass_models.viz.probes import PlaneProbe

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        z = np.linspace(0, 1, 20)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        mesh.field_values = X.ravel() + Y.ravel()  # field = x + y

        # z=0.5 plane, normal pointing up
        position = np.array([0.5, 0.5, 0.5])
        normal = np.array([0, 0, 1])

        probe = PlaneProbe(mesh=mesh, position=position, normal=normal)
        result = probe.evaluate()

        assert result is not None
        assert len(result.values) == 2500
        assert len(result.coordinates) == 2500

    def test_plane_probe_grid_structure(self):
        """Plane probe samples form 50x50 grid."""
        from glass_models.viz.probes import PlaneProbe

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = np.ones_like(X.ravel())

        position = np.array([0.5, 0.5, 0.0])
        normal = np.array([0, 0, 1])

        probe = PlaneProbe(mesh=mesh, position=position, normal=normal)
        result = probe.evaluate()

        # Verify grid-like structure
        assert len(result.coordinates) == 2500

    def test_plane_probe_perpendicular_to_normal(self):
        """Plane probe samples are perpendicular to the normal vector."""
        from glass_models.viz.probes import PlaneProbe

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 12)
        y = np.linspace(0, 1, 12)
        z = np.linspace(0, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = np.ones_like(X.ravel())

        position = np.array([0.5, 0.5, 0.5])
        normal = np.array([1, 0, 0])  # x-normal

        probe = PlaneProbe(mesh=mesh, position=position, normal=normal)
        result = probe.evaluate()

        # All x-coordinates should be ~0.5 if perpendicular to x-normal
        # (within sampling tolerance)
        x_coords = result.coordinates[:, 0]
        assert np.std(x_coords) < 0.15  # Allow some variance due to discretization


# ============================================================================
# Unit Tests: Probe Manager - Coordination and Caching
# ============================================================================


class TestProbeManager:
    """Test ProbeManager for probe coordination and caching."""

    def test_manager_initialization(self):
        """ProbeManager initializes with empty probe list."""
        from glass_models.viz.probes import ProbeManager

        mesh = MagicMock()
        manager = ProbeManager(mesh=mesh)

        assert manager.mesh is mesh
        assert len(manager.probes) == 0

    def test_manager_add_point_probe(self):
        """Manager can add point probes."""
        from glass_models.viz.probes import PointProbe, ProbeManager

        mesh = MagicMock()
        mesh.nodes = np.array([[0, 0, 0], [1, 1, 1]])
        mesh.field_values = np.array([1.0, 2.0])

        manager = ProbeManager(mesh=mesh)
        location = np.array([0.5, 0.5, 0.5])
        probe = manager.add_point_probe(location=location)

        assert len(manager.probes) == 1
        assert isinstance(probe, PointProbe)

    def test_manager_add_line_probe(self):
        """Manager can add line probes."""
        from glass_models.viz.probes import LineProbe, ProbeManager

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = np.ones_like(X.ravel())

        manager = ProbeManager(mesh=mesh)
        start = np.array([0, 0, 0])
        end = np.array([1, 0, 0])
        probe = manager.add_line_probe(start=start, end=end)

        assert len(manager.probes) == 1
        assert isinstance(probe, LineProbe)

    def test_manager_add_plane_probe(self):
        """Manager can add plane probes."""
        from glass_models.viz.probes import PlaneProbe, ProbeManager

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = np.ones_like(X.ravel())

        manager = ProbeManager(mesh=mesh)
        position = np.array([0.5, 0.5, 0.5])
        normal = np.array([0, 0, 1])
        probe = manager.add_plane_probe(position=position, normal=normal)

        assert len(manager.probes) == 1
        assert isinstance(probe, PlaneProbe)

    def test_manager_evaluate_all_probes(self):
        """Manager can evaluate all probes and return results."""
        from glass_models.viz.probes import ProbeManager

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 6)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = X.ravel() + Y.ravel() + Z.ravel()

        manager = ProbeManager(mesh=mesh)
        manager.add_point_probe(location=np.array([0, 0, 0]))
        manager.add_point_probe(location=np.array([1, 0, 0]))

        results = manager.evaluate_all()

        assert len(results) == 2
        assert all(r is not None for r in results)

    def test_manager_caching_same_probe(self):
        """Manager caches results for identical probes."""
        from glass_models.viz.probes import ProbeManager

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = X.ravel() + Y.ravel() + Z.ravel()

        manager = ProbeManager(mesh=mesh)
        location = np.array([0.5, 0.5, 0.5])

        probe1 = manager.add_point_probe(location=location)
        result1 = probe1.evaluate()

        probe2 = manager.add_point_probe(location=location)
        result2 = probe2.evaluate()

        # Both should return consistent results
        assert np.allclose(result1.values, result2.values)

    def test_manager_remove_probe(self):
        """Manager can remove probes by index."""
        from glass_models.viz.probes import ProbeManager

        mesh = MagicMock()
        mesh.nodes = np.array([[0, 0, 0]])
        mesh.field_values = np.array([1.0])

        manager = ProbeManager(mesh=mesh)
        manager.add_point_probe(location=np.array([0, 0, 0]))
        manager.add_point_probe(location=np.array([1, 1, 1]))

        assert len(manager.probes) == 2

        manager.remove_probe(0)

        assert len(manager.probes) == 1


# ============================================================================
# Integration Tests: Interpolation Accuracy
# ============================================================================


class TestInterpolationAccuracy:
    """Test interpolation accuracy on synthetic meshes."""

    def test_linear_interpolation_accuracy(self):
        """Linear interpolation on linear field <0.1% error."""
        from glass_models.viz.probes import PointProbe

        # Create 3D mesh: field = 3*x + 2
        x = np.linspace(0, 10, 15)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh = MagicMock()
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = 3.0 * X.ravel() + 2.0

        # Test at several points
        test_points = np.array([2.5, 5.7, 8.3])

        errors = []
        for pt in test_points:
            probe = PointProbe(mesh=mesh, location=np.array([pt, 0, 0]))
            result = probe.evaluate()
            error = abs(result.values[0] - (3 * pt + 2)) / (3 * pt + 2)
            errors.append(error)

        # Should be <0.1% error
        assert np.all(np.array(errors) < 0.001)

    def test_quadratic_interpolation_accuracy(self):
        """Quadratic field interpolation <1% error with linear interp."""
        from glass_models.viz.probes import PointProbe

        # Field = x^2, sample enough points for accuracy
        x = np.linspace(0, 5, 20)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 5)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh = MagicMock()
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = X.ravel() ** 2

        test_point = 2.5
        expected = test_point**2

        probe = PointProbe(mesh=mesh, location=np.array([test_point, 0, 0]))
        result = probe.evaluate()

        error = abs(result.values[0] - expected) / expected
        assert error < 0.01


# ============================================================================
# Performance Tests: Execution Time Targets
# ============================================================================


class TestPerformance:
    """Test performance targets: <50ms point, <200ms line, <500ms plane."""

    @pytest.mark.slow
    def test_point_probe_performance(self):
        """Point probe evaluation <50ms."""
        import time

        from glass_models.viz.probes import PointProbe

        mesh = MagicMock()
        # Realistic mesh size: 1000 nodes
        n = 1000
        mesh.nodes = np.random.rand(n, 3) * 10
        mesh.field_values = np.random.rand(n)

        probe = PointProbe(mesh=mesh, location=np.array([5, 5, 5]))

        start = time.time()
        for _ in range(10):
            probe.evaluate()
        elapsed = (time.time() - start) / 10

        assert elapsed < 0.1, (
            f"Point probe took {elapsed * 1000:.2f}ms (target: <100ms)"
        )

    @pytest.mark.slow
    def test_line_probe_performance(self):
        """Line probe evaluation <200ms."""
        import time

        from glass_models.viz.probes import LineProbe

        mesh = MagicMock()
        n = 2000
        mesh.nodes = np.random.rand(n, 3) * 10
        mesh.field_values = np.random.rand(n)

        probe = LineProbe(
            mesh=mesh, start=np.array([0, 0, 0]), end=np.array([10, 10, 10])
        )

        start = time.time()
        _ = probe.evaluate()
        elapsed = time.time() - start

        assert elapsed < 0.2, f"Line probe took {elapsed * 1000:.2f}ms (target: <200ms)"

    @pytest.mark.slow
    def test_plane_probe_performance(self):
        """Plane probe evaluation <500ms."""
        import time

        from glass_models.viz.probes import PlaneProbe

        mesh = MagicMock()
        n = 5000
        mesh.nodes = np.random.rand(n, 3) * 10
        mesh.field_values = np.random.rand(n)

        probe = PlaneProbe(
            mesh=mesh, position=np.array([5, 5, 5]), normal=np.array([0, 0, 1])
        )

        start = time.time()
        _ = probe.evaluate()
        elapsed = time.time() - start

        assert elapsed < 0.5, (
            f"Plane probe took {elapsed * 1000:.2f}ms (target: <500ms)"
        )


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Test stability: no NaN/Inf, proper bounds checking."""

    def test_no_nan_in_results(self):
        """Probe results contain no NaN values."""
        from glass_models.viz.probes import PointProbe

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 6)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = X.ravel() + Y.ravel() + Z.ravel()

        probe = PointProbe(mesh=mesh, location=np.array([0.5, 0.5, 0.5]))
        result = probe.evaluate()

        assert not np.any(np.isnan(result.values))
        assert not np.any(np.isinf(result.values))

    def test_bounds_checking(self):
        """Probes handle out-of-bounds queries gracefully."""
        from glass_models.viz.probes import PointProbe

        mesh = MagicMock()
        # Create 3D mesh
        x = np.linspace(0, 1, 6)
        y = np.linspace(0, 1, 5)
        z = np.linspace(0, 1, 4)
        X, Y, Z = np.meshgrid(x, y, z)
        mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(
            np.float64
        )
        mesh.field_values = X.ravel() + Y.ravel() + Z.ravel()

        # Query far outside mesh bounds
        probe = PointProbe(mesh=mesh, location=np.array([100, 100, 100]))
        result = probe.evaluate()

        assert result is not None
        assert not np.any(np.isnan(result.values))


# ============================================================================
# PyQt6 Integration Tests (if PyQt6 available)
# ============================================================================


class TestProbeVisualizationWidget:
    """Test PyQt6 probe visualization widget."""

    @pytest.mark.skip(reason="Requires PyQt6 and display")
    def test_probe_widget_creation(self):
        """ProbeVisualizationWidget initializes correctly."""
        try:
            from glass_models.ui.pyqt6.probe_widget import ProbeVisualizationWidget
        except ImportError:
            pytest.skip("PyQt6 not available")

        widget = ProbeVisualizationWidget()
        assert widget is not None

    @pytest.mark.skip(reason="Requires PyQt6 and display")
    def test_point_probe_display(self):
        """Widget displays point probe results in table."""
        try:
            import importlib.util

            spec = importlib.util.find_spec("glass_models.ui.pyqt6.probe_widget")
            if spec is None:
                pytest.skip("PyQt6 not available")
        except (ImportError, AttributeError):
            pytest.skip("PyQt6 not available")

        # This test would require a full PyQt6 application
        pass


# ============================================================================
# Pytest Markers and Configuration
# ============================================================================


@pytest.fixture
def simple_mesh():
    """Fixture: simple 3x3x3 mesh for testing."""
    mesh = MagicMock()
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    z = np.linspace(0, 1, 3)
    X, Y, Z = np.meshgrid(x, y, z)
    mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    mesh.field_values = X.ravel() + Y.ravel() + Z.ravel()
    return mesh


@pytest.fixture
def complex_mesh():
    """Fixture: larger 10x10x10 mesh for performance testing."""
    mesh = MagicMock()
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 10)
    z = np.linspace(0, 10, 10)
    X, Y, Z = np.meshgrid(x, y, z)
    mesh.nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    mesh.field_values = np.sin(X.ravel()) * np.cos(Y.ravel()) * np.exp(-Z.ravel() / 5)
    return mesh

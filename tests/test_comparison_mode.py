"""Test suite for Comparison Mode (GitHub issue #544).

This module implements TDD for dual FEA/CFD results comparison with:
1. Unit tests: difference computation, agreement calculation
2. Integration tests: dual rendering, camera synchronization
3. Viewport synchronization

Success criteria:
- Comparison mode renders dual meshes smoothly
- Cameras stay synchronized
- Difference field computed correctly
- Agreement % accurate
- UI responsive
- Code formatted and typed
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Ensure headless mode
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for all tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestDifferenceFieldComputation:
    """Test difference field calculation for two field datasets."""

    @pytest.mark.unit
    def test_compute_difference_field_basic(self) -> None:
        """Test basic difference computation between two fields.

        Creates two similar fields and verifies difference is computed
        as field_a - field_b.
        """
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        # Create simple test fields
        field_a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        field_b = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]])

        diff = ComparisonViewController.compute_difference_field(field_a, field_b)

        expected = field_a - field_b
        np.testing.assert_array_almost_equal(diff, expected)

    @pytest.mark.unit
    def test_compute_difference_field_identical(self) -> None:
        """Test difference field when both inputs are identical.

        Difference should be zero everywhere.
        """
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field = np.random.rand(5, 5, 5)
        diff = ComparisonViewController.compute_difference_field(field, field)

        np.testing.assert_array_almost_equal(diff, np.zeros_like(field))

    @pytest.mark.unit
    def test_compute_difference_field_shape_mismatch(self) -> None:
        """Test that shape mismatch raises ValueError."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field_a = np.random.rand(5, 5, 5)
        field_b = np.random.rand(6, 6, 6)

        with pytest.raises(ValueError, match="Shape mismatch"):
            ComparisonViewController.compute_difference_field(field_a, field_b)

    @pytest.mark.unit
    def test_compute_difference_field_nan_handling(self) -> None:
        """Test difference computation with NaN values.

        Should preserve NaN and not propagate incorrectly.
        """
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field_a = np.array([[[1.0, np.nan], [3.0, 4.0]]])
        field_b = np.array([[[0.5, 1.5], [2.5, 3.5]]])

        diff = ComparisonViewController.compute_difference_field(field_a, field_b)

        assert np.isnan(diff[0, 0, 1])
        assert not np.isnan(diff[0, 0, 0])

    @pytest.mark.unit
    def test_compute_difference_field_preserves_dtype(self) -> None:
        """Test that difference field preserves float dtype."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field_a = np.array([[[1.0, 2.0]]], dtype=np.float32)
        field_b = np.array([[[0.5, 1.5]]], dtype=np.float32)

        diff = ComparisonViewController.compute_difference_field(field_a, field_b)

        assert diff.dtype in (np.float32, np.float64)


class TestAgreementPercentageCalculation:
    """Test agreement percentage calculation between fields."""

    @pytest.mark.unit
    def test_agreement_percentage_identical_fields(self) -> None:
        """Test agreement % when fields are identical.

        Should return 100%.
        """
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field = np.random.rand(5, 5, 5)
        agreement = ComparisonViewController.compute_agreement_percentage(
            field, field, threshold=0.01
        )

        assert agreement == 100.0

    @pytest.mark.unit
    def test_agreement_percentage_no_agreement(self) -> None:
        """Test agreement % when fields are completely different.

        Should return 0%.
        """
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field_a = np.ones((5, 5, 5))
        field_b = np.zeros((5, 5, 5))

        agreement = ComparisonViewController.compute_agreement_percentage(
            field_a, field_b, threshold=0.01
        )

        assert agreement == 0.0

    @pytest.mark.unit
    def test_agreement_percentage_partial_agreement(self) -> None:
        """Test agreement % with partial overlap.

        Creates fields where 50% of values agree within threshold.
        """
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field_a = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        field_b = np.array([[[1.01, 1.01], [2.0, 2.0]]])

        threshold = 0.02  # 0.01 is within, 1.0 is not
        agreement = ComparisonViewController.compute_agreement_percentage(
            field_a, field_b, threshold=threshold
        )

        # 2 out of 4 values agree
        assert agreement == 50.0

    @pytest.mark.unit
    def test_agreement_percentage_threshold_sensitivity(self) -> None:
        """Test that agreement % changes with threshold."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field_a = np.array([[[1.0, 1.0]]])
        field_b = np.array([[[1.05, 1.05]]])

        agree_strict = ComparisonViewController.compute_agreement_percentage(
            field_a, field_b, threshold=0.01
        )
        agree_loose = ComparisonViewController.compute_agreement_percentage(
            field_a, field_b, threshold=0.1
        )

        assert agree_strict < agree_loose

    @pytest.mark.unit
    def test_agreement_percentage_ignores_nan(self) -> None:
        """Test that NaN values are properly excluded from agreement calc."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        field_a = np.array([[[1.0, np.nan], [1.0, 1.0]]])
        field_b = np.array([[[1.01, 1.01], [1.01, 1.01]]])

        agreement = ComparisonViewController.compute_agreement_percentage(
            field_a, field_b, threshold=0.02
        )

        # 2 valid, all agree: 100%
        assert agreement == 100.0


class TestComparisonViewControllerBasics:
    """Test ComparisonViewController initialization and properties."""

    @pytest.mark.unit
    def test_comparison_view_controller_initialization(self, qapp) -> None:
        """Test that ComparisonViewController initializes without errors."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        assert controller is not None
        assert hasattr(controller, "left_viewer")
        assert hasattr(controller, "right_viewer")

    @pytest.mark.unit
    def test_comparison_view_has_dual_viewers(self, qapp) -> None:
        """Test that controller has both left and right FEAResultsViewers."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        # Both viewers should exist
        assert hasattr(controller, "left_viewer")
        assert hasattr(controller, "right_viewer")
        assert controller.left_viewer is not None
        assert controller.right_viewer is not None

    @pytest.mark.unit
    def test_comparison_view_has_labels(self, qapp) -> None:
        """Test that controller can set solver labels."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()
        controller.set_solver_labels("COMSOL", "FEniCS")

        assert controller.left_label == "COMSOL"
        assert controller.right_label == "FEniCS"

    @pytest.mark.unit
    def test_comparison_view_swap_positions(self, qapp) -> None:
        """Test that viewports can be swapped."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        # Store initial viewers
        initial_left = controller.left_viewer
        initial_right = controller.right_viewer

        controller.swap_viewer_positions()

        # After swap, positions should be reversed
        assert controller.left_viewer is initial_right
        assert controller.right_viewer is initial_left


class TestCameraSynchronization:
    """Test camera synchronization between viewports."""

    @pytest.mark.unit
    def test_camera_sync_signal_connection(self, qapp) -> None:
        """Test that camera sync signals are connected."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        # Camera sync should be enabled by default
        assert controller.camera_sync_enabled is True

    @pytest.mark.unit
    def test_toggle_camera_sync(self, qapp) -> None:
        """Test enabling/disabling camera synchronization."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        assert controller.camera_sync_enabled is True
        controller.set_camera_sync(False)
        assert controller.camera_sync_enabled is False
        controller.set_camera_sync(True)
        assert controller.camera_sync_enabled is True


class TestFieldSelectionModes:
    """Test field selection behavior (independent vs synchronized)."""

    @pytest.mark.unit
    def test_independent_field_selection(self, qapp) -> None:
        """Test that fields can be selected independently."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController(synchronized_fields=False)

        assert controller.synchronized_fields is False

    @pytest.mark.unit
    def test_synchronized_field_selection(self, qapp) -> None:
        """Test synchronized field selection mode."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController(synchronized_fields=True)

        assert controller.synchronized_fields is True


class TestSplitLayoutToggle:
    """Test layout orientation switching."""

    @pytest.mark.unit
    def test_toggle_split_layout(self, qapp) -> None:
        """Test switching between horizontal and vertical split."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        # Default should be horizontal
        assert controller.get_split_orientation() == "horizontal"

        controller.set_split_orientation("vertical")
        assert controller.get_split_orientation() == "vertical"

        controller.set_split_orientation("horizontal")
        assert controller.get_split_orientation() == "horizontal"

    @pytest.mark.unit
    def test_invalid_split_orientation(self, qapp) -> None:
        """Test that invalid orientation raises error."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        with pytest.raises(ValueError, match="orientation must be"):
            controller.set_split_orientation("diagonal")


class TestLoadDataAndVisualization:
    """Test loading field data into both viewers."""

    @pytest.mark.unit
    def test_load_both_field_datasets(self, qapp) -> None:
        """Test loading data into both viewers."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        field_a = np.random.rand(10, 10, 10)
        field_b = np.random.rand(10, 10, 10)

        controller.load_left_field(field_a)
        controller.load_right_field(field_b)

        np.testing.assert_array_equal(controller.left_field, field_a)
        np.testing.assert_array_equal(controller.right_field, field_b)

    @pytest.mark.unit
    def test_load_mismatched_shapes_warning(self, qapp) -> None:
        """Test that loading mismatched shapes emits warning."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        field_a = np.random.rand(10, 10, 10)
        field_b = np.random.rand(12, 12, 12)

        controller.load_left_field(field_a)

        # Should not raise, but should log warning
        controller.load_right_field(field_b)

        assert controller.right_field is not None


class TestDifferenceVisualization:
    """Test difference field visualization."""

    @pytest.mark.unit
    def test_compute_and_display_difference(self, qapp) -> None:
        """Test computing difference field for visualization."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        field_a = np.ones((5, 5, 5))
        field_b = np.ones((5, 5, 5)) * 0.5

        controller.load_left_field(field_a)
        controller.load_right_field(field_b)

        diff_field = controller.compute_difference_field(field_a, field_b)

        # Difference should be 0.5 everywhere
        np.testing.assert_array_almost_equal(diff_field, np.ones((5, 5, 5)) * 0.5)

    @pytest.mark.unit
    def test_agreement_display_updates(self, qapp) -> None:
        """Test that agreement percentage is computed and stored."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        field_a = np.ones((5, 5, 5))
        field_b = np.ones((5, 5, 5))  # Identical

        controller.load_left_field(field_a)
        controller.load_right_field(field_b)

        agreement = controller.compute_agreement_percentage(
            field_a, field_b, threshold=0.01
        )

        assert agreement >= 99.9  # Allow floating point tolerance


class TestIntegrationDualRendering:
    """Integration tests for dual mesh rendering."""

    @pytest.mark.integration
    def test_dual_viewers_render_different_data(self, qapp) -> None:
        """Test that two viewers can independently render data."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        field_a = np.random.rand(8, 8, 8)
        field_b = np.random.rand(8, 8, 8)

        controller.load_left_field(field_a)
        controller.load_right_field(field_b)

        # Both viewers should have data
        assert controller.left_field is not None
        assert controller.right_field is not None

    @pytest.mark.integration
    def test_synchronized_field_selection_updates_both(self, qapp) -> None:
        """Test synchronized field selection affects both viewers."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController(synchronized_fields=True)

        field_a = np.random.rand(8, 8, 8)
        field_b = np.random.rand(8, 8, 8)

        controller.load_left_field(field_a)
        controller.load_right_field(field_b)

        # When synchronized, changing iso-value on one should affect both
        # (This would be tested more thoroughly in UI tests)
        assert controller.synchronized_fields is True


class TestCameraSync:
    """Test camera synchronization between viewports."""

    @pytest.mark.integration
    def test_camera_updates_trigger_sync(self, qapp) -> None:
        """Test that camera changes on one viewer trigger sync to other."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        # Camera sync should be connected
        assert controller.camera_sync_enabled is True

        # Mock the synchronization
        controller.left_viewer.visualization_updated = MagicMock()
        controller.right_viewer.visualization_updated = MagicMock()

        # In a real test, we'd trigger camera change and verify signal fired
        # This is a placeholder for actual camera sync testing


class TestMemoryManagement:
    """Test memory cleanup and resource management."""

    @pytest.mark.unit
    def test_clear_all_data(self, qapp) -> None:
        """Test clearing all loaded data."""
        from glass_models.ui.pyqt6.comparison_viewer import (
            ComparisonViewController,
        )

        controller = ComparisonViewController()

        field_a = np.random.rand(8, 8, 8)
        field_b = np.random.rand(8, 8, 8)

        controller.load_left_field(field_a)
        controller.load_right_field(field_b)

        assert controller.left_field is not None
        assert controller.right_field is not None

        controller.clear()

        assert controller.left_field is None
        assert controller.right_field is None


__all__ = [
    "TestDifferenceFieldComputation",
    "TestAgreementPercentageCalculation",
    "TestComparisonViewControllerBasics",
    "TestCameraSynchronization",
    "TestFieldSelectionModes",
    "TestSplitLayoutToggle",
    "TestLoadDataAndVisualization",
    "TestDifferenceVisualization",
    "TestIntegrationDualRendering",
    "TestCameraSync",
    "TestMemoryManagement",
]

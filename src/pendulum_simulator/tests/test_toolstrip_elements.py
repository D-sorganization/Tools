"""Tests for the ToolStrip widget — verify all critical UI elements exist.

These tests ensure that the playback slider, torque vector checkboxes,
and moment of force display options are ALWAYS present in the toolbar.

Closes #1207: Tests for playback slider existence.
Closes #1208: Tests for torque/MoF/sum moments checkboxes.
Closes #1209: Tests that gravity checkbox is REMOVED.

Design by Contract:
- Every test verifies a specific widget EXISTS as a child of ToolStrip.
- Tests are independent and fast (no simulation required).
"""

from __future__ import annotations

import os
import sys

import pytest
from PyQt6.QtWidgets import QApplication, QCheckBox, QSlider

from double_pendulum_golf.gui.toolstrip_widget import ToolStrip

# Skip entire module when no Qt platform backend is available.
# QT_QPA_PLATFORM=offscreen is set in CI to enable headless widget tests.
_QT_AVAILABLE = (
    sys.platform in ("win32", "darwin")
    or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    or bool(os.environ.get("DISPLAY"))
    or bool(os.environ.get("WAYLAND_DISPLAY"))
)
pytestmark = pytest.mark.skipif(
    not _QT_AVAILABLE, reason="No Qt platform available (set QT_QPA_PLATFORM=offscreen)"
)

# ---------------------------------------------------------------------------
# Fixture: QApplication + ToolStrip instance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def toolstrip(qapp: QApplication) -> ToolStrip:
    """Create a fresh ToolStrip instance."""
    ts = ToolStrip()
    return ts


# ---------------------------------------------------------------------------
# #1207: Playback slider MUST exist
# ---------------------------------------------------------------------------


class TestPlaybackSlider:
    """The playback frame slider must exist and be functional."""

    def test_frame_slider_exists(self, toolstrip: ToolStrip) -> None:
        """ToolStrip must have a _frame_slider attribute that is a QSlider."""
        assert hasattr(
            toolstrip, "_frame_slider"
        ), "ToolStrip is missing _frame_slider attribute"
        assert isinstance(
            toolstrip._frame_slider, QSlider
        ), f"_frame_slider is {type(toolstrip._frame_slider)}, expected QSlider"

    def test_frame_slider_is_child(self, toolstrip: ToolStrip) -> None:
        """Frame slider must be a descendant widget of the ToolStrip."""
        all_sliders = toolstrip.findChildren(QSlider)
        assert (
            toolstrip._frame_slider in all_sliders
        ), "Frame slider is not a child widget of ToolStrip"

    def test_frame_slider_has_minimum_width(self, toolstrip: ToolStrip) -> None:
        """Frame slider must have a minimum width >= 200px for visibility."""
        assert toolstrip._frame_slider.minimumWidth() >= 200, (
            f"Frame slider minimumWidth is {toolstrip._frame_slider.minimumWidth()}, "
            "expected >= 200"
        )

    def test_frame_slider_initial_range(self, toolstrip: ToolStrip) -> None:
        """Frame slider starts with range 0-0 (no simulation yet)."""
        assert toolstrip._frame_slider.minimum() == 0
        assert toolstrip._frame_slider.maximum() == 0

    def test_set_frame_range(self, toolstrip: ToolStrip) -> None:
        """set_frame_range should update the slider's maximum."""
        toolstrip.set_frame_range(100)
        assert toolstrip._frame_slider.maximum() == 99

    def test_set_frame(self, toolstrip: ToolStrip) -> None:
        """set_frame should update the slider position without emitting signals."""
        toolstrip.set_frame_range(100)
        toolstrip.set_frame(50)
        assert toolstrip._frame_slider.value() == 50

    def test_frame_scrubbed_signal(self, toolstrip: ToolStrip) -> None:
        """Scrubbing the slider must emit the frame_scrubbed signal."""
        toolstrip.set_frame_range(100)
        received: list[int] = []
        toolstrip.frame_scrubbed.connect(received.append)
        toolstrip._frame_slider.setValue(42)
        assert 42 in received, "frame_scrubbed signal not emitted on slider change"


# ---------------------------------------------------------------------------
# #1208: Torque/MoF/Sum checkboxes MUST exist
# ---------------------------------------------------------------------------


class TestTorqueCheckboxes:
    """Torque vector, moment of force, and sum of moments checkboxes."""

    def test_torque_checkbox_exists(self, toolstrip: ToolStrip) -> None:
        """ToolStrip must have a chk_torque checkbox."""
        assert hasattr(toolstrip, "chk_torque"), "ToolStrip missing chk_torque"
        assert isinstance(toolstrip.chk_torque, QCheckBox)

    def test_moment_of_force_checkbox_exists(self, toolstrip: ToolStrip) -> None:
        """ToolStrip must have a chk_mof checkbox."""
        assert hasattr(toolstrip, "chk_mof"), "ToolStrip missing chk_mof"
        assert isinstance(toolstrip.chk_mof, QCheckBox)

    def test_sum_moments_checkbox_exists(self, toolstrip: ToolStrip) -> None:
        """ToolStrip must have a chk_sum_moments checkbox."""
        assert hasattr(
            toolstrip, "chk_sum_moments"
        ), "ToolStrip missing chk_sum_moments"
        assert isinstance(toolstrip.chk_sum_moments, QCheckBox)

    def test_torque_signal_connected(self, toolstrip: ToolStrip) -> None:
        """Checking the torque checkbox must emit torque_vectors_toggled."""
        received: list[bool] = []
        toolstrip.torque_vectors_toggled.connect(received.append)
        toolstrip.chk_torque.setChecked(True)
        assert True in received, "torque_vectors_toggled not emitted"

    def test_mof_signal_connected(self, toolstrip: ToolStrip) -> None:
        """Checking the MoF checkbox must emit moment_of_force_toggled."""
        received: list[bool] = []
        toolstrip.moment_of_force_toggled.connect(received.append)
        toolstrip.chk_mof.setChecked(True)
        assert True in received, "moment_of_force_toggled not emitted"

    def test_sum_signal_connected(self, toolstrip: ToolStrip) -> None:
        """Checking the sum checkbox must emit sum_moments_toggled."""
        received: list[bool] = []
        toolstrip.sum_moments_toggled.connect(received.append)
        toolstrip.chk_sum_moments.setChecked(True)
        assert True in received, "sum_moments_toggled not emitted"


# ---------------------------------------------------------------------------
# #1209: Gravity checkbox MUST NOT exist
# ---------------------------------------------------------------------------


class TestNoGravityCheckbox:
    """Gravity checkbox must be permanently removed."""

    def test_no_gravity_checkbox_in_toolstrip(self, toolstrip: ToolStrip) -> None:
        """ToolStrip must NOT have a chk_gravity attribute."""
        assert not hasattr(
            toolstrip, "chk_gravity"
        ), "chk_gravity still exists in ToolStrip — it must be removed (#1209)"

    def test_no_gravity_toggled_signal(self, toolstrip: ToolStrip) -> None:
        """ToolStrip must NOT have gravity_toggled signal."""
        assert not hasattr(
            toolstrip, "gravity_toggled"
        ), "gravity_toggled signal still exists — must be removed (#1209)"


# ---------------------------------------------------------------------------
# Existing overlay checkboxes still present
# ---------------------------------------------------------------------------


class TestExistingOverlays:
    """Verify that existing overlay checkboxes are still present."""

    def test_forces_checkbox(self, toolstrip: ToolStrip) -> None:
        assert hasattr(toolstrip, "chk_forces")
        assert isinstance(toolstrip.chk_forces, QCheckBox)

    def test_mobility_checkbox(self, toolstrip: ToolStrip) -> None:
        assert hasattr(toolstrip, "chk_mob")
        assert isinstance(toolstrip.chk_mob, QCheckBox)

    def test_force_ellipsoid_checkbox(self, toolstrip: ToolStrip) -> None:
        assert hasattr(toolstrip, "chk_force_ell")
        assert isinstance(toolstrip.chk_force_ell, QCheckBox)

    def test_zero_torque_checkbox(self, toolstrip: ToolStrip) -> None:
        assert hasattr(toolstrip, "chk_zero_torque")
        assert isinstance(toolstrip.chk_zero_torque, QCheckBox)

    def test_com_checkbox(self, toolstrip: ToolStrip) -> None:
        assert hasattr(toolstrip, "chk_com")
        assert isinstance(toolstrip.chk_com, QCheckBox)

    def test_loop_checkbox(self, toolstrip: ToolStrip) -> None:
        assert hasattr(toolstrip, "chk_loop")
        assert isinstance(toolstrip.chk_loop, QCheckBox)

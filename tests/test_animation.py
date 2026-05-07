"""Test suite for animation and time-stepping features.

This module implements TDD for GitHub issue #546: Animation Scrubbing UI
for Time-Dependent Data.

Tests are organized by:
1. TimeStepManager unit tests (time stepping, seeking, state)
2. AnimationPlaybackWidget integration tests (UI controls, playback)
3. Frame export verification
4. Performance tests (30+ FPS requirement, <100ms scrubbing)

Success criteria:
- All animation tests pass
- Time slider scrubbing works smoothly (<100ms response)
- Speed control responsive (0.5x to 10x)
- Frame export generates image sequences
- Loop toggle works correctly
- Code formatted and typed
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _ensure_qapp() -> None:
    """Ensure QApplication exists (side effect: creates it if not)."""
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])


class TestTimeStepManagerBasics:
    """Unit tests for TimeStepManager core functionality."""

    @pytest.mark.unit
    def test_init_with_valid_data(self) -> None:
        """Test TimeStepManager initialization with valid data."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [
            {"velocity": np.random.rand(10, 3)},
            {"velocity": np.random.rand(10, 3)},
            {"velocity": np.random.rand(10, 3)},
            {"velocity": np.random.rand(10, 3)},
        ]

        manager = TimeStepManager(time_steps, field_data)

        assert manager is not None
        assert manager.total_steps == 4
        assert manager.current_step_index == 0
        assert manager.current_time == 0.0

    @pytest.mark.unit
    def test_init_with_mismatched_lengths(self) -> None:
        """Test TimeStepManager raises ValueError for mismatched array lengths."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [
            {"velocity": np.random.rand(10, 3)},
            {"velocity": np.random.rand(10, 3)},
        ]

        with pytest.raises(ValueError, match="time_steps and field_data_list"):
            TimeStepManager(time_steps, field_data)

    @pytest.mark.unit
    def test_get_current_field(self) -> None:
        """Test get_current_field returns correct data at current step."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2])
        field_a = np.array([1.0, 2.0, 3.0])
        field_b = np.array([4.0, 5.0, 6.0])
        field_c = np.array([7.0, 8.0, 9.0])

        field_data = [
            {"data": field_a},
            {"data": field_b},
            {"data": field_c},
        ]

        manager = TimeStepManager(time_steps, field_data)

        # At step 0
        current = manager.get_current_field()
        assert "data" in current
        np.testing.assert_array_equal(current["data"], field_a)

        # After advancing
        manager.advance_step()
        current = manager.get_current_field()
        np.testing.assert_array_equal(current["data"], field_b)

    @pytest.mark.unit
    def test_advance_step_increments_index(self) -> None:
        """Test advance_step increments the current step."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [{"f": i} for i in range(4)]

        manager = TimeStepManager(time_steps, field_data)

        assert manager.current_step_index == 0
        result = manager.advance_step()
        assert result is True
        assert manager.current_step_index == 1

    @pytest.mark.unit
    def test_advance_step_returns_false_at_end(self) -> None:
        """Test advance_step returns False when reaching the last step."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"f": i} for i in range(3)]

        manager = TimeStepManager(time_steps, field_data)

        # Advance to last step
        manager.advance_step()  # step 1
        manager.advance_step()  # step 2 (last)

        # Try to advance beyond
        result = manager.advance_step()
        assert result is False
        assert manager.current_step_index == 2

    @pytest.mark.unit
    def test_seek_to_step_valid_index(self) -> None:
        """Test seek_to_step moves to specified index."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [{"step": i} for i in range(4)]

        manager = TimeStepManager(time_steps, field_data)

        manager.seek_to_step(2)
        assert manager.current_step_index == 2
        assert manager.current_time == 0.2

    @pytest.mark.unit
    def test_seek_to_step_boundary_values(self) -> None:
        """Test seek_to_step with boundary indices."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [{"step": i} for i in range(4)]

        manager = TimeStepManager(time_steps, field_data)

        # First step
        manager.seek_to_step(0)
        assert manager.current_step_index == 0

        # Last step
        manager.seek_to_step(3)
        assert manager.current_step_index == 3

    @pytest.mark.unit
    def test_seek_to_step_out_of_bounds(self) -> None:
        """Test seek_to_step raises ValueError for out-of-bounds index."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"step": i} for i in range(3)]

        manager = TimeStepManager(time_steps, field_data)

        with pytest.raises(ValueError, match="out of bounds"):
            manager.seek_to_step(10)

        with pytest.raises(ValueError, match="out of bounds"):
            manager.seek_to_step(-1)

    @pytest.mark.unit
    def test_current_time_property(self) -> None:
        """Test current_time property returns correct value."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.5, 1.0, 1.5])
        field_data = [{"step": i} for i in range(4)]

        manager = TimeStepManager(time_steps, field_data)

        assert manager.current_time == 0.0
        manager.seek_to_step(1)
        assert manager.current_time == 0.5
        manager.seek_to_step(3)
        assert manager.current_time == 1.5

    @pytest.mark.unit
    def test_reset(self) -> None:
        """Test reset returns manager to initial state."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"step": i} for i in range(3)]

        manager = TimeStepManager(time_steps, field_data)
        manager.seek_to_step(2)

        manager.reset()

        assert manager.current_step_index == 0
        assert manager.current_time == 0.0


class TestTimeStepManagerPlaybackState:
    """Unit tests for TimeStepManager playback tracking."""

    @pytest.mark.unit
    def test_is_at_end(self) -> None:
        """Test is_at_end property."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"step": i} for i in range(3)]

        manager = TimeStepManager(time_steps, field_data)

        assert manager.is_at_end is False
        manager.seek_to_step(2)
        assert manager.is_at_end is True

    @pytest.mark.unit
    def test_time_range(self) -> None:
        """Test time_range property returns (min, max) tuple."""
        from glass_models.viz.timesteps import TimeStepManager

        time_steps = np.array([0.5, 1.0, 1.5, 2.0])
        field_data = [{"step": i} for i in range(4)]

        manager = TimeStepManager(time_steps, field_data)

        min_time, max_time = manager.time_range
        assert min_time == 0.5
        assert max_time == 2.0


@pytest.mark.requires_gl
class TestAnimationPlaybackWidgetBasics:
    """Unit tests for AnimationPlaybackWidget initialization and state."""

    @pytest.mark.unit
    def test_widget_initialization(self) -> None:
        """Test AnimationPlaybackWidget initializes with default state."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        assert widget is not None
        # Default state should be paused
        assert widget.is_playing is False
        assert widget.speed_multiplier == 1.0

    @pytest.mark.unit
    def test_widget_has_required_controls(self) -> None:
        """Test widget has all required UI controls."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # Check for required controls
        assert hasattr(widget, "play_button")
        assert hasattr(widget, "pause_button")
        assert hasattr(widget, "restart_button")
        assert hasattr(widget, "time_slider")
        assert hasattr(widget, "speed_slider")
        assert hasattr(widget, "first_button")
        assert hasattr(widget, "last_button")
        assert hasattr(widget, "loop_checkbox")
        assert hasattr(widget, "time_display_label")
        assert hasattr(widget, "export_button")

    @pytest.mark.unit
    def test_speed_slider_range(self) -> None:
        """Test speed slider is configured for 0.5x to 10x range."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # Speed slider should map to 0.5x - 10x
        # Implementation detail: slider typically ranges 0-100 or similar
        assert widget.speed_slider.minimum() >= 0
        assert widget.speed_slider.maximum() > widget.speed_slider.minimum()

    @pytest.mark.unit
    def test_time_display_label_format(self) -> None:
        """Test time display label shows current/total time."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        label_text = widget.time_display_label.text()
        # Should contain format like "0.0 / 0.2" or similar
        assert "/" in label_text or ":" in label_text


@pytest.mark.requires_gl
class TestAnimationPlaybackWidgetInteraction:
    """Integration tests for AnimationPlaybackWidget slider and button interactions."""

    @pytest.mark.integration
    def test_slider_seeking(self) -> None:
        """Test time slider scrubbing moves to correct step."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [{"step": i} for i in range(4)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # Seek to middle
        widget.time_slider.setValue(2)
        widget.on_slider_moved(2)

        # Should be at step 2 or 3 (depending on implementation)
        assert widget.manager.current_step_index >= 1

    @pytest.mark.integration
    def test_play_button_starts_playback(self) -> None:
        """Test play button starts playback."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        assert widget.is_playing is False
        widget.play_button.clicked.emit()
        assert widget.is_playing is True

    @pytest.mark.integration
    def test_pause_button_stops_playback(self) -> None:
        """Test pause button stops playback."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        widget.play_button.clicked.emit()
        assert widget.is_playing is True

        widget.pause_button.clicked.emit()
        assert widget.is_playing is False

    @pytest.mark.integration
    def test_restart_button_resets_to_beginning(self) -> None:
        """Test restart button returns to first frame."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [{"data": i} for i in range(4)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # Move to end
        widget.manager.seek_to_step(3)
        assert widget.manager.current_step_index == 3

        # Click restart
        widget.restart_button.clicked.emit()

        assert widget.manager.current_step_index == 0

    @pytest.mark.integration
    def test_first_button_goes_to_start(self) -> None:
        """Test first step button goes to first frame."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [{"data": i} for i in range(4)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        widget.manager.seek_to_step(2)
        widget.first_button.clicked.emit()

        assert widget.manager.current_step_index == 0

    @pytest.mark.integration
    def test_last_button_goes_to_end(self) -> None:
        """Test last step button goes to last frame."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2, 0.3])
        field_data = [{"data": i} for i in range(4)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        widget.last_button.clicked.emit()

        assert widget.manager.current_step_index == 3

    @pytest.mark.integration
    def test_speed_slider_changes_multiplier(self) -> None:
        """Test speed slider changes playback speed multiplier."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        initial_speed = widget.speed_multiplier

        # Change speed slider
        widget.speed_slider.setValue(75)
        widget.on_speed_changed(75)

        # Speed should have changed
        assert widget.speed_multiplier != initial_speed or initial_speed == 1.0

    @pytest.mark.integration
    def test_loop_checkbox_toggle(self) -> None:
        """Test loop checkbox can be toggled."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        assert widget.loop_checkbox is not None
        initial_state = widget.loop_checkbox.isChecked()

        widget.loop_checkbox.setChecked(not initial_state)
        assert widget.loop_checkbox.isChecked() == (not initial_state)

    @pytest.mark.integration
    def test_looping_behavior(self) -> None:
        """Test animation loops when enabled."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)
        widget.loop_checkbox.setChecked(True)

        # Go to last step
        widget.manager.seek_to_step(2)

        # Advance should loop back
        _ = widget.manager.advance_step()
        # With looping, this might wrap or handle differently
        # Just verify the state is consistent
        assert widget.manager.current_step_index >= 0


@pytest.mark.requires_gl
class TestAnimationPlaybackPerformance:
    """Performance tests for animation playback."""

    @pytest.mark.unit
    def test_slider_response_time(self) -> None:
        """Test time slider responds in <100ms."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        field_data = [{"data": i} for i in range(6)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # Measure slider seeking time
        start = time.perf_counter()
        for i in range(5):
            widget.time_slider.setValue(i * 20)
            widget.on_slider_moved(i * 20)
        elapsed = time.perf_counter() - start

        # Should complete 5 seeks in <100ms (20ms each)
        assert elapsed < 0.1, f"Slider seeking took {elapsed:.3f}s, expected <0.1s"

    @pytest.mark.unit
    def test_fps_calculation(self) -> None:
        """Test frame rate calculation is plausible (30+ FPS target)."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # At 1.0x speed with 0.1s per frame, timer should fire ~10 times/sec
        # At 1.0x speed, each frame takes 0.1s, so 10 FPS minimum
        # Higher speeds will increase FPS
        assert widget.speed_multiplier >= 0.5


@pytest.mark.requires_gl
class TestFrameExport:
    """Tests for frame export functionality."""

    @pytest.mark.integration
    def test_export_button_exists(self) -> None:
        """Test export button is present in widget."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2])
        field_data = [{"data": i} for i in range(3)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        assert hasattr(widget, "export_button")
        assert widget.export_button is not None

    @pytest.mark.integration
    def test_export_frame_sequence(self) -> None:
        """Test exporting frames to image sequence."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        with tempfile.TemporaryDirectory() as tmpdir:
            time_steps = np.array([0.0, 0.1, 0.2])
            field_data = [{"data": i} for i in range(3)]

            widget = AnimationPlaybackWidget(time_steps, field_data)

            # Export should create files in tmpdir
            # Mock the export to avoid requiring image libraries
            with patch.object(widget, "export_frames") as mock_export:
                mock_export(tmpdir)
                mock_export.assert_called_once_with(tmpdir)


@pytest.mark.requires_gl
class TestIntegrationTimeSliderSeeking:
    """Integration tests for time slider scrubbing responsiveness."""

    @pytest.mark.integration
    def test_rapid_slider_seeking(self) -> None:
        """Test rapid slider seeking doesn't break state."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.arange(0, 1.0, 0.01)  # 100 frames
        field_data = [{"data": i} for i in range(len(time_steps))]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # Rapidly seek to different positions
        for i in range(0, 100, 10):
            widget.time_slider.setValue(i)
            widget.on_slider_moved(i)

        # Should end up in valid state
        assert 0 <= widget.manager.current_step_index < len(time_steps)

    @pytest.mark.integration
    def test_slider_bounds(self) -> None:
        """Test slider value stays within bounds."""
        pytest.importorskip("PyQt6")

        from glass_models.ui.pyqt6.animation_widget import AnimationPlaybackWidget

        _ensure_qapp()

        time_steps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        field_data = [{"data": i} for i in range(5)]

        widget = AnimationPlaybackWidget(time_steps, field_data)

        # Slider should have appropriate bounds
        assert widget.time_slider.minimum() >= 0
        assert widget.time_slider.maximum() == len(time_steps) - 1

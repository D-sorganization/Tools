"""Test suite for advanced camera controls and saved viewpoints.

This module tests the viewpoint management system with emphasis on:
- Viewpoint dataclass creation and validation
- Standard view positioning (Top, Bottom, Front, Back, Left, Right, Isometric)
- Viewpoint save/restore with JSON serialization
- Smooth camera transitions (1.0s default with interpolation)
- ViewpointManager lifecycle and state management
- Design by Contract: camera positions validated for standard views
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from glass_models.viz.viewpoints import (
    Viewpoint,
    ViewpointManager,
)


class TestViewpointDataclass:
    """Unit tests for the Viewpoint dataclass."""

    def test_viewpoint_creation_basic(self) -> None:
        """Test basic viewpoint creation."""
        position = np.array([0.0, 0.0, 10.0])
        focal_point = np.array([0.0, 0.0, 0.0])
        up_vector = np.array([0.0, 1.0, 0.0])

        viewpoint = Viewpoint(
            name="Test View",
            position=position,
            focal_point=focal_point,
            up_vector=up_vector,
        )

        assert viewpoint.name == "Test View"
        assert np.allclose(viewpoint.position, position)
        assert np.allclose(viewpoint.focal_point, focal_point)
        assert np.allclose(viewpoint.up_vector, up_vector)

    def test_viewpoint_position_validation(self) -> None:
        """Test that position must be 3D array."""
        with pytest.raises(TypeError):
            Viewpoint(
                name="Bad View",
                position=np.array([0.0, 0.0]),  # Wrong shape
                focal_point=np.array([0.0, 0.0, 0.0]),
                up_vector=np.array([0.0, 1.0, 0.0]),
            )

    def test_viewpoint_focal_point_validation(self) -> None:
        """Test that focal_point must be 3D array."""
        with pytest.raises(TypeError):
            Viewpoint(
                name="Bad View",
                position=np.array([0.0, 0.0, 10.0]),
                focal_point=np.array([0.0, 0.0]),  # Wrong shape
                up_vector=np.array([0.0, 1.0, 0.0]),
            )

    def test_viewpoint_up_vector_validation(self) -> None:
        """Test that up_vector must be 3D array."""
        with pytest.raises(TypeError):
            Viewpoint(
                name="Bad View",
                position=np.array([0.0, 0.0, 10.0]),
                focal_point=np.array([0.0, 0.0, 0.0]),
                up_vector=np.array([0.0, 1.0]),  # Wrong shape
            )

    def test_viewpoint_up_vector_normalization(self) -> None:
        """Test that up_vector is normalized."""
        up_vector = np.array([0.0, 2.0, 0.0])
        viewpoint = Viewpoint(
            name="Test",
            position=np.array([0.0, 0.0, 10.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=up_vector,
        )
        # Should be normalized
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(viewpoint.up_vector, expected)

    def test_viewpoint_non_zero_up_vector(self) -> None:
        """Test that up_vector cannot be zero vector."""
        with pytest.raises(ValueError, match="up_vector cannot be zero"):
            Viewpoint(
                name="Test",
                position=np.array([0.0, 0.0, 10.0]),
                focal_point=np.array([0.0, 0.0, 0.0]),
                up_vector=np.array([0.0, 0.0, 0.0]),
            )


class TestStandardViews:
    """Test standard view definitions."""

    def test_standard_view_top(self) -> None:
        """Test Top standard view positioning."""
        manager = ViewpointManager()
        top_view = manager.get_standard_view("Top")

        assert top_view is not None
        # Top view: looking down at object from positive Z
        assert np.isclose(top_view.position[2], 10.0)  # z positive
        assert np.isclose(top_view.focal_point[2], -10.0)  # looking down
        assert np.isclose(top_view.up_vector[1], 1.0)  # Y is up

    def test_standard_view_bottom(self) -> None:
        """Test Bottom standard view positioning."""
        manager = ViewpointManager()
        bottom_view = manager.get_standard_view("Bottom")

        assert bottom_view is not None
        # Bottom view: looking up at object from negative Z
        assert np.isclose(bottom_view.position[2], -10.0)  # z negative
        assert np.isclose(bottom_view.focal_point[2], 10.0)  # looking up
        assert np.isclose(bottom_view.up_vector[1], -1.0)  # Y inverted

    def test_standard_view_front(self) -> None:
        """Test Front standard view positioning."""
        manager = ViewpointManager()
        front_view = manager.get_standard_view("Front")

        assert front_view is not None
        # Front view: looking along Y axis
        assert np.isclose(front_view.position[1], 10.0)  # ahead on Y
        assert np.isclose(front_view.focal_point[1], 0.0)  # centered
        assert np.isclose(front_view.up_vector[2], 1.0)  # Z is up

    def test_standard_view_back(self) -> None:
        """Test Back standard view positioning."""
        manager = ViewpointManager()
        back_view = manager.get_standard_view("Back")

        assert back_view is not None
        # Back view: looking opposite of front
        assert np.isclose(back_view.position[1], -10.0)
        assert np.isclose(back_view.focal_point[1], 0.0)
        assert np.isclose(back_view.up_vector[2], 1.0)

    def test_standard_view_left(self) -> None:
        """Test Left standard view positioning."""
        manager = ViewpointManager()
        left_view = manager.get_standard_view("Left")

        assert left_view is not None
        # Left view: looking along X axis from left
        assert np.isclose(left_view.position[0], -10.0)
        assert np.isclose(left_view.focal_point[0], 0.0)
        assert np.isclose(left_view.up_vector[2], 1.0)

    def test_standard_view_right(self) -> None:
        """Test Right standard view positioning."""
        manager = ViewpointManager()
        right_view = manager.get_standard_view("Right")

        assert right_view is not None
        # Right view: looking along X axis from right
        assert np.isclose(right_view.position[0], 10.0)
        assert np.isclose(right_view.focal_point[0], 0.0)
        assert np.isclose(right_view.up_vector[2], 1.0)

    def test_standard_view_isometric(self) -> None:
        """Test Isometric standard view positioning."""
        manager = ViewpointManager()
        iso_view = manager.get_standard_view("Isometric")

        assert iso_view is not None
        # Isometric: equal angles on all axes
        position = iso_view.position
        # All position components should have similar magnitude
        abs_pos = np.abs(position)
        assert np.isclose(abs_pos[0], abs_pos[1])
        assert np.isclose(abs_pos[1], abs_pos[2])

    def test_get_standard_view_invalid(self) -> None:
        """Test getting invalid standard view."""
        manager = ViewpointManager()
        with pytest.raises(ValueError, match="Unknown standard view"):
            manager.get_standard_view("Invalid")

    def test_all_standard_views_available(self) -> None:
        """Test that all 7 standard views are available."""
        manager = ViewpointManager()
        standard_views = [
            "Top",
            "Bottom",
            "Front",
            "Back",
            "Left",
            "Right",
            "Isometric",
        ]
        for view_name in standard_views:
            view = manager.get_standard_view(view_name)
            assert view is not None
            assert view.name == view_name


class TestViewpointManager:
    """Test ViewpointManager class."""

    def test_manager_creation(self) -> None:
        """Test basic manager creation."""
        manager = ViewpointManager()
        assert isinstance(manager.viewpoints, dict)
        assert len(manager.viewpoints) == 0

    def test_save_viewpoint(self) -> None:
        """Test saving a viewpoint."""
        manager = ViewpointManager()
        viewpoint = Viewpoint(
            name="Test View",
            position=np.array([1.0, 2.0, 3.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        manager.save_viewpoint(viewpoint)
        assert "Test View" in manager.viewpoints
        assert manager.viewpoints["Test View"] == viewpoint

    def test_load_viewpoint(self) -> None:
        """Test loading a saved viewpoint."""
        manager = ViewpointManager()
        original = Viewpoint(
            name="Saved View",
            position=np.array([1.0, 2.0, 3.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )
        manager.save_viewpoint(original)

        loaded = manager.load_viewpoint("Saved View")
        assert loaded is not None
        assert loaded.name == original.name
        assert np.allclose(loaded.position, original.position)
        assert np.allclose(loaded.focal_point, original.focal_point)

    def test_load_viewpoint_not_found(self) -> None:
        """Test loading a non-existent viewpoint."""
        manager = ViewpointManager()
        with pytest.raises(KeyError):
            manager.load_viewpoint("Nonexistent")

    def test_delete_viewpoint(self) -> None:
        """Test deleting a viewpoint."""
        manager = ViewpointManager()
        viewpoint = Viewpoint(
            name="To Delete",
            position=np.array([1.0, 2.0, 3.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )
        manager.save_viewpoint(viewpoint)
        assert "To Delete" in manager.viewpoints

        manager.delete_viewpoint("To Delete")
        assert "To Delete" not in manager.viewpoints

    def test_delete_nonexistent_viewpoint(self) -> None:
        """Test deleting a non-existent viewpoint."""
        manager = ViewpointManager()
        with pytest.raises(KeyError):
            manager.delete_viewpoint("Nonexistent")

    def test_list_viewpoints(self) -> None:
        """Test listing saved viewpoints."""
        manager = ViewpointManager()
        vp1 = Viewpoint(
            "View1",
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        vp2 = Viewpoint(
            "View2",
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

        manager.save_viewpoint(vp1)
        manager.save_viewpoint(vp2)

        names = manager.list_viewpoints()
        assert set(names) == {"View1", "View2"}


class TestViewpointSerialization:
    """Test viewpoint JSON serialization."""

    def test_viewpoint_to_dict(self) -> None:
        """Test converting viewpoint to dictionary."""
        viewpoint = Viewpoint(
            name="Test",
            position=np.array([1.0, 2.0, 3.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        data = viewpoint.to_dict()
        assert data["name"] == "Test"
        assert np.allclose(data["position"], [1.0, 2.0, 3.0])
        assert np.allclose(data["focal_point"], [0.0, 0.0, 0.0])
        assert np.allclose(data["up_vector"], [0.0, 1.0, 0.0])

    def test_viewpoint_from_dict(self) -> None:
        """Test creating viewpoint from dictionary."""
        data = {
            "name": "From Dict",
            "position": [1.0, 2.0, 3.0],
            "focal_point": [0.0, 0.0, 0.0],
            "up_vector": [0.0, 1.0, 0.0],
        }

        viewpoint = Viewpoint.from_dict(data)
        assert viewpoint.name == "From Dict"
        assert np.allclose(viewpoint.position, [1.0, 2.0, 3.0])

    def test_save_viewpoints_to_json(self) -> None:
        """Test saving viewpoints to JSON file."""
        manager = ViewpointManager()
        vp1 = Viewpoint(
            "View1",
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        vp2 = Viewpoint(
            "View2",
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

        manager.save_viewpoint(vp1)
        manager.save_viewpoint(vp2)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "viewpoints.json"
            manager.save_to_json(filepath)
            assert filepath.exists()

            # Verify JSON content
            with open(filepath) as f:
                data = json.load(f)
            assert "View1" in data
            assert "View2" in data

    def test_load_viewpoints_from_json(self) -> None:
        """Test loading viewpoints from JSON file."""
        manager = ViewpointManager()
        vp1 = Viewpoint(
            "View1",
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

        manager.save_viewpoint(vp1)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "viewpoints.json"
            manager.save_to_json(filepath)

            # Create new manager and load
            manager2 = ViewpointManager()
            manager2.load_from_json(filepath)

            loaded = manager2.load_viewpoint("View1")
            assert loaded is not None
            assert loaded.name == "View1"
            assert np.allclose(loaded.position, [1.0, 0.0, 0.0])

    def test_json_roundtrip(self) -> None:
        """Test that save/load cycle preserves data."""
        original = Viewpoint(
            name="Roundtrip",
            position=np.array([1.5, 2.5, 3.5]),
            focal_point=np.array([0.1, 0.2, 0.3]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            manager1 = ViewpointManager()
            manager1.save_viewpoint(original)
            manager1.save_to_json(filepath)

            manager2 = ViewpointManager()
            manager2.load_from_json(filepath)

            loaded = manager2.load_viewpoint("Roundtrip")
            assert loaded is not None
            assert np.allclose(loaded.position, original.position)
            assert np.allclose(loaded.focal_point, original.focal_point)
            assert np.allclose(loaded.up_vector, original.up_vector)


class TestViewpointAnimation:
    """Test smooth camera transitions."""

    def test_animate_to_viewpoint_basic(self) -> None:
        """Test basic viewpoint animation."""
        manager = ViewpointManager()
        start = Viewpoint(
            name="Start",
            position=np.array([0.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )
        end = Viewpoint(
            name="End",
            position=np.array([10.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        frames = list(manager.animate_to_viewpoint(start, end, duration=1.0))
        assert len(frames) > 0

        # First frame should be close to start
        assert np.allclose(frames[0].position, start.position, atol=0.1)

        # Last frame should be close to end
        assert np.allclose(frames[-1].position, end.position, atol=0.1)

    def test_animate_default_duration(self) -> None:
        """Test that default duration is 1.0 seconds."""
        manager = ViewpointManager()
        start = Viewpoint(
            name="Start",
            position=np.array([0.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )
        end = Viewpoint(
            name="End",
            position=np.array([10.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        # Call without duration parameter
        frames = list(manager.animate_to_viewpoint(start, end))
        assert len(frames) > 0

    def test_animate_smooth_interpolation(self) -> None:
        """Test that animation interpolates smoothly."""
        manager = ViewpointManager()
        start = Viewpoint(
            name="Start",
            position=np.array([0.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )
        end = Viewpoint(
            name="End",
            position=np.array([10.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        frames = list(manager.animate_to_viewpoint(start, end, duration=1.0))

        # Check that positions are monotonically increasing on X axis
        x_positions = [f.position[0] for f in frames]
        for i in range(len(x_positions) - 1):
            # Should not decrease
            assert x_positions[i] <= x_positions[i + 1] + 0.01  # small tolerance

    def test_animate_focal_point_interpolation(self) -> None:
        """Test that focal point animates as well."""
        manager = ViewpointManager()
        start = Viewpoint(
            name="Start",
            position=np.array([0.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )
        end = Viewpoint(
            name="End",
            position=np.array([10.0, 0.0, 0.0]),
            focal_point=np.array([5.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        frames = list(manager.animate_to_viewpoint(start, end, duration=1.0))

        # First focal point should be close to start
        assert np.allclose(frames[0].focal_point, start.focal_point, atol=0.1)

        # Last focal point should be close to end
        assert np.allclose(frames[-1].focal_point, end.focal_point, atol=0.1)

    def test_animate_duration_affects_frame_count(self) -> None:
        """Test that longer duration produces more frames."""
        manager = ViewpointManager()
        start = Viewpoint(
            name="Start",
            position=np.array([0.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )
        end = Viewpoint(
            name="End",
            position=np.array([10.0, 0.0, 0.0]),
            focal_point=np.array([0.0, 0.0, 0.0]),
            up_vector=np.array([0.0, 1.0, 0.0]),
        )

        frames_short = list(manager.animate_to_viewpoint(start, end, duration=0.5))
        frames_long = list(manager.animate_to_viewpoint(start, end, duration=2.0))

        assert len(frames_long) > len(frames_short)


class TestSetStandardView:
    """Test set_standard_view convenience method."""

    def test_set_standard_view_top(self) -> None:
        """Test setting standard Top view."""
        manager = ViewpointManager()
        viewpoint = manager.set_standard_view("Top")
        assert viewpoint.name == "Top"

    def test_set_standard_view_saves(self) -> None:
        """Test that set_standard_view saves the view."""
        manager = ViewpointManager()
        manager.set_standard_view("Front")
        assert "Front" in manager.viewpoints

    def test_set_standard_view_invalid(self) -> None:
        """Test setting invalid standard view."""
        manager = ViewpointManager()
        with pytest.raises(ValueError, match="Unknown standard view"):
            manager.set_standard_view("Invalid")

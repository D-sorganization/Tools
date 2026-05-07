"""Advanced camera controls and saved viewpoints system.

This module provides camera viewpoint management with:
- Viewpoint dataclass for saving camera state (position, focal point, up vector)
- ViewpointManager for managing saved viewpoints and standard views
- Smooth camera transition animation over specified duration
- JSON serialization for persistent viewpoint storage
- Standard views: Top, Bottom, Front, Back, Left, Right, Isometric

Design patterns:
- Design by Contract: camera positions validated for standard views
- Separation of Concerns: storage separate from animation logic
- DRY: single animation implementation with interpolation
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Viewpoint:
    """Represents a camera viewpoint with position, focus, and orientation.

    Attributes:
        name: Human-readable name for this viewpoint
        position: Camera position in 3D space as np.ndarray of shape (3,)
        focal_point: Point the camera is looking at, shape (3,)
        up_vector: Direction considered "up", shape (3,), auto-normalized
    """

    name: str
    position: np.ndarray
    focal_point: np.ndarray
    up_vector: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize viewpoint on creation."""
        # Validate position
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.position.shape != (3,):
            raise TypeError(f"position must have shape (3,), got {self.position.shape}")

        # Validate focal_point
        self.focal_point = np.asarray(self.focal_point, dtype=np.float64)
        if self.focal_point.shape != (3,):
            raise TypeError(
                f"focal_point must have shape (3,), got {self.focal_point.shape}"
            )

        # Validate and normalize up_vector
        self.up_vector = np.asarray(self.up_vector, dtype=np.float64)
        if self.up_vector.shape != (3,):
            raise TypeError(
                f"up_vector must have shape (3,), got {self.up_vector.shape}"
            )

        # Check for zero vector
        up_magnitude = np.linalg.norm(self.up_vector)
        if np.isclose(up_magnitude, 0.0):
            raise ValueError("up_vector cannot be zero vector")

        # Normalize up_vector
        self.up_vector = self.up_vector / up_magnitude

    def to_dict(self) -> dict:
        """Convert viewpoint to dictionary for serialization.

        Returns:
            Dictionary with name and 3D coordinate lists
        """
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "focal_point": self.focal_point.tolist(),
            "up_vector": self.up_vector.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Viewpoint:
        """Create viewpoint from dictionary.

        Args:
            data: Dictionary with keys: name, position, focal_point, up_vector

        Returns:
            Viewpoint instance

        Raises:
            KeyError: If required keys are missing
            TypeError: If values have wrong shapes
        """
        return cls(
            name=data["name"],
            position=np.array(data["position"], dtype=np.float64),
            focal_point=np.array(data["focal_point"], dtype=np.float64),
            up_vector=np.array(data["up_vector"], dtype=np.float64),
        )


class ViewpointManager:
    """Manages saved viewpoints and standard camera views.

    Provides:
    - Save/load/delete user viewpoints
    - Access to 7 standard views (Top, Bottom, Front, Back, Left, Right, Isometric)
    - Smooth animation between viewpoints
    - JSON serialization for persistence
    """

    # Standard view definitions
    _STANDARD_VIEWS = {
        "Top": {
            "position": np.array([0.0, 0.0, 10.0]),
            "focal_point": np.array([0.0, 0.0, -10.0]),
            "up_vector": np.array([0.0, 1.0, 0.0]),
        },
        "Bottom": {
            "position": np.array([0.0, 0.0, -10.0]),
            "focal_point": np.array([0.0, 0.0, 10.0]),
            "up_vector": np.array([0.0, -1.0, 0.0]),
        },
        "Front": {
            "position": np.array([0.0, 10.0, 0.0]),
            "focal_point": np.array([0.0, 0.0, 0.0]),
            "up_vector": np.array([0.0, 0.0, 1.0]),
        },
        "Back": {
            "position": np.array([0.0, -10.0, 0.0]),
            "focal_point": np.array([0.0, 0.0, 0.0]),
            "up_vector": np.array([0.0, 0.0, 1.0]),
        },
        "Left": {
            "position": np.array([-10.0, 0.0, 0.0]),
            "focal_point": np.array([0.0, 0.0, 0.0]),
            "up_vector": np.array([0.0, 0.0, 1.0]),
        },
        "Right": {
            "position": np.array([10.0, 0.0, 0.0]),
            "focal_point": np.array([0.0, 0.0, 0.0]),
            "up_vector": np.array([0.0, 0.0, 1.0]),
        },
        "Isometric": {
            "position": np.array([10.0, 10.0, 10.0]),
            "focal_point": np.array([0.0, 0.0, 0.0]),
            "up_vector": np.array([-1.0, 1.0, -1.0]),
        },
    }

    def __init__(self) -> None:
        """Initialize viewpoint manager with empty saved viewpoints."""
        self.viewpoints: dict[str, Viewpoint] = {}
        logger.debug("ViewpointManager initialized")

    def get_standard_view(self, view_name: str) -> Viewpoint:
        """Get a standard view by name.

        Args:
            view_name: One of Top, Bottom, Front, Back, Left, Right, Isometric

        Returns:
            Viewpoint instance for the standard view

        Raises:
            ValueError: If view_name is not a recognized standard view
        """
        if view_name not in self._STANDARD_VIEWS:
            valid = ", ".join(self._STANDARD_VIEWS.keys())
            raise ValueError(
                f"Unknown standard view: {view_name}. Must be one of: {valid}"
            )

        view_data = self._STANDARD_VIEWS[view_name]
        return Viewpoint(
            name=view_name,
            position=view_data["position"].copy(),
            focal_point=view_data["focal_point"].copy(),
            up_vector=view_data["up_vector"].copy(),
        )

    def set_standard_view(self, view_name: str) -> Viewpoint:
        """Set and save a standard view.

        Args:
            view_name: Standard view name

        Returns:
            Viewpoint instance (also saved to viewpoints dict)

        Raises:
            ValueError: If view_name is invalid
        """
        viewpoint = self.get_standard_view(view_name)
        self.save_viewpoint(viewpoint)
        logger.debug("Set standard view: %s", view_name)
        return viewpoint

    def save_viewpoint(self, viewpoint: Viewpoint) -> None:
        """Save a viewpoint.

        Args:
            viewpoint: Viewpoint to save
        """
        self.viewpoints[viewpoint.name] = viewpoint
        logger.debug("Saved viewpoint: %s", viewpoint.name)

    def load_viewpoint(self, name: str) -> Viewpoint:
        """Load a saved viewpoint by name.

        Args:
            name: Name of viewpoint to load

        Returns:
            Viewpoint instance

        Raises:
            KeyError: If viewpoint doesn't exist
        """
        if name not in self.viewpoints:
            raise KeyError(f"Viewpoint '{name}' not found")
        return self.viewpoints[name]

    def delete_viewpoint(self, name: str) -> None:
        """Delete a saved viewpoint.

        Args:
            name: Name of viewpoint to delete

        Raises:
            KeyError: If viewpoint doesn't exist
        """
        if name not in self.viewpoints:
            raise KeyError(f"Viewpoint '{name}' not found")
        del self.viewpoints[name]
        logger.debug("Deleted viewpoint: %s", name)

    def list_viewpoints(self) -> list[str]:
        """Get list of saved viewpoint names.

        Returns:
            List of viewpoint names
        """
        return list(self.viewpoints.keys())

    def animate_to_viewpoint(
        self,
        start: Viewpoint,
        end: Viewpoint,
        duration: float = 1.0,
        frame_rate: float = 60.0,
    ) -> Generator[Viewpoint, None, None]:
        """Animate smoothly from one viewpoint to another.

        Uses linear interpolation for all components. Yields intermediate
        viewpoints at specified frame rate.

        Args:
            start: Starting viewpoint
            end: Ending viewpoint
            duration: Animation duration in seconds (default 1.0)
            frame_rate: Frames per second (default 60)

        Yields:
            Viewpoint instances for each frame of animation

        Raises:
            ValueError: If duration <= 0
        """
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")

        num_frames = max(2, int(duration * frame_rate))
        logger.debug(
            "Animating viewpoint transition over %.1fs (%d frames)",
            duration,
            num_frames,
        )

        for i in range(num_frames):
            # Interpolation factor: 0 at start, 1 at end
            t = i / (num_frames - 1)

            # Linear interpolation for each component
            position = (1 - t) * start.position + t * end.position
            focal_point = (1 - t) * start.focal_point + t * end.focal_point

            # Interpolate up_vector and normalize
            up_vector = (1 - t) * start.up_vector + t * end.up_vector
            up_magnitude = np.linalg.norm(up_vector)
            if not np.isclose(up_magnitude, 0.0):
                up_vector = up_vector / up_magnitude

            frame = Viewpoint(
                name=f"{start.name}→{end.name}:{t:.2%}",
                position=position,
                focal_point=focal_point,
                up_vector=up_vector,
            )
            yield frame

    def save_to_json(self, filepath: Path | str) -> None:
        """Save all viewpoints to JSON file.

        Args:
            filepath: Path where to save JSON file
        """
        filepath = Path(filepath)
        data = {name: vp.to_dict() for name, vp in self.viewpoints.items()}

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug("Saved %d viewpoints to %s", len(data), filepath)

    def load_from_json(self, filepath: Path | str) -> None:
        """Load viewpoints from JSON file.

        Loaded viewpoints are added to existing viewpoints (not replaced).

        Args:
            filepath: Path to JSON file to load
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning("Viewpoint file not found: %s", filepath)
            return

        with open(filepath) as f:
            data = json.load(f)

        for _name, vp_data in data.items():
            viewpoint = Viewpoint.from_dict(vp_data)
            self.save_viewpoint(viewpoint)

        logger.debug("Loaded %d viewpoints from %s", len(data), filepath)

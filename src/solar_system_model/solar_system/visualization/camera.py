"""
Camera System
=============

Provides multiple camera modes for viewing the solar system:
- Free camera: User-controlled position and orientation
- Heliocentric: Fixed at sun, looking outward
- Planet-centric: Follow a planet, looking at other objects
- Spacecraft-following: Track a spacecraft trajectory
- Earth sky view: View from Earth's surface
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.celestial_body import CelestialBody, Spacecraft


class CameraMode(Enum):
    """Available camera viewing modes."""

    FREE = "free"
    HELIOCENTRIC = "heliocentric"
    PLANET_CENTRIC = "planet_centric"
    SPACECRAFT_FOLLOW = "spacecraft_follow"
    EARTH_SKY = "earth_sky"
    TOP_DOWN = "top_down"


@dataclass
class CameraState:
    """Current state of the camera."""

    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float = 60.0
    near: float = 0.001
    far: float = 1000.0

    def __post_init__(self) -> None:
        self.position = np.array(self.position, dtype=np.float64)
        self.target = np.array(self.target, dtype=np.float64)
        self.up = np.array(self.up, dtype=np.float64)


class Camera:
    """
    Sophisticated camera system with multiple viewing modes.

    Supports smooth transitions between modes and interactive control.
    """

    def __init__(
        self,
        position: tuple[float, float, float] = (0, 1500, 4500),
        target: tuple[float, float, float] = (0, 0, 0),
        up: tuple[float, float, float] = (0, 1, 0),
        fov: float = 60.0,
    ):
        """
        Initialize the camera.

        Args:
            position: Initial camera position (default shows entire solar system)
            target: Initial look-at target
            up: Up vector
            fov: Field of view in degrees
        """
        if not (position is not None):
            raise ValueError("position must be provided")
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)
        self.fov = fov

        # Spherical coordinates for orbital camera
        self._distance = np.linalg.norm(self.position - self.target)
        self._azimuth = 0.0  # Horizontal angle
        self._elevation = 0.0  # Vertical angle

        # Calculate initial angles
        self._update_angles_from_position()

        # Camera mode and tracking
        self.mode = CameraMode.FREE
        self.tracked_body: CelestialBody | None = None
        self.tracked_spacecraft: Spacecraft | None = None

        # Movement parameters
        self.move_speed = 0.5
        self.rotate_speed = 0.005
        self.zoom_speed = 0.1
        self.smooth_factor = 0.1

        # View bounds
        self.min_distance = 0.01
        self.max_distance = 25000.0  # Increased to view Oort cloud potential

        # Near/far clipping planes
        self.near = 0.0001
        self.far = 100000.0  # Increased significantly

        # Animation state
        self._target_position = self.position.copy()
        self._target_target = self.target.copy()
        self._animating = False

    def snapshot(self) -> CameraState:
        """Capture the current camera parameters."""
        return CameraState(
            position=self.position.copy(),
            target=self.target.copy(),
            up=self.up.copy(),
            fov=self.fov,
            near=self.near,
            far=self.far,
        )

    def apply_state(self, state: CameraState) -> None:
        """Apply a stored camera state."""
        # Use array assignment instead of copy for better performance
        if not (state is not None):
            raise ValueError("state must be provided")
        self.position[:] = state.position
        self.target[:] = state.target
        self.up[:] = state.up
        self.fov = state.fov
        self.near = state.near
        self.far = state.far

    def stereo_states(
        self, eye_separation: float = 0.4
    ) -> tuple[CameraState, CameraState]:
        """Generate left/right stereo eye offsets for VR-style rendering."""

        if not (eye_separation is not None):
            raise ValueError("eye_separation must be provided")
        base = self.snapshot()
        forward = base.target - base.position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, base.up)
        right = right / np.linalg.norm(right)

        offset = right * (eye_separation / 2.0)
        left_state = CameraState(
            position=base.position - offset,
            target=base.target - offset,
            up=base.up,
            fov=base.fov,
            near=base.near,
            far=base.far,
        )
        right_state = CameraState(
            position=base.position + offset,
            target=base.target + offset,
            up=base.up,
            fov=base.fov,
            near=base.near,
            far=base.far,
        )
        return left_state, right_state

    def _update_angles_from_position(self) -> None:
        """Calculate spherical coordinates from current position."""
        direction = self.position - self.target
        self._distance = np.linalg.norm(direction)

        if self._distance > 0:
            direction = direction / self._distance
            self._azimuth = math.atan2(direction[0], direction[2])
            self._elevation = math.asin(np.clip(direction[1], -1, 1))

    def _update_position_from_angles(self) -> None:
        """Update position from spherical coordinates."""
        x = self._distance * math.cos(self._elevation) * math.sin(self._azimuth)
        y = self._distance * math.sin(self._elevation)
        z = self._distance * math.cos(self._elevation) * math.cos(self._azimuth)

        self.position = self.target + np.array([x, y, z])

    def set_mode(
        self, mode: CameraMode, target_body: CelestialBody | None = None
    ) -> None:
        """
        Change camera mode.

        Args:
            mode: New camera mode
            target_body: Body to track (for planet-centric modes)
        """
        if not (mode is not None):
            raise ValueError("mode must be provided")
        self.mode = mode
        self.tracked_body = target_body

        if mode == CameraMode.HELIOCENTRIC:
            self._animate_to(position=np.array([0, 5, 10]), target=np.array([0, 0, 0]))
        elif mode == CameraMode.TOP_DOWN:
            self._animate_to(position=np.array([0, 50, 0]), target=np.array([0, 0, 0]))
            self.up = np.array([0, 0, -1])
        elif mode == CameraMode.PLANET_CENTRIC and target_body:
            # Will be updated in update() method
            pass

    def _animate_to(self, position: np.ndarray, target: np.ndarray) -> None:
        """Start smooth animation to new position."""
        if not (position is not None):
            raise ValueError("position must be provided")
        self._target_position = position
        self._target_target = target
        self._animating = True

    def orbit(self, delta_azimuth: float, delta_elevation: float) -> None:
        """
        Orbit the camera around the target point.

        Args:
            delta_azimuth: Change in horizontal angle (radians)
            delta_elevation: Change in vertical angle (radians)
        """
        if not (delta_azimuth is not None):
            raise ValueError("delta_azimuth must be provided")
        self._azimuth += delta_azimuth * self.rotate_speed
        self._elevation += delta_elevation * self.rotate_speed

        # Clamp elevation to prevent flipping
        self._elevation = np.clip(
            self._elevation, -math.pi / 2 + 0.01, math.pi / 2 - 0.01
        )

        self._update_position_from_angles()

    def zoom(self, delta: float) -> None:
        """
        Zoom camera in or out.

        Args:
            delta: Zoom amount (positive = zoom in)
        """
        if not (delta is not None):
            raise ValueError("delta must be provided")
        self._distance *= 1 - delta * self.zoom_speed
        self._distance = np.clip(self._distance, self.min_distance, self.max_distance)
        self._update_position_from_angles()

    def zoom_at(
        self, delta: float, mouse_ndc: tuple[float, float], aspect_ratio: float
    ) -> None:
        """
        Zoom towards a specific point on the screen (mouse cursor).

        Args:
           delta: Zoom amount (positive = zoom in, negative = zoom out).
           mouse_ndc: Mouse coordinates in Normalized Device Coordinates (-1 to 1).
           aspect_ratio: Screen aspect ratio.
        """
        if not (delta is not None):
            raise ValueError("delta must be provided")
        factor = 1.0 - delta * self.zoom_speed
        new_distance = self._distance * factor
        new_distance = np.clip(new_distance, self.min_distance, self.max_distance)

        # Calculate how much we zoomed
        # Calculate how much we zoomed
        actual_factor = new_distance / self._distance if self._distance > 0 else 1.0

        # If zooming in (factor < 1), we want to move the target towards the mouse ray
        # If zooming out, we usually just pull back

        # Simplified "zoom to cursor" for orbit camera:
        # We need to pan the camera so the point under cursor remains stable.
        # This effectively shifts the 'target'

        # Calculate Right and Up vectors
        forward = self.target - self.position
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 0:
            forward /= forward_norm
        else:
            forward = np.array([0, 0, -1])

        right = np.cross(forward, self.up)
        right_norm = np.linalg.norm(right)

        if right_norm > 1e-6:
            right /= right_norm
        else:
            # Handle degenerate case (looking straight up/down)
            # Choose an arbitrary "right" vector orthogonal to forward
            if abs(forward[1]) < 0.99:
                right = np.cross(forward, np.array([0, 1, 0]))
            else:
                right = np.array([1, 0, 0])

            # Re-normalize just in case
            rn = np.linalg.norm(right)
            if rn > 0:
                right /= rn

        up = np.cross(right, forward)

        # Calculate viewport dimensions at the target depth
        # box height at distance D = 2 * D * tan(fov/2)
        fov_rad = math.radians(self.fov)
        view_height = 2.0 * self._distance * math.tan(fov_rad / 2.0)
        view_width = view_height * aspect_ratio

        # Mouse offset from center (0,0) in NDC
        mx, my = mouse_ndc

        # Calculate shift in world space
        # We want to shift the target such that the point under cursor stays
        # fixed relative to camera frame?
        # Actually, standard "Google Earth" style:
        # offset = (mouse_pos_world_on_plane - camera_pos) * (1 - scale_factor)

        # Shift amount in camera plane
        shift_x = mx * (view_width / 2.0) * (1 - actual_factor)
        shift_y = my * (view_height / 2.0) * (1 - actual_factor)

        offset = right * shift_x + up * shift_y

        self.target += offset
        self.position += offset

        # Apply distance change
        self._distance = new_distance
        self._update_position_from_angles()

    def pan(self, delta_x: float, delta_y: float) -> None:
        """
        Pan the camera (move target and position together).

        Args:
            delta_x: Horizontal pan amount
            delta_y: Vertical pan amount
        """
        # Get right and up vectors in view space
        if not (delta_x is not None):
            raise ValueError("delta_x must be provided")
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Scale by distance for consistent feel
        scale = self._distance * 0.001

        offset = right * delta_x * scale + up * delta_y * scale

        self.position += offset
        self.target += offset

    def move_forward(self, amount: float) -> None:
        """Move camera forward/backward."""
        if not (amount is not None):
            raise ValueError("amount must be provided")
        direction = self.target - self.position
        direction = direction / np.linalg.norm(direction)
        self.position += direction * amount * self.move_speed
        self.target += direction * amount * self.move_speed

    def reset(self) -> None:
        """Reset camera to default view."""
        # Position camera far enough to see entire solar system (Neptune at ~4500 units)
        self.position = np.array([0, 1500, 4500], dtype=np.float64)
        self.target = np.array([0, 0, 0], dtype=np.float64)
        self.up = np.array([0, 1, 0], dtype=np.float64)
        self._update_angles_from_position()

    def update(self, julian_date: float, scale: float = 1e-9) -> None:
        """
        Update camera based on current mode and time.

        Args:
            julian_date: Current simulation time
            scale: Scale factor for converting positions
        """
        # Handle smooth animation
        if not (julian_date is not None):
            raise ValueError("julian_date must be provided")
        if self._animating:
            self.position = (
                self.position
                + (self._target_position - self.position) * self.smooth_factor
            )
            self.target = (
                self.target + (self._target_target - self.target) * self.smooth_factor
            )

            if np.linalg.norm(self.position - self._target_position) < 0.01:
                self._animating = False

        # Update based on mode
        if self.mode == CameraMode.PLANET_CENTRIC and self.tracked_body:
            state = self.tracked_body.get_state_at_time(julian_date)
            body_pos = state.position * scale

            # Position camera behind and above the body
            velocity = state.velocity
            if np.linalg.norm(velocity) > 0:
                forward = velocity / np.linalg.norm(velocity)
            else:
                forward = np.array([1, 0, 0])

            camera_offset = -forward * self._distance + np.array(
                [0, self._distance * 0.3, 0]
            )

            self._target_target = body_pos
            self._target_position = body_pos + camera_offset

            # Smooth follow
            self.target = self.target + (self._target_target - self.target) * 0.1
            self.position = (
                self.position + (self._target_position - self.position) * 0.1
            )

        elif self.mode == CameraMode.SPACECRAFT_FOLLOW and self.tracked_spacecraft:
            state = self.tracked_spacecraft.get_state_at_time(julian_date)
            spacecraft_pos = state.position * scale

            velocity = state.velocity
            if np.linalg.norm(velocity) > 0:
                forward = velocity / np.linalg.norm(velocity)
            else:
                forward = np.array([1, 0, 0])

            camera_offset = -forward * self._distance * 0.5 + np.array(
                [0, self._distance * 0.2, 0]
            )

            self.target = spacecraft_pos
            self.position = spacecraft_pos + camera_offset

        elif self.mode == CameraMode.EARTH_SKY and self.tracked_body:
            # View from Earth's surface looking up
            state = self.tracked_body.get_state_at_time(julian_date)
            earth_pos = state.position * scale

            # Position on Earth's surface (simplified)
            surface_offset = np.array(
                [0.0001, 0, 0]
            )  # Small offset representing surface

            self.position = earth_pos + surface_offset
            self.target = earth_pos + np.array([0, 0, 1])  # Looking up
            self.up = np.array([1, 0, 0])  # Radial direction is "up"

    def get_view_matrix(self) -> np.ndarray:
        """
        Calculate the view matrix.

        Returns:
            4x4 view matrix
        """
        # Calculate camera basis vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Build view matrix
        view = np.eye(4, dtype=np.float64)

        view[0, 0:3] = right
        view[1, 0:3] = up
        view[2, 0:3] = -forward

        view[0, 3] = -np.dot(right, self.position)
        view[1, 3] = -np.dot(up, self.position)
        view[2, 3] = np.dot(forward, self.position)

        return view

    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """
        Calculate the perspective projection matrix.

        Args:
            aspect_ratio: Width / height of viewport

        Returns:
            4x4 projection matrix
        """
        if not (aspect_ratio is not None):
            raise ValueError("aspect_ratio must be provided")
        fov_rad = math.radians(self.fov)
        f = 1.0 / math.tan(fov_rad / 2)

        projection = np.zeros((4, 4), dtype=np.float64)

        projection[0, 0] = f / aspect_ratio
        projection[1, 1] = f
        projection[2, 2] = (self.far + self.near) / (self.near - self.far)
        projection[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        projection[3, 2] = -1

        return projection

    def get_state(self) -> CameraState:
        """Get current camera state."""
        return CameraState(
            position=self.position.copy(),
            target=self.target.copy(),
            up=self.up.copy(),
            fov=self.fov,
            near=self.near,
            far=self.far,
        )

    def set_state(self, state: CameraState) -> None:
        """Restore camera from state."""
        if not (state is not None):
            raise ValueError("state must be provided")
        self.position = state.position.copy()
        self.target = state.target.copy()
        self.up = state.up.copy()
        self.fov = state.fov
        self.near = state.near
        self.far = state.far
        self._update_angles_from_position()

    def look_at(self, position: np.ndarray, smooth: bool = True) -> None:
        """
        Point camera at a specific position.

        Args:
            position: Position to look at
            smooth: Whether to animate smoothly
        """
        if smooth:
            self._target_target = position
            self._animating = True
        else:
            self.target = position
            self._update_angles_from_position()

    def set_distance(self, distance: float) -> None:
        """Set distance from target."""
        if not (distance is not None):
            raise ValueError("distance must be provided")
        self._distance = np.clip(distance, self.min_distance, self.max_distance)
        self._update_position_from_angles()

    @property
    def yaw(self) -> float:
        """Get the camera's horizontal angle (yaw) in radians."""
        return self._azimuth

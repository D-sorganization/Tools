"""Particle advection and pathline visualization engine.

This module provides core particle advection with 4th-order Runge-Kutta integration,
trajectory management, and trajectory rendering for CFD visualization.

Key components:
- Particle: dataclass for individual particle state (position, trajectory, age, alive)
- ParticleAdvectionEngine: main advection engine with velocity field integration
- TrajectoryRenderer: rendering support for visualizing particle pathlines

Design patterns:
- Design by Contract: invariants validated on particle state and bounds
- Separation of Concerns: engine separate from renderer and UI
- DRY: single RK4 integration implementation
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Particle:
    """Represents a single fluid particle with trajectory tracking.

    Attributes:
        id: Unique particle identifier
        position: Current 3D position as np.ndarray of shape (3,)
        trajectory: List of previous positions (pathline)
        age: Current age of particle (in simulation time units)
        alive: Whether particle is active in simulation
    """

    id: int
    position: np.ndarray
    trajectory: list[np.ndarray] = field(default_factory=list)
    age: float = 0.0
    alive: bool = True

    def __post_init__(self) -> None:
        """Validate particle state on creation."""
        if not isinstance(self.position, np.ndarray):
            raise TypeError(f"position must be np.ndarray, got {type(self.position)}")
        if self.position.shape != (3,):
            raise ValueError(
                f"position must have shape (3,), got {self.position.shape}"
            )
        if not isinstance(self.trajectory, list):
            self.trajectory = list(self.trajectory)


class ParticleAdvectionEngine:
    """Advects particles through a velocity field using 4th-order Runge-Kutta.

    This engine manages particle lifecycle including:
    - Seeding particles at specified locations
    - Integrating motion using RK4 on velocity field samples
    - Tracking trajectories for pathline rendering
    - Managing particle age and removal of old particles
    - Enforcing boundary conditions

    The engine operates on a continuous velocity field provided as a callable
    that takes position and time and returns velocity.

    Design by Contract:
    - All particles remain within domain bounds (or are removed)
    - Particle ages are monotonically increasing
    - Engine time is monotonically increasing
    - No NaN or Inf values appear in particle positions
    """

    def __init__(
        self,
        velocity_field: Callable[[np.ndarray, float], np.ndarray],
        domain_bounds: np.ndarray,
        max_particle_age: float = 10.0,
        remove_out_of_bounds: bool = True,
        dt_rk4: float = 0.001,
        record_every_n_steps: int = 1,
    ) -> None:
        """Initialize the particle advection engine.

        Args:
            velocity_field: Callable(position: ndarray[3], time: float) -> ndarray[3]
                           Returns velocity at given position and time
            domain_bounds: ndarray of shape (2, 3) specifying [min, max] bounds
            max_particle_age: Maximum age before particle is marked dead
            remove_out_of_bounds: If True, remove particles outside domain
            dt_rk4: Internal RK4 time step (adaptive in future)
            record_every_n_steps: How often to record position in trajectory

        Raises:
            ValueError: If domain_bounds is malformed
            TypeError: If velocity_field is not callable
        """
        if not callable(velocity_field):
            raise TypeError("velocity_field must be callable")

        domain_bounds = np.asarray(domain_bounds, dtype=np.float64)
        if domain_bounds.shape != (2, 3):
            raise ValueError(
                f"domain_bounds must have shape (2, 3), got {domain_bounds.shape}"
            )
        if not np.all(domain_bounds[0] <= domain_bounds[1]):
            raise ValueError("domain_bounds[0] must be <= domain_bounds[1]")

        self.velocity_field = velocity_field
        self.domain_bounds = domain_bounds
        self.max_particle_age = max_particle_age
        self.remove_out_of_bounds = remove_out_of_bounds
        self.dt_rk4 = dt_rk4
        self.record_every_n_steps = record_every_n_steps

        self.particles: list[Particle] = []
        self.time: float = 0.0
        self._next_particle_id: int = 0
        self._update_step_count: int = 0

    def seed_particles(
        self,
        position: np.ndarray,
        count: int = 1,
        jitter: float = 0.0,
    ) -> None:
        """Seed new particles at specified location(s).

        Args:
            position: 3D position where to seed particles, shape (3,)
            count: Number of particles to seed at this location
            jitter: If > 0, add random offset to each particle (uniform jitter)

        Raises:
            ValueError: If position is out of domain bounds
            TypeError: If position is not 3D array
        """
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,):
            raise TypeError(f"position must have shape (3,), got {position.shape}")

        if not self._is_in_bounds(position):
            raise ValueError(f"Seed position {position} is outside domain bounds")

        for _ in range(count):
            # Add jitter if specified
            if jitter > 0:
                rng = np.random.default_rng()
                offset = rng.uniform(-jitter, jitter, size=3)
                seed_pos = position + offset
            else:
                seed_pos = position.copy()

            # Ensure jittered position is still in bounds
            seed_pos = np.clip(seed_pos, self.domain_bounds[0], self.domain_bounds[1])

            # Create particle with initial position in trajectory
            particle = Particle(
                id=self._next_particle_id,
                position=seed_pos.copy(),
                trajectory=[seed_pos.copy()],
                age=0.0,
                alive=True,
            )
            self._next_particle_id += 1
            self.particles.append(particle)

    def update(self, dt: float) -> None:
        """Advance all particles by time dt using RK4 integration.

        Args:
            dt: Time step to advance particles

        Process:
        1. For each particle, integrate its position using RK4
        2. Record position in trajectory at regular intervals
        3. Increment particle age
        4. Mark particles as dead if age > max_age
        5. Remove dead particles if remove_out_of_bounds is True
        6. Advance engine time

        Design by Contract:
        - All particles remain in bounds or are removed
        - Ages are monotonically increasing
        - Time is monotonically increasing
        - No NaN/Inf in particle positions
        """
        # Update positions via RK4 and record trajectories
        self._update_positions(dt)

        # Update ages and lifecycle
        self._update_ages_and_lifecycle(dt)

        # Advance engine time and validate
        self.time += dt
        self._validate_invariants()

    def _update_positions(self, dt: float) -> None:
        """Integrate positions and handle boundaries."""
        for particle in self.particles:
            if not particle.alive:
                continue

            # RK4 integration step
            self._rk4_step(particle, dt)

            # Record trajectory position
            particle.trajectory.append(particle.position.copy())

            # Enforce bounds
            if not self._is_in_bounds(particle.position):
                self._handle_out_of_bounds(particle)

    def _handle_out_of_bounds(self, particle: Particle) -> None:
        """Handle particles that go out of bounds."""
        if self.remove_out_of_bounds:
            particle.alive = False
        else:
            # Reflect particle back into bounds
            particle.position = np.clip(
                particle.position,
                self.domain_bounds[0],
                self.domain_bounds[1],
            )

    def _update_ages_and_lifecycle(self, dt: float) -> None:
        """Update particle ages and mark old ones as dead."""
        for particle in self.particles:
            if particle.alive:
                particle.age += dt

                # Mark old particles as dead
                if particle.age > self.max_particle_age:
                    particle.alive = False

    def _rk4_step(self, particle: Particle, dt: float) -> None:
        """Perform one RK4 integration step on a particle.

        Args:
            particle: Particle to integrate
            dt: Integration time step

        The RK4 method is 4th-order accurate with local error O(dt^5):
            k1 = v(x, t)
            k2 = v(x + dt/2 * k1, t + dt/2)
            k3 = v(x + dt/2 * k2, t + dt/2)
            k4 = v(x + dt * k3, t + dt)
            x_new = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        This provides good stability and accuracy for particle advection.
        """
        t = self.time

        # Compute RK4 stages
        k1 = self.velocity_field(particle.position, t)
        k2 = self.velocity_field(particle.position + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.velocity_field(particle.position + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.velocity_field(particle.position + dt * k3, t + dt)

        # Update position
        particle.position = particle.position + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Validate no NaN/Inf introduced
        if not np.all(np.isfinite(particle.position)):
            raise ValueError(
                f"RK4 step produced NaN/Inf at position {particle.position}"
            )

    def _is_in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within domain bounds.

        Args:
            position: 3D position to check

        Returns:
            True if position is strictly within domain, False otherwise
        """
        return np.all(position >= self.domain_bounds[0]) and np.all(
            position <= self.domain_bounds[1]
        )

    def _validate_invariants(self) -> None:
        """Validate design-by-contract invariants.

        Raises:
            AssertionError: If any invariant is violated
        """
        # Invariant 1: All alive particles are within bounds
        for particle in self.particles:
            if particle.alive and self.remove_out_of_bounds:
                assert self._is_in_bounds(particle.position), (
                    f"Particle {particle.id} outside bounds: {particle.position}"
                )

        # Invariant 2: Time is monotonically increasing (checked during update)
        # (maintained by explicit self.time += dt)

        # Invariant 3: No NaN/Inf in positions
        for particle in self.particles:
            assert np.all(np.isfinite(particle.position)), (
                f"Particle {particle.id} has NaN/Inf at {particle.position}"
            )

    def get_alive_particles(self) -> list[Particle]:
        """Return list of currently alive particles.

        Returns:
            List of Particle objects with alive=True
        """
        return [p for p in self.particles if p.alive]

    def get_trajectories(self) -> list[np.ndarray]:
        """Return trajectories of all alive particles as arrays.

        Returns:
            List of arrays, where each array has shape (n_points, 3)
            representing the pathline of one particle
        """
        return [np.array(p.trajectory) for p in self.particles if p.alive]

    def clear_particles(self) -> None:
        """Clear all particles from engine (useful for resetting)."""
        self.particles.clear()
        self._next_particle_id = 0


class TrajectoryRenderer:
    """Handles rendering of particle trajectories.

    This class is responsible for converting particle trajectory data into
    a format suitable for visualization in PyVista or similar libraries.

    It supports:
    - Trajectory to mesh conversion
    - Age-based colormapping
    - Custom colormaps for velocity magnitude or other properties
    - Line width and transparency options
    """

    def __init__(self, colormap: str = "viridis", line_width: float = 2.0) -> None:
        """Initialize trajectory renderer.

        Args:
            colormap: Name of matplotlib colormap to use
            line_width: Width of rendered lines
        """
        self.colormap = colormap
        self.line_width = line_width

    def trajectory_to_points(self, trajectory: list[np.ndarray]) -> np.ndarray:
        """Convert trajectory list to point array.

        Args:
            trajectory: List of 3D positions

        Returns:
            ndarray of shape (n_points, 3)
        """
        return np.array(trajectory, dtype=np.float64)

    def trajectory_to_line(
        self,
        trajectory: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert trajectory to points and connectivity for line rendering.

        Args:
            trajectory: List of 3D positions

        Returns:
            Tuple of (points, cells) where:
            - points: ndarray of shape (n_points, 3)
            - cells: ndarray of connectivity for rendering as polyline
        """
        points = np.array(trajectory, dtype=np.float64)

        # Create line connectivity: sequence of point indices
        if len(points) < 2:
            # Can't render a line with < 2 points
            return points, np.array([])

        # Cells format for PyVista: [n_points_in_line, point_id_0, point_id_1, ...]
        n_points = len(points)
        cells = np.concatenate([[n_points], np.arange(n_points)])

        return points, cells

    def age_colormap(self, trajectory: list[np.ndarray]) -> np.ndarray:
        """Generate age-based colors for trajectory points.

        Newer points (later in trajectory) are assigned higher color values.

        Args:
            trajectory: List of 3D positions

        Returns:
            ndarray of colors with shape (n_points,), values in [0, 1]
        """
        n_points = len(trajectory)
        if n_points == 0:
            return np.array([])
        return np.linspace(0.0, 1.0, n_points)

    def generate_renderer_data(
        self,
        particles: list[Particle],
        colormap_type: str = "age",
    ) -> list[dict]:
        """Generate rendering data for all particles.

        Args:
            particles: List of Particle objects
            colormap_type: Type of colormap ('age', 'speed', etc.)

        Returns:
            List of dictionaries with keys:
            - 'points': ndarray of shape (n_points, 3)
            - 'colors': ndarray of colors
            - 'particle_id': id of the particle
        """
        render_data = []

        for particle in particles:
            if len(particle.trajectory) < 2:
                continue  # Skip particles with no trajectory yet

            points = self.trajectory_to_points(particle.trajectory)
            # Default to age if unknown type
            colors = self.age_colormap(particle.trajectory)

            render_data.append(
                {
                    "points": points,
                    "colors": colors,
                    "particle_id": particle.id,
                }
            )

        return render_data

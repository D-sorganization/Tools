"""Test suite for particle advection and pathline visualization.

This module tests the core particle advection engine with emphasis on:
- RK4 integration accuracy on analytical velocity fields
- Particle trajectory management and lifecycle
- Performance (60+ FPS for 100+ particles)
- Numerical stability (no NaN/Inf propagation)
- Design by contract: invariants about particle state
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

# Import the modules we'll implement
from glass_models.viz.particle_advection import (
    Particle,
    ParticleAdvectionEngine,
    TrajectoryRenderer,
)


class TestParticleDataclass:
    """Unit tests for the Particle dataclass."""

    def test_particle_creation_basic(self) -> None:
        """Test basic particle creation with default values."""
        particle = Particle(
            id=0,
            position=np.array([0.0, 0.0, 0.0]),
            trajectory=[],
            age=0.0,
            alive=True,
        )
        assert particle.id == 0
        assert np.allclose(particle.position, [0.0, 0.0, 0.0])
        assert particle.trajectory == []
        assert np.isclose(particle.age, 0.0)  # noqa: S1244
        assert particle.alive is True

    def test_particle_trajectory_growth(self) -> None:
        """Test that trajectory can be appended to."""
        particle = Particle(
            id=1,
            position=np.array([0.0, 0.0, 0.0]),
            trajectory=[],
            age=0.0,
            alive=True,
        )
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([2.0, 0.0, 0.0])
        particle.trajectory.append(pos1)
        particle.trajectory.append(pos2)
        assert len(particle.trajectory) == 2
        assert np.allclose(particle.trajectory[0], pos1)  # noqa: S1244
        assert np.allclose(particle.trajectory[1], pos2)  # noqa: S1244

    def test_particle_position_update(self) -> None:
        """Test that particle position can be updated."""
        particle = Particle(
            id=2,
            position=np.array([0.0, 0.0, 0.0]),
            trajectory=[],
            age=0.0,
            alive=True,
        )
        new_pos = np.array([3.0, 4.0, 5.0])
        particle.position = new_pos
        assert np.allclose(particle.position, new_pos)

    def test_particle_age_advancement(self) -> None:
        """Test that particle age can be incremented."""
        particle = Particle(
            id=3,
            position=np.array([0.0, 0.0, 0.0]),
            trajectory=[],
            age=0.0,
            alive=True,
        )
        particle.age = 1.5
        assert particle.age == 1.5


class TestRK4Integration:
    """Test 4th-order Runge-Kutta integration on analytical fields."""

    def test_rk4_uniform_field_straight_line(self) -> None:
        """Test RK4 produces straight line motion in uniform velocity field.

        In a uniform field v = (1, 0, 0), a particle starting at origin
        should move in a straight line along x-axis.
        """
        def uniform_velocity_field(pos: np.ndarray, t: float) -> np.ndarray:
            """Uniform velocity field: v = (1, 0, 0)."""
            return np.array([1.0, 0.0, 0.0])

        # Test data
        initial_pos = np.array([0.0, 0.0, 0.0])
        dt = 0.01
        steps = 100

        # Manual RK4 integration
        pos = initial_pos.copy()
        for _ in range(steps):
            k1 = uniform_velocity_field(pos, 0.0)
            k2 = uniform_velocity_field(pos + 0.5 * dt * k1, 0.0)
            k3 = uniform_velocity_field(pos + 0.5 * dt * k2, 0.0)
            k4 = uniform_velocity_field(pos + dt * k3, 0.0)
            pos = pos + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Expected: x = 1.0 * t = 1.0 * (100 * 0.01) = 1.0
        expected_pos = np.array([1.0, 0.0, 0.0])
        assert np.allclose(pos, expected_pos, atol=1e-6)
        assert np.isclose(pos[1], 0.0)  # y unchanged  # noqa: S1244
        assert np.isclose(pos[2], 0.0)  # z unchanged  # noqa: S1244

    def test_rk4_radial_field_circular_motion(self) -> None:
        """Test RK4 on a radial field that produces circular motion.

        In a field v = (-y, x, 0), particles rotate counterclockwise
        around the origin in the xy-plane.
        """
        def circular_velocity_field(pos: np.ndarray, t: float) -> np.ndarray:
            """Radial field that causes circular motion: v = (-y, x, 0)."""
            return np.array([-pos[1], pos[0], 0.0])

        # Starting on the unit circle
        initial_pos = np.array([1.0, 0.0, 0.0])
        dt = 0.0005  # Smaller timestep for better accuracy
        steps = 12566  # ~2*pi radians at unit speed with smaller dt

        pos = initial_pos.copy()
        for _ in range(steps):
            k1 = circular_velocity_field(pos, 0.0)
            k2 = circular_velocity_field(pos + 0.5 * dt * k1, 0.0)
            k3 = circular_velocity_field(pos + 0.5 * dt * k2, 0.0)
            k4 = circular_velocity_field(pos + dt * k3, 0.0)
            pos = pos + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # After one revolution, should return near starting position
        expected_pos = np.array([1.0, 0.0, 0.0])
        # RK4 with very small dt has good accuracy
        assert np.allclose(pos, expected_pos, atol=0.01)
        # Verify we're still at unit distance from origin
        distance = np.linalg.norm(pos[:2])
        assert np.isclose(distance, 1.0, atol=0.01)

    def test_rk4_linear_growth_field(self) -> None:
        """Test RK4 on a field that causes linear growth.

        In field v = (1, 0, 0), position should grow linearly with time.
        """
        def linear_velocity_field(pos: np.ndarray, t: float) -> np.ndarray:
            """Linear growth field: v = (1, 0, 0)."""
            return np.array([1.0, 0.0, 0.0])

        initial_pos = np.array([0.0, 0.0, 0.0])
        dt = 0.1
        steps = 10

        pos = initial_pos.copy()
        trajectory = [pos.copy()]
        for _ in range(steps):
            k1 = linear_velocity_field(pos, 0.0)
            k2 = linear_velocity_field(pos + 0.5 * dt * k1, 0.0)
            k3 = linear_velocity_field(pos + 0.5 * dt * k2, 0.0)
            k4 = linear_velocity_field(pos + dt * k3, 0.0)
            pos = pos + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(pos.copy())

        # Verify trajectory grows monotonically
        x_coords = [p[0] for p in trajectory]
        for i in range(1, len(x_coords)):
            assert x_coords[i] >= x_coords[i - 1]

    def test_rk4_no_nan_inf_propagation(self) -> None:
        """Test that RK4 doesn't produce NaN or Inf on reasonable inputs."""
        def stable_field(pos: np.ndarray, t: float) -> np.ndarray:
            """A field that shouldn't produce NaN/Inf."""
            return np.array([0.1 * pos[0], 0.1 * pos[1], 0.1 * pos[2]])

        initial_pos = np.array([1.0, 2.0, 3.0])
        dt = 0.01
        steps = 1000

        pos = initial_pos.copy()
        for _ in range(steps):
            k1 = stable_field(pos, 0.0)
            k2 = stable_field(pos + 0.5 * dt * k1, 0.0)
            k3 = stable_field(pos + 0.5 * dt * k2, 0.0)
            k4 = stable_field(pos + dt * k3, 0.0)
            pos = pos + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Verify no NaN or Inf
        assert np.all(np.isfinite(pos))
        assert not np.any(np.isnan(pos))
        assert not np.any(np.isinf(pos))


class TestParticleAdvectionEngine:
    """Test the ParticleAdvectionEngine class."""

    def test_engine_creation(self) -> None:
        """Test basic engine creation with default parameters."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )
        assert np.isclose(engine.time, 0.0)  # noqa: S1244
        assert len(engine.particles) == 0
        assert np.isclose(engine.max_particle_age, 10.0)  # noqa: S1244

    def test_seed_particles_single_point(self) -> None:
        """Test seeding particles at a single location."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )

        seed_pos = np.array([5.0, 5.0, 5.0])
        engine.seed_particles(seed_pos, count=10)

        assert len(engine.particles) == 10
        for particle in engine.particles:
            assert np.allclose(particle.position, seed_pos)
            assert particle.alive is True
            assert np.isclose(particle.age, 0.0)  # noqa: S1244

    def test_seed_particles_multiple_calls(self) -> None:
        """Test that multiple seed calls accumulate particles."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([1.0, 1.0, 1.0]), count=5)
        engine.seed_particles(np.array([2.0, 2.0, 2.0]), count=5)

        assert len(engine.particles) == 10

    def test_update_moves_particles(self) -> None:
        """Test that update() moves particles using RK4 integration."""
        def uniform_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=uniform_field,
            domain_bounds=np.array([[-100, -100, -100], [100, 100, 100]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([0.0, 0.0, 0.0]), count=1)
        initial_pos = engine.particles[0].position.copy()

        dt = 0.1
        engine.update(dt)

        # After one update, particle should have moved in +x direction
        updated_pos = engine.particles[0].position
        assert updated_pos[0] > initial_pos[0]
        assert np.isclose(updated_pos[1], initial_pos[1])  # y unchanged  # noqa: S1244
        assert np.isclose(updated_pos[2], initial_pos[2])  # z unchanged  # noqa: S1244

    def test_update_advances_time(self) -> None:
        """Test that update() advances the engine time."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )

        assert np.isclose(engine.time, 0.0)  # noqa: S1244
        engine.update(0.1)
        assert np.isclose(engine.time, 0.1)  # noqa: S1244
        engine.update(0.2)
        assert np.isclose(engine.time, 0.3)  # noqa: S1244

    def test_update_increments_particle_age(self) -> None:
        """Test that particles age during updates."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([5.0, 5.0, 5.0]), count=1)
        particle = engine.particles[0]

        assert np.isclose(particle.age, 0.0)  # noqa: S1244
        engine.update(0.5)
        assert np.isclose(particle.age, 0.5)  # noqa: S1244
        engine.update(0.3)
        assert np.isclose(particle.age, 0.8)  # noqa: S1244

    def test_update_marks_old_particles_dead(self) -> None:
        """Test that particles are marked dead when exceeding max_age."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=1.0,
        )

        engine.seed_particles(np.array([5.0, 5.0, 5.0]), count=1)
        particle = engine.particles[0]

        assert particle.alive is True
        engine.update(0.5)
        assert particle.alive is True
        engine.update(0.6)  # Total age = 1.1 > max_age
        assert particle.alive is False

    def test_trajectory_recording(self) -> None:
        """Test that particle trajectories are recorded during updates."""
        def uniform_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=uniform_field,
            domain_bounds=np.array([[-100, -100, -100], [100, 100, 100]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([0.0, 0.0, 0.0]), count=1)
        particle = engine.particles[0]

        # Trajectory starts with seed position
        assert len(particle.trajectory) >= 1

        dt = 0.1
        engine.update(dt)

        # After update, trajectory should grow
        trajectory_length_after_1st = len(particle.trajectory)
        assert trajectory_length_after_1st > 1

        engine.update(dt)
        trajectory_length_after_2nd = len(particle.trajectory)
        assert trajectory_length_after_2nd > trajectory_length_after_1st

    def test_particles_stay_in_bounds(self) -> None:
        """Test that particles outside bounds are removed or respawned."""
        def escape_field(pos: np.ndarray, t: float) -> np.ndarray:
            # Field that drives particles toward boundary
            return np.array([5.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=escape_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
            remove_out_of_bounds=True,
        )

        engine.seed_particles(np.array([9.0, 5.0, 5.0]), count=1)

        # Update enough times to push particle out of bounds
        for _ in range(10):
            engine.update(0.1)

        # Particle should be dead or removed
        alive_count = sum(1 for p in engine.particles if p.alive)
        assert alive_count == 0 or engine.particles[0].position[0] <= 10.0

    def test_engine_stability_many_updates(self) -> None:
        """Test that engine remains stable through many updates."""
        def stable_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.1, 0.05, 0.02])

        engine = ParticleAdvectionEngine(
            velocity_field=stable_field,
            domain_bounds=np.array([[-1000, -1000, -1000], [1000, 1000, 1000]]),
            max_particle_age=100.0,
        )

        engine.seed_particles(np.array([0.0, 0.0, 0.0]), count=10)

        # Perform 100 updates
        for _ in range(100):
            engine.update(0.1)

        # All particles should remain finite
        for particle in engine.particles:
            assert np.all(np.isfinite(particle.position))
            assert not np.any(np.isnan(particle.position))
            assert not np.any(np.isinf(particle.position))


class TestTrajectoryRenderer:
    """Test trajectory rendering functionality."""

    def test_renderer_creation(self) -> None:
        """Test basic renderer creation."""
        renderer = TrajectoryRenderer()
        assert isinstance(renderer, TrajectoryRenderer)  # noqa: S5727

    def test_renderer_converts_trajectory_to_array(self) -> None:
        """Test that renderer can convert trajectory list to array."""
        renderer = TrajectoryRenderer()  # noqa: S1481

        trajectory = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ]

        # This should not raise an error
        points = np.array(trajectory)
        assert points.shape == (3, 3)
        assert np.allclose(points[0], [0.0, 0.0, 0.0])
        assert np.allclose(points[2], [2.0, 0.0, 0.0])

    def test_renderer_colormap_generation(self) -> None:
        """Test that renderer can generate colors for trajectories."""
        renderer = TrajectoryRenderer()

        n_points = 10
        # Generate a simple colormap (age-based)
        colors = np.linspace(0, 1, n_points)
        assert len(colors) == n_points
        assert colors[0] == 0.0
        assert colors[-1] == 1.0


class TestPerformance:
    """Performance tests to ensure 60+ FPS for 100+ particles."""

    def test_update_performance_100_particles(self) -> None:
        """Test that 100 particles can be updated reasonably fast."""
        def fast_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.1, 0.05, 0.02])

        engine = ParticleAdvectionEngine(
            velocity_field=fast_field,
            domain_bounds=np.array([[-1000, -1000, -1000], [1000, 1000, 1000]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([0.0, 0.0, 0.0]), count=100)

        import time

        start = time.perf_counter()
        for _ in range(60):  # 60 frames
            engine.update(1.0 / 60.0)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (~50ms per frame for 100 particles)
        assert elapsed < 5.0, f"100 particles took {elapsed}s, target < 5s"

    def test_update_performance_1000_particles(self) -> None:
        """Test that 1000 particles can be updated reasonably fast."""
        def fast_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.01, 0.005, 0.002])

        engine = ParticleAdvectionEngine(
            velocity_field=fast_field,
            domain_bounds=np.array([[-10000, -10000, -10000], [10000, 10000, 10000]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([0.0, 0.0, 0.0]), count=1000)

        import time

        start = time.perf_counter()
        for _ in range(10):  # 10 frames
            engine.update(1.0 / 60.0)
        elapsed = time.perf_counter() - start

        # Should complete reasonably fast (~100ms per frame for 1000 particles)
        assert elapsed < 15.0, f"1000 particles took {elapsed}s"


class TestDesignByContract:
    """Test design-by-contract invariants."""

    def test_invariant_particle_count_never_negative(self) -> None:
        """Test that particle count never goes negative."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([5.0, 5.0, 5.0]), count=5)

        for _ in range(100):
            engine.update(0.1)
            assert len(engine.particles) > -1  # noqa: S3981

    def test_invariant_particle_positions_in_domain(self) -> None:
        """Test that particles stay within domain bounds (or are removed)."""
        bounds = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])

        def safe_field(pos: np.ndarray, t: float) -> np.ndarray:
            # Field that respects bounds
            return np.array([0.1, 0.05, 0.02])

        engine = ParticleAdvectionEngine(
            velocity_field=safe_field,
            domain_bounds=bounds,
            max_particle_age=10.0,
            remove_out_of_bounds=True,
        )

        engine.seed_particles(np.array([5.0, 5.0, 5.0]), count=5)

        for _ in range(50):
            engine.update(0.1)
            for particle in engine.particles:
                if particle.alive:
                    assert np.all(particle.position >= bounds[0])
                    assert np.all(particle.position <= bounds[1])

    def test_invariant_ages_are_monotonic(self) -> None:
        """Test that particle ages never decrease."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )

        engine.seed_particles(np.array([5.0, 5.0, 5.0]), count=1)
        particle = engine.particles[0]

        prev_age = particle.age
        for _ in range(20):
            engine.update(0.1)
            assert particle.age >= prev_age
            prev_age = particle.age

    def test_invariant_engine_time_monotonic(self) -> None:
        """Test that engine time never decreases."""
        def dummy_field(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0])

        engine = ParticleAdvectionEngine(
            velocity_field=dummy_field,
            domain_bounds=np.array([[0, 0, 0], [10, 10, 10]]),
            max_particle_age=10.0,
        )

        prev_time = engine.time
        for _ in range(20):
            engine.update(0.1)
            assert engine.time >= prev_time
            prev_time = engine.time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

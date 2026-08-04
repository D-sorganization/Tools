# ruff: noqa: E501
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

#!/usr/bin/env python3
"""Asteroid-field RRT path planner with optional 3D visualization."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import pygame
    from pygame.locals import DOUBLEBUF, K_ESCAPE, K_SPACE, KEYDOWN, OPENGL, QUIT, K_c

    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    DOUBLEBUF = K_c = K_ESCAPE = K_SPACE = KEYDOWN = OPENGL = QUIT = 0
    PYGAME_AVAILABLE = False

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None
    TRIMESH_AVAILABLE = False

try:
    from OpenGL.GL import (
        GL_AMBIENT,
        GL_COLOR_BUFFER_BIT,
        GL_COLOR_MATERIAL,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_DIFFUSE,
        GL_LIGHT0,
        GL_LIGHTING,
        GL_LINE_STRIP,
        GL_MODELVIEW,
        GL_NORMALIZE,
        GL_POINTS,
        GL_POSITION,
        GL_PROJECTION,
        GL_QUADS,
        GL_TRIANGLES,
        glBegin,
        glClear,
        glColor3f,
        glDisable,
        glEnable,
        glEnd,
        glLightfv,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glNormal3f,
        glPointSize,
        glPopMatrix,
        glPushMatrix,
        glTranslatef,
        glVertex3f,
    )
    from OpenGL.GLU import (
        gluDeleteQuadric,
        gluLookAt,
        gluNewQuadric,
        gluPerspective,
        gluSphere,
    )

    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

Vector3 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Obstacle:
    """Obstacle representation for the asteroid field."""

    type: int  # 0=sphere, 1=cube
    position: Vector3
    size: float
    color: tuple[float, float, float]


@dataclass
class Ship:
    """Ship state used by the planner and renderer."""

    position: Vector3
    orientation: npt.NDArray[np.float64]
    velocity: Vector3
    model: Any | None = None
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class PathMetrics:
    """Educational summary statistics for a planned route."""

    waypoint_count: int
    path_length: float
    straight_line_distance: float
    efficiency: float
    min_clearance: float
    mean_turn_angle_deg: float


def _coerce_bounds(bounds: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Validate and normalize the bounds array."""
    if len(bounds) != 6:
        raise ValueError(
            "Bounds must contain 6 elements: [xmin, xmax, ymin, ymax, zmin, zmax]"
        )

    bounds_array = np.asarray(bounds, dtype=np.float64)
    if any(bounds_array[i] >= bounds_array[i + 1] for i in range(0, 6, 2)):
        raise ValueError("Invalid bounds: min must be less than max for each dimension")
    return bounds_array


def _as_point(name: str, point: npt.NDArray[np.float64]) -> Vector3:
    """Validate that a point is a 3D vector."""
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (3,):
        raise ValueError(f"{name} must be a 3D point with shape (3,)")
    return point_array


def distance_to_obstacle_surface(
    point: npt.NDArray[np.float64], obstacle: Obstacle
) -> float:
    """Return signed clearance from a point to an obstacle surface."""
    if point is None:
        raise TypeError("point must not be None")
    point_array = _as_point("point", point)

    if obstacle.type == 0:
        return float(np.linalg.norm(point_array - obstacle.position) - obstacle.size)

    delta = np.abs(point_array - obstacle.position) - obstacle.size / 2
    outside = np.maximum(delta, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    inside = float(np.max(delta))
    return outside_norm if outside_norm > 0 else inside


def generate_asteroid_field(
    bounds: npt.NDArray[np.float64],
    num_obstacles: int,
    *,
    rng: np.random.Generator | None = None,
    reserved_points: list[npt.NDArray[np.float64]] | None = None,
    clearance: float = 0.0,
) -> list[Obstacle]:
    """Generate an asteroid field while preserving safe launch corridors."""
    if bounds is None:
        raise TypeError("bounds must not be None")
    bounds_array = _coerce_bounds(bounds)
    generator = rng or np.random.default_rng()
    protected_points = [_as_point("reserved_point", p) for p in (reserved_points or [])]

    obstacles: list[Obstacle] = []
    max_attempts = max(200, num_obstacles * 50)
    attempts = 0

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        obstacle_type = int(generator.integers(0, 2))
        position = np.array(
            [
                generator.uniform(bounds_array[0], bounds_array[1]),
                generator.uniform(bounds_array[2], bounds_array[3]),
                generator.uniform(bounds_array[4], bounds_array[5]),
            ],
            dtype=np.float64,
        )
        size = float(generator.uniform(0.02, 0.08))
        color = tuple(float(v) for v in generator.uniform(0.45, 1.0, size=3))
        obstacle = Obstacle(obstacle_type, position, size, color)  # type: ignore[arg-type]

        if all(
            distance_to_obstacle_surface(point, obstacle) >= clearance
            for point in protected_points
        ):
            obstacles.append(obstacle)

    return obstacles


class RRTPlanner:
    """Rapidly-exploring random tree planner with analysis helpers."""

    def __init__(
        self,
        bounds: npt.NDArray[np.float64],
        max_iterations: int = 5000,
        *,
        seed: int | None = None,
    ) -> None:
        """Initialize the planner with validated bounds and parameters."""
        if bounds is None:
            raise TypeError("bounds must not be None")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        self.bounds = _coerce_bounds(bounds)
        self.max_iterations = max_iterations
        self.step_size = 0.05
        self.goal_radius = 0.1
        self.goal_bias = 0.2
        self._rng = random.Random(seed)
        self.last_plan_metrics: PathMetrics | None = None

    def _point_within_bounds(self, point: Vector3) -> bool:
        """Return True when a point lies inside the configured search bounds."""
        return bool(
            self.bounds[0] <= point[0] <= self.bounds[1]
            and self.bounds[2] <= point[1] <= self.bounds[3]
            and self.bounds[4] <= point[2] <= self.bounds[5]
        )

    def _validate_waypoint(self, name: str, point: npt.NDArray[np.float64]) -> Vector3:
        """Validate planner inputs against DbC preconditions."""
        point_array = _as_point(name, point)
        if not self._point_within_bounds(point_array):
            raise ValueError(f"{name} must lie within planner bounds")
        return point_array

    def _sample_point(self, goal: Vector3) -> Vector3:
        """Sample a random point with configurable goal bias."""
        if goal is None:
            raise TypeError("goal must not be None")
        if self._rng.random() < self.goal_bias:
            return goal.copy()

        return np.array(
            [
                self._rng.uniform(self.bounds[0], self.bounds[1]),
                self._rng.uniform(self.bounds[2], self.bounds[3]),
                self._rng.uniform(self.bounds[4], self.bounds[5]),
            ],
            dtype=np.float64,
        )

    def _nearest_node_index(
        self, nodes: list[npt.NDArray[np.float64]], sample: Vector3
    ) -> int:
        """Return the index of the nearest existing tree node.

        Kept for backward compatibility and ad-hoc queries: it rebuilds the
        coordinate array from ``nodes`` on each call. The hot planning loop uses
        :meth:`_nearest_in_coords` against an incrementally maintained array to
        avoid the O(N^2) full-array reconstruction (see issue #3683).
        """
        if nodes is None:
            raise TypeError("nodes must not be None")
        coordinates = np.array([node[:3] for node in nodes], dtype=np.float64)
        return self._nearest_in_coords(coordinates, sample)

    @staticmethod
    def _nearest_in_coords(
        coordinates: npt.NDArray[np.float64],
        sample: npt.NDArray[np.float64],
    ) -> int:
        """Return the index of the row in ``coordinates`` nearest to ``sample``."""
        distances = np.linalg.norm(coordinates - sample, axis=1)
        return int(np.argmin(distances))

    def _steer(self, origin: Vector3, target: Vector3) -> Vector3:
        """Step from one point toward another by at most ``step_size``."""
        if origin is None:
            raise TypeError("origin must not be None")
        direction = target - origin
        distance = float(np.linalg.norm(direction))
        if distance == 0.0:
            return origin.copy()
        return origin + self.step_size * direction / distance

    def _check_collision(
        self, point: npt.NDArray[np.float64], obstacles: list[Obstacle]
    ) -> bool:
        """Return True if a point lies inside any obstacle."""
        if point is None:
            raise TypeError("point must not be None")
        point_array = _as_point("point", point)
        return any(
            distance_to_obstacle_surface(point_array, obstacle) <= 0.0
            for obstacle in obstacles
        )

    def _segment_is_collision_free(
        self,
        start: npt.NDArray[np.float64],
        end: npt.NDArray[np.float64],
        obstacles: list[Obstacle],
    ) -> bool:
        """Sample a segment to ensure shortcutting does not cut through asteroids."""
        if start is None:
            raise TypeError("start must not be None")
        start_point = _as_point("start", start)
        end_point = _as_point("end", end)
        segment_length = float(np.linalg.norm(end_point - start_point))
        samples = max(2, int(math.ceil(segment_length / max(self.step_size / 2, 1e-6))))

        for fraction in np.linspace(0.0, 1.0, samples):
            probe = start_point + fraction * (end_point - start_point)
            if self._check_collision(probe, obstacles):
                return False
        return True

    def plan_path(
        self,
        start: npt.NDArray[np.float64],
        goal: npt.NDArray[np.float64],
        obstacles: list[Obstacle],
    ) -> npt.NDArray[np.float64] | None:
        """Plan a collision-free path using the RRT algorithm."""
        if start is None:
            raise TypeError("start must not be None")
        start_point = self._validate_waypoint("start", start)
        goal_point = self._validate_waypoint("goal", goal)

        if self._check_collision(start_point, obstacles):
            return None
        if self._check_collision(goal_point, obstacles):
            return None

        nodes = [np.append(start_point, -1.0)]

        # Maintain the tree's coordinates in an incrementally grown buffer so the
        # nearest-neighbour query never rebuilds the full coordinate array from
        # ``nodes`` (issue #3683). The buffer doubles in capacity as needed,
        # giving amortized O(1) appends instead of the previous O(N) rebuild that
        # made planning O(N^2) in the number of tree nodes.
        capacity = max(16, self.max_iterations + 1)
        coords: npt.NDArray[np.float64] = np.empty((capacity, 3), dtype=np.float64)
        coords[0] = start_point
        node_count = 1

        for _iteration in range(self.max_iterations):
            sample = self._sample_point(goal_point)
            nearest_idx = self._nearest_in_coords(coords[:node_count], sample)
            nearest_point = np.asarray(nodes[nearest_idx][:3], dtype=np.float64)
            new_point = self._steer(nearest_point, sample)

            if self._check_collision(new_point, obstacles):
                continue
            if not self._segment_is_collision_free(nearest_point, new_point, obstacles):
                continue

            nodes.append(np.append(new_point, float(nearest_idx)))
            if node_count >= coords.shape[0]:
                coords = np.resize(coords, (coords.shape[0] * 2, 3))
            coords[node_count] = new_point
            node_count += 1
            if np.linalg.norm(new_point - goal_point) <= self.goal_radius:
                path = self._extract_path(nodes, len(nodes) - 1)
                self.last_plan_metrics = self.analyze_path(path, obstacles)
                return path

        self.last_plan_metrics = None
        return None

    def _extract_path(
        self, nodes: list[npt.NDArray[np.float64]], goal_idx: int
    ) -> npt.NDArray[np.float64]:
        """Backtrack from a tree node to recover the final route."""
        if nodes is None:
            raise TypeError("nodes must not be None")
        path: list[npt.NDArray[np.float64]] = []
        current_idx = goal_idx

        while current_idx != -1:
            path.append(np.asarray(nodes[current_idx][:3], dtype=np.float64))
            current_idx = int(nodes[current_idx][3])

        return np.array(path[::-1], dtype=np.float64)

    def analyze_path(
        self, path: npt.NDArray[np.float64], obstacles: list[Obstacle]
    ) -> PathMetrics:
        """Compute route metrics for educational and debugging displays."""
        if path is None:
            raise TypeError("path must not be None")
        if len(path) == 0:
            return PathMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

        diffs = np.diff(path, axis=0)
        segment_lengths = (
            np.linalg.norm(diffs, axis=1) if len(path) > 1 else np.array([])
        )
        path_length = float(np.sum(segment_lengths))
        straight_line_distance = (
            float(np.linalg.norm(path[-1] - path[0])) if len(path) > 1 else 0.0
        )
        efficiency = straight_line_distance / path_length if path_length > 0 else 1.0

        if obstacles:
            min_clearance = min(
                distance_to_obstacle_surface(point, obstacle)
                for point in path
                for obstacle in obstacles
            )
        else:
            min_clearance = float("inf")

        turn_angles: list[float] = []
        for idx in range(1, len(path) - 1):
            incoming = path[idx] - path[idx - 1]
            outgoing = path[idx + 1] - path[idx]
            if np.linalg.norm(incoming) == 0 or np.linalg.norm(outgoing) == 0:
                continue

            incoming_unit = incoming / np.linalg.norm(incoming)
            outgoing_unit = outgoing / np.linalg.norm(outgoing)
            cosine = float(np.clip(np.dot(incoming_unit, outgoing_unit), -1.0, 1.0))
            turn_angles.append(math.degrees(math.acos(cosine)))

        mean_turn_angle = float(np.mean(turn_angles)) if turn_angles else 0.0
        return PathMetrics(
            waypoint_count=len(path),
            path_length=path_length,
            straight_line_distance=straight_line_distance,
            efficiency=efficiency,
            min_clearance=min_clearance,
            mean_turn_angle_deg=mean_turn_angle,
        )

    def smooth_path(
        self,
        path: npt.NDArray[np.float64],
        obstacles: list[Obstacle],
        *,
        iterations: int = 64,
    ) -> npt.NDArray[np.float64]:
        """Shortcut a path while preserving endpoints and collision safety."""
        if path is None:
            raise TypeError("path must not be None")
        if len(path) <= 2:
            return np.array(path, dtype=np.float64)

        smoothed = [np.asarray(point, dtype=np.float64) for point in path]
        for _ in range(max(iterations, 0)):
            if len(smoothed) <= 2:
                break

            start_idx = self._rng.randint(0, len(smoothed) - 3)
            end_idx = self._rng.randint(start_idx + 2, len(smoothed) - 1)

            if self._segment_is_collision_free(
                smoothed[start_idx], smoothed[end_idx], obstacles
            ):
                smoothed = smoothed[: start_idx + 1] + smoothed[end_idx:]

        return np.array(smoothed, dtype=np.float64)

    def format_metrics(self, metrics: PathMetrics | None) -> str:
        """Return a compact science-facing summary string for the UI."""
        if metrics is None:
            return "No route yet"

        clearance = (
            "clear"
            if math.isinf(metrics.min_clearance)
            else f"{metrics.min_clearance:.2f}"
        )
        return (
            f"Waypoints {metrics.waypoint_count} | "
            f"Length {metrics.path_length:.2f} | "
            f"Efficiency {metrics.efficiency:.1%} | "
            f"Min clearance {clearance} | "
            f"Mean turn {metrics.mean_turn_angle_deg:.1f} deg"
        )


class PursuitAI:
    """Simple target behavior model for the pursuit scenario."""

    def __init__(
        self, bounds: npt.NDArray[np.float64], *, seed: int | None = None
    ) -> None:
        if bounds is None:
            raise TypeError("bounds must not be None")
        self.bounds = _coerce_bounds(bounds)
        self.evasion_radius = 0.15
        self.capture_radius = 0.05
        self.pursuer_speed = 0.02
        self.target_speed = 0.015
        self._rng = random.Random(seed)

    def update_target_behavior(
        self, target: Ship, pursuer: Ship, obstacles: list[Obstacle]
    ) -> Vector3:
        """Move the target away from danger while staying inside the bounds."""
        if target is None:
            raise TypeError("target must not be None")
        del obstacles
        distance = float(np.linalg.norm(target.position - pursuer.position))

        if distance < self.evasion_radius:
            direction = target.position - pursuer.position
        else:
            direction = self._generate_random_goal() - target.position

        if np.linalg.norm(direction) == 0:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        direction = direction / np.linalg.norm(direction)
        new_pos = target.position + self.target_speed * direction
        return np.clip(
            new_pos,
            [self.bounds[0], self.bounds[2], self.bounds[4]],
            [self.bounds[1], self.bounds[3], self.bounds[5]],
        )

    def _generate_random_goal(self) -> Vector3:
        """Generate a new point inside the navigation volume."""
        return np.array(
            [
                self._rng.uniform(self.bounds[0], self.bounds[1]),
                self._rng.uniform(self.bounds[2], self.bounds[3]),
                self._rng.uniform(self.bounds[4], self.bounds[5]),
            ],
            dtype=np.float64,
        )


class StarWarsRenderer:
    """Optional OpenGL renderer for the asteroid navigator."""

    def __init__(self, width: int = 1600, height: int = 900) -> None:
        """Initialize the renderer and fail gracefully if dependencies are missing."""
        if not isinstance(width, int) or width <= 0:
            raise TypeError("width must be a positive int")
        if not PYGAME_AVAILABLE or not OPENGL_AVAILABLE:
            raise RuntimeError(
                "Visualization requires pygame and PyOpenGL. "
                "Install them with: pip install pygame PyOpenGL PyOpenGL_accelerate"
            )

        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("RRT Asteroid Navigator")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)

        self.camera_pos = np.array([0.0, -2.0, 1.0], dtype=np.float64)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (width / height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

        self.stars, self.star_brightness = self._generate_starfield(1200)

    def _generate_starfield(
        self, num_stars: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate a stable starfield shell around the scene."""
        if not isinstance(num_stars, int) or num_stars <= 0:
            raise TypeError("num_stars must be a positive int")
        rng = np.random.default_rng(42)
        directions = rng.standard_normal((num_stars, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.where(norms == 0, 1, norms)
        radii = rng.uniform(8.0, 15.0, (num_stars, 1))
        brightness = rng.uniform(0.45, 1.0, num_stars)
        return directions * radii, brightness

    def render_frame(
        self,
        ships: list[Ship],
        obstacles: list[Obstacle],
        paths: list[npt.NDArray[np.float64]],
        *,
        camera_mode: str = "cinematic",
    ) -> None:
        """Render a frame using the requested camera mode."""
        if ships is None:
            raise TypeError("ships must not be None")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self._update_camera(camera_mode, ships)
        self._render_starfield()

        for obstacle in obstacles:
            self._render_obstacle(obstacle)
        for ship in ships:
            self._render_ship(ship)
        for path in paths:
            if len(path) > 1:
                self._render_path(path)

        pygame.display.flip()

    def _update_camera(self, mode: str, ships: list[Ship]) -> None:
        """Switch among a few simple cinematic camera presets."""
        if not isinstance(mode, str):
            raise TypeError("mode must be a str")
        if mode == "top_down":
            gluLookAt(0.0, 0.0, 2.8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        elif mode == "chase" and ships:
            ship_pos = ships[0].position
            camera_pos = ship_pos + np.array([-1.6, -0.2, 0.8], dtype=np.float64)
            gluLookAt(
                camera_pos[0],
                camera_pos[1],
                camera_pos[2],
                ship_pos[0],
                ship_pos[1],
                ship_pos[2],
                0,
                0,
                1,
            )
        else:
            gluLookAt(
                self.camera_pos[0],
                self.camera_pos[1],
                self.camera_pos[2],
                self.camera_target[0],
                self.camera_target[1],
                self.camera_target[2],
                self.camera_up[0],
                self.camera_up[1],
                self.camera_up[2],
            )

        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, -2.0, 2.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.35, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])

    def _render_starfield(self) -> None:
        """Draw the ambient starfield."""
        glDisable(GL_LIGHTING)
        glPointSize(2.0)
        glBegin(GL_POINTS)
        for star, brightness in zip(self.stars, self.star_brightness, strict=False):
            glColor3f(brightness, brightness, brightness)
            glVertex3f(star[0], star[1], star[2])  # type: ignore[index]
        glEnd()
        glEnable(GL_LIGHTING)

    def _render_obstacle(self, obstacle: Obstacle) -> None:
        """Render a sphere or cube obstacle."""
        if obstacle is None:
            raise TypeError("obstacle must not be None")
        glPushMatrix()
        glTranslatef(obstacle.position[0], obstacle.position[1], obstacle.position[2])
        glColor3f(
            min(obstacle.color[0] + 0.15, 1.0),
            min(obstacle.color[1] + 0.15, 1.0),
            min(obstacle.color[2] + 0.15, 1.0),
        )

        if obstacle.type == 0:
            quad = gluNewQuadric()
            gluSphere(quad, obstacle.size, 16, 16)
            gluDeleteQuadric(quad)
        else:
            size = obstacle.size / 2
            glBegin(GL_QUADS)
            glNormal3f(0, 0, 1)
            glVertex3f(-size, -size, size)
            glVertex3f(size, -size, size)
            glVertex3f(size, size, size)
            glVertex3f(-size, size, size)
            glNormal3f(0, 0, -1)
            glVertex3f(-size, -size, -size)
            glVertex3f(-size, size, -size)
            glVertex3f(size, size, -size)
            glVertex3f(size, -size, -size)
            glNormal3f(0, 1, 0)
            glVertex3f(-size, size, -size)
            glVertex3f(-size, size, size)
            glVertex3f(size, size, size)
            glVertex3f(size, size, -size)
            glNormal3f(0, -1, 0)
            glVertex3f(-size, -size, -size)
            glVertex3f(size, -size, -size)
            glVertex3f(size, -size, size)
            glVertex3f(-size, -size, size)
            glNormal3f(1, 0, 0)
            glVertex3f(size, -size, -size)
            glVertex3f(size, size, -size)
            glVertex3f(size, size, size)
            glVertex3f(size, -size, size)
            glNormal3f(-1, 0, 0)
            glVertex3f(-size, -size, -size)
            glVertex3f(-size, -size, size)
            glVertex3f(-size, size, size)
            glVertex3f(-size, size, -size)
            glEnd()

        glPopMatrix()

    def _render_ship(self, ship: Ship) -> None:
        """Render a simple arrowhead spacecraft."""
        if ship is None:
            raise TypeError("ship must not be None")
        glPushMatrix()
        glTranslatef(ship.position[0], ship.position[1], ship.position[2])
        glColor3f(*ship.color)
        glDisable(GL_LIGHTING)
        glBegin(GL_TRIANGLES)
        size = 0.08
        glVertex3f(-size, 0.0, 0.01)
        glVertex3f(size, -size / 2, 0.0)
        glVertex3f(size, size / 2, 0.0)
        glVertex3f(-size, 0.0, -0.01)
        glVertex3f(size, size / 2, 0.0)
        glVertex3f(size, -size / 2, 0.0)
        glEnd()
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def _render_path(self, path: npt.NDArray[np.float64]) -> None:
        """Render the active path and waypoint markers."""
        if path is None:
            raise TypeError("path must not be None")
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.9, 0.2)
        glLineWidth(3.0)
        glBegin(GL_LINE_STRIP)
        for point in path:
            glVertex3f(point[0], point[1], point[2])
        glEnd()

        glPointSize(5.0)
        glBegin(GL_POINTS)
        for point in path:
            glVertex3f(point[0], point[1], point[2])
        glEnd()
        glEnable(GL_LIGHTING)


class StarWarsRRTApp:
    """Application wrapper that ties planner, AI, and rendering together."""

    def __init__(self, *, width: int = 1600, height: int = 900, seed: int = 7) -> None:
        self.bounds = np.array([-1.0, 1.0, -0.6, 0.6, -0.3, 0.3], dtype=np.float64)
        self.seed = seed
        self.planner = RRTPlanner(self.bounds, seed=seed)
        self.pursuit_ai = PursuitAI(self.bounds, seed=seed + 1)
        self.renderer = StarWarsRenderer(width=width, height=height)

        self.ships: list[Ship] = []
        self.obstacles: list[Obstacle] = []
        self.paths: list[npt.NDArray[np.float64]] = []
        self.mode = "single"
        self.camera_modes = ["cinematic", "chase", "top_down"]
        self.camera_mode_index = 0
        self.running = True
        self.clock = pygame.time.Clock()
        self.ship_models = self._load_ship_models()

    def _load_ship_models(self) -> dict[str, Any]:
        """Load optional STL ship models when the dependency is available."""
        if not TRIMESH_AVAILABLE:
            return {}

        models: dict[str, Any] = {}
        model_path = Path(__file__).with_name("falcon_clean_fixed.stl")
        if not model_path.exists():
            return models

        try:
            models["falcon"] = trimesh.load(model_path)
            logging.info("Loaded ship model from %s", model_path)
        except (
            Exception
        ) as exc:  # noqa: BLE001  # pragma: no cover - visualization-only fallback
            logging.warning("Could not load STL model %s: %s", model_path, exc)
        return models

    def _cycle_camera_mode(self) -> None:
        """Cycle through the available camera modes."""
        self.camera_mode_index = (self.camera_mode_index + 1) % len(self.camera_modes)
        self._update_window_caption(self.planner.last_plan_metrics)

    @property
    def camera_mode(self) -> str:
        """Return the currently active camera mode."""
        return self.camera_modes[self.camera_mode_index]

    def setup_scenario(self, mode: str = "single") -> None:
        """Set up either the single-route or pursuit scenario."""
        if not isinstance(mode, str):
            raise TypeError("mode must be a str")
        self.mode = mode
        if mode == "single":
            self._setup_single_navigation()
        else:
            self._setup_pursuit_scenario()

    def _setup_single_navigation(self) -> None:
        """Prepare a deterministic asteroid-field navigation challenge."""
        start = np.array([-0.8, 0.0, 0.0], dtype=np.float64)
        goal = np.array([0.8, 0.0, 0.0], dtype=np.float64)
        self.obstacles = generate_asteroid_field(
            self.bounds,
            30,
            rng=np.random.default_rng(self.seed),
            reserved_points=[start, goal],
            clearance=0.12,
        )
        ship = Ship(position=start.copy(), orientation=np.eye(3), velocity=np.zeros(3))
        if "falcon" in self.ship_models:
            ship.model = self.ship_models["falcon"]
        self.ships = [ship]

        raw_path = self.planner.plan_path(start, goal, self.obstacles)
        if raw_path is None:
            self.paths = []
            logging.warning("No path found through the asteroid field")
            self._update_window_caption(None)
            return

        smooth_path = self.planner.smooth_path(raw_path, self.obstacles, iterations=96)
        metrics = self.planner.analyze_path(smooth_path, self.obstacles)
        self.planner.last_plan_metrics = metrics
        self.paths = [smooth_path]
        logging.info("Route summary: %s", self.planner.format_metrics(metrics))
        self._update_window_caption(metrics)

    def _setup_pursuit_scenario(self) -> None:
        """Prepare a two-ship chase with protected spawn points."""
        pursuer_start = np.array([-0.8, -0.3, 0.0], dtype=np.float64)
        target_start = np.array([-0.8, 0.3, 0.0], dtype=np.float64)
        self.obstacles = generate_asteroid_field(
            self.bounds,
            24,
            rng=np.random.default_rng(self.seed + 99),
            reserved_points=[pursuer_start, target_start],
            clearance=0.10,
        )
        pursuer = Ship(
            pursuer_start.copy(), np.eye(3), np.zeros(3), color=(0.85, 0.85, 0.85)
        )
        target = Ship(
            target_start.copy(), np.eye(3), np.zeros(3), color=(0.65, 0.8, 1.0)
        )
        self.ships = [pursuer, target]
        self.paths = []
        self._update_window_caption(None)

    def _update_window_caption(self, metrics: PathMetrics | None) -> None:
        """Push a compact science summary into the window title bar."""
        if not PYGAME_AVAILABLE:
            return

        caption = (
            f"RRT Asteroid Navigator | Mode: {self.mode} | Camera: {self.camera_mode}"
        )
        if metrics is not None:
            caption += f" | {self.planner.format_metrics(metrics)}"
        pygame.display.set_caption(caption)

    def run(self) -> None:
        """Run the interactive event loop."""
        logging.info("Starting RRT Asteroid Navigator")
        logging.info("SPACE toggles single/pursuit mode")
        logging.info("C cycles camera modes")
        logging.info("ESC quits")

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_SPACE:
                        next_mode = "pursuit" if self.mode == "single" else "single"
                        self.setup_scenario(next_mode)
                    elif event.key == K_c:
                        self._cycle_camera_mode()

            if self.mode == "pursuit":
                self._update_pursuit()

            self.renderer.render_frame(
                self.ships,
                self.obstacles,
                self.paths,
                camera_mode=self.camera_mode,
            )
            self.clock.tick(60)

        pygame.quit()

    def _update_pursuit(self) -> None:
        """Advance the simple pursuit simulation."""
        if len(self.ships) < 2:
            return

        pursuer, target = self.ships[0], self.ships[1]
        target.position = self.pursuit_ai.update_target_behavior(
            target, pursuer, self.obstacles
        )

        direction = target.position - pursuer.position
        distance = float(np.linalg.norm(direction))
        if distance > 0:
            direction = direction / distance
            pursuer.position += self.pursuit_ai.pursuer_speed * direction

        if distance < self.pursuit_ai.capture_radius:
            logging.info("Target captured")
            target.color = (1.0, 0.2, 0.2)


def main() -> int:
    """Run the asteroid navigator application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        app = StarWarsRRTApp()
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    app.setup_scenario("single")
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

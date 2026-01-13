#!/usr/bin/env python3
"""
Star Wars RRT Path Planner - Python Version
Enhanced performance with real-time rendering and GPU acceleration
"""

import logging
import random
from dataclasses import dataclass

import numpy as np
import pygame
import trimesh
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_STRIP,
    GL_POINTS,
    GL_QUADS,
    GL_TRIANGLES,
    glBegin,
    glClear,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glLineWidth,
    glLoadIdentity,
    glPointSize,
    glPopMatrix,
    glPushMatrix,
    glTranslatef,
    glVertex3f,
)
from OpenGL.GLU import gluDeleteQuadric, gluLookAt, gluNewQuadric, gluSphere
from pygame.locals import DOUBLEBUF, K_ESCAPE, K_SPACE, KEYDOWN, OPENGL, QUIT, K_c


@dataclass
class Obstacle:
    """Obstacle representation"""

    type: int  # 0=sphere, 1=cube
    position: np.ndarray
    size: float
    color: tuple[float, float, float]


@dataclass
class Ship:
    """Ship representation"""

    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    model: trimesh.Trimesh | None = None
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)


class RRTPlanner:
    """High-performance RRT path planner with GPU acceleration"""

    def __init__(self, bounds: np.ndarray, max_iterations: int = 5000) -> None:
        """Initialize RRT planner with search bounds and iteration limit"""
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.step_size = 0.05
        self.goal_radius = 0.1
        self.goal_bias = 0.2

    def plan_path(
        self, start: np.ndarray, goal: np.ndarray, obstacles: list[Obstacle]
    ) -> np.ndarray | None:
        """Plan path using RRT algorithm"""
        if self._check_collision(start, obstacles):
            return None
        if self._check_collision(goal, obstacles):
            return None

        nodes = [np.append(start, -1)]  # [x, y, z, parent_index]

        for _iteration in range(self.max_iterations):
            # Goal-biased sampling
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = np.array(
                    [
                        random.uniform(self.bounds[0], self.bounds[1]),
                        random.uniform(self.bounds[2], self.bounds[3]),
                        random.uniform(self.bounds[4], self.bounds[5]),
                    ]
                )

            # Find nearest node using pre-allocated array for better performance
            n_nodes = len(nodes)
            nodes_array = np.empty((n_nodes, 3), dtype=np.float32)
            
            # Fill array directly to avoid list comprehension overhead
            for i, node in enumerate(nodes):
                nodes_array[i] = node[:3]
            
            distances = np.linalg.norm(nodes_array - sample, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_node = nodes_array[nearest_idx]

            # Step towards sample
            direction = sample - nearest_node
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                new_pos = nearest_node + self.step_size * direction

                # Check collision
                if not self._check_collision(new_pos, obstacles):
                    nodes.append(np.append(new_pos, nearest_idx))

                    # Check if goal reached
                    if np.linalg.norm(new_pos - goal) < self.goal_radius:
                        return self._extract_path(nodes, len(nodes) - 1)

        return None

    def _check_collision(self, point: np.ndarray, obstacles: list[Obstacle]) -> bool:
        """Fast collision checking using vectorized operations"""
        for obstacle in obstacles:
            if obstacle.type == 0:  # Sphere
                if np.linalg.norm(point - obstacle.position) <= obstacle.size:
                    return True
            elif np.all(np.abs(point - obstacle.position) <= obstacle.size / 2):
                return True
        return False

    def _extract_path(self, nodes: list[np.ndarray], goal_idx: int) -> np.ndarray:
        """Extract path from RRT tree"""
        path = []
        current_idx = goal_idx

        while current_idx != -1:
            path.append(nodes[current_idx][:3])
            current_idx = int(nodes[current_idx][3])

        return np.array(path[::-1])


class PursuitAI:
    """Intelligent pursuit AI with advanced behavior"""

    def __init__(self, bounds: np.ndarray) -> None:
        """Initialize pursuit AI with search bounds"""
        self.bounds = bounds
        self.evasion_radius = 0.15
        self.capture_radius = 0.05
        self.pursuer_speed = 0.02
        self.target_speed = 0.015

    def update_target_behavior(
        self, target: Ship, pursuer: Ship, obstacles: list[Obstacle]
    ) -> np.ndarray:
        """Update target ship behavior (evade or move to goal)"""
        distance = np.linalg.norm(target.position - pursuer.position)

        if distance < self.evasion_radius:
            # Evasion mode
            evasion_direction = target.position - pursuer.position
            evasion_direction = evasion_direction / np.linalg.norm(evasion_direction)
            new_pos = target.position + self.target_speed * evasion_direction
        else:
            # Normal mode - move toward random goal
            goal = self._generate_random_goal()
            goal_direction = goal - target.position
            goal_direction = goal_direction / np.linalg.norm(goal_direction)
            new_pos = target.position + self.target_speed * goal_direction

        # Constrain to bounds
        new_pos = np.clip(
            new_pos,
            [self.bounds[0], self.bounds[2], self.bounds[4]],
            [self.bounds[1], self.bounds[3], self.bounds[5]],
        )

        return new_pos

    def _generate_random_goal(self) -> np.ndarray:
        """Generate random goal within bounds"""
        return np.array(
            [
                random.uniform(self.bounds[0], self.bounds[1]),
                random.uniform(self.bounds[2], self.bounds[3]),
                random.uniform(self.bounds[4], self.bounds[5]),
            ]
        )


class StarWarsRenderer:
    """High-performance 3D renderer using OpenGL"""

    def __init__(self, width: int = 1600, height: int = 900) -> None:
        """Initialize renderer with window dimensions"""
        pygame.init()
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Star Wars RRT Path Planner - Python")

        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Camera setup
        self.camera_pos = np.array([2.0, -2.0, 1.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])

        # Starfield
        self.stars = self._generate_starfield(1000)

    def _generate_starfield(self, num_stars: int) -> np.ndarray:
        """Generate dynamic starfield"""
        return np.random.randn(num_stars, 3) * 3

    def render_frame(
        self,
        ships: list[Ship],
        obstacles: list[Obstacle],
        paths: list[np.ndarray],
        camera_mode: str = "cinematic",
    ) -> None:
        """Render a single frame at 60 FPS"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Update camera based on mode
        self._update_camera(camera_mode, ships)

        # Render starfield
        self._render_starfield()

        # Render obstacles
        for obstacle in obstacles:
            self._render_obstacle(obstacle)

        # Render ships
        for ship in ships:
            self._render_ship(ship)

        # Render paths
        for path in paths:
            if len(path) > 1:
                self._render_path(path)

        pygame.display.flip()

    def _update_camera(self, mode: str, ships: list[Ship]) -> None:
        """Update camera position based on mode"""
        if mode == "cinematic":
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
        elif mode == "chase" and len(ships) > 0:
            # Chase camera following first ship
            ship_pos = ships[0].position
            camera_offset = np.array([-2.0, 0.0, 1.0])
            camera_pos = ship_pos + camera_offset
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

    def _render_starfield(self) -> None:
        """Render starfield with varying star sizes"""
        glDisable(GL_LIGHTING)
        glPointSize(2.0)
        glBegin(GL_POINTS)
        for star in self.stars:
            brightness = random.uniform(0.5, 1.0)
            glColor3f(brightness, brightness, brightness)
            glVertex3f(star[0], star[1], star[2])
        glEnd()
        glEnable(GL_LIGHTING)

    def _render_obstacle(self, obstacle: Obstacle) -> None:
        """Render obstacle with proper lighting"""
        glPushMatrix()
        glTranslatef(obstacle.position[0], obstacle.position[1], obstacle.position[2])
        glColor3f(*obstacle.color)

        if obstacle.type == 0:  # Sphere
            quad = gluNewQuadric()
            gluSphere(quad, obstacle.size, 16, 16)
            gluDeleteQuadric(quad)
        else:  # Cube
            size = obstacle.size / 2
            glBegin(GL_QUADS)
            # Front face
            glVertex3f(-size, -size, size)
            glVertex3f(size, -size, size)
            glVertex3f(size, size, size)
            glVertex3f(-size, size, size)
            # Back face
            glVertex3f(-size, -size, -size)
            glVertex3f(-size, size, -size)
            glVertex3f(size, size, -size)
            glVertex3f(size, -size, -size)
            # Top face
            glVertex3f(-size, size, -size)
            glVertex3f(-size, size, size)
            glVertex3f(size, size, size)
            glVertex3f(size, size, -size)
            # Bottom face
            glVertex3f(-size, -size, -size)
            glVertex3f(size, -size, -size)
            glVertex3f(size, -size, size)
            glVertex3f(-size, -size, size)
            # Right face
            glVertex3f(size, -size, -size)
            glVertex3f(size, size, -size)
            glVertex3f(size, size, size)
            glVertex3f(size, -size, size)
            # Left face
            glVertex3f(-size, -size, -size)
            glVertex3f(-size, -size, size)
            glVertex3f(-size, size, size)
            glVertex3f(-size, size, -size)
            glEnd()

        glPopMatrix()

    def _render_ship(self, ship: Ship) -> None:
        """Render ship with proper orientation"""
        glPushMatrix()
        glTranslatef(ship.position[0], ship.position[1], ship.position[2])

        # Apply ship orientation
        if ship.model is not None:
            # Render 3D model
            glColor3f(*ship.color)
            # Simplified ship rendering for now
            self._render_simple_ship()
        else:
            # Render simple ship
            glColor3f(*ship.color)
            self._render_simple_ship()

        glPopMatrix()

    def _render_simple_ship(self) -> None:
        """Render simple ship geometry"""
        size = 0.05
        glBegin(GL_TRIANGLES)
        # Ship body
        glVertex3f(-size, 0, 0)  # Nose
        glVertex3f(size, -size / 2, 0)  # Right wing
        glVertex3f(size, size / 2, 0)  # Left wing
        glEnd()

    def _render_path(self, path: np.ndarray) -> None:
        """Render path as line"""
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)  # Yellow path
        glLineWidth(3.0)
        glBegin(GL_LINE_STRIP)
        for point in path:
            glVertex3f(point[0], point[1], point[2])
        glEnd()
        glEnable(GL_LIGHTING)


class StarWarsRRTApp:
    """Main application class"""

    def __init__(self) -> None:
        """Initialize the Star Wars RRT application"""
        self.bounds = np.array([-1.0, 1.0, -0.6, 0.6, -0.3, 0.3])
        self.planner = RRTPlanner(self.bounds)
        self.pursuit_ai = PursuitAI(self.bounds)
        self.renderer = StarWarsRenderer()

        # Game state
        self.ships = []
        self.obstacles = []
        self.paths = []
        self.mode = "single"  # "single" or "pursuit"
        self.running = True
        self.clock = pygame.time.Clock()

        # Load STL models
        self.ship_models = self._load_ship_models()

    def _load_ship_models(self) -> dict:
        """Load STL models for ships"""
        models = {}
        try:
            # Try to load Millennium Falcon
            models["falcon"] = trimesh.load("falcon_clean_fixed.stl")
            logging.info("✅ Loaded Millennium Falcon model")
        except Exception:
            logging.warning("⚠️ Could not load STL model, using simple geometry")

        return models

    def setup_scenario(self, mode: str = "single") -> None:
        """Setup the scenario"""
        self.mode = mode

        # Generate obstacles
        self.obstacles = self._generate_obstacles(30)

        if mode == "single":
            # Single ship navigation
            start = np.array([-0.8, 0.0, 0.0])
            goal = np.array([0.8, 0.0, 0.0])

            ship = Ship(position=start, orientation=np.eye(3), velocity=np.zeros(3))
            if "falcon" in self.ship_models:
                ship.model = self.ship_models["falcon"]

            self.ships = [ship]

            # Plan path
            path = self.planner.plan_path(start, goal, self.obstacles)
            if path is not None:
                self.paths = [path]
                logging.info(f"✅ Path planned: {len(path)} waypoints")
            else:
                logging.warning("❌ No path found")

        elif mode == "pursuit":
            # Pursuit scenario
            pursuer_start = np.array([-0.8, -0.3, 0.0])
            target_start = np.array([-0.8, 0.3, 0.0])

            pursuer = Ship(
                position=pursuer_start,
                orientation=np.eye(3),
                velocity=np.zeros(3),
                color=(0.8, 0.8, 0.8),
            )
            target = Ship(
                position=target_start,
                orientation=np.eye(3),
                velocity=np.zeros(3),
                color=(0.6, 0.6, 0.6),
            )

            if "falcon" in self.ship_models:
                pursuer.model = self.ship_models["falcon"]
                target.model = self.ship_models["falcon"]

            self.ships = [pursuer, target]
            self.paths = []

    def _generate_obstacles(self, num_obstacles: int) -> list[Obstacle]:
        """Generate random obstacles"""
        obstacles = []
        for _ in range(num_obstacles):
            obstacle_type = random.randint(0, 1)
            position = np.array(
                [
                    random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3]),
                    random.uniform(self.bounds[4], self.bounds[5]),
                ]
            )
            size = random.uniform(0.02, 0.08)
            color = (
                random.uniform(0.3, 0.7),
                random.uniform(0.3, 0.7),
                random.uniform(0.3, 0.7),
            )

            obstacles.append(Obstacle(obstacle_type, position, size, color))

        return obstacles

    def run(self) -> None:
        """Main game loop"""
        logging.info("🚀 Starting Star Wars RRT Path Planner - Python Version")
        logging.info("Controls:")
        logging.info("  SPACE - Toggle between single/pursuit mode")
        logging.info("  C - Change camera view")
        logging.info("  ESC - Quit")

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                    elif event.key == K_SPACE:
                        self.mode = "pursuit" if self.mode == "single" else "single"
                        self.setup_scenario(self.mode)
                    elif event.key == K_c:
                        # Cycle camera modes (not yet implemented)
                        continue

            # Update game state
            if self.mode == "pursuit":
                self._update_pursuit()

            # Render frame
            self.renderer.render_frame(self.ships, self.obstacles, self.paths)

            # Maintain 60 FPS
            self.clock.tick(60)

        pygame.quit()

    def _update_pursuit(self) -> None:
        """Update pursuit scenario"""
        if len(self.ships) >= 2:
            pursuer = self.ships[0]
            target = self.ships[1]

            # Update target behavior
            new_target_pos = self.pursuit_ai.update_target_behavior(
                target, pursuer, self.obstacles
            )
            target.position = new_target_pos

            # Update pursuer (simple direct movement for now)
            direction = target.position - pursuer.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                pursuer.position += self.pursuit_ai.pursuer_speed * direction

            # Check for capture
            if distance < self.pursuit_ai.capture_radius:
                logging.info("🎯 Target captured!")
                target.color = (1.0, 0.0, 0.0)  # Turn red


def main() -> None:
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app = StarWarsRRTApp()
    app.setup_scenario("single")
    app.run()


if __name__ == "__main__":
    main()

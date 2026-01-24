import os
import sys
import unittest

import numpy as np

# Use shared path utility
try:
    from utils.path_helpers import ensure_utils_in_path

    ensure_utils_in_path()
except ImportError:
    # Fallback
    sys.path.append(os.path.abspath(Path(Path(__file__).parent, "../src")))

from star_wars_rrt import Obstacle, PursuitAI, RRTPlanner, Ship


class TestRRTPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = np.array([-10, 10, -10, 10, -10, 10])
        self.planner = RRTPlanner(self.bounds, max_iterations=2000)
        # Increase goal radius for test stability
        self.planner.goal_radius = 0.5

    def test_plan_path_no_obstacles(self) -> None:
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0, 0.0])
        obstacles: list[Obstacle] = []
        path = self.planner.plan_path(start, goal, obstacles)
        self.assertIsNotNone(path)
        if path is not None:
            # Start and end should be close to requested
            self.assertTrue(np.allclose(path[0], start, atol=1e-5))
            # Last point should be within goal_radius of goal
            self.assertTrue(np.linalg.norm(path[-1] - goal) < self.planner.goal_radius)

    def test_collision_sphere(self) -> None:
        obstacle = Obstacle(
            type=0, position=np.array([0.5, 0, 0]), size=0.2, color=(1, 1, 1)
        )
        # Point inside
        self.assertTrue(
            self.planner._check_collision(np.array([0.5, 0.1, 0]), [obstacle])
        )
        # Point outside
        self.assertFalse(
            self.planner._check_collision(np.array([0.0, 0.0, 0]), [obstacle])
        )

    def test_collision_cube(self) -> None:
        obstacle = Obstacle(
            type=1, position=np.array([0.5, 0, 0]), size=0.2, color=(1, 1, 1)
        )
        # Box extends from 0.4 to 0.6 in x, -0.1 to 0.1 in y/z
        # Point inside
        self.assertTrue(
            self.planner._check_collision(np.array([0.5, 0.05, 0.05]), [obstacle])
        )
        # Point outside
        self.assertFalse(
            self.planner._check_collision(np.array([0.65, 0, 0]), [obstacle])
        )

    def test_plan_path_collision_start(self) -> None:
        obstacle = Obstacle(
            type=0, position=np.array([0.0, 0, 0]), size=0.2, color=(1, 1, 1)
        )
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0, 0.0])
        path = self.planner.plan_path(start, goal, [obstacle])
        self.assertIsNone(path)


class TestPursuitAI(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = np.array([-10, 10, -10, 10, -10, 10])
        self.ai = PursuitAI(self.bounds)

    def test_evasion(self) -> None:
        target = Ship(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.eye(3),
            velocity=np.zeros(3),
        )
        pursuer = Ship(
            position=np.array([0.1, 0.0, 0.0]),
            orientation=np.eye(3),
            velocity=np.zeros(3),
        )
        # Within evasion radius (0.15)
        new_pos = self.ai.update_target_behavior(target, pursuer, [])

        # Should move away from pursuer (negative x)
        direction = new_pos - target.position
        # target at 0, pursuer at 0.1 (x). Evasion should be towards -x.
        self.assertTrue(direction[0] < 0)


if __name__ == "__main__":
    unittest.main()

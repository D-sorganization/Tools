"""
Comprehensive Tests for RRT Path Planner with DbC Principles.
"""

import unittest
import numpy as np
from star_wars_rrt import Obstacle, PursuitAI, RRTPlanner, Ship


class TestRRTPlannerDbC(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        self.planner = RRTPlanner(self.bounds, max_iterations=2000)
        self.planner.goal_radius = 0.1

    def test_initialization_contracts(self) -> None:
        """Verify initialization preconditions."""
        # Valid bounds
        self.assertEqual(len(self.planner.bounds), 6)
        self.assertTrue(
            all(self.bounds[i] < self.bounds[i + 1] for i in range(0, 6, 2))
        )

        # Invalid bounds (precondition check)
        invalid_bounds = np.array([1.0, -1.0, 0, 1.0, 0, 1.0])
        with self.assertRaises(ValueError):
            RRTPlanner(invalid_bounds)

    def test_plan_path_postconditions(self) -> None:
        """Verify that successful paths meet all postconditions."""
        start = np.array([-0.8, -0.8, 0.0])
        goal = np.array([0.8, 0.8, 0.0])
        obstacles = [Obstacle(0, np.array([0, 0, 0]), 0.2, (1, 1, 1))]

        path = self.planner.plan_path(start, goal, obstacles)

        if path is not None:
            # Postcondition: Path starts at start
            self.assertTrue(np.allclose(path[0], start))
            # Postcondition: Path ends near goal
            self.assertTrue(np.linalg.norm(path[-1] - goal) <= self.planner.goal_radius)
            # Postcondition: No point in path collides
            for point in path:
                self.assertFalse(self.planner._check_collision(point, obstacles))
            # Postcondition: Path segments are small (<= step_size)
            for i in range(len(path) - 1):
                dist = np.linalg.norm(path[i + 1] - path[i])
                self.assertLessEqual(dist, self.planner.step_size + 1e-5)

    def test_collision_logic_completeness(self) -> None:
        """Exhaustive test of collision detection logic."""
        # Sphere collision
        sphere = Obstacle(0, np.array([0, 0, 0]), 0.5, (1, 0, 0))
        self.assertTrue(
            self.planner._check_collision(np.array([0.1, 0.1, 0.1]), [sphere])
        )
        self.assertFalse(
            self.planner._check_collision(np.array([0.6, 0.6, 0.6]), [sphere])
        )

        # Cube collision
        cube = Obstacle(1, np.array([0, 0, 0]), 0.5, (0, 1, 0))
        # Inside
        self.assertTrue(
            self.planner._check_collision(np.array([0.2, 0.2, 0.2]), [cube])
        )
        # Outside (just barely)
        self.assertFalse(self.planner._check_collision(np.array([0.3, 0, 0]), [cube]))

    def test_edge_cases(self) -> None:
        """Test planner resilience to edge cases."""
        start = np.array([-0.8, 0.0, 0.0])
        goal = np.array([0.8, 0.0, 0.0])

        # 1. No obstacles
        path = self.planner.plan_path(start, goal, [])
        self.assertIsNotNone(path)

        # 2. Start inside obstacle
        bad_obs = [Obstacle(0, start, 0.5, (1, 1, 1))]
        self.assertIsNone(self.planner.plan_path(start, goal, bad_obs))

        # 3. Goal inside obstacle
        bad_obs_goal = [Obstacle(0, goal, 0.5, (1, 1, 1))]
        self.assertIsNone(self.planner.plan_path(start, goal, bad_obs_goal))

    def test_pursuit_ai_invariants(self) -> None:
        """Verify PursuitAI behavior stays within bounds."""
        ai = PursuitAI(self.bounds)
        ship = Ship(np.array([0, 0, 0]), np.eye(3), np.zeros(3))
        pursuer = Ship(np.array([2, 0, 0]), np.eye(3), np.zeros(3))

        for _ in range(100):
            new_pos = ai.update_target_behavior(ship, pursuer, [])
            # Invariant: position must be within bounds
            self.assertTrue(self.bounds[0] <= new_pos[0] <= self.bounds[1])
            self.assertTrue(self.bounds[2] <= new_pos[1] <= self.bounds[3])
            self.assertTrue(self.bounds[4] <= new_pos[2] <= self.bounds[5])


if __name__ == "__main__":
    unittest.main()

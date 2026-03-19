"""
Comprehensive Tests for RRT Path Planner with DbC Principles.
"""

import unittest

import numpy as np
from star_wars_rrt import (
    Obstacle,
    PursuitAI,
    RRTPlanner,
    Ship,
    distance_to_obstacle_surface,
    generate_asteroid_field,
)


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

    def test_path_metrics_capture_efficiency(self) -> None:
        """Path analysis should quantify route quality for educational overlays."""
        path = np.array(
            [
                [-0.8, -0.8, 0.0],
                [-0.2, -0.2, 0.0],
                [0.2, 0.2, 0.0],
                [0.8, 0.8, 0.0],
            ]
        )

        metrics = self.planner.analyze_path(path, [])

        self.assertEqual(metrics.waypoint_count, 4)
        self.assertGreater(metrics.path_length, 0.0)
        self.assertAlmostEqual(metrics.efficiency, 1.0, delta=0.05)
        self.assertGreaterEqual(metrics.min_clearance, 0.0)

    def test_smoothing_preserves_endpoints(self) -> None:
        """Path smoothing should keep the mission endpoints intact."""
        path = np.array(
            [
                [-0.8, -0.8, 0.0],
                [-0.2, -0.5, 0.0],
                [0.2, 0.5, 0.0],
                [0.8, 0.8, 0.0],
            ]
        )

        smoothed = self.planner.smooth_path(path, [], iterations=32)

        self.assertTrue(np.allclose(smoothed[0], path[0]))
        self.assertTrue(np.allclose(smoothed[-1], path[-1]))
        self.assertLessEqual(len(smoothed), len(path))

    def test_generated_asteroid_field_respects_reserved_points(self) -> None:
        """Obstacle generation should leave launch and destination corridors open."""
        start = np.array([-0.8, 0.0, 0.0])
        goal = np.array([0.8, 0.0, 0.0])

        obstacles = generate_asteroid_field(
            self.bounds,
            25,
            rng=np.random.default_rng(7),
            reserved_points=[start, goal],
            clearance=0.12,
        )

        for obstacle in obstacles:
            self.assertGreaterEqual(distance_to_obstacle_surface(start, obstacle), 0.12)
            self.assertGreaterEqual(distance_to_obstacle_surface(goal, obstacle), 0.12)


class TestDbCNoneInputs(unittest.TestCase):
    """Verify that all DbC-guarded functions raise TypeError on None inputs."""

    def setUp(self) -> None:
        self.bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        self.planner = RRTPlanner(self.bounds)
        self.obstacle = Obstacle(0, np.array([0.0, 0.0, 0.0]), 0.1, (1.0, 0.0, 0.0))
        self.valid_point = np.array([0.5, 0.5, 0.5])
        self.valid_path = np.array(
            [[-0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]
        )

    def test_distance_to_obstacle_surface_none_point(self) -> None:
        """distance_to_obstacle_surface raises TypeError when point is None."""
        with self.assertRaises(TypeError):
            distance_to_obstacle_surface(None, self.obstacle)

    def test_generate_asteroid_field_none_bounds(self) -> None:
        """generate_asteroid_field raises TypeError when bounds is None."""
        with self.assertRaises(TypeError):
            generate_asteroid_field(None, 5)

    def test_rrt_planner_init_none_bounds(self) -> None:
        """RRTPlanner raises TypeError when bounds is None."""
        with self.assertRaises(TypeError):
            RRTPlanner(None)

    def test_plan_path_none_start(self) -> None:
        """plan_path raises TypeError when start is None."""
        goal = np.array([0.8, 0.0, 0.0])
        with self.assertRaises(TypeError):
            self.planner.plan_path(None, goal, [])

    def test_analyze_path_none_path(self) -> None:
        """analyze_path raises TypeError when path is None."""
        with self.assertRaises(TypeError):
            self.planner.analyze_path(None, [])

    def test_smooth_path_none_path(self) -> None:
        """smooth_path raises TypeError when path is None."""
        with self.assertRaises(TypeError):
            self.planner.smooth_path(None, [])

    def test_pursuit_ai_init_none_bounds(self) -> None:
        """PursuitAI raises TypeError when bounds is None."""
        with self.assertRaises(TypeError):
            PursuitAI(None)

    def test_update_target_behavior_none_target(self) -> None:
        """update_target_behavior raises TypeError when target is None."""
        ai = PursuitAI(self.bounds)
        pursuer = Ship(np.array([0.0, 0.0, 0.0]), np.eye(3), np.zeros(3))
        with self.assertRaises(TypeError):
            ai.update_target_behavior(None, pursuer, [])

    def test_check_collision_none_point(self) -> None:
        """_check_collision raises TypeError when point is None."""
        with self.assertRaises(TypeError):
            self.planner._check_collision(None, [])

    def test_nearest_node_index_none_nodes(self) -> None:
        """_nearest_node_index raises TypeError when nodes is None."""
        with self.assertRaises(TypeError):
            self.planner._nearest_node_index(None, self.valid_point)

    def test_steer_none_origin(self) -> None:
        """_steer raises TypeError when origin is None."""
        with self.assertRaises(TypeError):
            self.planner._steer(None, self.valid_point)

    def test_segment_is_collision_free_none_start(self) -> None:
        """_segment_is_collision_free raises TypeError when start is None."""
        with self.assertRaises(TypeError):
            self.planner._segment_is_collision_free(None, self.valid_point, [])

    def test_extract_path_none_nodes(self) -> None:
        """_extract_path raises TypeError when nodes is None."""
        with self.assertRaises(TypeError):
            self.planner._extract_path(None, 0)


if __name__ == "__main__":
    unittest.main()

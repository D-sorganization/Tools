"""Regression tests for the incremental RRT nearest-neighbour query (issue #3683).

The planner used to rebuild the full coordinate array from ``nodes`` on every
iteration, making nearest-neighbour selection O(N^2) over the tree. These tests
pin the behaviour after switching to an incrementally maintained coordinate
buffer: nearest-neighbour selection must still match a brute-force search, and
the planner must still find a valid collision-free path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
# The planner source lives outside the default ``tests`` collection path, so add
# it explicitly (mirrors how the data_processor_io contract tests bootstrap).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RRT_SRC = _REPO_ROOT / "src" / "rrt_path_planner" / "python" / "src"
if str(_RRT_SRC) not in sys.path:
    sys.path.insert(0, str(_RRT_SRC))

from star_wars_rrt import Obstacle, RRTPlanner  # noqa: E402


@pytest.fixture
def planner() -> RRTPlanner:
    bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    p = RRTPlanner(bounds, max_iterations=2000)
    p.goal_radius = 0.1
    return p


@pytest.mark.unit
def test_nearest_in_coords_matches_brute_force(planner: RRTPlanner) -> None:
    """Incremental nearest-neighbour matches an explicit brute-force search."""
    nodes = [
        np.append(np.array([0.0, 0.0, 0.0]), -1.0),
        np.append(np.array([0.5, 0.1, 0.0]), 0.0),
        np.append(np.array([-0.3, 0.7, 0.2]), 0.0),
        np.append(np.array([0.9, -0.4, 0.1]), 1.0),
        np.append(np.array([-0.6, -0.6, -0.3]), 2.0),
    ]
    coords = np.array([node[:3] for node in nodes], dtype=np.float64)

    samples = [
        np.array([0.4, 0.0, 0.0]),
        np.array([-0.2, 0.6, 0.1]),
        np.array([0.85, -0.35, 0.05]),
        np.array([-0.55, -0.5, -0.2]),
        np.array([0.0, 0.0, 0.0]),
    ]
    for sample in samples:
        brute = int(np.argmin([np.linalg.norm(c - sample) for c in coords]))
        # Both the legacy nodes-based query and the new coords-based query must
        # agree with the brute-force result.
        assert planner._nearest_node_index(nodes, sample) == brute
        assert planner._nearest_in_coords(coords, sample) == brute


@pytest.mark.unit
def test_plan_path_still_finds_valid_path(planner: RRTPlanner) -> None:
    """The planner returns a collision-free path that starts and ends correctly."""
    start = np.array([-0.8, -0.8, 0.0])
    goal = np.array([0.8, 0.8, 0.0])
    obstacles = [Obstacle(0, np.array([0, 0, 0]), 0.2, (1, 1, 1))]

    path = planner.plan_path(start, goal, obstacles)

    assert path is not None
    assert np.allclose(path[0], start)
    assert np.linalg.norm(path[-1] - goal) <= planner.goal_radius
    for point in path:
        assert not planner._check_collision(point, obstacles)
    for i in range(len(path) - 1):
        step = np.linalg.norm(path[i + 1] - path[i])
        assert step <= planner.step_size + 1e-5


@pytest.mark.unit
def test_plan_path_no_obstacles_succeeds(planner: RRTPlanner) -> None:
    """A clear corridor must always yield a path."""
    start = np.array([-0.8, 0.0, 0.0])
    goal = np.array([0.8, 0.0, 0.0])
    path = planner.plan_path(start, goal, [])
    assert path is not None
    assert np.allclose(path[0], start)

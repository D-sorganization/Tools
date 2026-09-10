"""Independent piecewise-linear three-mass event and work reference."""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq


def generator(active: bool) -> np.ndarray:
    mass = np.zeros((3, 3))
    mass[:2, :2] = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0.03, 0.1])
    mass[2, 2] = 0.046
    stiffness = np.zeros((3, 3))
    stiffness[:2, :2] = 1000 * np.array([[1, -1], [-1, 1]]) + np.diag([400, 0])
    normal = np.array([0, 1, -1])
    stiffness += active * 2e4 * np.outer(normal, normal)
    damping = np.diag([2.0, 0, 0]) + active * 3 * np.outer(normal, normal)
    matrix = np.zeros((9, 9))
    matrix[:3, 3:6] = np.eye(3)
    matrix[3:6, :3] = -np.linalg.solve(mass, stiffness)
    matrix[3:6, 3:6] = -np.linalg.solve(mass, damping)
    forcing = np.array([[-4 + 0.6 + 0.012, 120 + 0.8, 80], [0.7, 0, 0], [0, 0, 0]])
    matrix[3:6, 6:] = np.linalg.solve(mass, forcing)
    matrix[7, 6], matrix[8, 7] = 1, 2
    return matrix


def _powers(state: np.ndarray, rate: np.ndarray, active: bool) -> np.ndarray:
    q, v, time_s = state[:3], state[3:6], state[7]
    anchor_q, anchor_v = -0.01 + 0.3 * time_s + 0.2 * time_s**2, 0.3 + 0.4 * time_s
    effort = 0.03 * (rate[3] - 0.4) + 2 * (v[0] - anchor_v) + 400 * (q[0] - anchor_q)
    compression, speed = q[1] - q[2], v[1] - v[2]
    return np.array(
        [
            0.7 * v[1],
            effort * anchor_v,
            2 * (v[0] - anchor_v) ** 2,
            3 * speed**2 if active else 0,
            -2e4 * compression * speed if not active and compression > 0 else 0,
        ]
    )


def reference(end_s: float = 0.003) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    state = np.array([0.01, 0.03, 0.0301, 0.2, -0.1, -0.4, 1, 0, 0])
    time_s, work, events = 0.0, np.zeros(5), []
    for active, force_event, bracket in (
        (False, False, (0.0003, 0.0005)),
        (True, True, (0.002, 0.0024)),
        (False, False, (0.0001, 0.0002)),
        (False, False, None),
    ):
        matrix = generator(active)

        def value(
            dt: float,
            matrix: np.ndarray = matrix,
            initial: np.ndarray = state,
            force: bool = force_event,
        ) -> float:
            point = expm(dt * matrix) @ initial
            compression, speed = point[1] - point[2], point[4] - point[5]
            return float(2e4 * compression + 3 * speed if force else compression)

        dt = end_s - time_s if bracket is None else brentq(value, *bracket, xtol=1e-15)
        assert dt > 0, "reference segments must advance time"
        if bracket is not None:
            # Separate SI residual checks: newtons for release, metres for gap.
            tolerance = 1e-8 if force_event else 1e-11
            assert abs(value(dt)) < tolerance, "reference event root is unresolved"
        for index in range(5):

            def power(
                t: float,
                matrix: np.ndarray = matrix,
                initial: np.ndarray = state,
                active: bool = active,
                index: int = index,
            ) -> float:
                point = expm(t * matrix) @ initial
                return float(_powers(point, matrix @ point, active)[index])

            work[index] += quad(power, 0, dt, epsabs=1e-13, epsrel=1e-11)[0]
        state = expm(dt * matrix) @ state
        time_s += dt
        if bracket is not None:
            events.append(time_s)
    return state[:6], work, tuple(events)

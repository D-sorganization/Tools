"""Rust-accelerated ball-flight facade (GRACEFUL dual-path posture).

Mirrors UpstreamDrift ``physics/aerodynamics/_rust_facade.py``: the
``tools_core`` wheel is probed at import time and callers choose the fast
path via :func:`is_rust_available`. Unlike the swing-dynamics facade
(strict, bilateral_rust posture), flight is perfectly serviceable in
scipy, so absence of the wheel never raises at import — the scipy models
in :mod:`shared.python.swing_sim.flight.models` are the fallback.

The canonical Rust implementation is
``rust_core/tools-core/src/ball_flight.rs`` (its header notes it replaces
the UpstreamDrift Numba version). Its native frame is the app frame
(x downrange, y up, z lateral/right); this facade converts results into
the flight frame (x forward, y left, z up) used across this package.

Requires a ``tools_core`` wheel that exposes ``simulate_trajectory``
(added alongside this package); older wheels that only ship the classes
report unavailable.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .frames import from_flight_frame, to_flight_frame
from .types import (
    FlightResult,
    LaunchConditions,
    TrajectoryPoint,
    compute_flight_metrics,
)

logger = logging.getLogger(__name__)

RUST_MODEL_NAME = "tools-core Rust RK4"

_RUST_AVAILABLE = False
_rust: Any = None

try:
    import tools_core

    _rust = tools_core
    _RUST_AVAILABLE = bool(
        hasattr(_rust, "simulate_trajectory")
        and hasattr(_rust, "BallProperties")
        and hasattr(_rust, "EnvironmentalConditions")
        and hasattr(_rust, "LaunchConditions")
    )
    if _RUST_AVAILABLE:
        logger.info("tools_core ball_flight kernel loaded (Rust fast path)")
    else:
        logger.info(
            "tools_core installed but simulate_trajectory missing (older wheel) "
            "— using pure-Python scipy flight models."
        )
except ImportError:  # pragma: no cover - exercised on machines without wheel
    logger.info(
        "tools_core not installed — using pure-Python scipy flight models. "
        "Install with `pip install rust_core/tools-core`."
    )


def is_rust_available() -> bool:
    """Return True when the Rust ball-flight kernel is loaded."""
    return _RUST_AVAILABLE


def _build_rust_inputs(
    launch: LaunchConditions,
) -> tuple[Any, Any, Any]:
    """Construct ``tools_core`` ball/env/launch objects from a launch spec."""
    ball = _rust.BallProperties()
    ball.mass = float(launch.ball_mass)
    ball.diameter = 2.0 * float(launch.ball_radius)

    env = _rust.EnvironmentalConditions()
    env.air_density = float(launch.air_density)
    env.gravity = float(launch.gravity)
    wind_flight = launch.get_wind_vector()
    if float(np.linalg.norm(wind_flight)) > 0.0:
        wind_app = from_flight_frame(wind_flight)
        env.set_wind(float(wind_app[0]), float(wind_app[1]), float(wind_app[2]))

    rust_launch = _rust.LaunchConditions(
        float(launch.ball_speed),
        math.degrees(launch.launch_angle),
        math.degrees(launch.azimuth_angle),
        float(launch.spin_rate),
    )
    spin_vec_flight = launch.get_spin_vector()
    spin_mag = float(np.linalg.norm(spin_vec_flight))
    if spin_mag > 0.0:
        axis_app = from_flight_frame(spin_vec_flight / spin_mag)
        rust_launch.set_spin_axis(
            float(axis_app[0]), float(axis_app[1]), float(axis_app[2])
        )
    return ball, env, rust_launch


def simulate_trajectory_rust(
    launch: LaunchConditions,
    max_time: float = 10.0,
    dt: float = 0.01,
) -> FlightResult:
    """Simulate a trajectory with the ``tools_core`` Rust RK4 kernel.

    Args:
        launch: Flight-frame launch conditions (radians / RPM / SI).
        max_time: Maximum simulated time [s], > 0.
        dt: Fixed RK4 step [s], > 0.

    Returns:
        :class:`FlightResult` with the trajectory converted to the flight
        frame (x forward, y left, z up).

    Raises:
        ImportError: If the Rust kernel is unavailable (use
            :func:`is_rust_available` and fall back to the scipy models).
        ValueError: If ``launch.ball_speed``, ``max_time``, or ``dt`` is
            not positive (validated here so the Rust precondition
            ``assert!`` never panics across the FFI boundary).
    """
    if not _RUST_AVAILABLE:
        raise ImportError(
            "tools_core ball-flight kernel is not available; fall back to "
            "shared.python.swing_sim.flight.models (scipy). Build the wheel "
            "with `pip install rust_core/tools-core`."
        )
    if launch is None:
        raise ValueError("launch must be provided")
    if launch.wind_scenario is not None and not launch.wind_scenario.is_steady:
        raise ValueError(
            "Rust flight currently supports steady wind only; use a Python "
            "model for shear, gusts, or turbulence"
        )
    if launch.ball_speed <= 0.0:
        raise ValueError(f"ball_speed must be > 0; got {launch.ball_speed!r}")
    if not (math.isfinite(max_time) and max_time > 0.0):
        raise ValueError(f"max_time must be finite and > 0; got {max_time!r}")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be finite and > 0; got {dt!r}")

    ball, env, rust_launch = _build_rust_inputs(launch)
    raw_points = _rust.simulate_trajectory(ball, env, rust_launch, max_time, dt)

    points: list[TrajectoryPoint] = []
    for p in raw_points:
        pos_app = np.array([p.x, p.y, p.z])
        vel_app = np.array([p.vx, p.vy, p.vz])
        points.append(
            TrajectoryPoint(
                time=float(p.time),
                position=to_flight_frame(pos_app),
                velocity=to_flight_frame(vel_app),
            )
        )

    return compute_flight_metrics(points, RUST_MODEL_NAME)


__all__ = [
    "RUST_MODEL_NAME",
    "is_rust_available",
    "simulate_trajectory_rust",
]

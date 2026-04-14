import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicsConfig:
    """Design by Contract container for Pendulum physics parameters."""

    m1: float = 1.0
    m2: float = 2.5
    l1: float = 1.0
    l2: float = 1.0
    gravity: float = 9.81

    # Initial State
    theta1: float = np.pi / 1.1
    omega1: float = 2.5
    theta2: float = np.pi / 1.5
    omega2: float = 4.0

    # Non-conservative forces
    damp1: float = 0.0
    damp2: float = 0.0
    amp1: float = 0.0
    freq1: float = 1.0
    amp2: float = 0.0
    freq2: float = 1.0

    def __post_init__(self) -> None:
        """DbC: Pre-condition invariants."""
        if self.m1 <= 0:
            raise ValueError("Mass 1 must be positive")
        if self.m2 <= 0:
            raise ValueError("Mass 2 must be positive")
        if self.l1 <= 0:
            raise ValueError("Length 1 must be positive")
        if self.l2 <= 0:
            raise ValueError("Length 2 must be positive")
        if self.gravity <= 0:
            raise ValueError("Gravity must be positive")
        if self.damp1 < 0:
            raise ValueError("Damping must be non-negative")
        if self.damp2 < 0:
            raise ValueError("Damping must be non-negative")


@dataclass
class RenderConfig:
    """DbC container for visualization settings."""

    save_path: str | None = None
    fps: int = 120
    duration: int = 30
    history_sec: float = 10.0

    def __post_init__(self) -> None:
        """DbC: Pre-condition invariants."""
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.history_sec <= 0:
            raise ValueError("History timeframe must be positive")


def parse_args() -> tuple[PhysicsConfig, RenderConfig]:
    """Parse unified arguments."""
    parser = argparse.ArgumentParser(description="Chaotic Pendulum Screensaver")
    parser.add_argument("--save", type=str, default=None, help="Path for output")
    parser.add_argument("--fps", type=int, default=120, help="Frames per second")
    parser.add_argument("--duration", type=int, default=30, help="Simulation time (s)")

    parser.add_argument("--m1", type=float, default=1.0, help="Mass 1")
    parser.add_argument("--m2", type=float, default=2.5, help="Mass 2")
    parser.add_argument("--l1", type=float, default=1.0, help="Length 1")
    parser.add_argument("--l2", type=float, default=1.0, help="Length 2")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity")

    parser.add_argument("--theta1", type=float, default=np.pi / 1.1)
    parser.add_argument("--omega1", type=float, default=2.5)
    parser.add_argument("--theta2", type=float, default=np.pi / 1.5)
    parser.add_argument("--omega2", type=float, default=4.0)

    parser.add_argument("--damp1", type=float, default=0.0)
    parser.add_argument("--damp2", type=float, default=0.0)
    parser.add_argument("--amp1", type=float, default=0.0)
    parser.add_argument("--freq1", type=float, default=1.0)
    parser.add_argument("--amp2", type=float, default=0.0)
    parser.add_argument("--freq2", type=float, default=1.0)

    args = parser.parse_args()

    physics = PhysicsConfig(
        m1=args.m1,
        m2=args.m2,
        l1=args.l1,
        l2=args.l2,
        gravity=args.gravity,
        theta1=args.theta1,
        omega1=args.omega1,
        theta2=args.theta2,
        omega2=args.omega2,
        damp1=args.damp1,
        damp2=args.damp2,
        amp1=args.amp1,
        freq1=args.freq1,
        amp2=args.amp2,
        freq2=args.freq2,
    )

    render = RenderConfig(save_path=args.save, fps=args.fps, duration=args.duration)

    return physics, render

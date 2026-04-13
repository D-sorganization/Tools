"""Pure physics engine for the Asteroid Jumper simulation.

All state is represented as plain dataclasses; no Qt dependencies.
Physics uses Newtonian rigid-body mechanics in 2-D.

Design-by-Contract:
  - Every public function validates its preconditions via assertions.
  - Physical quantities carry SI-like units (mass in kg, length in m, etc.)
    but we deliberately keep "toy" scales: asteroid radius ~10 m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class Vec2(NamedTuple):
    """Immutable 2-D vector."""

    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: object) -> Vec2:  # noqa: PYI034  # override with wider type
        assert isinstance(other, Vec2), "Vec2 + Vec2 required"
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: object) -> Vec2:  # noqa: PYI034  # override with wider type
        assert isinstance(other, Vec2), "Vec2 - Vec2 required"
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: object) -> Vec2:
        assert isinstance(scalar, int | float), "Vec2 * scalar required"
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: object) -> Vec2:
        return self.__mul__(scalar)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def dot(self, other: Vec2) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vec2) -> float:
        """Scalar z-component of 3-D cross product."""
        return self.x * other.y - self.y * other.x

    def length(self) -> float:
        """Euclidean magnitude."""
        return math.hypot(self.x, self.y)

    def normalize(self) -> Vec2:
        """Unit vector; returns zero vector if magnitude is zero."""
        mag = self.length()
        if mag < 1e-12:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def rotate(self, angle_rad: float) -> Vec2:
        """Rotate counter-clockwise by *angle_rad*."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def perp(self) -> Vec2:
        """90-degree counter-clockwise perpendicular."""
        return Vec2(-self.y, self.x)


# ---------------------------------------------------------------------------
# Rigid body state
# ---------------------------------------------------------------------------


@dataclass
class RigidBody:
    """Mutable state for a single 2-D rigid body.

    Invariant: mass > 0, moment_of_inertia > 0.
    """

    mass: float  # kg
    moment_of_inertia: float  # kgÂ·mÂ²
    pos: Vec2 = field(default_factory=Vec2)
    vel: Vec2 = field(default_factory=Vec2)
    angle: float = 0.0  # radians
    angular_vel: float = 0.0  # rad/s

    def __post_init__(self) -> None:
        assert self.mass > 0, f"mass must be positive, got {self.mass}"
        assert self.moment_of_inertia > 0, (
            f"moment_of_inertia must be positive, got {self.moment_of_inertia}"
        )

    @property
    def speed(self) -> float:
        """Translational speed (m/s)."""
        return self.vel.length()

    @property
    def kinetic_energy_trans(self) -> float:
        """Translational kinetic energy (J)."""
        return 0.5 * self.mass * self.speed**2

    @property
    def kinetic_energy_rot(self) -> float:
        """Rotational kinetic energy (J)."""
        return 0.5 * self.moment_of_inertia * self.angular_vel**2


# ---------------------------------------------------------------------------
# Asteroid / jumper geometry helpers
# ---------------------------------------------------------------------------


def moment_of_inertia_ellipse(mass: float, a: float, b: float) -> float:
    """Moment of inertia for a solid ellipse with semi-axes a, b."""
    assert mass > 0
    assert a > 0 and b > 0
    return 0.25 * mass * (a**2 + b**2)


def moment_of_inertia_disk(mass: float, radius: float) -> float:
    """Moment of inertia for a solid disk."""
    assert mass > 0 and radius > 0
    return 0.5 * mass * radius**2


def moment_of_inertia_rod(mass: float, length: float) -> float:
    """Moment of inertia for a thin rod about its centre."""
    assert mass > 0 and length > 0
    return mass * length**2 / 12.0


# ---------------------------------------------------------------------------
# Jump impulse physics
# ---------------------------------------------------------------------------


def compute_jump_impulse(
    force_magnitude: float,
    force_direction_rad: float,
    contact_point: Vec2,
    asteroid_com: Vec2,
    jumper_com: Vec2,
) -> tuple[Vec2, float, float]:
    """Compute the impulse vector and torques for both bodies.

    The jumper pushes on the asteroid with *force_magnitude* in the direction
    given by *force_direction_rad* (angle from +x axis in world frame).  By
    Newton's third law the asteroid receives an equal-and-opposite impulse.

    Args:
        force_magnitude: Magnitude of the impulse (NÂ·s, positive).
        force_direction_rad: Direction the *jumper* moves after the jump.
        contact_point: World-frame contact point (where feet leave asteroid).
        asteroid_com: World-frame position of asteroid centre of mass.
        jumper_com: World-frame position of jumper centre of mass.

    Returns:
        (jumper_impulse, asteroid_torque_impulse, jumper_torque_impulse)
        where torques are scalar (z-component of r Ã— J).
    """
    assert force_magnitude >= 0, "force_magnitude must be non-negative"

    J = Vec2(
        force_magnitude * math.cos(force_direction_rad),
        force_magnitude * math.sin(force_direction_rad),
    )

    # Asteroid receives -J (reaction)
    r_asteroid = contact_point - asteroid_com
    asteroid_angular_impulse = r_asteroid.cross(-J)

    # Jumper receives +J
    r_jumper = contact_point - jumper_com
    jumper_angular_impulse = r_jumper.cross(J)

    return J, asteroid_angular_impulse, jumper_angular_impulse


def apply_impulse(body: RigidBody, impulse: Vec2, torque_impulse: float) -> None:
    """Apply a linear and angular impulse to *body* (mutates in place)."""
    body.vel = body.vel + impulse * (1.0 / body.mass)
    body.angular_vel += torque_impulse / body.moment_of_inertia


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

GRAVITY: Vec2 = Vec2(0.0, 0.0)  # Deep space â€” no gravity by default


def integrate_body(body: RigidBody, dt: float) -> None:
    """Semi-implicit Euler integration step for *body* (mutates in place)."""
    assert dt > 0, f"dt must be positive, got {dt}"
    body.pos = body.pos + body.vel * dt
    body.angle += body.angular_vel * dt


# ---------------------------------------------------------------------------
# Spring launch model
# ---------------------------------------------------------------------------


@dataclass
class SpringLaunch:
    """Models a spring-like leg push over multiple frames.

    The force ramps up then down following a half-sine profile so the
    simulation looks smooth and the total impulse equals the desired value.

    Invariant: duration > 0, remaining >= 0.
    """

    total_impulse: float  # NÂ·s
    force_direction_rad: float  # rad
    contact_point: Vec2
    asteroid_com: Vec2
    jumper_com: Vec2
    duration: float  # seconds
    elapsed: float = 0.0

    def __post_init__(self) -> None:
        assert self.total_impulse >= 0
        assert self.duration > 0

    @property
    def is_complete(self) -> bool:
        """True when the spring push has finished."""
        return self.elapsed >= self.duration

    def step(self, dt: float) -> tuple[Vec2, float, float] | None:
        """Advance the spring by *dt* seconds.

        Returns the (impulse, asteroid_torque, jumper_torque) for this step,
        or None if the launch is already complete.
        """
        assert dt > 0
        if self.is_complete:
            return None
        remaining = self.duration - self.elapsed
        actual_dt = min(dt, remaining)
        phase = math.pi * self.elapsed / self.duration
        # Half-sine force profile: integral = total_impulse over duration
        instantaneous_force = (
            math.pi * self.total_impulse / (2.0 * self.duration) * math.sin(phase)
        )
        step_impulse = instantaneous_force * actual_dt
        self.elapsed += actual_dt
        return compute_jump_impulse(
            step_impulse,
            self.force_direction_rad,
            self.contact_point,
            self.asteroid_com,
            self.jumper_com,
        )


# ---------------------------------------------------------------------------
# Full simulation state
# ---------------------------------------------------------------------------


@dataclass
class SimState:
    """Complete state of the asteroid-jumper system.

    Invariant: asteroid and jumper are distinct objects with positive mass.
    """

    asteroid: RigidBody
    jumper: RigidBody
    spring: SpringLaunch | None = None
    phase: str = "ready"  # "ready" | "jumping" | "flight"
    time: float = 0.0

    def __post_init__(self) -> None:
        assert self.asteroid is not self.jumper, "asteroid and jumper must differ"

    @property
    def total_linear_momentum(self) -> Vec2:
        """Conservation check: total linear momentum of the system."""
        return (
            self.asteroid.vel * self.asteroid.mass + self.jumper.vel * self.jumper.mass
        )

    @property
    def total_angular_momentum(self) -> float:
        """Conservation check: total angular momentum about world origin."""
        ast_L = (
            self.asteroid.pos.cross(self.asteroid.vel * self.asteroid.mass)
            + self.asteroid.moment_of_inertia * self.asteroid.angular_vel
        )
        jmp_L = (
            self.jumper.pos.cross(self.jumper.vel * self.jumper.mass)
            + self.jumper.moment_of_inertia * self.jumper.angular_vel
        )
        return ast_L + jmp_L


def step_simulation(state: SimState, dt: float) -> None:
    """Advance the simulation by *dt* seconds (mutates *state*)."""
    assert dt > 0

    if state.spring is not None and not state.spring.is_complete:
        result = state.spring.step(dt)
        if result is not None:
            jumper_impulse, ast_torque, jmp_torque = result
            apply_impulse(state.asteroid, -jumper_impulse, ast_torque)
            apply_impulse(state.jumper, jumper_impulse, jmp_torque)
        if state.spring.elapsed >= state.spring.duration:
            state.spring = None
            state.phase = "flight"

    integrate_body(state.asteroid, dt)
    integrate_body(state.jumper, dt)
    state.time += dt


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def off_centre_ratio(
    contact_point: Vec2, asteroid_com: Vec2, jumper_com: Vec2
) -> float:
    """Fraction [0, 1] of how far the force is off the line joining COMs.

    0 = perfectly through both centres (maximum translational efficiency).
    1 = maximally off-centre (maximum spin, minimum translation per impulse).
    """
    line_dir = (jumper_com - asteroid_com).normalize()
    r = contact_point - asteroid_com
    r_perp = r - line_dir * r.dot(line_dir)
    line_len = (jumper_com - asteroid_com).length()
    if line_len < 1e-9:
        return 0.0
    return min(r_perp.length() / (line_len * 0.5), 1.0)

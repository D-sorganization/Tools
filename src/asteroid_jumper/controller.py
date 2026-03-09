"""Simulation controller for the Asteroid Jumper.

Owns the SimState and exposes high-level actions:
  - configure_asteroid(mass, shape_kind, ...)
  - set_force_angle(angle_rad)
  - set_impulse(N_s)
  - start_jump()
  - tick(dt)
  - reset()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from asteroid_jumper.asteroid_shape import (
    AsteroidShape,
    ShapeKind,
    make_circle,
    make_ellipse,
    make_random,
    surface_point_at_angle,
)
from asteroid_jumper.physics import (
    RigidBody,
    SimState,
    SpringLaunch,
    Vec2,
    moment_of_inertia_disk,
    moment_of_inertia_ellipse,
    step_simulation,
)

# ---------------------------------------------------------------------------
# Jumper constants
# ---------------------------------------------------------------------------

JUMPER_MASS: float = 80.0  # kg (average human)
JUMPER_HEIGHT: float = 1.8  # m
JUMPER_RADIUS: float = 0.3  # m — approximate for MoI
JUMP_DURATION: float = 0.4  # seconds — spring push time
DEFAULT_IMPULSE: float = 500.0  # N·s  (~6× body weight for 0.4 s)

# Asteroid defaults
ASTEROID_MASS_FACTOR: float = 2.0  # relative to JUMPER_MASS
DEFAULT_ASTEROID_RADIUS: float = 10.0  # m
ASTEROID_DENSITY: float = 2000.0  # kg/m³ (rocky)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class SimController:
    """Owns all simulation state and exposes high-level control API."""

    # ---- configurable parameters ----
    asteroid_mass: float = JUMPER_MASS * ASTEROID_MASS_FACTOR
    asteroid_shape_kind: ShapeKind = ShapeKind.ELLIPSE
    asteroid_semi_a: float = DEFAULT_ASTEROID_RADIUS
    asteroid_semi_b: float = DEFAULT_ASTEROID_RADIUS * 0.6
    force_angle_deg: float = 90.0  # where on asteroid surface the jump lands
    jump_direction_deg: float = 90.0  # direction jumper travels
    impulse_magnitude: float = DEFAULT_IMPULSE
    spring_duration: float = JUMP_DURATION

    # ---- derived / live ----
    shape: AsteroidShape = field(init=False)
    state: SimState = field(init=False)
    _initial_momentum: float = 0.0

    def __post_init__(self) -> None:
        self.shape = self._build_shape()
        self.state = self._build_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(
        self,
        asteroid_mass: float | None = None,
        shape_kind: ShapeKind | None = None,
        semi_a: float | None = None,
        semi_b: float | None = None,
        impulse: float | None = None,
    ) -> None:
        """Reconfigure and reset — call before jumping."""
        assert self.state.phase == "ready", "Cannot reconfigure mid-flight"
        if asteroid_mass is not None:
            assert asteroid_mass > 0
            self.asteroid_mass = asteroid_mass
        if shape_kind is not None:
            self.asteroid_shape_kind = shape_kind
        if semi_a is not None:
            assert semi_a > 0
            self.asteroid_semi_a = semi_a
        if semi_b is not None:
            assert semi_b > 0
            self.asteroid_semi_b = semi_b
        if impulse is not None:
            assert impulse >= 0
            self.impulse_magnitude = impulse
        self.shape = self._build_shape()
        self.state = self._build_state()

    def set_force_angle(self, angle_deg: float) -> None:
        """Set where on the asteroid surface the jumper stands (degrees)."""
        self.force_angle_deg = float(angle_deg)

    def set_jump_direction(self, angle_deg: float) -> None:
        """Set the direction the jumper launches (degrees from +x)."""
        self.jump_direction_deg = float(angle_deg)

    def set_impulse(self, n_s: float) -> None:
        """Set the total impulse of the jump."""
        assert n_s >= 0
        self.impulse_magnitude = float(n_s)

    def start_jump(self) -> None:
        """Begin the spring-launch sequence."""
        assert self.state.phase == "ready", "Already jumping or in flight"
        contact_pt = self._contact_point()
        asteroid_com = self.state.asteroid.pos
        jumper_com = self.state.jumper.pos
        jump_rad = math.radians(self.jump_direction_deg)
        self.state.spring = SpringLaunch(
            total_impulse=self.impulse_magnitude,
            force_direction_rad=jump_rad,
            contact_point=contact_pt,
            asteroid_com=asteroid_com,
            jumper_com=jumper_com,
            duration=self.spring_duration,
        )
        self.state.phase = "jumping"
        self._initial_momentum = self._total_momentum_magnitude()

    def tick(self, dt: float) -> None:
        """Advance simulation by *dt* seconds."""
        assert dt > 0
        step_simulation(self.state, dt)

    def reset(self) -> None:
        """Return to initial ready state."""
        self.shape = self._build_shape()
        self.state = self._build_state()

    # ------------------------------------------------------------------
    # Read-only metrics
    # ------------------------------------------------------------------

    def jumper_speed(self) -> float:
        """Current translational speed of the jumper (m/s)."""
        return float(self.state.jumper.speed)

    def jumper_angular_speed(self) -> float:
        """Absolute angular speed of the jumper (rad/s)."""
        return float(abs(self.state.jumper.angular_vel))

    def asteroid_speed(self) -> float:
        """Current translational speed of the asteroid (m/s)."""
        return float(self.state.asteroid.speed)

    def asteroid_angular_speed(self) -> float:
        """Absolute angular speed of the asteroid (rad/s)."""
        return float(abs(self.state.asteroid.angular_vel))

    def off_centre_fraction(self) -> float:
        """How off-centre the jump is: 0 = through COM, 1 = maximally off."""
        from asteroid_jumper.physics import off_centre_ratio

        contact = self._contact_point()
        return float(
            off_centre_ratio(
                contact,
                self.state.asteroid.pos,
                self.state.jumper.pos,
            )
        )

    def leg_phase(self) -> float:
        """Normalised spring phase [0, 1] for leg animation."""
        if self.state.spring is None:
            if self.state.phase == "jumping":
                return 1.0
            return 0.0
        elapsed = self.state.spring.elapsed
        dur = self.state.spring.duration
        return min(elapsed / dur, 1.0) if dur > 0 else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_shape(self) -> AsteroidShape:
        """Construct the asteroid shape from current parameters."""
        kind = self.asteroid_shape_kind
        a, b = self.asteroid_semi_a, self.asteroid_semi_b
        if kind == ShapeKind.CIRCLE:
            return make_circle(a)
        if kind == ShapeKind.ELLIPSE:
            return make_ellipse(a, b)
        return make_random(a, roughness=0.35, seed=42)

    def _build_state(self) -> SimState:
        """Construct the initial SimState from current configuration."""
        a, b = self.shape.semi_a, self.shape.semi_b
        asteroid_moi = moment_of_inertia_ellipse(self.asteroid_mass, a, b)
        asteroid = RigidBody(
            mass=self.asteroid_mass,
            moment_of_inertia=asteroid_moi,
            pos=Vec2(0.0, 0.0),
        )
        # Jumper positioned on surface
        surface_angle_rad = math.radians(self.force_angle_deg)
        sx, sy = surface_point_at_angle(self.shape, surface_angle_rad)
        # Normal offset: place jumper COM above surface by half height
        offset = JUMPER_HEIGHT / 2.0
        nx = math.cos(surface_angle_rad)
        ny = math.sin(surface_angle_rad)
        jx = sx + nx * offset
        jy = sy + ny * offset
        jumper_moi = moment_of_inertia_disk(JUMPER_MASS, JUMPER_RADIUS)
        jumper = RigidBody(
            mass=JUMPER_MASS,
            moment_of_inertia=jumper_moi,
            pos=Vec2(jx, jy),
        )
        return SimState(asteroid=asteroid, jumper=jumper)

    def _contact_point(self) -> Vec2:
        """World-frame contact point between jumper feet and asteroid."""
        surface_angle_rad = math.radians(self.force_angle_deg)
        sx, sy = surface_point_at_angle(self.shape, surface_angle_rad)
        return Vec2(sx, sy)

    def _total_momentum_magnitude(self) -> float:
        """Total system linear momentum magnitude."""
        total = self.state.total_linear_momentum
        return float(total.length())

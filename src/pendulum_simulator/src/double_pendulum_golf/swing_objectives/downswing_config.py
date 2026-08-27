"""Configuration for a downswing optimization, shared by every objective.

Everything that must be held fixed for the objective comparison to mean anything
lives in one immutable object: the golfer, the top-of-backswing posture, the
torque limits, the physiological slew-rate limits, the duration, the anatomical
wrist range, and the impact condition. Only the objective changes between runs,
so any difference in the resulting swing is attributable to the objective alone.

Closes #4769.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from double_pendulum_golf.swing_objectives.actuation import SwingActuation
from double_pendulum_golf.physics import (
    JointLimits,
    PendulumParams,
    TorqueClamp,
    mass_matrix,
)

__all__ = ["DownswingConfig", "RATE_SCALE", "DEFAULT_TORQUE_RATE_LIMITS"]

FloatArray = npt.NDArray[np.float64]

#: Characteristic angular rate in rad/s used to non-dimensionalize the NLP.
RATE_SCALE: float = 30.0

#: Default torque slew-rate limits ``(hub, wrist)`` in N·m/s. Muscle cannot
#: switch torque instantaneously; these correspond to reaching full torque in
#: roughly 75 ms, and without them a collocation optimizer will discover
#: bang-bang swings that stop the arms dead at impact.
DEFAULT_TORQUE_RATE_LIMITS: tuple[float, float] = (2400.0, 300.0)

_STATE_WIDTH = 4
_MIN_NODES = 5
_ANGLE_SPAN_MARGIN_RAD = 1.0
_MAX_ANGULAR_RATE = 80.0
_MAX_UNCOCK_RATE = 120.0


@dataclass(frozen=True, slots=True)
class DownswingConfig:
    """Immutable settings for one downswing optimization problem.

    Attributes:
        params: Double pendulum physical parameters.
        node_count: Number of collocation knot points across the downswing.
        duration_s: Downswing duration from the top of the backswing to impact.
            Tour players sit in the 0.23-0.30 s range.
        initial_state: Top-of-backswing state ``[theta1, phi, dtheta1, dphi]``.
        impact_theta1_rad: Arm angle at impact. Zero puts the hands at the bottom
            of the arc.
        torque_clamp: Peak joint torque magnitudes the golfer can produce.
        joint_limits: Anatomical joint ranges; only the bounds are used here, not
            the penalty stiffness.
        torque_rate_limits: Maximum ``(hub, wrist)`` torque slew in N·m/s.
        require_release: Whether to constrain the club to be fully released at
            impact, which is what physically must happen for the face to reach
            the ball.
        limit_torque_rate: Whether to enforce the slew-rate limits. Disabling it
            is instructive but is not a fair comparison.
        use_variable_scaling: Whether to solve in non-dimensional variables.
            Disabling it reproduces the badly conditioned formulation and is
            retained so tests can demonstrate why scaling is required.
        effort_weight: Weight on a small control-effort regularizer. Not a
            competing objective — it keeps the NLP well posed and stops the
            solver chattering torques between bounds. Identical across objectives.
        collocation_method: ``"hermite_simpson"`` or ``"trapezoidal"``.
        max_iterations: NLP iteration cap.
        tolerance: Solver convergence tolerance. Must stay tight; at SciPy's
            default the solver stops as soon as it finds a feasible trajectory
            and returns the initial guess unchanged.
        min_hand_speed_ms: Optional floor on hand speed at impact, in m/s. Real
            golfers arrive with 6-9 m/s (Nesbit 2005); the unconstrained optimum
            arrives near zero. Raising this floor is how epic #4775 measures the
            model's structural coupling between releasing the club and stopping
            the hands.
        actuation: Optional Hill-type actuation limits. When supplied, torque
            capacity falls with joint speed and braking is restricted to the
            weaker antagonist budget, replacing the flat symmetric clamp as the
            binding limit. ``torque_clamp`` still supplies the box bounds the
            NLP is scaled against. See epic #4775: without this the
            speed-optimal downswing brakes the arms to a standstill at impact.
    """

    params: PendulumParams
    initial_state: FloatArray
    torque_clamp: TorqueClamp
    joint_limits: JointLimits
    node_count: int = 21
    duration_s: float = 0.28
    impact_theta1_rad: float = 0.0
    torque_rate_limits: tuple[float, float] = DEFAULT_TORQUE_RATE_LIMITS
    require_release: bool = True
    limit_torque_rate: bool = True
    use_variable_scaling: bool = True
    effort_weight: float = 1e-3
    collocation_method: str = "hermite_simpson"
    max_iterations: int = 400
    tolerance: float = 1e-9
    min_hand_speed_ms: float | None = None
    actuation: SwingActuation | None = None
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the configuration.

        Pre: none.
        Post: every field is usable by the optimizer without further checking.
        """
        self._validate_grid()
        self._validate_initial_state()
        self._validate_limits()
        self._validate_sweep_is_reachable()

    def _validate_grid(self) -> None:
        """Check the collocation grid and solver settings."""
        if self.node_count < _MIN_NODES:
            raise ValueError(
                f"node_count must be at least {_MIN_NODES}, got {self.node_count}"
            )
        if not (self.duration_s > 0.0 and np.isfinite(self.duration_s)):
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.collocation_method not in ("hermite_simpson", "trapezoidal"):
            raise ValueError(
                f"collocation_method must be hermite_simpson or trapezoidal, "
                f"got {self.collocation_method!r}"
            )
        if self.effort_weight < 0.0:
            raise ValueError(f"effort_weight must be non-negative, got {self.effort_weight}")

    def _validate_initial_state(self) -> None:
        """Check the top-of-backswing posture."""
        state = np.asarray(self.initial_state, dtype=np.float64)
        if state.shape != (_STATE_WIDTH,):
            raise ValueError(f"initial_state must have shape ({_STATE_WIDTH},)")
        if not np.all(np.isfinite(state)):
            raise ValueError(f"initial_state must be finite, got {state}")
        wrist_angle = float(state[1])
        if not self.joint_limits.phi_min <= wrist_angle <= self.joint_limits.phi_max:
            raise ValueError(
                f"initial wrist cock {wrist_angle:.3f} rad lies outside the wrist "
                f"limits [{self.joint_limits.phi_min}, {self.joint_limits.phi_max}]"
            )

    def _validate_limits(self) -> None:
        """Check torque slew-rate limits."""
        rates = np.asarray(self.torque_rate_limits, dtype=np.float64)
        if rates.shape != (2,) or not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
            raise ValueError(
                f"torque_rate_limits must be two positive values, got {self.torque_rate_limits}"
            )

    def _validate_sweep_is_reachable(self) -> None:
        """Reject a downswing the golfer's torque budget provably cannot deliver.

        This is a *necessary* condition, not a sufficient one: it ignores the
        torque slew ramp, gravity, and the work of releasing the wrists, all of
        which push the true minimum higher. Its purpose is to turn an obscure
        "positive directional derivative for linesearch" solver failure into a
        statement about the golfer.
        """
        if self.duration_s < self.minimum_sweep_duration_s:
            raise ValueError(
                f"duration_s={self.duration_s:.3f} s is below the "
                f"{self.minimum_sweep_duration_s:.3f} s needed for a "
                f"{self.arm_sweep_rad:.3f} rad sweep at "
                f"{self.torque_clamp.max_torque1:.0f} N*m against "
                f"{self.hub_inertia_at_start:.3f} kg*m^2. Lengthen the downswing, "
                f"raise the hub torque limit, or start from a shorter backswing. "
                f"Note this bound is necessary, not sufficient: the slew-rate "
                f"limit and the wrist release both push the true minimum higher."
            )

    # --- Derived quantities ---------------------------------------------------

    @property
    def arm_sweep_rad(self) -> float:
        """Total arm angle travelled from the top of the backswing to impact."""
        return abs(float(self.initial_state[0]) - self.impact_theta1_rad)

    @property
    def hub_inertia_at_start(self) -> float:
        """Effective hub inertia at the top of the backswing, in kg*m^2."""
        return float(mass_matrix(float(self.initial_state[1]), self.params)[0, 0])

    @property
    def minimum_sweep_duration_s(self) -> float:
        """Necessary-condition lower bound on the downswing duration, in s.

        Assumes the whole torque budget accelerates the arm from rest for the
        entire downswing against its starting inertia, which is optimistic in
        every respect. See :meth:`_validate_sweep_is_reachable`.
        """
        peak_acceleration = self.torque_clamp.max_torque1 / self.hub_inertia_at_start
        return float(np.sqrt(2.0 * self.arm_sweep_rad / peak_acceleration))

    @property
    def time_grid(self) -> FloatArray:
        """Uniform collocation time grid in s."""
        grid: FloatArray = np.linspace(0.0, self.duration_s, self.node_count)
        return grid

    @property
    def time_step(self) -> float:
        """Spacing between adjacent collocation nodes in s."""
        return self.duration_s / (self.node_count - 1)

    @property
    def torque_limit_vector(self) -> FloatArray:
        """Peak torque magnitudes ``[hub, wrist]`` in N·m."""
        return np.array(
            [self.torque_clamp.max_torque1, self.torque_clamp.max_torque2],
            dtype=np.float64,
        )

    @property
    def torque_rate_vector(self) -> FloatArray:
        """Torque slew-rate limits ``[hub, wrist]`` in N·m/s."""
        return np.asarray(self.torque_rate_limits, dtype=np.float64)

    @property
    def state_scale(self) -> FloatArray:
        """Characteristic magnitude of each state variable."""
        return np.array([1.0, 1.0, RATE_SCALE, RATE_SCALE], dtype=np.float64)

    def state_bounds(self) -> list[tuple[float, float]]:
        """Per-node state bounds ``[theta1, phi, dtheta1, dphi]``.

        The wrist bounds are the anatomical range; the arm bound is generous
        because the swing arc, not anatomy, limits it.
        """
        arm_span = abs(float(self.initial_state[0])) + _ANGLE_SPAN_MARGIN_RAD
        return [
            (-arm_span, arm_span),
            (self.joint_limits.phi_min, self.joint_limits.phi_max),
            (-_MAX_ANGULAR_RATE, _MAX_ANGULAR_RATE),
            (-_MAX_UNCOCK_RATE, _MAX_UNCOCK_RATE),
        ]

    def torque_bounds(self) -> list[tuple[float, float]]:
        """Per-node torque bounds ``[hub, wrist]`` in N·m."""
        hub, wrist = self.torque_limit_vector
        return [(-float(hub), float(hub)), (-float(wrist), float(wrist))]

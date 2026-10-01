"""Hill-type joint actuation limits for a golf downswing.

The shipped :class:`~double_pendulum_golf.physics.TorqueClamp` gives the golfer a
symmetric, velocity-independent torque budget. Under that budget the
speed-optimal downswing drives the arms to the limit and then reverses the hub
torque to brake them to a standstill at impact — 32% of the downswing spent
braking, hands arriving at 0.36 m/s against a measured 6-9 m/s.

Two pieces of physiology are missing, and this module supplies both.

Torque falls with joint speed
-----------------------------
Muscle force declines hyperbolically with shortening velocity
(`Hill 1938 <https://doi.org/10.1098/rspb.1938.0050>`_). At a joint this becomes
a torque-angular-velocity relation; the normalised form used here is

.. code-block:: text

    tau_max(w) = tau0 * (w_max - w) / (w_max + curvature * w)      0 <= w <= w_max

which is ``tau0`` at rest, falls monotonically, and reaches zero at the
unloaded velocity ``w_max``. ``curvature`` is Hill's shape term: larger values
bend the curve down sooner. Golf-swing forward-dynamics models have used
torque-velocity limits of this kind since
`Sprigings & Neal 2000 <https://doi.org/10.1123/jab.16.4.356>`_ and
`MacKenzie & Sprigings 2009 <https://doi.org/10.1007/s12283-009-0020-9>`_;
without one, a simulated golfer accelerates the arms far past what a person can.

Braking is not free
-------------------
The muscles that decelerate the arms are not the ones that drive them. Modelling
the brake as the same budget in the other direction is what lets the optimizer
stop the hands. Braking capacity is therefore a fraction ``brake_fraction`` of
the driving peak, raised by ``eccentric_gain`` because lengthening muscle is
stronger than isometric.

Sign convention matches the rest of the package: a downswing runs with
``omega1 < 0``, so the driving limit applies to negative hub torque and the
braking limit to positive hub torque. :meth:`JointActuation.torque_bounds`
resolves that from the sign of the joint rate, so callers never have to.

Closes #4777.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "JointActuation",
    "SwingActuation",
    "tour_hub_actuation",
    "tour_wrist_actuation",
]

FloatArray = npt.NDArray[np.float64]

#: Peak hub torque for a strong player, N*m. Consistent with the 200-250 N*m
#: range reported for the torso/shoulder contribution in
#: Nesbit & Serrano's work-and-power analysis of the swing.
_TOUR_HUB_PEAK_NM = 225.0

#: Unloaded angular velocity of the arm about the hub, rad/s. Loaded tour arm
#: rates peak near 15-20 rad/s, so the zero-torque asymptote sits above that.
_TOUR_HUB_MAX_RATE = 30.0

#: Hill shape term. Larger bends the torque-velocity curve down sooner.
_TOUR_HUB_CURVATURE = 4.0

#: Antagonist braking capacity as a fraction of the driving peak.
_TOUR_HUB_BRAKE_FRACTION = 0.30

#: Eccentric (lengthening) strength relative to isometric.
_TOUR_ECCENTRIC_GAIN = 1.30

#: Wrist actuation. An order of magnitude below the hub, which is what makes the
#: release predominantly passive rather than driven.
_TOUR_WRIST_PEAK_NM = 20.0
_TOUR_WRIST_MAX_RATE = 45.0
_TOUR_WRIST_CURVATURE = 3.0
_TOUR_WRIST_BRAKE_FRACTION = 0.45


@dataclass(frozen=True, slots=True)
class JointActuation:
    """Velocity-dependent, direction-asymmetric torque capacity for one joint.

    Attributes:
        peak_torque_nm: Isometric peak torque ``tau0``, in N*m.
        max_rate_rad_s: Unloaded angular velocity ``w_max`` at which driving
            capacity reaches zero, in rad/s.
        curvature: Hill shape term; larger values bend the curve down sooner.
        brake_fraction: Antagonist braking capacity as a fraction of
            ``peak_torque_nm``, in (0, 1].
        eccentric_gain: Eccentric strength relative to isometric, at least 1.
    """

    peak_torque_nm: float
    max_rate_rad_s: float
    curvature: float
    brake_fraction: float
    eccentric_gain: float

    def __post_init__(self) -> None:
        """Validate that every parameter is physically meaningful.

        Pre: none.
        Post: all fields are finite and inside their physical ranges.
        """
        if not (self.peak_torque_nm > 0.0 and np.isfinite(self.peak_torque_nm)):
            raise ValueError(f"peak_torque_nm must be positive, got {self.peak_torque_nm}")
        if not (self.max_rate_rad_s > 0.0 and np.isfinite(self.max_rate_rad_s)):
            raise ValueError(f"max_rate_rad_s must be positive, got {self.max_rate_rad_s}")
        if not (self.curvature >= 0.0 and np.isfinite(self.curvature)):
            raise ValueError(f"curvature must be non-negative, got {self.curvature}")
        if not 0.0 < self.brake_fraction <= 1.0:
            raise ValueError(f"brake_fraction must lie in (0, 1], got {self.brake_fraction}")
        if not (self.eccentric_gain >= 1.0 and np.isfinite(self.eccentric_gain)):
            raise ValueError(f"eccentric_gain must be at least 1, got {self.eccentric_gain}")

    # --- Scalar contract ------------------------------------------------------

    def driving_limit(self, joint_rate_rad_s: float) -> float:
        """Torque available *with* the direction of motion, in N*m.

        Args:
            joint_rate_rad_s: Joint angular speed; only its magnitude matters.

        Returns:
            Available driving torque, falling to zero at ``max_rate_rad_s`` and
            staying at zero beyond it.

        Pre: the rate is finite.
        Post: the result lies in ``[0, peak_torque_nm]``.
        """
        if not np.isfinite(joint_rate_rad_s):
            raise ValueError(f"joint rate must be finite, got {joint_rate_rad_s}")
        speed = abs(float(joint_rate_rad_s))
        if speed >= self.max_rate_rad_s:
            return 0.0
        numerator = self.max_rate_rad_s - speed
        denominator = self.max_rate_rad_s + self.curvature * speed
        return float(self.peak_torque_nm * numerator / denominator)

    def braking_limit(self, joint_rate_rad_s: float) -> float:
        """Torque available *against* the direction of motion, in N*m.

        Modelled as the antagonist capacity raised by the eccentric gain. It is
        deliberately not velocity-faded: eccentric capacity is broadly flat with
        lengthening speed, and the constraint that matters is that it is small.

        Args:
            joint_rate_rad_s: Joint angular speed; only its magnitude matters.

        Returns:
            Available braking torque, in N*m.

        Pre: the rate is finite.
        Post: the result is positive.
        """
        if not np.isfinite(joint_rate_rad_s):
            raise ValueError(f"joint rate must be finite, got {joint_rate_rad_s}")
        return float(self.peak_torque_nm * self.brake_fraction * self.eccentric_gain)

    def torque_bounds(self, joint_rate_rad_s: float) -> tuple[float, float]:
        """Admissible torque interval at a given joint rate.

        Args:
            joint_rate_rad_s: Signed joint angular velocity.

        Returns:
            ``(lower, upper)`` torque bounds in N*m. Driving capacity applies in
            the direction the joint is already moving, braking capacity against
            it; at rest the interval is symmetric on the driving limit.

        Pre: the rate is finite.
        Post: ``lower < 0 < upper``.
        """
        if not np.isfinite(joint_rate_rad_s):
            raise ValueError(f"joint rate must be finite, got {joint_rate_rad_s}")
        driving = self.driving_limit(joint_rate_rad_s)
        braking = self.braking_limit(joint_rate_rad_s)
        if joint_rate_rad_s < 0.0:
            return -driving, braking
        if joint_rate_rad_s > 0.0:
            return -braking, driving
        return -driving, driving

    def margins(self, joint_rate_rad_s: float, torque_nm: float) -> tuple[float, float]:
        """Signed distances from a torque to each of its bounds.

        Positive on both entries means the torque is admissible. This is the form
        the NLP consumes as an inequality constraint.

        Args:
            joint_rate_rad_s: Signed joint angular velocity.
            torque_nm: Applied joint torque.

        Returns:
            ``(torque - lower, upper - torque)``.
        """
        lower, upper = self.torque_bounds(joint_rate_rad_s)
        return torque_nm - lower, upper - torque_nm

    # --- Vectorized contract --------------------------------------------------

    def batch_margins(self, rates: FloatArray, torques: FloatArray) -> FloatArray:
        """Compute :meth:`margins` for a whole trajectory at once.

        Args:
            rates: ``(N,)`` signed joint angular velocities.
            torques: ``(N,)`` applied joint torques.

        Returns:
            ``(N, 2)`` array of lower and upper margins.

        Pre: both arrays are finite and the same length.
        Post: the result is finite.
        """
        rate_array = np.asarray(rates, dtype=np.float64)
        torque_array = np.asarray(torques, dtype=np.float64)
        if rate_array.shape != torque_array.shape:
            raise ValueError("rates and torques must have the same shape")
        if not (np.all(np.isfinite(rate_array)) and np.all(np.isfinite(torque_array))):
            raise ValueError("rates and torques must be finite")

        speed = np.abs(rate_array)
        driving = np.where(
            speed >= self.max_rate_rad_s,
            0.0,
            self.peak_torque_nm
            * (self.max_rate_rad_s - speed)
            / (self.max_rate_rad_s + self.curvature * speed),
        )
        braking = np.full_like(
            speed, self.peak_torque_nm * self.brake_fraction * self.eccentric_gain
        )
        moving = rate_array != 0.0
        upper = np.where(rate_array < 0.0, braking, driving)
        lower = -np.where(rate_array > 0.0, braking, driving)
        upper = np.where(moving, upper, driving)
        lower = np.where(moving, lower, -driving)
        return np.column_stack([torque_array - lower, upper - torque_array])


@dataclass(frozen=True, slots=True)
class SwingActuation:
    """Actuation limits for both joints of a downswing.

    Attributes:
        hub: Torso/shoulder actuation driving the arms.
        wrist: Wrist actuation, an order of magnitude weaker.
    """

    hub: JointActuation
    wrist: JointActuation

    def batch_margins(self, rates: FloatArray, torques: FloatArray) -> FloatArray:
        """Compute margins for both joints across a trajectory.

        Args:
            rates: ``(N, 2)`` signed joint rates ``[omega1, phidot]``.
            torques: ``(N, 2)`` joint torques ``[hub, wrist]``.

        Returns:
            ``(N, 4)`` array of margins, hub pair then wrist pair.

        Pre: both arrays are ``(N, 2)`` and finite.
        """
        rate_array = np.asarray(rates, dtype=np.float64)
        torque_array = np.asarray(torques, dtype=np.float64)
        if rate_array.shape != torque_array.shape or rate_array.ndim != 2:
            raise ValueError("rates and torques must both be (N, 2)")
        return np.hstack(
            [
                self.hub.batch_margins(rate_array[:, 0], torque_array[:, 0]),
                self.wrist.batch_margins(rate_array[:, 1], torque_array[:, 1]),
            ]
        )

    @property
    def peak_torques_nm(self) -> FloatArray:
        """Isometric peaks ``[hub, wrist]``, for scaling the NLP."""
        return np.array([self.hub.peak_torque_nm, self.wrist.peak_torque_nm], dtype=np.float64)


def tour_hub_actuation() -> JointActuation:
    """Torso/shoulder actuation for a strong player driving the arms."""
    return JointActuation(
        peak_torque_nm=_TOUR_HUB_PEAK_NM,
        max_rate_rad_s=_TOUR_HUB_MAX_RATE,
        curvature=_TOUR_HUB_CURVATURE,
        brake_fraction=_TOUR_HUB_BRAKE_FRACTION,
        eccentric_gain=_TOUR_ECCENTRIC_GAIN,
    )


def tour_wrist_actuation() -> JointActuation:
    """Wrist actuation for a strong player, far below the hub."""
    return JointActuation(
        peak_torque_nm=_TOUR_WRIST_PEAK_NM,
        max_rate_rad_s=_TOUR_WRIST_MAX_RATE,
        curvature=_TOUR_WRIST_CURVATURE,
        brake_fraction=_TOUR_WRIST_BRAKE_FRACTION,
        eccentric_gain=_TOUR_ECCENTRIC_GAIN,
    )

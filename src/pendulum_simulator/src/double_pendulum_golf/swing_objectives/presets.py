"""Default golfer presets for the swing objective comparison.

The preset exists so the surface and the CLI start from the same, feasible
configuration rather than each inventing one.

**The club masses are real masses on the canonical distributed club.**
:mod:`double_pendulum_golf.physics` models the club as a uniform shaft plus a
head point mass at the tip (Tools #5431). The shaft and head masses are solved
from the measured driver so that the model reproduces the real club's total mass
(0.310 kg) and its inertia about the wrist (0.2881 kg m^2) exactly:
``m2 = 0.108 kg`` (shaft plus grip) and ``mClub = 0.202 kg`` (head) at the
modelled 1.10 m shaft; see
:func:`~double_pendulum_golf.swing_objectives.club_equivalence.distributed_club_split`.
The earlier preset used a 0.238 kg point-mass "equivalent tip mass", which was a
workaround for the old lumped model (#4785) and would now double-correct.

Duration and torque budget carry deliberate **slack** above the golfer's minimum
sweep time: run the comparison too close to that bound and the constraints pin
the trajectory, every objective returns the identical swing, and the
cross-evaluation matrix fills with 100% entries that look like agreement but are
an artifact of the configuration.

Closes #4771, #4785.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from double_pendulum_golf.physics import JointLimits, PendulumParams, TorqueClamp
from double_pendulum_golf.swing_objectives.club_equivalence import (
    DRIVER_SPEC,
    distributed_club_split,
)
from double_pendulum_golf.swing_objectives.downswing_config import DownswingConfig

__all__ = ["GolferPreset", "SwingBudget", "DEFAULT_PRESET", "build_config"]

#: Anatomical wrist range: the wrists cannot uncock far past straight, and
#: anatomy caps how far they can cock.
_WRIST_MIN_RAD = -0.175
_WRIST_MAX_RAD = 2.094

#: Arm-angle bound, generous because the swing arc rather than anatomy limits it.
_ARM_BOUND_RAD = 4.0

#: Modelled shaft length, wrist to clubhead, in m.
_SHAFT_LENGTH_M = 1.10

#: Shaft and head masses reproducing a real driver's mass and wrist inertia.
_DRIVER_SHAFT_MASS_KG, _DRIVER_HEAD_MASS_KG = distributed_club_split(
    DRIVER_SPEC, shaft_length_m=_SHAFT_LENGTH_M
)


@dataclass(frozen=True, slots=True)
class GolferPreset:
    """A named, feasible starting point for a comparison.

    Attributes:
        name: Human-readable preset name.
        arm_mass_kg: Lumped arm mass, modeled as a point mass at the hands.
        shaft_mass_kg: Uniform shaft mass (including grip). With
            ``clubhead_mass_kg`` it reproduces a real club's mass and inertia
            about the wrist.
        clubhead_mass_kg: Head point mass at the tip.
        arm_length_m: Hub-to-hands distance.
        club_length_m: Club length.
        top_arm_angle_rad: Arm angle at the top of the backswing.
        top_wrist_cock_rad: Wrist cock angle at the top of the backswing.
        duration_s: Downswing duration, chosen with slack above the minimum.
        hub_torque_nm: Peak hub torque.
        wrist_torque_nm: Peak wrist torque.
        node_count: Collocation nodes.
    """

    name: str
    arm_mass_kg: float = 5.0
    shaft_mass_kg: float = _DRIVER_SHAFT_MASS_KG
    clubhead_mass_kg: float = _DRIVER_HEAD_MASS_KG
    arm_length_m: float = 0.65
    club_length_m: float = _SHAFT_LENGTH_M
    top_arm_angle_rad: float = 2.618
    top_wrist_cock_rad: float = 1.745
    duration_s: float = 0.28
    hub_torque_nm: float = 250.0
    wrist_torque_nm: float = 20.0
    node_count: int = 21

    def to_params(self) -> PendulumParams:
        """Build the physics parameters for this preset."""
        return PendulumParams(
            m1=self.arm_mass_kg,
            m2=self.shaft_mass_kg,
            L1=self.arm_length_m,
            L2=self.club_length_m,
            mClub=self.clubhead_mass_kg,
        )


#: Tour-plausible golfer with enough torque and time for the objectives to differ.
DEFAULT_PRESET = GolferPreset(name="Tour driver (comparison default)")


@dataclass(frozen=True, slots=True)
class SwingBudget:
    """The shared effort budget a comparison runs under.

    Grouped into one object so callers pass a single meaningful argument rather
    than four positional numbers.

    Attributes:
        duration_s: Downswing duration in s.
        hub_torque_nm: Peak hub torque in N·m.
        wrist_torque_nm: Peak wrist torque in N·m.
        node_count: Collocation node count.
    """

    duration_s: float = DEFAULT_PRESET.duration_s
    hub_torque_nm: float = DEFAULT_PRESET.hub_torque_nm
    wrist_torque_nm: float = DEFAULT_PRESET.wrist_torque_nm
    node_count: int = DEFAULT_PRESET.node_count


def build_config(
    budget: SwingBudget | None = None,
    preset: GolferPreset = DEFAULT_PRESET,
) -> DownswingConfig:
    """Build a downswing configuration from a preset and a shared effort budget.

    Args:
        budget: Duration, torque limits and node count. Defaults to the preset's.
        preset: Base golfer preset.

    Returns:
        A validated configuration.

    Raises:
        ValueError: If the resulting downswing is one the golfer's torque budget
            provably cannot deliver; the message states the required minimum.
    """
    settings = budget if budget is not None else SwingBudget()
    return DownswingConfig(
        params=preset.to_params(),
        node_count=settings.node_count,
        duration_s=settings.duration_s,
        initial_state=np.array(
            [preset.top_arm_angle_rad, preset.top_wrist_cock_rad, 0.0, 0.0],
            dtype=np.float64,
        ),
        impact_theta1_rad=0.0,
        torque_clamp=TorqueClamp(
            max_torque1=settings.hub_torque_nm,
            max_torque2=settings.wrist_torque_nm,
        ),
        joint_limits=JointLimits(
            phi_min=_WRIST_MIN_RAD,
            phi_max=_WRIST_MAX_RAD,
            theta1_min=-_ARM_BOUND_RAD,
            theta1_max=_ARM_BOUND_RAD,
        ),
    )

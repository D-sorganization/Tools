"""Default golfer presets for the swing objective comparison.

The preset exists so the surface and the CLI start from the same, feasible
configuration rather than each inventing one. Its duration and torque budget are
deliberately set with **slack** above the golfer's minimum sweep time: run the
comparison too close to that bound and the constraints pin the trajectory, every
objective returns the identical swing, and the cross-evaluation matrix fills with
100% entries that look like agreement but are an artifact of the configuration.

Closes #4771.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from double_pendulum_golf.physics import JointLimits, PendulumParams, TorqueClamp
from double_pendulum_golf.swing_objectives.downswing_config import DownswingConfig

__all__ = ["GolferPreset", "DEFAULT_PRESET", "build_config"]

#: Anatomical wrist range: the wrists cannot uncock far past straight, and
#: anatomy caps how far they can cock.
_WRIST_MIN_RAD = -0.175
_WRIST_MAX_RAD = 2.094

#: Arm-angle bound, generous because the swing arc rather than anatomy limits it.
_ARM_BOUND_RAD = 4.0


@dataclass(frozen=True, slots=True)
class GolferPreset:
    """A named, feasible starting point for a comparison.

    Attributes:
        name: Human-readable preset name.
        arm_mass_kg: Lumped arm mass, modeled as a point mass at the hands.
        shaft_mass_kg: Club shaft mass.
        clubhead_mass_kg: Clubhead mass, modeled as a point mass at the tip.
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
    shaft_mass_kg: float = 0.30
    clubhead_mass_kg: float = 0.20
    arm_length_m: float = 0.65
    club_length_m: float = 1.10
    top_arm_angle_rad: float = 2.618
    top_wrist_cock_rad: float = 1.745
    duration_s: float = 0.36
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


def build_config(
    duration_s: float | None = None,
    hub_torque_nm: float | None = None,
    wrist_torque_nm: float | None = None,
    node_count: int | None = None,
    preset: GolferPreset = DEFAULT_PRESET,
) -> DownswingConfig:
    """Build a downswing configuration from the preset with optional overrides.

    Args:
        duration_s: Downswing duration override, in s.
        hub_torque_nm: Hub torque limit override, in N·m.
        wrist_torque_nm: Wrist torque limit override, in N·m.
        node_count: Collocation node count override.
        preset: Base preset.

    Returns:
        A validated configuration.

    Raises:
        ValueError: If the resulting downswing is one the golfer's torque budget
            provably cannot deliver; the message states the required minimum.
    """
    return DownswingConfig(
        params=preset.to_params(),
        node_count=node_count if node_count is not None else preset.node_count,
        duration_s=duration_s if duration_s is not None else preset.duration_s,
        initial_state=np.array(
            [preset.top_arm_angle_rad, preset.top_wrist_cock_rad, 0.0, 0.0],
            dtype=np.float64,
        ),
        impact_theta1_rad=0.0,
        torque_clamp=TorqueClamp(
            max_torque1=(hub_torque_nm if hub_torque_nm is not None else preset.hub_torque_nm),
            max_torque2=(
                wrist_torque_nm if wrist_torque_nm is not None else preset.wrist_torque_nm
            ),
        ),
        joint_limits=JointLimits(
            phi_min=_WRIST_MIN_RAD,
            phi_max=_WRIST_MAX_RAD,
            theta1_min=-_ARM_BOUND_RAD,
            theta1_max=_ARM_BOUND_RAD,
        ),
    )

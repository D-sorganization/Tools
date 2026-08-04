"""Impact model value types.

Ported from UpstreamDrift ``src/shared/python/physics/impact_model/types.py``
(epic #4103 / issue #4106), rewritten self-contained against the vendored
:mod:`.constants`.

Changes from the UpstreamDrift source:

- ``PreImpactState`` gains ``clubhead_moi_tensor`` — an optional 3x3 club
  MOI tensor enabling the full 3-D effective-mass treatment
  ``J = (1/m_ball + 1/m_club + (r x n) . I^-1 (r x n))^-1`` in
  :mod:`.models` (scalar ``clubhead_moi`` fallback preserved).
- ``ImpactParameters`` drops the three empirical gear-effect scaling
  constants (``gear_effect_factor`` / ``h_scale`` / ``v_scale``); gear
  effect is now derived from first principles in :mod:`.gear_effect` and
  needs only ``cg_depth`` (CG distance behind the face plane).

Frame note: the impact model itself is frame-agnostic — all vectors must
simply share one right-handed Cartesian frame. The delivery front-end
(:mod:`.delivery`) produces vectors in the AffineDrift frame
(x = target line, y = up, z = right).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from .constants import (
    DRIVER_CG_DEPTH_M,
    DRIVER_COR,
    DRIVER_LIE_RAD,
    DRIVER_LOFT_RAD,
    DRIVER_MASS_KG,
    DRIVER_MOI_KG_M2,
    TYPICAL_CONTACT_DURATION_S,
)


class ImpactModelType(Enum):
    """Types of impact physics models."""

    RIGID_BODY = auto()  # Instantaneous impulse with COR
    SPRING_DAMPER = auto()  # Kelvin-Voigt viscoelastic
    FINITE_TIME = auto()  # Impulse-momentum with duration


@dataclass
class PreImpactState:
    """State of ball and clubhead immediately before impact.

    Attributes:
        clubhead_velocity: Clubhead velocity [m/s] (3,)
        clubhead_angular_velocity: Clubhead angular velocity [rad/s] (3,)
        clubhead_orientation: Clubface normal vector [unitless] (3,)
        ball_position: Ball center position [m] (3,)
        ball_velocity: Ball velocity [m/s] (3,)
        ball_angular_velocity: Ball spin [rad/s] (3,)
        clubhead_mass: Effective clubhead mass [kg]
        clubhead_loft: Clubface loft angle [rad]
        clubhead_lie: Clubface lie angle [rad]
        clubhead_moi: Clubhead scalar moment of inertia about CG [kg.m^2]
        impact_offset: Impact location offset from CG on clubface [m] (2,)
            [horizontal (+ = toe side), vertical (+ = high on face)]
        clubhead_moi_tensor: Optional full 3x3 clubhead MOI tensor about the
            CG [kg.m^2], expressed in the same frame as the vectors. When
            provided it replaces the scalar ``clubhead_moi`` in the
            effective-mass computation for off-center hits.
    """

    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    clubhead_orientation: np.ndarray
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray
    clubhead_mass: float = DRIVER_MASS_KG
    clubhead_loft: float = DRIVER_LOFT_RAD
    clubhead_lie: float = DRIVER_LIE_RAD
    clubhead_moi: float = DRIVER_MOI_KG_M2
    impact_offset: np.ndarray | None = None
    clubhead_moi_tensor: np.ndarray | None = None


@dataclass
class PostImpactState:
    """State of ball and clubhead immediately after impact.

    Attributes:
        ball_velocity: Ball launch velocity [m/s] (3,)
        ball_angular_velocity: Ball spin [rad/s] (3,)
        clubhead_velocity: Clubhead velocity after impact [m/s] (3,)
        clubhead_angular_velocity: Clubhead angular velocity after [rad/s] (3,)
        contact_duration: Duration of contact [s]
        energy_transfer: Kinetic energy transferred to ball [J]
        impact_location: Location of impact on clubface [m] (2,) [x, y]
    """

    ball_velocity: np.ndarray
    ball_angular_velocity: np.ndarray
    clubhead_velocity: np.ndarray
    clubhead_angular_velocity: np.ndarray
    contact_duration: float
    energy_transfer: float
    impact_location: np.ndarray


@dataclass
class ImpactParameters:
    """Parameters for impact model.

    Attributes:
        cor: Coefficient of restitution (0-1)
        contact_duration: Contact time [s]
        contact_stiffness: Spring stiffness for compliant model [N/m]
        contact_damping: Damping for compliant model [N.s/m]
        friction_coefficient: Ball-face friction
        cg_depth: Clubhead CG distance behind the face plane [m]; the
            front-to-back lever arm driving physics-based gear effect
            (:mod:`.gear_effect`).
    """

    cor: float = DRIVER_COR
    contact_duration: float = TYPICAL_CONTACT_DURATION_S
    contact_stiffness: float = 1e6  # [N/m]
    contact_damping: float = 1e3  # [N.s/m]
    friction_coefficient: float = 0.4
    cg_depth: float = DRIVER_CG_DEPTH_M


@dataclass
class ImpactEvent:
    """Complete record of a single impact event.

    Attributes:
        timestamp: Simulation time when impact occurred [s]
        pre_state: State before impact
        post_state: State after impact
        energy_balance: Energy analysis results
        impact_id: Unique identifier for this impact
        model_type: Type of impact model used
    """

    timestamp: float
    pre_state: PreImpactState
    post_state: PostImpactState
    energy_balance: dict[str, float]
    impact_id: int
    model_type: ImpactModelType

"""Private free-body Newton/Euler response using canonical shaft mechanics.

This is an instantaneous response, not a contact trajectory or calibrated
ball model. Full mass properties remain explicit, with no assumed club role.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...golf_club._grip_contracts import Vector6, finite_array, vector6
from ...golf_club._shaft_mass_solve import solve_material_mass
from ...golf_club._shaft_se3 import twist_ad
from ...golf_club._shaft_spatial_element import _material_spatial_inertia
from ...golf_club._shaft_spectrum import SpectrumScales
from ...golf_club._validation import (
    Matrix3,
    Vector3,
    require_finite_float,
    require_identifier,
    require_inertia,
    require_vector3,
)


@dataclass(frozen=True)
class RigidBodyResponse:
    """Owned body-twist rate, kinetic energy and external mechanical power.

    Material linear twist rate is not origin acceleration: add omega cross v
    before rotating to observer axes. Transport contributes no energy loss.
    """

    twist_rate: Vector6
    kinetic_energy_j: float
    power_w: float
    relative_residual: float


@dataclass(frozen=True)
class RigidBodyInertia:
    """SI mass properties in a named material frame with an explicit COM datum.

    Center of mass is measured from the material origin. The complete inertia
    tensor is about that COM and expressed in the same material axes. A sphere
    contact geometry requires its material origin to be the geometric center;
    the COM can differ if independently specified. Zero/singular inertia can
    be represented but is refused by the six-axis ODE response. Symmetry and
    principal-moment realizability are checked both in SI and after scaling
    by the largest tensor entry; small units cannot mask an invalid tensor.
    """

    material_frame_id: str
    mass_kg: float
    center_of_mass_m: Vector3
    inertia_at_com_kg_m2: Matrix3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "material_frame_id",
            require_identifier(self.material_frame_id, "material_frame_id"),
        )
        object.__setattr__(
            self,
            "mass_kg",
            require_finite_float(self.mass_kg, "mass_kg", positive=True),
        )
        object.__setattr__(
            self,
            "center_of_mass_m",
            require_vector3(self.center_of_mass_m, "center_of_mass_m"),
        )
        inertia = require_inertia(self.inertia_at_com_kg_m2)
        magnitude = float(np.max(np.abs(inertia)))
        if magnitude > 0:
            require_inertia(np.asarray(inertia) / magnitude)
        object.__setattr__(self, "inertia_at_com_kg_m2", inertia)

    def response(
        self,
        twist: object,
        wrench: object,
        material_frame_id: str,
        scales: SpectrumScales,
    ) -> RigidBodyResponse:
        """Evaluate M vdot = wrench + ad(v).T M v in declared material axes.

        Wrench is force [N], then moment [N m] about the material origin;
        twist is origin velocity [m/s], then angular velocity [rad/s]. The
        caller must match the material-frame identifier for both inputs.
        The same scaled, positive-mass solve as the shaft is used unchanged.
        """
        if (
            require_identifier(material_frame_id, "material_frame_id")
            != self.material_frame_id
        ):
            raise ValueError("response material frame must match mass properties")
        velocity = finite_array(twist, (6,), "body twist")
        load = finite_array(wrench, (6,), "body wrench")
        mass = _material_spatial_inertia(self)
        momentum = mass @ velocity
        rates, residual = solve_material_mass(
            mass, load + twist_ad(velocity).T @ momentum, scales
        )
        return RigidBodyResponse(
            vector6(rates, "body twist rate"),
            require_finite_float(float(0.5 * velocity @ momentum), "kinetic energy"),
            require_finite_float(float(velocity @ load), "external power"),
            residual,
        )


__all__ = ()

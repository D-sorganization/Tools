"""Stationary shaft assembly inputs; no inferred axial or polar properties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import require_finite_float
from .grip_impedance import PassiveGripImpedance
from .shaft_profile import ShaftProfile, ShaftProfileProvenance
from .types import ComponentMassProperties


@dataclass(frozen=True)
class ShaftRodProperties:
    """Additional measured rod properties on the existing raw station grid.

    Axial stiffness EA has units N. Polar mass moment per unit length has
    units kg m (not kg m²). Both arrays must contain one strictly positive
    finite value per profile station. They are not inferred from EI, GJ,
    diameter or density; composite sections need independent qualification.
    The supplied provenance describes these two supplemental arrays.
    """

    profile: ShaftProfile
    axial_stiffness_n: tuple[float, ...]
    polar_mass_per_length_kg_m: tuple[float, ...]
    provenance: ShaftProfileProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ShaftProfile):
            raise TypeError("profile must be ShaftProfile")
        if not isinstance(self.provenance, ShaftProfileProvenance):
            raise TypeError("provenance must be ShaftProfileProvenance")
        for name in ("axial_stiffness_n", "polar_mass_per_length_kg_m"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if len(values) != len(self.profile.stations):
                raise ValueError(f"{name} must match the profile station count")
            object.__setattr__(
                self,
                name,
                tuple(
                    require_finite_float(value, name, positive=True) for value in values
                ),
            )


@dataclass(frozen=True)
class ShaftAttachments:
    """Optional tip body and fixed-reference butt grip in the shaft frame.

    Tip-body COM is relative to the exposed tip, not the raw butt. Its full
    COM inertia is retained. Grip displacements are relative to a stationary
    anchor at the exposed butt. A moving base requires additional operators.
    """

    tip_body: ComponentMassProperties | None = None
    grip: PassiveGripImpedance | None = None

    def __post_init__(self) -> None:
        if self.tip_body is not None and not isinstance(
            self.tip_body, ComponentMassProperties
        ):
            raise TypeError("tip_body must be ComponentMassProperties or None")
        if self.grip is not None and not isinstance(self.grip, PassiveGripImpedance):
            raise TypeError("grip must be PassiveGripImpedance or None")


@dataclass(frozen=True)
class _LinearOperators:
    """Assembly-owned snapshot; returned arrays are independent writable copies.

    DOFs at each node are [ux, uy, uz, theta_x, theta_y, theta_z]. No constraints
    are eliminated. Zero grip leaves six rigid modes in the unprestressed beam.
    This private result is constructed only after validated finite assembly.
    """

    frame_id: str
    profile_id: str
    positions_m: tuple[float, ...]
    mass: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    profile_provenance: ShaftProfileProvenance
    rod_provenance: ShaftProfileProvenance
    attachments: ShaftAttachments
    model_name: str = "stationary_3d_euler_bernoulli/1"
    assumptions: tuple[str, ...] = (
        "small motion of a straight stationary Euler-Bernoulli shaft",
        "linear axial/torsional and cubic bending element fields",
        "midpoint properties; measured profile damping is not applied",
        "polar torsional inertia retained; transverse section rotary inertia omitted",
        "no shear deformation, warping or constitutive bend-twist coupling",
        "exposed span only; inserted mass belongs in the supplied tip body",
        "no prestress, rotating transport, contact or acoustic operator",
        "no experimentally qualified bandwidth or equipment prediction",
    )


__all__: list[str] = []

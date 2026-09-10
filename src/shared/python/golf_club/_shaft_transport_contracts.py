"""Measured section inertia and model contracts for distributed transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._rotating_body_contracts import RotatingFrameState
from ._shaft_linear_contracts import ShaftAttachments, ShaftRodProperties
from ._validation import require_identifier
from .shaft_profile import ShaftProfileProvenance


@dataclass(frozen=True)
class ShaftRotaryInertia:
    """Section mass moments per length in the declared shaft axes, in kg m.

    Each raw-profile station supplies (Jxx, Jyy, Jxy). These are tensor entries,
    not unsigned area products. The transverse tensor must be positive definite.
    A planar centered section has Jzz = Jxx + Jyy; the assembled model checks
    agreement with the separately declared polar mass density. No section COM
    offset or transverse/axial product of inertia is inferred.
    """

    profile_id: str
    frame_id: str
    transverse_kg_m: tuple[tuple[float, float, float], ...]
    provenance: ShaftProfileProvenance

    def __post_init__(self) -> None:
        for name in ("profile_id", "frame_id"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        if not isinstance(self.provenance, ShaftProfileProvenance):
            raise TypeError("provenance must be ShaftProfileProvenance")
        if not isinstance(self.transverse_kg_m, tuple) or len(self.transverse_kg_m) < 2:
            raise ValueError("transverse_kg_m must be a tuple of at least two stations")
        array = finite_array(
            self.transverse_kg_m, (len(self.transverse_kg_m), 3), "section inertia"
        )
        xx, yy, xy = array.T
        # Normalize the determinant test to avoid overflow or loss to underflow.
        scale = np.max(np.abs(array), axis=1)
        if np.any(scale == 0) or np.any(xx <= 0) or np.any(yy <= 0):
            raise ValueError("section transverse inertia must be positive definite")
        if np.any((xx / scale) * (yy / scale) <= (xy / scale) ** 2):
            raise ValueError("section transverse inertia must be positive definite")
        object.__setattr__(
            self,
            "transverse_kg_m",
            tuple(tuple(float(x) for x in row) for row in array),
        )


@dataclass(frozen=True)
class ShaftRotatingModel:
    """Distributed rod and explicit rotary inertia, with optional head and grip.

    The grip remains an ideal relative-coordinate impedance. Its inertance
    is not transported as if it were absolute hand mass. Both transverse
    section inertia and the full tip body participate in physical transport.
    """

    rod: ShaftRodProperties
    rotary_inertia: ShaftRotaryInertia
    attachments: ShaftAttachments = ShaftAttachments()

    def __post_init__(self) -> None:
        if not isinstance(self.rod, ShaftRodProperties):
            raise TypeError("rod must be ShaftRodProperties")
        if not isinstance(self.rotary_inertia, ShaftRotaryInertia):
            raise TypeError("rotary_inertia must be ShaftRotaryInertia")
        if not isinstance(self.attachments, ShaftAttachments):
            raise TypeError("attachments must be ShaftAttachments")
        profile = self.rod.profile
        inertia = self.rotary_inertia
        if inertia.profile_id != profile.shaft_id:
            raise ValueError("rotary inertia must identify the same profile")
        if inertia.frame_id != profile.frame_id:
            raise ValueError("rotary inertia must use the profile frame")
        if len(inertia.transverse_kg_m) != len(profile.stations):
            raise ValueError("rotary inertia must match the profile station count")
        array = np.asarray(inertia.transverse_kg_m)
        with np.errstate(over="ignore"):
            polar = array[:, 0] + array[:, 1]
        if not np.allclose(
            polar, self.rod.polar_mass_per_length_kg_m, rtol=1e-10, atol=0
        ):
            raise ValueError(
                "transverse trace must agree with declared polar mass density"
            )


@dataclass(frozen=True)
class _ShaftTransport:
    """Straight-reference transport and elastic matrices; not a loaded solve.

    The output retains the entire source model and frame sample. Missing
    geometric prestress, loaded shape and boundary work must not be inferred
    from these coefficients. No constraints or modal truncation are applied.
    """

    model: ShaftRotatingModel
    frame: RotatingFrameState
    positions_m: tuple[float, ...]
    elastic_stiffness: np.ndarray
    damping: np.ndarray
    mass: np.ndarray
    gyroscopic: np.ndarray
    centrifugal_stiffness: np.ndarray
    euler_stiffness: np.ndarray
    acceleration_stiffness: np.ndarray
    equilibrium_force: np.ndarray
    model_name: str = "straight_rayleigh_shaft_transport/1"
    assumptions: tuple[str, ...] = (
        "small canonical translation/rotation perturbations about a straight reference",
        "cubic bending, linear axial/torsional fields; no shear deformation or warping",
        "centerline is section COM; explicit full centered planar section mass inertia",
        "four-point Gauss integration split at profile property knots",
        "midpoint elastic stiffness; stored profile damping is not applied",
        "relative-coordinate grip impedance has no implicit absolute hand inertia",
        "no geometric prestress or loaded initial shape has been supplied",
        "no equilibrium, bandwidth, impact or acoustic qualification established",
    )


__all__: list[str] = []

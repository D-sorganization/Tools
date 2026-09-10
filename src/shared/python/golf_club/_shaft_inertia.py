"""Consistent kinetic quadrature for a private finite-rotation section element.

Samples reuse the existing physical mass-property and spatial-inertia kernels.
This supplies inertia and material transport, not a loaded operating point,
elastic law, time integrator or experimentally qualified frequency bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_se3 import (
    _material_fraction,
    _relative_maps,
    _rigid_pose,
    _SectionVelocityKinematics,
    log_pose,
    twist_ad,
)
from ._shaft_spatial_element import tip_spatial_inertia
from ._validation import require_finite_float
from .types import ComponentMassProperties


@dataclass(frozen=True)
class InertiaSample:
    """Integrated physical mass at a declared reference-section fraction.

    Body mass [kg] and COM rotary inertia [kg m²] already include the quadrature
    length/weight. COM offset [m] is from the interpolated section origin in
    its material axes. Do not add segment centerline spread to the section's
    COM inertia: its spatial distribution is represented by sample locations.
    The frame ID identifies the material-axis convention, not observer axes.
    Sampling accuracy and source provenance remain separate obligations.
    """

    fraction: float
    body: ComponentMassProperties

    def __post_init__(self) -> None:
        object.__setattr__(self, "fraction", _material_fraction(self.fraction))
        if not isinstance(self.body, ComponentMassProperties):
            raise TypeError("sample body must be ComponentMassProperties")


@dataclass(frozen=True)
class SectionKinetics:
    """Fresh material-coordinate inertia snapshot in linear-first SI ordering.

    The nodal inertial wrench is M*a + bias for nodal material accelerations a.
    Along nodal material velocity v, v.T*bias = v.T*M_dot*v/2. Bias alone is
    not generally zero-work; geometric energy exchange is not dissipation.
    Rank, stability and quadrature accuracy are not certified by this record.
    """

    mass: np.ndarray
    bias: np.ndarray
    mass_rate: np.ndarray
    kinetic_energy_j: float


def _sample_kinetics(
    sample: InertiaSample,
    kinematics: _SectionVelocityKinematics,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping, rate = kinematics.at(sample.fraction)
    inertia = tip_spatial_inertia(sample.body)
    motion = mapping @ velocity
    mass = mapping.T @ inertia @ mapping
    mass_rate = rate.T @ inertia @ mapping + mapping.T @ inertia @ rate
    bias = mapping.T @ (
        inertia @ rate @ velocity - twist_ad(motion).T @ inertia @ motion
    )
    return mass, bias, mass_rate


@dataclass(frozen=True)
class SectionInertia:
    """Mass quadrature using the same objective interpolation as section strain.

    Samples are copied into an immutable tuple and must use one material-axis
    convention. Repeated fractions are permitted for distinct physical pieces.
    No weights, density, section inertia, damping or empirical validity are
    inferred. A sparse/lumped quadrature can be singular; the returned matrix
    is never repaired or promoted to a qualified dynamic model.
    """

    samples: tuple[InertiaSample, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("section inertia must contain at least one sample")
        if any(not isinstance(sample, InertiaSample) for sample in samples):
            raise TypeError("samples must be InertiaSample records")
        first_body = samples[0].body
        for sample in samples:
            body = sample.body
            if body.frame_id != first_body.frame_id:
                raise ValueError("sample material frames must agree")
        object.__setattr__(self, "samples", samples)

    def evaluate(self, poses: object, nodal_velocity: object) -> SectionKinetics:
        """Return inertia, bias and mass-rate at two proper common-frame poses.

        Twelve nodal material velocity entries are [v_left,omega_left,v_right,
        omega_right] in m/s and rad/s. These are physical material twists, not
        unconverted finite rotation-vector rates. Inputs are not modified.
        """
        current = finite_array(poses, (2, 4, 4), "section poses")
        left, right = _rigid_pose(current[0]), _rigid_pose(current[1])
        relative = log_pose(np.linalg.solve(left, right))
        velocity = finite_array(nodal_velocity, (12,), "nodal velocity")
        relative_map, _, _ = _relative_maps(relative)
        relative_rate = relative_map @ velocity
        kinematics = _SectionVelocityKinematics(relative, relative_rate)
        mass, bias, mass_rate = np.zeros((12, 12)), np.zeros(12), np.zeros((12, 12))
        with np.errstate(over="ignore", invalid="ignore"):
            for sample in self.samples:
                sample_mass, sample_bias, sample_rate = _sample_kinetics(
                    sample, kinematics, velocity
                )
                mass += sample_mass
                bias += sample_bias
                mass_rate += sample_rate
            energy = float(velocity @ mass @ velocity / 2)
        return SectionKinetics(
            finite_array(mass, (12, 12), "section mass"),
            finite_array(bias, (12,), "section inertia bias"),
            finite_array(mass_rate, (12, 12), "section mass rate"),
            float(require_finite_float(energy, "section kinetic energy")),
        )


__all__ = ()

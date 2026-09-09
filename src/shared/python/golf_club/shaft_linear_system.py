"""Stationary full-shaft assembly for the rotating-model reference limits."""

from __future__ import annotations

import numpy as np

from ._shaft_linear_contracts import (
    ShaftAttachments,
    ShaftRodProperties,
    _LinearOperators,
)
from ._shaft_spatial_element import spatial_element, tip_spatial_inertia
from .shaft_dynamics import ShaftModalSettings

_DEFAULT_SETTINGS = ShaftModalSettings()
_NO_ATTACHMENTS = ShaftAttachments()


def _element_length(span: float, count: int) -> float:
    length = span / count
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        cube = np.float64(length) ** 3
    if not np.isfinite(cube) or cube <= 0:
        raise ValueError("element length must have a representable positive cube")
    return length


def _validate_matrices(matrices: tuple[np.ndarray, ...]) -> None:
    if not all(np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("shaft assembly must produce finite matrices")
    try:
        np.linalg.cholesky(matrices[0])
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "assembled mass must be numerically positive definite"
        ) from error


def _validate(
    rod: ShaftRodProperties, settings: ShaftModalSettings, attachments: ShaftAttachments
) -> None:
    if not isinstance(rod, ShaftRodProperties):
        raise TypeError("rod must be ShaftRodProperties")
    if not isinstance(settings, ShaftModalSettings):
        raise TypeError("settings must be ShaftModalSettings")
    if not isinstance(attachments, ShaftAttachments):
        raise TypeError("attachments must be ShaftAttachments")
    for attachment in (attachments.tip_body, attachments.grip):
        if attachment is not None and attachment.frame_id != rod.profile.frame_id:
            raise ValueError("attachments must use the shaft frame")


def _attach(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray], attachments: ShaftAttachments
) -> None:
    mass, stiffness, damping = matrices
    if attachments.tip_body is not None:
        mass[-6:, -6:] += tip_spatial_inertia(attachments.tip_body)
    if attachments.grip is not None:
        grip = attachments.grip
        for matrix, factor in (
            (mass, grip.inertance_factor),
            (stiffness, grip.stiffness_factor),
            (damping, grip.damping_factor),
        ):
            array = np.asarray(factor)
            matrix[:6, :6] += array.T @ array


def assemble_shaft_linear_system(
    rod: ShaftRodProperties,
    settings: ShaftModalSettings = _DEFAULT_SETTINGS,
    attachments: ShaftAttachments = _NO_ATTACHMENTS,
) -> _LinearOperators:
    """Assemble finite stationary M, K, C in SI six-axis nodal coordinates.

    Preconditions: explicit positive EA/polar properties, validated profile,
    common attachment frame. No load, rotation or boundary condition is inferred.
    Postconditions: independent finite matrices; symmetric constitutive assembly
    and consistent positive mass. Scope and source records accompany the result.
    Profile damping ratios are not converted to an arbitrary damping matrix.
    """
    _validate(rod, settings, attachments)
    count = settings.element_count
    profile = rod.profile
    length = _element_length(profile.flexible_length_m, count)
    positions = np.linspace(0, profile.flexible_length_m, count + 1)
    mass, stiffness, damping = (np.zeros((6 * (count + 1),) * 2) for _ in range(3))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for element in range(count):
            midpoint = profile.butt_trim_m + (element + 0.5) * length
            local_stiffness, local_mass = spatial_element(rod, midpoint, length)
            indices = slice(6 * element, 6 * element + 12)
            stiffness[indices, indices] += local_stiffness
            mass[indices, indices] += local_mass
        _attach((mass, stiffness, damping), attachments)
    _validate_matrices((mass, stiffness, damping))
    return _LinearOperators(
        profile.frame_id,
        profile.shaft_id,
        tuple(float(s) for s in positions),
        mass,
        stiffness,
        damping,
        profile.provenance,
        rod.provenance,
        attachments,
    )


__all__ = ["ShaftRodProperties", "ShaftAttachments", "assemble_shaft_linear_system"]

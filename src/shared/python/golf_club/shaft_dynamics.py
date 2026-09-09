"""Auditable modal finite-element reference for flexible golf shafts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._beam_fem import assemble_bending_axis, generalized_eigenvalues
from .shaft_profile import ShaftProfile

_MODEL_NAME = "euler_bernoulli_bending_fem/1"
_ASSUMPTIONS = (
    "small transverse deflection",
    "Euler-Bernoulli bending without shear deformation or rotary inertia",
    "clamped trimmed butt and free exposed tip",
    "consistent distributed shaft mass without head or grip coupling",
    "undamped natural frequencies; stored damping is not applied",
)


@dataclass(frozen=True)
class ShaftModalSettings:
    """Validated spatial resolution and retained-mode count."""

    element_count: int = 16
    mode_count: int = 3

    def __post_init__(self) -> None:
        for name in ("element_count", "mode_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.element_count < 2:
            raise ValueError("element_count must be >= 2")
        if self.mode_count < 1:
            raise ValueError("mode_count must be >= 1")
        if self.mode_count > 2 * self.element_count:
            raise ValueError(
                "mode_count exceeds the available bending degrees of freedom"
            )


_DEFAULT_MODAL_SETTINGS = ShaftModalSettings()


@dataclass(frozen=True)
class ShaftModalResponse:
    """Undamped natural frequencies for both transverse shaft-frame axes.

    ``frequencies_x_hz`` describes deflection along shaft-frame x and
    therefore uses bending stiffness about y. ``frequencies_y_hz`` uses
    bending stiffness about x.
    """

    frequencies_x_hz: tuple[float, ...]
    frequencies_y_hz: tuple[float, ...]
    flexible_length_m: float
    element_count: int
    model_name: str = _MODEL_NAME
    assumptions: tuple[str, ...] = _ASSUMPTIONS

    def __post_init__(self) -> None:
        if self.element_count < 2:
            raise ValueError("element_count must be >= 2")
        if not math.isfinite(self.flexible_length_m) or self.flexible_length_m <= 0:
            raise ValueError("flexible_length_m must be finite and > 0")
        for name in ("frequencies_x_hz", "frequencies_y_hz"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{name} must contain finite positive values")
            if values != tuple(sorted(values)):
                raise ValueError(f"{name} must be sorted")


def solve_shaft_bending_modes(
    profile: ShaftProfile,
    settings: ShaftModalSettings = _DEFAULT_MODAL_SETTINGS,
) -> ShaftModalResponse:
    """Solve two uncoupled Euler-Bernoulli bending eigenproblems.

    Station properties are interpolated at element midpoints. A standard
    two-node cubic beam element and its consistent mass matrix are assembled
    in SI units. This is a verification-oriented linear reference, not a
    nonlinear swing transient or a calibrated composite-laminate model.
    """
    if not isinstance(profile, ShaftProfile):
        raise TypeError("profile must be ShaftProfile")
    if not isinstance(settings, ShaftModalSettings):
        raise TypeError("settings must be ShaftModalSettings")
    frequencies_x = _solve_axis(
        profile,
        settings,
        stiffness_name="ei_about_y_n_m2",
    )
    frequencies_y = _solve_axis(
        profile,
        settings,
        stiffness_name="ei_about_x_n_m2",
    )
    return ShaftModalResponse(
        frequencies_x_hz=frequencies_x,
        frequencies_y_hz=frequencies_y,
        flexible_length_m=profile.flexible_length_m,
        element_count=settings.element_count,
    )


def _solve_axis(
    profile: ShaftProfile,
    settings: ShaftModalSettings,
    *,
    stiffness_name: str,
) -> tuple[float, ...]:
    stiffness, mass = assemble_bending_axis(
        profile, settings.element_count, stiffness_name
    )
    eigenvalues = generalized_eigenvalues(stiffness[2:, 2:], mass[2:, 2:])
    positive = eigenvalues[eigenvalues > np.finfo(float).eps]
    if len(positive) < settings.mode_count:
        raise RuntimeError("modal solve did not return enough positive eigenvalues")
    return tuple(
        float(math.sqrt(value) / (2.0 * math.pi))
        for value in positive[: settings.mode_count]
    )


__all__ = [
    "ShaftModalResponse",
    "ShaftModalSettings",
    "solve_shaft_bending_modes",
]

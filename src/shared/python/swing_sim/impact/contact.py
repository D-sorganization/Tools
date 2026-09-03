"""Canonical unilateral Kelvin-Voigt contact-force law."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KelvinVoigtContactLaw:
    """Linear spring and dashpot acting only while surfaces overlap."""

    stiffness_n_per_m: float
    damping_n_s_per_m: float
    maximum_force_n: float = 1.0e7

    def __post_init__(self) -> None:
        for name in (
            "stiffness_n_per_m",
            "damping_n_s_per_m",
            "maximum_force_n",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.stiffness_n_per_m <= 0.0:
            raise ValueError("stiffness_n_per_m must be positive")
        if self.damping_n_s_per_m < 0.0:
            raise ValueError("damping_n_s_per_m must be non-negative")
        if self.maximum_force_n <= 0.0:
            raise ValueError("maximum_force_n must be positive")

    @classmethod
    def from_restitution(
        cls,
        stiffness_n_per_m: float,
        restitution: float,
        effective_mass_kg: float,
        maximum_force_n: float = 1.0e7,
    ) -> KelvinVoigtContactLaw:
        """Construct damping from the linear-oscillator restitution limit."""
        if not 0.0 < restitution <= 1.0:
            raise ValueError("restitution must be in (0, 1]")
        if not math.isfinite(effective_mass_kg) or effective_mass_kg <= 0.0:
            raise ValueError("effective_mass_kg must be finite and positive")
        log_e = math.log(restitution)
        damping_ratio = -log_e / math.sqrt(math.pi**2 + log_e**2)
        damping = 2.0 * damping_ratio * math.sqrt(stiffness_n_per_m * effective_mass_kg)
        return cls(stiffness_n_per_m, damping, maximum_force_n)

    def normal_force(self, compression_m: float, compression_rate_mps: float) -> float:
        """Return compressive normal force for the current overlap state."""
        if not math.isfinite(compression_m) or not math.isfinite(compression_rate_mps):
            raise ValueError("contact state must be finite")
        if compression_m <= 0.0:
            return 0.0
        force = (
            self.stiffness_n_per_m * compression_m
            + self.damping_n_s_per_m * compression_rate_mps
        )
        return max(0.0, min(float(force), self.maximum_force_n))


__all__ = ["KelvinVoigtContactLaw"]

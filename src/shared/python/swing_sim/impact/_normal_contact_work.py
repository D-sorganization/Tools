"""Private normal-contact work port with explicit unilateral-cutoff losses."""

from __future__ import annotations

from dataclasses import dataclass

from ...golf_club._validation import require_finite_float
from .contact import KelvinVoigtContactLaw


def _normal_loss_rates(
    law: KelvinVoigtContactLaw, compression: float, rate: float, force: float
) -> tuple[float, float]:
    """Reuse the legacy disjoint loss algebra for already validated states."""
    active = compression > 0 and force > 0
    viscous = law.damping_n_s_per_m * rate**2 if active else 0.0
    cutoff = (
        -law.stiffness_n_per_m * compression * rate
        if compression > 0 and not active
        else 0.0
    )
    return viscous, cutoff


@dataclass(frozen=True)
class NormalContactWork:
    """Instantaneous SI ledger; input power is positive into the contact.

    Cutoff power accounts for the unilateral model's storage removal, not
    independently measured material damping or acoustic radiation. Finite
    closure is reported separately and is never added to physical loss.
    """

    force_n: float
    elastic_energy_j: float
    elastic_power_w: float
    viscous_power_w: float
    cutoff_power_w: float
    input_power_w: float

    def __post_init__(self) -> None:
        nonnegative = (
            "force_n",
            "elastic_energy_j",
            "viscous_power_w",
            "cutoff_power_w",
        )
        for name in (*nonnegative, "elastic_power_w", "input_power_w"):
            value = require_finite_float(getattr(self, name), name)
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        require_finite_float(self.power_residual_w, "normal contact power residual")

    @property
    def power_residual_w(self) -> float:
        return (
            self.input_power_w
            - self.elastic_power_w
            - self.viscous_power_w
            - self.cutoff_power_w
        )


def normal_contact_work(
    law: KelvinVoigtContactLaw, compression_m: object, compression_rate_mps: object
) -> NormalContactWork:
    """Evaluate the canonical law without silently accepting a force ceiling.

    Nonpositive compression is clearance, retaining the legacy zero-force
    value at first touch. Its nonzero dashpot right-hand limit must be handled
    by an event/peak calculation. No restitution calibration, friction,
    trajectory resolution or physical qualification follows from this port.
    """
    if not isinstance(law, KelvinVoigtContactLaw):
        raise TypeError("law must be KelvinVoigtContactLaw")
    compression = require_finite_float(compression_m, "compression_m")
    rate = require_finite_float(compression_rate_mps, "compression_rate_mps")
    if compression <= 0:
        return NormalContactWork(0, 0, 0, 0, 0, 0)
    raw = require_finite_float(
        law.unclipped_normal_force(compression, rate), "unclipped normal force"
    )
    if raw > law.maximum_force_n:
        raise ValueError("normal contact force ceiling invalidates this work model")
    force = law.normal_force(compression, rate)
    try:
        viscous, cutoff = _normal_loss_rates(law, compression, rate, force)
        stiffness = law.stiffness_n_per_m
        return NormalContactWork(
            force,
            0.5 * stiffness * compression * compression,
            stiffness * compression * rate,
            viscous,
            cutoff,
            force * rate,
        )
    except OverflowError as error:
        raise ValueError("normal contact work must be finite") from error


__all__ = ()

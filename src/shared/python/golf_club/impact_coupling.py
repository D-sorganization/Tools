"""Coupled ball–head–hands transient impact model (heavy-hit H1, #4563).

Quantifies the question the heavy-hit epic asks: during the ~500 µs of
club–ball contact, how much can the golfer's hands and body change the
impact? The chain, along the hit direction:

    ball ←KV contact (k_c, c_c)→ head ←shaft (k_s, c_s)→ hands ←grip (k_g, c_g)→ body

**Frame.** The simulation runs in the *body frame*: the grip anchor is
fixed, the head and hands start at rest, and the ball approaches at the
declared head speed. This is Galilean-equivalent to the lab picture and
makes energy accounting exact — the fixed anchor does no work, so with
zero damping the initial ball kinetic energy is conserved across the
reported components (a test gate). Reported ball speed is converted back
to the lab frame.

**Upper-bound semantics.** At contact timescales a real shaft transmits
force through its local impedance; any lumped ``k_s`` is an
approximation. Callers therefore sweep ``k_s`` up to a rigid-link bound —
"even a perfectly rigid shaft changes ball speed by X%" — with the static
tip stiffness (``solve_cantilever_tip_response``) as the realistic low
end. Reality lies below the rigid bound.

**Consistency, not coincidence.** The contact spring/damper defaults are
the impact package's own Kelvin-Voigt parameters, and the detached limit
(``k_s = 0``) is gated against :class:`SpringDamperImpactModel`'s ball
exit speed for identical contact parameters.

Integration is semi-implicit Euler at the impact model's ``dt = 1e-7 s``
from first contact until the contact force releases; the contact force is
clamped non-adhesive (``F_c ≥ 0``), matching the shipped model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

from shared.python.contracts import require
from shared.python.swing_sim.impact.constants import GOLF_BALL_MASS_KG

from ._validation import require_finite_float, require_identifier

_DEFAULT_CONTACT_STIFFNESS_N_M = 1.0e6
_DEFAULT_CONTACT_DAMPING_N_S_M = 1.0e3
_DEFAULT_DT_S = 1.0e-7
_DEFAULT_MAX_TIME_S = 0.005
IMPACT_COUPLING_REPORT_FORMAT = "golf_club.impact_coupling_report/1"

__all__ = [
    "IMPACT_COUPLING_REPORT_FORMAT",
    "CoupledImpactConfig",
    "CoupledImpactResult",
    "GripBoundary",
    "impact_coupling_report",
    "simulate_coupled_impact",
]


@dataclass(frozen=True)
class GripBoundary:
    """The hand-side boundary as the club sees it, with provenance.

    Attributes:
        effective_mass_kg: Hands + forearm mass moving with the grip (> 0).
        stiffness_n_m: Grip-to-body stiffness (>= 0; 0 = free hands).
        damping_n_s_m: Grip-to-body damping (>= 0).
        provenance: Where these numbers came from (literature, an H2
            engine-model reduction, or a measurement).
    """

    effective_mass_kg: float
    stiffness_n_m: float
    damping_n_s_m: float
    provenance: str

    def __post_init__(self) -> None:
        for name in ("effective_mass_kg", "stiffness_n_m", "damping_n_s_m"):
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )
        if self.effective_mass_kg <= 0.0:
            raise ValueError("effective_mass_kg must be > 0")
        if self.stiffness_n_m < 0.0 or self.damping_n_s_m < 0.0:
            raise ValueError("stiffness and damping must be >= 0")
        object.__setattr__(
            self, "provenance", require_identifier(self.provenance, "provenance")
        )


@dataclass(frozen=True)
class CoupledImpactConfig:
    """One coupled-impact scenario, SI, hit-direction 1-D."""

    head_mass_kg: float
    head_speed_mps: float
    shaft_stiffness_n_m: float
    grip: GripBoundary
    shaft_damping_n_s_m: float = 0.0
    ball_mass_kg: float = GOLF_BALL_MASS_KG
    contact_stiffness_n_m: float = _DEFAULT_CONTACT_STIFFNESS_N_M
    contact_damping_n_s_m: float = _DEFAULT_CONTACT_DAMPING_N_S_M
    dt_s: float = _DEFAULT_DT_S
    max_time_s: float = _DEFAULT_MAX_TIME_S

    def __post_init__(self) -> None:
        if not isinstance(self.grip, GripBoundary):
            raise TypeError("grip must be GripBoundary")
        for name in (
            "head_mass_kg",
            "head_speed_mps",
            "shaft_stiffness_n_m",
            "shaft_damping_n_s_m",
            "ball_mass_kg",
            "contact_stiffness_n_m",
            "contact_damping_n_s_m",
            "dt_s",
            "max_time_s",
        ):
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )
        if self.head_mass_kg <= 0.0 or self.ball_mass_kg <= 0.0:
            raise ValueError("masses must be > 0")
        if self.head_speed_mps <= 0.0:
            raise ValueError("head_speed_mps must be > 0")
        if self.shaft_stiffness_n_m < 0.0 or self.shaft_damping_n_s_m < 0.0:
            raise ValueError("shaft stiffness and damping must be >= 0")
        if self.contact_stiffness_n_m <= 0.0 or self.contact_damping_n_s_m < 0.0:
            raise ValueError("contact stiffness must be > 0, damping >= 0")
        if self.dt_s <= 0.0 or self.max_time_s <= self.dt_s:
            raise ValueError("dt_s must be > 0 and max_time_s > dt_s")


@dataclass(frozen=True)
class CoupledImpactResult:
    """Coupled outcome plus the internally computed free-head reference.

    Energy fields are in the simulation (body) frame, where the fixed grip
    anchor does no work; ``ball_speed_mps`` is the lab-frame exit speed.
    """

    ball_speed_mps: float
    free_head_ball_speed_mps: float
    decoupling_fraction: float
    contact_time_s: float
    peak_contact_force_n: float
    ball_kinetic_energy_j: float
    head_kinetic_energy_j: float
    grip_side_kinetic_energy_j: float
    stored_spring_energy_j: float
    energy_balance_fraction: float
    grip_provenance: str


def _integrate(config: CoupledImpactConfig, *, coupled: bool) -> tuple[float, ...]:
    """Body-frame semi-implicit transient; returns terminal state summary."""
    m_b = config.ball_mass_kg
    m_h = config.head_mass_kg
    m_g = config.grip.effective_mass_kg
    k_c, c_c = config.contact_stiffness_n_m, config.contact_damping_n_s_m
    k_s = config.shaft_stiffness_n_m if coupled else 0.0
    c_s = config.shaft_damping_n_s_m if coupled else 0.0
    k_g, c_g = config.grip.stiffness_n_m, config.grip.damping_n_s_m
    dt = config.dt_s

    # Body frame: ball approaches at -v0 toward the head face at x = 0.
    v0 = config.head_speed_mps
    x_b, v_b = 0.0, -v0
    x_h, v_h = 0.0, 0.0
    x_g, v_g = 0.0, 0.0

    time_s = 0.0
    peak_force = 0.0
    was_in_contact = True  # touching at t = 0
    while time_s < config.max_time_s:
        # Ball approaches from +x; overlap grows as it moves left of the face.
        overlap = x_h - x_b
        compression_rate = v_h - v_b
        if overlap > 0.0:
            force_contact = max(k_c * overlap + c_c * compression_rate, 0.0)
        else:
            force_contact = 0.0
            if was_in_contact and time_s > 0.0:
                break
        was_in_contact = overlap > 0.0
        peak_force = max(peak_force, force_contact)

        # Contact pushes the ball toward +x and recoils the head toward -x.
        force_shaft = k_s * (x_g - x_h) + c_s * (v_g - v_h)
        force_grip = -k_g * x_g - c_g * v_g

        a_b = force_contact / m_b
        a_h = (-force_contact + force_shaft) / m_h
        a_g = (-force_shaft + force_grip) / m_g

        v_b += a_b * dt
        v_h += a_h * dt
        v_g += a_g * dt
        x_b += v_b * dt
        x_h += v_h * dt
        x_g += v_g * dt
        time_s += dt

    spring_energy = 0.5 * k_s * (x_g - x_h) ** 2 + 0.5 * k_g * x_g**2
    return (v_b, v_h, v_g, time_s, peak_force, spring_energy)


def simulate_coupled_impact(config: CoupledImpactConfig) -> CoupledImpactResult:
    """Run the coupled transient and its free-head reference.

    Raises:
        TypeError: If ``config`` is not a :class:`CoupledImpactConfig`.
    """
    if not isinstance(config, CoupledImpactConfig):
        raise TypeError("config must be CoupledImpactConfig")

    v_b, v_h, v_g, contact_time, peak_force, spring_energy = _integrate(
        config, coupled=True
    )
    free_v_b = _integrate(config, coupled=False)[0]

    v0 = config.head_speed_mps
    ball_speed_lab = abs(v_b + v0)
    free_speed_lab = abs(free_v_b + v0)
    influence = abs(ball_speed_lab - free_speed_lab) / free_speed_lab
    decoupling = max(0.0, min(1.0, 1.0 - influence))

    initial_energy = 0.5 * config.ball_mass_kg * v0**2
    ball_ke = 0.5 * config.ball_mass_kg * v_b**2
    head_ke = 0.5 * config.head_mass_kg * v_h**2
    grip_ke = 0.5 * config.grip.effective_mass_kg * v_g**2
    total = ball_ke + head_ke + grip_ke + spring_energy
    balance = total / initial_energy if initial_energy > 0.0 else math.nan

    return CoupledImpactResult(
        ball_speed_mps=ball_speed_lab,
        free_head_ball_speed_mps=free_speed_lab,
        decoupling_fraction=decoupling,
        contact_time_s=contact_time,
        peak_contact_force_n=peak_force,
        ball_kinetic_energy_j=ball_ke,
        head_kinetic_energy_j=head_ke,
        grip_side_kinetic_energy_j=grip_ke,
        stored_spring_energy_j=spring_energy,
        energy_balance_fraction=balance,
        grip_provenance=config.grip.provenance,
    )


def _result_payload(result: CoupledImpactResult) -> dict[str, float | str]:
    return {
        "ball_speed_mps": result.ball_speed_mps,
        "free_head_ball_speed_mps": result.free_head_ball_speed_mps,
        "decoupling_fraction": result.decoupling_fraction,
        "contact_time_s": result.contact_time_s,
        "peak_contact_force_n": result.peak_contact_force_n,
        "grip_provenance": result.grip_provenance,
    }


def impact_coupling_report(
    baseline: CoupledImpactConfig,
    *,
    grip_stiffness_grid_n_m: tuple[float, ...],
    grip_mass_grid_kg: tuple[float, ...],
    shaft_stiffness_grid_n_m: tuple[float, ...],
) -> str:
    """Counterfactual sweep -> deterministic JSON report (H3, #4565).

    Evaluates the baseline plus one counterfactual per grid value (one
    axis varied at a time, the others held at baseline), reporting each
    outcome and its decoupling fraction. Serialization is sorted-keys
    deterministic: identical inputs produce byte-identical reports, and
    the grip provenance travels with every row so a report names the
    engine model it came from.

    Raises:
        TypeError: If ``baseline`` is not a :class:`CoupledImpactConfig`.
        PreconditionError: If any grid is empty or holds non-finite or
            negative values (mass grid must be strictly positive).
    """
    if not isinstance(baseline, CoupledImpactConfig):
        raise TypeError("baseline must be CoupledImpactConfig")
    for grid, name, strict in (
        (grip_stiffness_grid_n_m, "grip_stiffness_grid_n_m", False),
        (grip_mass_grid_kg, "grip_mass_grid_kg", True),
        (shaft_stiffness_grid_n_m, "shaft_stiffness_grid_n_m", False),
    ):
        require(
            isinstance(grid, tuple) and len(grid) >= 1,
            f"{name} must be a nonempty tuple",
        )
        for value in grid:
            require(
                isinstance(value, (float, int))
                and math.isfinite(value)
                and (float(value) > 0.0 if strict else float(value) >= 0.0),
                f"{name} values must be finite and {'>' if strict else '>='} 0",
            )

    counterfactuals: list[dict[str, object]] = []
    for stiffness in grip_stiffness_grid_n_m:
        config = replace(
            baseline, grip=replace(baseline.grip, stiffness_n_m=float(stiffness))
        )
        counterfactuals.append(
            {
                "axis": "grip_stiffness_n_m",
                "value": float(stiffness),
                **_result_payload(simulate_coupled_impact(config)),
            }
        )
    for mass in grip_mass_grid_kg:
        config = replace(
            baseline, grip=replace(baseline.grip, effective_mass_kg=float(mass))
        )
        counterfactuals.append(
            {
                "axis": "grip_mass_kg",
                "value": float(mass),
                **_result_payload(simulate_coupled_impact(config)),
            }
        )
    for stiffness in shaft_stiffness_grid_n_m:
        config = replace(baseline, shaft_stiffness_n_m=float(stiffness))
        counterfactuals.append(
            {
                "axis": "shaft_stiffness_n_m",
                "value": float(stiffness),
                **_result_payload(simulate_coupled_impact(config)),
            }
        )

    payload = {
        "format": IMPACT_COUPLING_REPORT_FORMAT,
        "baseline": _result_payload(simulate_coupled_impact(baseline)),
        "counterfactuals": counterfactuals,
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)

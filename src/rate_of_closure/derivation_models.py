"""Sectioned full-model derivations for the Calculation Description tab.

V4 of epic #4120: the tab covers EVERY model in use, not just the
closure chain. This module assembles the sections from the per-domain
content modules (kept separate to honor the 500-LOC file budget):

- ``closure``  — the existing impact-point kinematics chain
  (:func:`rate_of_closure.derivation.derivation_steps`, always shown);
- ``impact``   — impulse-momentum with COR, MOI-tensor effective mass,
  the 2/7 friction cap, D-plane, gear effect
  (:mod:`rate_of_closure.derivation_impact`, gear step conditional);
- ``flight``   — flight EOM + the ACTIVE literature model's coefficient
  law with its citation (:mod:`rate_of_closure.derivation_flight`);
- ``swing``    — pendulum Lagrangian, mass matrix, Coriolis, plane-tilt
  gravity (:mod:`rate_of_closure.derivation_swing`), present only when
  a pendulum swing source is selected; the triple-pendulum step appears
  only for the triple source.

Sections render conditionally per the live configuration, mirrored by
``web/src/model/derivationModels.ts`` (section keys parity-tested).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._contracts import ensure
from .derivation import DerivationStep, derivation_steps
from .derivation_flight import flight_steps
from .derivation_impact import impact_steps
from .derivation_swing import swing_steps
from .model import ImpactScenario

__all__ = ["DerivationConfig", "DerivationSection", "derivation_sections"]


@dataclass(frozen=True)
class DerivationConfig:
    """The live configuration that selects which sections render.

    Attributes:
        flight_model: Registry key of the active flight model.
        swing_source: ``"manual"`` | ``"double_pendulum"`` |
            ``"triple_pendulum"`` (Simulation tab swing source).
        gear_effect: Whether the gear-effect step is included — the
            session pipeline always applies gear effect today, so the
            default is True; the flag is the configuration seam.
        plane_tilts_deg: Live ``(yaw, side, forward)`` swing-plane
            tilts substituted into the gravity-projection step.
    """

    flight_model: str = "waterloo_penner"
    swing_source: str = "manual"
    gear_effect: bool = True
    plane_tilts_deg: tuple[float, float, float] = (0.0, -45.0, 0.0)


@dataclass(frozen=True)
class DerivationSection:
    """One titled section of the Calculation Description tab."""

    key: str
    title: str
    intro: str
    steps: tuple[DerivationStep, ...]


def derivation_sections(
    scenario: ImpactScenario, config: DerivationConfig | None = None
) -> tuple[DerivationSection, ...]:
    """Every derivation section active under ``config``.

    Args:
        scenario: The live delivery (substitutes the closure and impact
            sections).
        config: Section-selection configuration; defaults match the
            Simulation tab's defaults.

    Returns:
        Ordered sections; the swing section appears only for pendulum
        sources.
    """
    cfg = config or DerivationConfig()
    sections = [
        DerivationSection(
            key="closure",
            title="Closure Chain — Impact-Point Kinematics",
            intro=(
                "The original derivation: from the frame conventions to "
                "the reported impact-point deviations and closure "
                "metrics, with the live scenario substituted."
            ),
            steps=derivation_steps(scenario),
        ),
        DerivationSection(
            key="impact",
            title="Impact Model — Impulse-Momentum With COR",
            intro=(
                "How ball speed and spin come out of the strike: the "
                "rigid-body impulse solve of swing_sim.impact, including "
                "the MOI-tensor effective mass and the friction spin cap."
            ),
            steps=impact_steps(scenario, gear_effect=cfg.gear_effect),
        ),
        DerivationSection(
            key="flight",
            title="Ball Flight — Aerodynamic Integration",
            intro=(
                "The trajectory ODE and the selected literature model's "
                "coefficient law; switching the flight model in the "
                "Simulation tab rewrites this section."
            ),
            steps=flight_steps(cfg.flight_model),
        ),
    ]
    if cfg.swing_source in ("double_pendulum", "triple_pendulum"):
        sections.append(
            DerivationSection(
                key="swing",
                title="Swing Model — Pendulum Dynamics",
                intro=(
                    "The pendulum swing source generating the delivery: "
                    "Lagrangian equations of motion in the tilted swing "
                    "plane, with the live plane tilts substituted."
                ),
                steps=swing_steps(cfg.swing_source, cfg.plane_tilts_deg),
            )
        )
    result = tuple(sections)
    ensure(
        [section.key for section in result][:3] == ["closure", "impact", "flight"],
        "core sections must always render",
    )
    return result

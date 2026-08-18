"""Swing-model derivation content for the Calculation Description tab.

Sectioned V4 coverage (#4120): the double-pendulum Lagrangian equations
of motion (mass matrix, Coriolis/centripetal, gravity), the swing-plane
gravity projection, and the triple-pendulum extension — sourced from
:mod:`swing_sim.reference` (the parity oracle for the Rust kernel) and
:mod:`rate_of_closure.simulation.sources`. Substitutes the live plane
tilts so the gravity-projection line tracks the Simulation tab.
"""

from __future__ import annotations

import math

from shared.python.swing_sim.reference import in_plane_gravity_from_tilts

from ._contracts import ensure
from .derivation import DerivationStep

__all__ = ["swing_steps"]

_G = 9.81


def swing_steps(
    swing_source: str,
    plane_tilts_deg: tuple[float, float, float] = (0.0, -45.0, 0.0),
) -> tuple[DerivationStep, ...]:
    """Pendulum swing-model derivation steps for the selected source.

    Args:
        swing_source: ``"double_pendulum"`` or ``"triple_pendulum"``
            (the manual constant-twist source has no pendulum section).
        plane_tilts_deg: Live ``(yaw, side, forward)`` plane tilts from
            the Simulation tab, substituted into the gravity projection.

    Returns:
        Ordered steps: Lagrangian EOM, mass matrix, Coriolis terms,
        plane-tilt gravity projection, and — when the triple pendulum
        is selected — its extension note.
    """
    yaw, side, fwd = plane_tilts_deg
    gx, gy = in_plane_gravity_from_tilts(
        math.radians(yaw), math.radians(side), math.radians(fwd), _G
    )

    steps = [
        DerivationStep(
            title="Lagrangian Equations of Motion",
            latex=(
                r"$M(\theta)\,\ddot{\theta} + C(\theta, \dot{\theta}) "
                r"+ G(\theta) + D\,\dot{\theta} = 0$"
            ),
            values=(
                r"$\theta = (\theta_1, \theta_2):\ \mathrm{arm\ and\ club}"
                r"\ \mathrm{links\ in\ the\ inclined\ swing\ plane}$"
            ),
            narrative=(
                "The double-pendulum swing model treats arms and club as "
                "two rigid links in an inclined plane. Its equations of "
                "motion follow from the Lagrangian: a configuration-"
                "dependent mass matrix, velocity-dependent Coriolis "
                "terms, gravity, and viscous damping "
                "(swing_sim.reference / rust swing-core, integrated with "
                "classical RK4)."
            ),
        ),
        DerivationStep(
            title="Mass Matrix",
            latex=(
                r"$M_{11} = I_1 + I_2 + m_2 l_1^2 "
                r"+ 2 m_2 l_1 l_{c2} \cos\theta_2,\quad "
                r"M_{12} = M_{21} = I_2 + m_2 l_1 l_{c2} \cos\theta_2,"
                r"\quad M_{22} = I_2$"
            ),
            values=(
                r"$\det M > 0\ \mathrm{enforced\ (singular\ mass\ matrix}"
                r"\ \mathrm{rejected\ by\ contract)}$"
            ),
            narrative=(
                "The symmetric 2×2 inertia matrix couples the links "
                "through the wrist angle θ₂: a straighter wrist "
                "(cos θ₂ → 1) maximizes the coupling. The integrator "
                "inverts M each step, with a determinant contract "
                "guarding singular configurations "
                "(swing_sim.reference.mass_matrix)."
            ),
        ),
        DerivationStep(
            title="Coriolis and Centripetal Terms",
            latex=(
                r"$h = -m_2 l_1 l_{c2} \sin\theta_2:\quad "
                r"C_1 = h\,(2\dot{\theta}_1\dot{\theta}_2 "
                r"+ \dot{\theta}_2^2),\quad "
                r"C_2 = -h\,\dot{\theta}_1^2$"
            ),
            values=(
                r"$\mathrm{late\ release:}\ \dot{\theta}_1^2\ "
                r"\mathrm{slings\ the\ club\ through}\ C_2$"
            ),
            narrative=(
                "The velocity-dependent forces of the rotating links: the "
                "centripetal −h·ω₁² term is the physics of the release — "
                "arm rotation slings the club link outward without any "
                "wrist torque (swing_sim.reference.coriolis_vector)."
            ),
        ),
        DerivationStep(
            title="Plane-Tilt Gravity Projection",
            latex=(
                r"$R = R_z(yaw)\,R_x(side)\,R_y(fwd),\qquad "
                r"\vec{g}_{plane} = \left(\vec{g}_{world} \cdot "
                r"\hat{e}_1,\ \vec{g}_{world} \cdot \hat{e}_3\right)$"
            ),
            values=(
                rf"$({yaw:.0f}^\circ,\ {side:.0f}^\circ,\ "
                rf"{fwd:.0f}^\circ) \Rightarrow \vec{{g}}_{{plane}} = "
                rf"({gx:+.2f},\ {gy:+.2f})\ \mathrm{{m/s^2}}$"
            ),
            narrative=(
                "The swing plane is oriented by yaw, side tilt, and "
                "forward tilt; world gravity is projected onto the "
                "plane's in-plane axes and the EOM consumes that "
                "2-vector directly — a steeper plane feels more in-plane "
                "gravity. The numbers substitute the Simulation tab's "
                "live tilts (swing_sim.reference.in_plane_gravity)."
            ),
        ),
    ]

    if swing_source == "triple_pendulum":
        steps.append(
            DerivationStep(
                title="Triple-Pendulum Extension",
                latex=(
                    r"$M(\theta)\,\ddot{\theta} + C(\theta, \dot{\theta})"
                    r" + G(\theta) = 0,\qquad \theta \in \mathbb{R}^3$"
                ),
                values=(
                    r"$\mathrm{links:}\ \mathrm{torso} \to \mathrm{arms}"
                    r" \to \mathrm{club\ (planar,\ same\ formalism)}$"
                ),
                narrative=(
                    "The triple pendulum adds a torso link ahead of arms "
                    "and club, solved with the same mass-matrix formalism "
                    "in the planar frame — a 3×3 M(θ) assembled from the "
                    "link inertias and solved each step "
                    "(rate_of_closure.simulation.sources)."
                ),
            )
        )

    result = tuple(steps)
    ensure(len(result) >= 4, "swing derivation must cover the Lagrangian chain")
    return result

"""Impact-model derivation content for the Calculation Description tab.

Sectioned V4 coverage (#4120): the rigid-body impulse-momentum impact
model, sourced from the derivations in the ``swing_sim.impact``
docstrings — :mod:`swing_sim.impact.models` (COR impulse, MOI-tensor
effective mass, 2/7 rolling friction cap), :mod:`swing_sim.impact.delivery`
(spin loft and the D-plane), and :mod:`swing_sim.impact.gear_effect`
(head-recoil gear spin). Numeric lines substitute the live scenario's
impact offsets and the vendored driver/ball constants.
"""

from __future__ import annotations

from shared.python.swing_sim.impact.constants import (
    DRIVER_CG_DEPTH_M,
    DRIVER_COR,
    DRIVER_MASS_KG,
    DRIVER_MOI_KG_M2,
    GOLF_BALL_MASS_KG,
)

from ._contracts import ensure
from .derivation import DerivationStep
from .model import ImpactScenario

__all__ = ["impact_steps"]


def impact_steps(
    scenario: ImpactScenario, *, gear_effect: bool = True
) -> tuple[DerivationStep, ...]:
    """Impact-model derivation steps for the live scenario.

    Args:
        scenario: The delivery whose offsets substitute into the lines.
        gear_effect: Include the gear-effect recoil derivation (the
            session pipeline always applies it; the flag mirrors the
            configuration seam).

    Returns:
        Ordered steps: impulse-momentum with COR, effective mass with
        the MOI-tensor triple product, the 2/7 friction spin cap, the
        D-plane, and (when enabled) the gear-effect recoil.
    """
    toe_mm = scenario.impact_offset_toe_mm
    high_mm = scenario.impact_offset_high_mm
    offset_mm = (toe_mm**2 + high_mm**2) ** 0.5

    steps = [
        DerivationStep(
            title="Impulse-Momentum Exchange With COR",
            latex=(
                r"$J = \frac{(1 + e)\,v_{rel}}"
                r"{\frac{1}{m_{ball}} + \frac{1}{m_{eff}}},\qquad "
                r"v_{ball} = \frac{J}{m_{ball}}$"
            ),
            values=(
                rf"$e = {DRIVER_COR:.2f}\ \mathrm{{(driver\ COR\ cap)}},\ "
                rf"m_{{ball}} = {GOLF_BALL_MASS_KG * 1000.0:.1f}\ \mathrm{{g}},\ "
                rf"m_{{club}} = {DRIVER_MASS_KG * 1000.0:.0f}\ \mathrm{{g}}$"
            ),
            narrative=(
                "The ball leaves with the momentum delivered by one normal "
                "impulse J over the ~450 µs contact. The coefficient of "
                "restitution e scales the separation speed; the club side "
                "enters through its effective mass, not its full mass "
                "(swing_sim.impact.models rigid-body COR model)."
            ),
        ),
        DerivationStep(
            title="Effective Mass — the MOI-Tensor Triple Product",
            latex=(
                r"$\frac{1}{m_{eff}} = \frac{1}{m_{club}} "
                r"+ (\vec{r} \times \hat{n})^T I^{-1} (\vec{r} \times \hat{n})$"
            ),
            values=(
                rf"$|\vec{{r}}| = {offset_mm:.1f}\ \mathrm{{mm}}\ "
                rf"\mathrm{{(toe\ {toe_mm:+.0f},\ high\ {high_mm:+.0f})}},\ "
                rf"I_{{scalar}} = {DRIVER_MOI_KG_M2 * 1e6:.0f}\ "
                r"\mathrm{g\,cm^2\ fallback:}\ "
                r"\frac{1}{m} + \frac{|\vec{r}|^2}{I}$"
            ),
            narrative=(
                "An off-center strike spends part of the impulse twisting "
                "the head: the exact club-side denominator is the triple "
                "product (r × n)ᵀ I⁻¹ (r × n) with the full 3×3 MOI tensor. "
                "A diagonal tensor I·eye(3) reproduces the scalar fallback "
                "1/m + |r|²/I exactly because r lies in the face plane "
                "(derivation in swing_sim.impact.models docstring)."
            ),
        ),
        DerivationStep(
            title="Friction Spin — the 2/7 Rolling Cap",
            latex=(
                r"$J_f = \min\!\left(\mu J,\ "
                r"\frac{2}{7}\,m_{ball}\,v_t\right),\qquad "
                r"\frac{2}{7} = \frac{1}{1 + \frac{5}{2}}$"
            ),
            values=(
                r"$I_{sphere} = \frac{2}{5} m R^2 \Rightarrow "
                r"J_f\left(\frac{1}{m} + \frac{R^2}{I}\right) = v_t "
                r"\Rightarrow J_f = \frac{2}{7}\,m\,v_t$"
            ),
            narrative=(
                "Friction converts the tangential approach speed into spin "
                "only until the contact point stops sliding (rolling "
                "without slip). For a uniform solid sphere that caps the "
                "friction impulse at (2/7)·m·v_t — beyond it no more spin "
                "is available (Cross 2002; SPHERE_ROLLING_CAP_FACTOR in "
                "swing_sim.impact.models). The physical spin axis is "
                "t × n — the sign fix documented in the port."
            ),
        ),
        DerivationStep(
            title="Spin Loft and the D-Plane",
            latex=(
                r"$\mathrm{spin\ loft} = \arccos(\hat{v} \cdot \hat{n}),"
                r"\qquad \hat{a}_{spin} = \widehat{\hat{v} \times \hat{n}}$"
            ),
            values=(
                r"$\hat{v} = (\cos AoA \cos path,\ \sin AoA,\ "
                r"\cos AoA \sin path),\ \hat{n} = "
                r"(\cos loft \cos face,\ \sin loft,\ \cos loft \sin face)$"
            ),
            narrative=(
                "The D-plane is spanned by the club-path vector and the "
                "delivered face normal: the ball launches close to the "
                "normal and spins about the plane's normal v̂ × n̂, so the "
                "face-minus-path difference tilts the spin axis "
                "(swing_sim.impact.delivery; Jorgensen; TrackMan D-plane "
                "literature)."
            ),
        ),
    ]

    if gear_effect:
        steps.append(
            DerivationStep(
                title="Gear Effect — Head Recoil Times CG Depth",
                latex=(
                    r"$\Delta\vec{\omega}_{head} = I^{-1}"
                    r"\left(\vec{r} \times (-J\hat{n})\right),\qquad "
                    r"\vec{v}_{surf} = \frac{1}{2}\,"
                    r"\Delta\vec{\omega}_{head} \times \vec{r}$"
                ),
                values=(
                    rf"$\vec{{r}} = \vec{{r}}_{{plane}} + d\,\hat{{n}},\quad "
                    rf"d = {DRIVER_CG_DEPTH_M * 1000.0:.0f}\ \mathrm{{mm}}"
                    r"\ \mathrm{(driver\ CG\ depth)}$"
                ),
                narrative=(
                    "The off-center impulse makes the head recoil in "
                    "rotation; because the CG sits a depth d behind the "
                    "face, the rotating face sweeps tangentially under the "
                    "ball (time-averaged at half the final recoil). "
                    "Friction gears the ball against that moving surface — "
                    "toe hits gain draw spin, high hits lose backspin — "
                    "capped by the same 2/7 rolling limit "
                    "(swing_sim.impact.gear_effect derivation)."
                ),
            )
        )

    result = tuple(steps)
    ensure(len(result) >= 4, "impact derivation must cover the model chain")
    return result

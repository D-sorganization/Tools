"""Step-by-step derivation of the impact-point calculation.

Single source of truth for the "Derivation & Traceability" tab: every
step carries the symbolic formula (matplotlib-mathtext-compatible
LaTeX), a plain-language narrative, and the numeric substitution for
the *live* scenario, so a user can follow each number in the results
panel back to the rigid-body kinematics that produced it.

Also owns ``RESULT_EXPLANATIONS`` — the click-through text behind every
result row in both the PyQt6 and web UIs.

Sources: AffineDrift Launch Monitor Technology Review
(``sections/02-parameters.tex`` frame and sign conventions), the
closure-rate derivation (d / R_ISA, deg/ft), and the closure-rate
literature dossier (Cheetham 2014 HTV / CCV figures).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._contracts import ensure
from .model import ImpactResult, ImpactScenario, solve

__all__ = [
    "METRIC_EXPLANATIONS",
    "RESULT_EXPLANATIONS",
    "DerivationStep",
    "derivation_steps",
]


@dataclass(frozen=True)
class DerivationStep:
    """One traceable step of the calculation.

    Attributes:
        title: Title Case step heading.
        latex: Symbolic formula, matplotlib-mathtext compatible.
        values: Numeric substitution for the live scenario (mathtext).
        narrative: Plain-language explanation of the step.
    """

    title: str
    latex: str
    values: str
    narrative: str


#: Click-through explanation for every result row, keyed by field name.
RESULT_EXPLANATIONS: dict[str, str] = {
    "path_deviation_deg": (
        "The horizontal angle between the impact point's velocity and the "
        "geometric center's velocity: atan2(v_z, v_x). Launch monitors "
        "report the GC path; the ball responds to the impact point's path. "
        "Negative = the impact point travels left of the reported path "
        "(standard launch-monitor sign convention: club path positive = "
        "in-to-out). Openly published launch-monitor material puts this "
        "gap near 3 degrees for a driver."
    ),
    "aoa_deviation_deg": (
        "The vertical analogue of the path deviation: atan2(v_y, "
        "sqrt(v_x^2 + v_z^2)). Positive = the impact point is travelling "
        "more upward than the reported delivery (a shallower effective "
        "attack angle). Driven mostly by the swing-plane rotation "
        "component."
    ),
    "tangential_speed_mph": (
        "The magnitude of omega x r: how fast the impact point moves "
        "relative to the geometric center purely because the head is "
        "rotating. For the forum's 35 mm / 2,000 deg/s case this is "
        "1.22 m/s = 2.73 mph — the number that was misread as 1.2 mph."
    ),
    "speed_delta_mph": (
        "How much the rotation changes the impact point's total speed. "
        "Because omega x r is nearly perpendicular to the delivery, it "
        "redirects the point without meaningfully speeding it up or "
        "slowing it down — which is why framing the effect as a percent "
        "of clubhead speed understates it. Direction changes; speed "
        "barely does."
    ),
    "closure_rate_dps": (
        "The vertical component of the angular velocity vector — the "
        "rate the face normal sweeps horizontally (closes). This is the "
        "literature's club closure velocity: CCV = HTV sin(lie) + "
        "SPV cos(lie). Cheetham 2014 tour driver data: HTV 1,307 +/- 304 "
        "deg/s about the shaft (range 652-2,432, n = 94); CCV mean near "
        "2,100 deg/s."
    ),
    "normalized_closure_deg_per_ft": (
        "Closure per foot of travel: omega / v, which equals 1 / R_ISA "
        "(the inverse distance to the instantaneous screw axis). It is "
        "speed-invariant: two deliveries with the same deg/ft have the "
        "same path-gap geometry regardless of clubhead speed, because "
        "the gap between two reference points is d / R_ISA."
    ),
    "closure_during_contact_deg": (
        "Face closure accumulated while the ball is on the face: CCV "
        "times the contact duration (about 450 microseconds for a "
        "driver). The face the ball leaves is not the face it met — "
        "roughly a degree at tour closure rates."
    ),
    "loft_gain_during_contact_deg": (
        "Dynamic loft gained during contact: the heel-toe component of "
        "omega times the contact duration. The swing-plane rotation "
        "keeps adding loft while the ball is on the face."
    ),
}


#: Click-through explanation for every common-literature closure metric.
METRIC_EXPLANATIONS: dict[str, str] = {
    "ccv_dps": (
        "Club closure velocity in degrees per second — the most common "
        "way golf research reports rate of closure. Identical to the "
        "closure rate above: CCV = HTV sin(lie) + SPV cos(lie). Tour "
        "driver mean near 2,100 deg/s (Cheetham 2014 dossier)."
    ),
    "closure_deg_per_ft": (
        "Closure per foot of clubhead travel — the speed-invariant "
        "normalization preferred in the AffineDrift derivation "
        "(omega / v = 1 / R_ISA). Two deliveries with the same deg/ft "
        "have identical path-gap geometry at any speed."
    ),
    "closure_deg_per_inch": (
        "The same speed-invariant closure quoted per inch of travel — a "
        "framing club fitters use when discussing strike-to-strike face "
        "variation across the hitting area."
    ),
    "closure_deg_per_ms": (
        "Closure per millisecond — the timing framing: how much the face "
        "angle changes for every millisecond of timing error in the "
        "release. Roughly 2 degrees/ms at tour closure rates, which is "
        "why closure rate behaves as a dispersion term."
    ),
    "r_isa_m": (
        "Distance from the clubhead to the instantaneous screw axis, "
        "v / omega, in metres. The smaller this radius, the faster the "
        "face sweeps for the same clubhead speed. Infinite when the face "
        "is not closing."
    ),
    "r_isa_ft": (
        "The same instantaneous-screw-axis distance in feet. The openly "
        "published ~3 degree GC-vs-face-center gap implies roughly "
        "2.5 ft at a 40 mm offset — closer than the hub radius, the "
        "tension the AffineDrift derivation documents."
    ),
    "time_to_square_from_1deg_open_ms": (
        "How long before impact the face was one degree open, at the "
        "current closure rate. At tour rates this is about half a "
        "millisecond — the timing window behind the classic 'a degree "
        "per half-millisecond' framing of release timing."
    ),
    "toe_heel_speed_delta_mph": (
        "Speed difference between the toe and heel ends of a 117 mm "
        "face due to rotation alone. The toe outruns the heel on every "
        "closing delivery — the same rigid-body effect that produces "
        "the reference-point path gap."
    ),
}


def _fmt_vec(vec: tuple[float, float, float], decimals: int = 3) -> str:
    """Render a 3-vector as a mathtext tuple."""
    return "(" + ",\\ ".join(f"{component:.{decimals}f}" for component in vec) + ")"


def derivation_steps(scenario: ImpactScenario) -> tuple[DerivationStep, ...]:
    """Build the full traceable derivation for one scenario.

    Args:
        scenario: The delivery to trace.

    Returns:
        Ordered steps from frame definition to the reported outputs,
        each with the live numeric substitution.
    """
    result: ImpactResult = solve(scenario)
    speed_mps = scenario.clubhead_speed_mph * 0.44704
    speed_fts = speed_mps / 0.3048
    lever = (
        scenario.com_to_face_mm / 1000.0,
        scenario.impact_offset_high_mm / 1000.0,
        scenario.impact_offset_toe_mm / 1000.0,
    )
    cross_term = (
        result.point_velocity_mps[0] - speed_mps,
        result.point_velocity_mps[1],
        result.point_velocity_mps[2],
    )

    steps = (
        DerivationStep(
            title="Frame and Sign Conventions",
            latex=(
                r"$\hat{x} \parallel \mathrm{target},\ "
                r"\hat{y} \parallel \mathrm{up},\ "
                r"\hat{z} \parallel \mathrm{right\ of\ target}$"
            ),
            values=(
                r"$\mathrm{club\ path} > 0 \Rightarrow \mathrm{in\!-\!to\!-\!out;}"
                r"\ \mathrm{deviations\ referenced\ at\ maximum\ compression}$"
            ),
            narrative=(
                "The AffineDrift house convention (Launch Monitor Technology "
                "Review, 02-parameters.tex, following standard "
                "launch-monitor definitions): x along the "
                "target line, y vertical, z right of the target line. All "
                "angles positive right and up. The tracked reference point is "
                "the geometric center (GC); the CG lies within ~6 mm of it."
            ),
        ),
        DerivationStep(
            title="Shaft Axis and Swing-Plane Normal",
            latex=(
                r"$\hat{s} = (0,\ \sin\beta,\ -\cos\beta),\quad "
                r"\hat{n} = \widehat{\hat{x} \times \hat{s}} "
                r"= (0,\ \cos\beta,\ \sin\beta)$"
            ),
            values=(
                rf"$\beta = {scenario.lie_angle_deg:.1f}^\circ:\ "
                rf"\hat{{s}} = {_fmt_vec(result.shaft_axis)},\ "
                rf"\hat{{n}} = {_fmt_vec(result.plane_normal)}$"
            ),
            narrative=(
                "The shaft leans from the head up toward the hands (up and "
                "left of the target line for a right-handed golfer) at the "
                "impact lie angle. The swing plane contains the shaft and "
                "the target line; its unit normal carries the in-plane "
                "rotation."
            ),
        ),
        DerivationStep(
            title="Angular Velocity Assembly",
            latex=(
                r"$\vec{\omega} = \omega_{plane}\,\hat{n} "
                r"+ \omega_{shaft}\,\hat{s}$"
            ),
            values=(
                rf"$({scenario.omega_plane_dps:.0f}\,\hat{{n}} "
                rf"+ {scenario.omega_shaft_dps:.0f}\,\hat{{s}})\ "
                rf"\mathrm{{deg/s}} \Rightarrow \vec{{\omega}} "
                rf"= {_fmt_vec(result.omega_dps, 0)}\ \mathrm{{deg/s}}$"
            ),
            narrative=(
                "The two rates reported by 3-D motion studies (Cheetham "
                "2014): swing-plane velocity (SPV) about the plane normal "
                "and horizontal turning velocity (HTV) about the shaft — "
                "the closing/release component. They add vectorially "
                "because the two axes are orthogonal."
            ),
        ),
        DerivationStep(
            title="Lever Arm to the Impact Point",
            latex=r"$\vec{r} = (d,\ h_{high},\ h_{toe})$",
            values=(
                rf"$\vec{{r}} = {_fmt_vec(lever)}\ \mathrm{{m}}"
                rf"\quad (d = {scenario.com_to_face_mm:.0f}\ \mathrm{{mm}}"
                r"\ \mathrm{GC\ to\ face\ center})$"
            ),
            narrative=(
                "The vector from the geometric center to the struck point: "
                "forward to the face center (published head data cites 25-50 mm for "
                "drivers; 40 mm is the AffineDrift worked-example value) "
                "plus any high/toe miss offsets."
            ),
        ),
        DerivationStep(
            title="Rigid-Body Point Velocity",
            latex=(r"$\vec{v}_P = \vec{v}_{GC} + \vec{\omega} \times \vec{r}$"),
            values=(
                rf"$\vec{{v}}_{{GC}} = ({speed_mps:.2f},\ 0,\ 0),\ "
                rf"\vec{{\omega}} \times \vec{{r}} = {_fmt_vec(cross_term)},"
                rf"\ \vec{{v}}_P = {_fmt_vec(result.point_velocity_mps)}"
                r"\ \mathrm{m/s}$"
            ),
            narrative=(
                "The twist relation: a rigid body has one velocity per "
                "point, and any point's velocity is the reference velocity "
                "plus omega cross the lever. The cross product is the "
                "rotation-induced velocity — "
                f"{result.tangential_speed_mph:.2f} mph here — and it is "
                "nearly perpendicular to the delivery."
            ),
        ),
        DerivationStep(
            title="Path and Attack-Angle Deviation",
            latex=(
                r"$\Delta\theta_{path} = \mathrm{atan2}(v_z,\ v_x),\quad "
                r"\Delta AoA = \mathrm{atan2}\!\left(v_y,\ "
                r"\sqrt{v_x^2 + v_z^2}\right)$"
            ),
            values=(
                rf"$\Delta\theta_{{path}} = "
                rf"{result.path_deviation_deg:+.2f}^\circ,\quad "
                rf"\Delta AoA = {result.aoa_deviation_deg:+.2f}^\circ$"
            ),
            narrative=(
                "The deliverables: how far the impact point's direction "
                "differs from the reported geometric-center delivery, "
                "horizontally and vertically. Negative path = left of the "
                "reported path."
            ),
        ),
        DerivationStep(
            title="Closure Rate — the CCV Identity",
            latex=(r"$CCV = \omega_y = HTV\,\sin\beta + SPV\,\cos\beta$"),
            values=(
                rf"${scenario.omega_shaft_dps:.0f}\sin"
                rf"{scenario.lie_angle_deg:.0f}^\circ + "
                rf"{scenario.omega_plane_dps:.0f}\cos"
                rf"{scenario.lie_angle_deg:.0f}^\circ = "
                rf"{result.closure_rate_dps:.0f}\ \mathrm{{deg/s}}$"
            ),
            narrative=(
                "The vertical omega component is exactly the literature's "
                "global club closure velocity — the dossier's "
                "reconciliation of shaft-axis and swing-plane rates "
                "(Cheetham tour mean near 2,100 deg/s)."
            ),
        ),
        DerivationStep(
            title="Speed-Invariant Closure and the Path Gap",
            latex=(
                r"$\frac{\omega}{v} = \frac{1}{R_{ISA}},\qquad "
                r"\Delta\theta_{path} \approx \frac{d}{R_{ISA}}$"
            ),
            values=(
                rf"$\frac{{{result.closure_rate_dps:.0f}\ \mathrm{{deg/s}}}}"
                rf"{{{speed_fts:.0f}\ \mathrm{{ft/s}}}} = "
                rf"{result.normalized_closure_deg_per_ft:.2f}\ "
                r"\mathrm{deg/ft}$"
            ),
            narrative=(
                "Closure per foot of travel is the inverse distance to the "
                "instantaneous screw axis, so the path gap between two "
                "points separated by d is d / R_ISA — independent of "
                "clubhead speed. This is the AffineDrift derivation's "
                "preferred unit."
            ),
        ),
        DerivationStep(
            title="Face Rotation During Contact",
            latex=(
                r"$\Delta\phi_{close} = CCV\,\Delta t,\qquad "
                r"\Delta\phi_{loft} = \omega_z\,\Delta t$"
            ),
            values=(
                rf"$\Delta t = {scenario.contact_duration_us:.0f}\ \mu s:\ "
                rf"\Delta\phi_{{close}} = "
                rf"{result.closure_during_contact_deg:.2f}^\circ,\ "
                rf"\Delta\phi_{{loft}} = "
                rf"{result.loft_gain_during_contact_deg:.2f}^\circ$"
            ),
            narrative=(
                "The ball stays on the face about 450 microseconds; the "
                "face keeps rotating the whole time. The face the ball "
                "leaves is not the face it met — a dispersion term, not a "
                "calibratable bias (Cheetham outcome correlation "
                "r = -.14)."
            ),
        ),
    )
    ensure(len(steps) >= 8, "derivation must cover the full chain")
    return steps

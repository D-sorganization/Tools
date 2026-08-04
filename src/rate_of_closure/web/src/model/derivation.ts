/**
 * Step-by-step derivation content — TypeScript mirror of
 * `src/rate_of_closure/derivation.py` (same steps, same explanations,
 * KaTeX-flavoured LaTeX). The vitest suite pins step count, order, and
 * live substitution so the two derivation surfaces cannot drift apart.
 */

import { solve, type ImpactScenario } from "./impact";

export interface DerivationStep {
  title: string;
  latex: string;
  values: string;
  narrative: string;
}

/** Click-through explanation for every result row, keyed by field. */
export const RESULT_EXPLANATIONS: Record<string, string> = {
  pathDeviationDeg:
    "The horizontal angle between the impact point's velocity and the " +
    "geometric center's velocity: atan2(v_z, v_x). Launch monitors report " +
    "the GC path; the ball responds to the impact point's path. Negative " +
    "= the impact point travels left of the reported path (standard " +
    "launch-monitor sign convention: club path positive = in-to-out). " +
    "Openly published launch-monitor material puts this gap near 3 " +
    "degrees for a driver.",
  aoaDeviationDeg:
    "The vertical analogue of the path deviation: atan2(v_y, " +
    "sqrt(v_x^2 + v_z^2)). Positive = the impact point is travelling more " +
    "upward than the reported delivery (a shallower effective attack " +
    "angle). Driven mostly by the swing-plane rotation component.",
  tangentialSpeedMph:
    "The magnitude of omega x r: how fast the impact point moves relative " +
    "to the geometric center purely because the head is rotating. For the " +
    "forum's 35 mm / 2,000 deg/s case this is 1.22 m/s = 2.73 mph — the " +
    "number that was misread as 1.2 mph.",
  speedDeltaMph:
    "How much the rotation changes the impact point's total speed. " +
    "Because omega x r is nearly perpendicular to the delivery, it " +
    "redirects the point without meaningfully changing its speed — which " +
    "is why framing the effect as a percent of clubhead speed understates " +
    "it. Direction changes; speed barely does.",
  closureRateDps:
    "The vertical component of the angular velocity vector — the rate the " +
    "face normal sweeps horizontally (closes). This is the literature's " +
    "club closure velocity: CCV = HTV sin(lie) + SPV cos(lie). Cheetham " +
    "2014 tour driver data: HTV 1,307 +/- 304 deg/s about the shaft " +
    "(range 652-2,432, n = 94); CCV mean near 2,100 deg/s.",
  normalizedClosureDegPerFt:
    "Closure per foot of travel: omega / v, which equals 1 / R_ISA (the " +
    "inverse distance to the instantaneous screw axis). It is " +
    "speed-invariant: two deliveries with the same deg/ft have the same " +
    "path-gap geometry regardless of clubhead speed, because the gap " +
    "between two reference points is d / R_ISA.",
  closureDuringContactDeg:
    "Face closure accumulated while the ball is on the face: CCV times " +
    "the contact duration (about 450 microseconds for a driver). The face " +
    "the ball leaves is not the face it met — roughly a degree at tour " +
    "closure rates.",
  loftGainDuringContactDeg:
    "Dynamic loft gained during contact: the heel-toe component of omega " +
    "times the contact duration. The swing-plane rotation keeps adding " +
    "loft while the ball is on the face.",
};

/** Click-through explanation for every common-literature metric. */
export const METRIC_EXPLANATIONS: Record<string, string> = {
  ccvDps:
    "Club closure velocity in degrees per second — the most common way " +
    "golf research reports rate of closure. Identical to the closure " +
    "rate above: CCV = HTV sin(lie) + SPV cos(lie). Tour driver mean " +
    "near 2,100 deg/s (Cheetham 2014 dossier).",
  closureDegPerFt:
    "Closure per foot of clubhead travel — the speed-invariant " +
    "normalization preferred in the AffineDrift derivation (omega / v = " +
    "1 / R_ISA). Two deliveries with the same deg/ft have identical " +
    "path-gap geometry at any speed.",
  closureDegPerInch:
    "The same speed-invariant closure quoted per inch of travel — a " +
    "framing club fitters use when discussing strike-to-strike face " +
    "variation across the hitting area.",
  closureDegPerMs:
    "Closure per millisecond — the timing framing: how much the face " +
    "angle changes for every millisecond of timing error in the " +
    "release. Roughly 2 degrees/ms at tour closure rates, which is why " +
    "closure rate behaves as a dispersion term.",
  rIsaM:
    "Distance from the clubhead to the instantaneous screw axis, " +
    "v / omega, in metres. The smaller this radius, the faster the face " +
    "sweeps for the same clubhead speed. Infinite when the face is not " +
    "closing.",
  rIsaFt:
    "The same instantaneous-screw-axis distance in feet. The openly " +
    "published ~3 degree GC-vs-face-center gap implies roughly 2.5 ft " +
    "at a 40 mm offset — closer than the hub radius, the tension the " +
    "AffineDrift derivation documents.",
  timeToSquareFrom1DegOpenMs:
    "How long before impact the face was one degree open, at the " +
    "current closure rate. At tour rates this is about half a " +
    "millisecond — the timing window behind the classic 'a degree per " +
    "half-millisecond' framing of release timing.",
  toeHeelSpeedDeltaMph:
    "Speed difference between the toe and heel ends of a 117 mm face " +
    "due to rotation alone. The toe outruns the heel on every closing " +
    "delivery — the same rigid-body effect that produces the " +
    "reference-point path gap.",
};

const fmtVec = (v: readonly number[], decimals = 3): string =>
  "(" + v.map((c) => c.toFixed(decimals)).join(",\\ ") + ")";

/** Build the full traceable derivation for one scenario. */
export function derivationSteps(scenario: ImpactScenario): DerivationStep[] {
  const result = solve(scenario);
  const speedMps = scenario.clubheadSpeedMph * 0.44704;
  const speedFts = speedMps / 0.3048;
  const lever = [
    scenario.comToFaceMm / 1000,
    scenario.impactOffsetHighMm / 1000,
    scenario.impactOffsetToeMm / 1000,
  ];
  const cross = result.pointVelocityMps.map(
    (c, i) => c - [speedMps, 0, 0][i],
  );

  return [
    {
      title: "Frame and Sign Conventions",
      latex:
        "\\hat{x} \\parallel \\text{target},\\quad \\hat{y} \\parallel " +
        "\\text{up},\\quad \\hat{z} \\parallel \\text{right of target}",
      values:
        "\\text{club path} > 0 \\Rightarrow \\text{in-to-out; referenced " +
        "at maximum compression}",
      narrative:
        "The AffineDrift house convention (Launch Monitor Technology " +
        "Review, 02-parameters.tex, following standard launch-monitor definitions): x along the " +
        "target line, y vertical, z right of the target line. All angles " +
        "positive right and up. The tracked reference point is the " +
        "geometric center (GC); the CG lies within ~6 mm of it.",
    },
    {
      title: "Shaft Axis and Swing-Plane Normal",
      latex:
        "\\hat{s} = (0,\\ \\sin\\beta,\\ -\\cos\\beta),\\quad " +
        "\\hat{n} = \\widehat{\\hat{x} \\times \\hat{s}} = " +
        "(0,\\ \\cos\\beta,\\ \\sin\\beta)",
      values:
        `\\beta = ${scenario.lieAngleDeg.toFixed(1)}^\\circ:\\ ` +
        `\\hat{s} = ${fmtVec(result.shaftAxis)},\\ ` +
        `\\hat{n} = ${fmtVec(result.planeNormal)}`,
      narrative:
        "The shaft leans from the head up toward the hands (up and left " +
        "of the target line for a right-handed golfer) at the impact lie " +
        "angle. The swing plane contains the shaft and the target line; " +
        "its unit normal carries the in-plane rotation.",
    },
    {
      title: "Angular Velocity Assembly",
      latex:
        "\\vec{\\omega} = \\omega_{plane}\\,\\hat{n} + " +
        "\\omega_{shaft}\\,\\hat{s}",
      values:
        `(${scenario.omegaPlaneDps.toFixed(0)}\\,\\hat{n} + ` +
        `${scenario.omegaShaftDps.toFixed(0)}\\,\\hat{s})\\ ` +
        `\\text{deg/s} \\Rightarrow \\vec{\\omega} = ` +
        `${fmtVec(result.omegaDps, 0)}\\ \\text{deg/s}`,
      narrative:
        "The two rates reported by 3-D motion studies (Cheetham 2014): " +
        "swing-plane velocity (SPV) about the plane normal and horizontal " +
        "turning velocity (HTV) about the shaft — the closing/release " +
        "component. They add vectorially because the axes are orthogonal.",
    },
    {
      title: "Lever Arm to the Impact Point",
      latex: "\\vec{r} = (d,\\ h_{high},\\ h_{toe})",
      values:
        `\\vec{r} = ${fmtVec(lever)}\\ \\text{m}\\quad (d = ` +
        `${scenario.comToFaceMm.toFixed(0)}\\ \\text{mm GC to face center})`,
      narrative:
        "The vector from the geometric center to the struck point: " +
        "forward to the face center (published head data cites 25-50 mm for drivers; " +
        "40 mm is the AffineDrift worked-example value) plus any high/toe " +
        "miss offsets.",
    },
    {
      title: "Rigid-Body Point Velocity",
      latex: "\\vec{v}_P = \\vec{v}_{GC} + \\vec{\\omega} \\times \\vec{r}",
      values:
        `\\vec{v}_{GC} = (${speedMps.toFixed(2)},\\ 0,\\ 0),\\ ` +
        `\\vec{\\omega} \\times \\vec{r} = ${fmtVec(cross)},\\ ` +
        `\\vec{v}_P = ${fmtVec(result.pointVelocityMps)}\\ \\text{m/s}`,
      narrative:
        "The twist relation: a rigid body has one velocity per point, and " +
        "any point's velocity is the reference velocity plus omega cross " +
        "the lever. The cross product is the rotation-induced velocity — " +
        `${result.tangentialSpeedMph.toFixed(2)} mph here — and it is ` +
        "nearly perpendicular to the delivery.",
    },
    {
      title: "Path and Attack-Angle Deviation",
      latex:
        "\\Delta\\theta_{path} = \\operatorname{atan2}(v_z, v_x),\\quad " +
        "\\Delta AoA = \\operatorname{atan2}\\!\\left(v_y, " +
        "\\sqrt{v_x^2 + v_z^2}\\right)",
      values:
        `\\Delta\\theta_{path} = ` +
        `${result.pathDeviationDeg >= 0 ? "+" : ""}` +
        `${result.pathDeviationDeg.toFixed(2)}^\\circ,\\quad \\Delta AoA ` +
        `= ${result.aoaDeviationDeg >= 0 ? "+" : ""}` +
        `${result.aoaDeviationDeg.toFixed(2)}^\\circ`,
      narrative:
        "The deliverables: how far the impact point's direction differs " +
        "from the reported geometric-center delivery, horizontally and " +
        "vertically. Negative path = left of the reported path.",
    },
    {
      title: "Closure Rate — the CCV Identity",
      latex: "CCV = \\omega_y = HTV\\,\\sin\\beta + SPV\\,\\cos\\beta",
      values:
        `${scenario.omegaShaftDps.toFixed(0)}\\sin` +
        `${scenario.lieAngleDeg.toFixed(0)}^\\circ + ` +
        `${scenario.omegaPlaneDps.toFixed(0)}\\cos` +
        `${scenario.lieAngleDeg.toFixed(0)}^\\circ = ` +
        `${result.closureRateDps.toFixed(0)}\\ \\text{deg/s}`,
      narrative:
        "The vertical omega component is exactly the literature's global " +
        "club closure velocity — the dossier's reconciliation of " +
        "shaft-axis and swing-plane rates (Cheetham tour mean near 2,100 " +
        "deg/s).",
    },
    {
      title: "Speed-Invariant Closure and the Path Gap",
      latex:
        "\\frac{\\omega}{v} = \\frac{1}{R_{ISA}},\\qquad " +
        "\\Delta\\theta_{path} \\approx \\frac{d}{R_{ISA}}",
      values:
        `\\frac{${result.closureRateDps.toFixed(0)}\\ \\text{deg/s}}` +
        `{${speedFts.toFixed(0)}\\ \\text{ft/s}} = ` +
        `${result.normalizedClosureDegPerFt.toFixed(2)}\\ \\text{deg/ft}`,
      narrative:
        "Closure per foot of travel is the inverse distance to the " +
        "instantaneous screw axis, so the path gap between two points " +
        "separated by d is d / R_ISA — independent of clubhead speed. " +
        "This is the AffineDrift derivation's preferred unit.",
    },
    {
      title: "Face Rotation During Contact",
      latex:
        "\\Delta\\phi_{close} = CCV\\,\\Delta t,\\qquad " +
        "\\Delta\\phi_{loft} = \\omega_z\\,\\Delta t",
      values:
        `\\Delta t = ${scenario.contactDurationUs.toFixed(0)}\\ \\mu s:\\ ` +
        `\\Delta\\phi_{close} = ` +
        `${result.closureDuringContactDeg.toFixed(2)}^\\circ,\\ ` +
        `\\Delta\\phi_{loft} = ` +
        `${result.loftGainDuringContactDeg.toFixed(2)}^\\circ`,
      narrative:
        "The ball stays on the face about 450 microseconds; the face " +
        "keeps rotating the whole time. The face the ball leaves is not " +
        "the face it met — a dispersion term, not a calibratable bias " +
        "(Cheetham outcome correlation r = -.14).",
    },
  ];
}

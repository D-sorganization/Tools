/**
 * Prevalidated text for one accepted `putting_result/2` record
 * (#4800 P5/P7) — the React twin of
 * `ui/pyqt6/putting_result_presentation.py`'s row builder.
 *
 * Every fallible scalar label is built before visual publication, from
 * the v2 document and the P3 twist diagnostic only — nothing here
 * re-simulates, and nothing reads the raw trajectory.
 *
 * `term` is the glossary key behind a row's click-through, present only
 * where the shared Python/TypeScript glossary already carries one; the
 * newer P1-P3 rows explain themselves inline rather than inventing
 * glossary keys this runtime would then own alone.
 */

import type { PutterTwist } from "../model/putterHead";
import type { PuttingResultDocument } from "../model/puttingResultWire";

export interface PuttingResultRow {
  readonly key: string;
  readonly label: string;
  readonly explanation: string;
  /** Glossary key, when the shared glossary carries one. */
  readonly term?: string;
}

export const PUTTING_RESULT_ROWS: readonly PuttingResultRow[] = [
  {
    key: "puttRolloutM",
    label: "Roll-Out Distance",
    term: "stimp",
    explanation:
      "How far the ball travels before stopping (or dropping). The skid " +
      "phase sheds speed at the sliding-friction rate, then pure roll " +
      "decelerates at the stimp-derived rolling rate — faster greens mean " +
      "a lower rolling coefficient and a longer roll-out for the same pace.",
  },
  {
    key: "puttSkidM",
    label: "Skid Distance",
    term: "skid",
    explanation:
      "Ground covered while the ball is still sliding rather than rolling. " +
      "A struck putt leaves the face with backspin, so friction must first " +
      "spin it up to pure roll; the transition happens where ball speed " +
      "equals surface spin speed (v = ωr).",
  },
  {
    key: "puttSkidPct",
    label: "Skid Share of Putt",
    term: "skid",
    explanation:
      "The skid distance as a share of the whole putt. Good strokes keep " +
      "this small — the classic no-spin result is pure roll at 5/7 of " +
      "launch speed, and more backspin extends the skid.",
  },
  {
    key: "puttTimeS",
    label: "Time To Rest",
    term: "pure_roll",
    explanation:
      "Elapsed time from impact until the ball stops or drops. Rolling " +
      "deceleration is constant on a uniform green, so time grows linearly " +
      "with the speed the roll phase starts at.",
  },
  {
    key: "puttBallSpeedMps",
    label: "Ball Speed",
    explanation:
      "Ball speed leaving the face. The putter transfers (1+e)·M/(M+m) of " +
      "its speed along the lofted face normal; an off-centre strike lowers " +
      "the head's effective mass to 1/(1/M + r²/I) and takes speed away.",
  },
  {
    key: "puttStartAzimuthDeg",
    label: "Start Line",
    explanation:
      "Direction the ball actually starts on, measured off the target line " +
      "(+ = right). Aim sets the reference, the face angle carries most of " +
      "the start line, and the face-to-path mismatch deflects it by " +
      "atan2((2/7)·sin(fp), T·cos(fp)) through the 2/7 rolling cap.",
  },
  {
    key: "puttSidespinRadS",
    label: "Launch Sidespin",
    explanation:
      "Spin about the vertical axis at launch (+ = the ball turns left). " +
      "It comes from the same face-to-path tangential impulse that bends " +
      "the start line; on a green it is a diagnostic of stroke quality " +
      "rather than a driver of the roll.",
  },
  {
    key: "puttFaceTwistDeg",
    label: "Face Twist At Impact",
    explanation:
      "Quasi-static face rotation during the ~0.5 ms contact window, " +
      "θ = J·r·τ/(2·I) about the head's vertical axis (+ = the face opens, " +
      "a toe strike). A library putter with no measured inertia tensor " +
      "reports the catalogue-MOI response.",
  },
  {
    key: "puttBreakM",
    label: "Break At Rest",
    term: "break",
    explanation:
      "Lateral offset of the ball off the starting line where it comes to " +
      "rest or drops (positive = left), caused by the in-plane component " +
      "of gravity on the sloped green. Break grows fastest late in the " +
      "putt, when the ball is slow.",
  },
  {
    key: "puttApexBreakM",
    label: "Apex Break",
    term: "break",
    explanation:
      "The largest lateral excursion anywhere along the putt, and how far " +
      "down the line it happens — the high point a player actually aims " +
      "at, which is never the same as the break measured at the hole.",
  },
  {
    key: "puttEntryAzimuthDeg",
    label: "Entry Angle",
    explanation:
      "Direction the ball is travelling at its closest approach to the " +
      "hole (+ = moving right). A steeply breaking putt enters across the " +
      "mouth rather than into it, which is what shrinks the usable part " +
      "of the lip.",
  },
  {
    key: "puttClosestApproachM",
    label: "Closest Approach",
    explanation:
      "Smallest distance between the ball's centre and the hole centre " +
      "anywhere along the trajectory — measured from the retained " +
      "integration samples, not re-simulated.",
  },
  {
    key: "puttCaptureMarginM",
    label: "Capture Margin",
    term: "capture_speed",
    explanation:
      "Effective hole radius at the approach speed minus the closest " +
      "approach. The published model shrinks the mouth as R·√(1−(v/vc)²) " +
      "(Holmes 1991; Penner 2002), so a positive margin is the width of " +
      "the lip the ball still had to spare.",
  },
  {
    key: "puttSpeedAtHoleMps",
    label: "Speed At The Hole",
    term: "capture_speed",
    explanation:
      "Ball speed when it first crosses the hole mouth. The putt drops " +
      "only if this is at or below the geometric capture bound — the ball " +
      "must fall half its diameter while crossing the opening.",
  },
  {
    key: "puttMargin",
    label: "Holed / Miss Margin",
    term: "capture_speed",
    explanation:
      "Holed putts: how far under the capture-speed bound the ball crossed " +
      "the hole. Missed putts: the distance from the ball's resting place " +
      "back to the hole — the length of the comebacker.",
  },
];

/** Build every fallible scalar label before visual publication. */
export function puttingResultValues(
  document: PuttingResultDocument,
  twist: PutterTwist,
  formatM: (value: number) => string,
): Record<string, string> {
  return {
    puttRolloutM: formatM(document.totalDistanceM),
    puttSkidM: formatM(document.skidDistanceM),
    puttSkidPct: `${(
      (100 * document.skidDistanceM) /
      Math.max(document.totalDistanceM, 1e-9)
    ).toFixed(1)} %`,
    puttTimeS: `${document.timeS.toFixed(2)} s`,
    puttBallSpeedMps: `${document.ballSpeedMps.toFixed(2)} m/s`,
    puttStartAzimuthDeg: `${document.startAzimuthDeg.toFixed(2)} ° ${
      document.startAzimuthDeg >= 0 ? "right" : "left"
    }`,
    puttSidespinRadS: `${document.sidespinRadS.toFixed(1)} rad/s`,
    puttFaceTwistDeg: `${twist.faceTwistOpenDeg.toFixed(3)} ° open`,
    puttBreakM: formatM(document.finalBreakM),
    puttApexBreakM: `${formatM(document.apexBreakM)} at ${formatM(
      document.apexBreakAtM,
    )}`,
    puttEntryAzimuthDeg: `${document.entryAzimuthDeg.toFixed(2)} °`,
    puttClosestApproachM: formatM(document.closestApproachM),
    puttCaptureMarginM: `${(1000 * document.captureMarginM).toFixed(0)} mm ` +
      `(radius ${(1000 * document.effectiveHoleRadiusM).toFixed(0)} mm)`,
    puttSpeedAtHoleMps:
      document.speedAtHoleMps !== null
        ? `${document.speedAtHoleMps.toFixed(2)} m/s`
        : "— (never reached)",
    puttMargin: document.holed
      ? `HOLED (+${(document.marginMps ?? 0).toFixed(2)} m/s under bound)`
      : `miss by ${formatM(document.missDistanceM ?? 0)}`,
  };
}

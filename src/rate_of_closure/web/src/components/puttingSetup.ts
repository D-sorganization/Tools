/**
 * Editable inputs behind one putt, and the model bounds each carries.
 *
 * Kept beside `PuttingControls.tsx` rather than inside it so the
 * control component exports a component and nothing else. Every bound
 * here is the bound the corresponding Python model validates, so a
 * rejected value surfaces as a refusal instead of a silent clamp.
 */

import type { CaptureModel } from "../model/puttingGreen";

export type PaceMode = "speed" | "backstroke";

/** Every editable input behind one putt. */
export interface PuttSetup {
  putterName: string;
  paceMode: PaceMode;
  speed: number;
  backstrokeCm: number;
  shaftLeanDeg: number;
  aimDeg: number;
  faceAngleDeg: number;
  pathAngleDeg: number;
  attackAngleDeg: number;
  strikeOffsetToeMm: number;
  strikeOffsetHighMm: number;
  stimp: number;
  grade: number;
  aspect: number;
  captureModel: CaptureModel;
  distance: number;
}

/** A square stroke on a flat stimp-10 green from three metres. */
export const DEFAULT_PUTT_SETUP: Omit<PuttSetup, "putterName"> = {
  paceMode: "speed",
  speed: 1.8,
  backstrokeCm: 30,
  shaftLeanDeg: 0,
  aimDeg: 0,
  faceAngleDeg: 0,
  pathAngleDeg: 0,
  attackAngleDeg: 0,
  strikeOffsetToeMm: 0,
  strikeOffsetHighMm: 0,
  stimp: 10,
  grade: 0,
  aspect: 90,
  captureModel: "effective_radius",
  distance: 3,
};

export interface FieldSpec {
  readonly key: keyof PuttSetup;
  readonly label: string;
  readonly suffix: string;
  readonly step: number;
  readonly bounds: readonly [min: number, max: number];
  readonly title: string;
}

/** Stroke parameters — the P1 impact solve's inputs, bound-for-bound. */
export const STROKE_FIELDS: readonly FieldSpec[] = [
  {
    key: "aimDeg",
    label: "Aim",
    suffix: "°",
    step: 0.1,
    bounds: [-45, 45],
    title:
      "Start-line aim off the target line; + = right. Aim sets the reference the face and path angles are measured from (swing_sim.putting.impact)",
  },
  {
    key: "faceAngleDeg",
    label: "Face angle",
    suffix: "°",
    step: 0.1,
    bounds: [-20, 20],
    title:
      "Face angle at impact relative to the aim line; + = open (right). The face carries most of the start line",
  },
  {
    key: "pathAngleDeg",
    label: "Putter path",
    suffix: "°",
    step: 0.1,
    bounds: [-20, 20],
    title:
      "Putter path through impact relative to the aim line; + = in-to-out. Face-to-path mismatch deflects the start line through the 2/7 rolling cap",
  },
  {
    key: "attackAngleDeg",
    label: "Attack angle",
    suffix: "°",
    step: 0.1,
    bounds: [-10, 10],
    title:
      "Vertical angle of approach; + = hitting up. Spin loft is the effective loft minus the attack angle",
  },
  {
    key: "shaftLeanDeg",
    label: "Shaft lean",
    suffix: "°",
    step: 0.1,
    bounds: [-10, 10],
    title:
      "Forward shaft lean at impact; adds to the putter's static loft to give the effective loft, which must stay within [-2, 15]°",
  },
  {
    key: "strikeOffsetToeMm",
    label: "Strike toward toe",
    suffix: "mm",
    step: 1,
    bounds: [-40, 40],
    title:
      "Strike location across the face; + = toward the toe. Off-centre strikes cut the head's effective mass to 1/(1/M + r²/I) and twist the face",
  },
  {
    key: "strikeOffsetHighMm",
    label: "Strike up the face",
    suffix: "mm",
    step: 1,
    bounds: [-20, 20],
    title:
      "Strike height on the face; + = high. A high strike adds dynamic loft through the same quasi-static twist",
  },
];

/** Green parameters — the P2 surface and capture model's inputs. */
export const GREEN_FIELDS: readonly FieldSpec[] = [
  {
    key: "stimp",
    label: "Green speed (stimp)",
    suffix: "ft",
    step: 0.5,
    bounds: [3, 16],
    title:
      "Stimpmeter reading; 7 slow - 13 tournament fast (USGA stimpmeter geometry, swing_sim.putting.roll)",
  },
  {
    key: "grade",
    label: "Slope grade",
    suffix: "%",
    step: 0.25,
    bounds: [0, 10],
    title:
      "Uniform green slope grade; greens rarely exceed ~5 % (swing_sim.putting.surface)",
  },
  {
    key: "aspect",
    label: "Downhill direction",
    suffix: "°",
    step: 5,
    bounds: [-360, 360],
    title:
      "Downhill direction relative to the putt line: 0° ahead, +90° low side left, 180° uphill",
  },
];

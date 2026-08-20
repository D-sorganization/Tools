/** Explicit contact policies and miss-safe outcomes for browser simulations. */

export type ContactMode = "delivery_inspection" | "fixed_ball_contact";
export type ImpactStatus = "hit" | "miss";

export interface ContactSample {
  t: number;
  position: readonly [number, number, number];
}

export interface ImpactOutcomeTs {
  mode: ContactMode;
  status: ImpactStatus;
  candidateTimeS: number;
  closestApproachM: number;
  contactThresholdM: number;
  contactMarginM: number;
  ballPositionM: [number, number, number];
  frame: "app_frame:x_target,y_up,z_right";
  geometryModel:
    | "forced_reference_point_alignment"
    | "sampled_reference_point_to_ball_sphere";
  geometryLimitations: string;
}

const FRAME = "app_frame:x_target,y_up,z_right" as const;
const FIXED_LIMITATIONS =
  "Sampled point-to-sphere proximity only; ignores the clubhead mesh, " +
  "face plane and curvature, swept contact between samples, and ball compression.";
const INSPECTION_LIMITATIONS =
  "Delivery-inspection mode translates the entire swing so the selected " +
  "clubhead reference point coincides with the ball center; this is not " +
  "geometric contact detection.";
const CONTACT_ABS_TOLERANCE_M = 1e-9;

function requireFinitePositive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${name} must be finite and greater than zero.`);
  }
}

function requireBall(
  ball: readonly [number, number, number],
): [number, number, number] {
  if (!ball.every(Number.isFinite)) {
    throw new RangeError("Ball position must contain three finite coordinates.");
  }
  return [ball[0], ball[1], ball[2]];
}

export function deliveryInspectionOutcome(
  candidateTimeS: number,
  ballPositionM: readonly [number, number, number],
  contactThresholdM: number,
): ImpactOutcomeTs {
  requireFinitePositive(contactThresholdM, "Contact threshold");
  if (!Number.isFinite(candidateTimeS) || candidateTimeS < 0) {
    throw new RangeError("Candidate time must be finite and non-negative.");
  }
  return {
    mode: "delivery_inspection",
    status: "hit",
    candidateTimeS,
    closestApproachM: 0,
    contactThresholdM,
    contactMarginM: contactThresholdM,
    ballPositionM: requireBall(ballPositionM),
    frame: FRAME,
    geometryModel: "forced_reference_point_alignment",
    geometryLimitations: INSPECTION_LIMITATIONS,
  };
}

export function assessFixedContact(
  samples: readonly ContactSample[],
  ballPositionM: readonly [number, number, number],
  contactThresholdM: number,
): ImpactOutcomeTs {
  requireFinitePositive(contactThresholdM, "Contact threshold");
  const ball = requireBall(ballPositionM);
  if (samples.length === 0) {
    throw new RangeError("Contact assessment requires at least one sample.");
  }

  let candidateTimeS = 0;
  let closestApproachM = Number.POSITIVE_INFINITY;
  for (const sample of samples) {
    if (!Number.isFinite(sample.t) || sample.t < 0 || !sample.position.every(Number.isFinite)) {
      throw new RangeError("Contact samples must have finite, non-negative times and positions.");
    }
    const distance = Math.hypot(
      sample.position[0] - ball[0],
      sample.position[1] - ball[1],
      sample.position[2] - ball[2],
    );
    if (distance < closestApproachM) {
      closestApproachM = distance;
      candidateTimeS = sample.t;
    }
  }

  const contactMarginM = contactThresholdM - closestApproachM;
  return {
    mode: "fixed_ball_contact",
    status: contactMarginM >= -CONTACT_ABS_TOLERANCE_M ? "hit" : "miss",
    candidateTimeS,
    closestApproachM,
    contactThresholdM,
    contactMarginM,
    ballPositionM: ball,
    frame: FRAME,
    geometryModel: "sampled_reference_point_to_ball_sphere",
    geometryLimitations: FIXED_LIMITATIONS,
  };
}

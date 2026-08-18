/** Passive reduced single-point turf contact, mirrored from shared Python. */

import type { Vec3 } from "./impactPhysics";

export type ReducedTurfStatus = "no_contact" | "separated" |
  "outside_calibrated_domain" | "step_limit";

export interface TurfContactProfileTs {
  profileId: string;
  normalStiffnessNM: number;
  normalDampingNsM: number;
  frictionCoefficient: number;
  frictionRegularizationMps: number;
  maxPenetrationM: number;
  calibrationStatus: "uncalibrated" | "illustrative" | "calibrated";
}

export interface ReducedTurfContactResultTs {
  status: ReducedTurfStatus;
  durationS: number;
  stepCount: number;
  peakPenetrationM: number;
  impulseWorldNs: Vec3;
  normalImpulseNs: number;
  finalContactVelocityMps: Vec3;
  initialKineticEnergyJ: number;
  finalKineticEnergyJ: number;
  dissipatedEnergyJ: number;
  separationLossEnergyJ: number;
  energyBalanceResidualJ: number;
}

export const FIRM_FAIRWAY_TURF: TurfContactProfileTs = {
  profileId: "illustrative-firm-fairway",
  normalStiffnessNM: 60_000,
  normalDampingNsM: 220,
  frictionCoefficient: 0.35,
  frictionRegularizationMps: 0.02,
  maxPenetrationM: 0.025,
  calibrationStatus: "illustrative",
};

const CONTACT_TOLERANCE_M = 1e-12;
const DEFAULT_TIME_STEP_S = 5e-6;
const DEFAULT_MAX_TIME_S = 0.1;
const MAX_STEPS = 200_000;
const MAX_STIFFNESS_N_M = 5_000_000;
const MAX_DAMPING_N_S_M = 100_000;
const MAX_FRICTION = 1;
const MAX_PENETRATION_M = 0.25;

const kineticEnergy = (velocity: Vec3, massKg: number): number =>
  0.5 * massKg * velocity.reduce((sum, value) => sum + value * value, 0);

interface ResultArguments {
  status: ReducedTurfStatus;
  stepCount: number;
  peakPenetrationM: number;
  impulse: Vec3;
  velocity: Vec3;
  massKg: number;
  initialEnergyJ: number;
  dissipatedEnergyJ: number;
  separationLossEnergyJ?: number;
  timeStepS: number;
}

function result(arguments_: ResultArguments): ReducedTurfContactResultTs {
  const separationLoss = arguments_.separationLossEnergyJ ?? 0;
  const finalEnergy = kineticEnergy(arguments_.velocity, arguments_.massKg);
  return {
    status: arguments_.status,
    durationS: arguments_.stepCount * arguments_.timeStepS,
    stepCount: arguments_.stepCount,
    peakPenetrationM: arguments_.peakPenetrationM,
    impulseWorldNs: arguments_.impulse,
    normalImpulseNs: arguments_.impulse[1],
    finalContactVelocityMps: arguments_.velocity,
    initialKineticEnergyJ: arguments_.initialEnergyJ,
    finalKineticEnergyJ: finalEnergy,
    dissipatedEnergyJ: arguments_.dissipatedEnergyJ + separationLoss,
    separationLossEnergyJ: separationLoss,
    energyBalanceResidualJ: arguments_.initialEnergyJ - finalEnergy
      - arguments_.dissipatedEnergyJ - separationLoss,
  };
}

function validateInputs(
  profile: TurfContactProfileTs, velocity: Vec3, massKg: number, timeStepS: number,
): void {
  if (profile === null || typeof profile !== "object") {
    throw new TypeError("profile must be a turf contact profile");
  }
  const profileValues = [
    profile.normalStiffnessNM,
    profile.normalDampingNsM,
    profile.frictionCoefficient,
    profile.frictionRegularizationMps,
    profile.maxPenetrationM,
  ];
  if (!profileValues.every(Number.isFinite)) throw new RangeError("turf profile must be finite");
  if (typeof profile.profileId !== "string" || profile.profileId.trim().length === 0) {
    throw new RangeError("profileId must be nonempty");
  }
  if (profile.normalStiffnessNM < 0 || profile.normalStiffnessNM > MAX_STIFFNESS_N_M) {
    throw new RangeError(`normalStiffnessNM must be in [0, ${MAX_STIFFNESS_N_M}]`);
  }
  if (profile.normalDampingNsM < 0 || profile.normalDampingNsM > MAX_DAMPING_N_S_M) {
    throw new RangeError(`normalDampingNsM must be in [0, ${MAX_DAMPING_N_S_M}]`);
  }
  if (profile.frictionCoefficient < 0 || profile.frictionCoefficient > MAX_FRICTION) {
    throw new RangeError(`frictionCoefficient must be in [0, ${MAX_FRICTION}]`);
  }
  if (profile.frictionRegularizationMps <= 0) {
    throw new RangeError("frictionRegularizationMps must be > 0");
  }
  if (profile.maxPenetrationM <= 0 || profile.maxPenetrationM > MAX_PENETRATION_M) {
    throw new RangeError(`maxPenetrationM must be in (0, ${MAX_PENETRATION_M}]`);
  }
  if (!["uncalibrated", "illustrative", "calibrated"].includes(profile.calibrationStatus)) {
    throw new RangeError("calibrationStatus is invalid");
  }
  if (!Array.isArray(velocity) || velocity.length !== 3 || !velocity.every(Number.isFinite)) {
    throw new RangeError("velocity must contain three finite components");
  }
  if (!Number.isFinite(massKg) || massKg <= 0) throw new RangeError("massKg must be > 0");
  if (!Number.isFinite(timeStepS) || timeStepS <= 0) {
    throw new RangeError("timeStepS must be > 0");
  }
}

/** Integrate a y-up effective contact mass until separation or a typed limit. */
export function simulateReducedTurfContact(
  profile: TurfContactProfileTs,
  initialVelocityMps: Vec3,
  massKg: number,
  timeStepS = DEFAULT_TIME_STEP_S,
): ReducedTurfContactResultTs {
  validateInputs(profile, initialVelocityMps, massKg, timeStepS);
  let velocity: Vec3 = [...initialVelocityMps];
  const initialEnergyJ = kineticEnergy(velocity, massKg);
  const impulse: Vec3 = [0, 0, 0];
  const early = (): ReducedTurfContactResultTs => result({
    status: "no_contact", stepCount: 0, peakPenetrationM: 0,
    impulse, velocity, massKg, initialEnergyJ, dissipatedEnergyJ: 0, timeStepS,
  });
  if (velocity[1] >= 0) return early();
  let penetration = 0;
  let peakPenetration = 0;
  let dissipatedEnergy = 0;
  const stepLimit = Math.min(Math.ceil(DEFAULT_MAX_TIME_S / timeStepS), MAX_STEPS);
  for (let step = 1; step <= stepLimit; step += 1) {
    const penetrationRate = -velocity[1];
    const effectivePenetration = Math.min(penetration, profile.maxPenetrationM);
    if (penetration > profile.maxPenetrationM) {
      return result({
        status: "outside_calibrated_domain", stepCount: step - 1,
        peakPenetrationM: peakPenetration, impulse, velocity, massKg,
        initialEnergyJ, dissipatedEnergyJ: dissipatedEnergy, timeStepS,
      });
    }
    const normalForce = Math.max(
      0,
      profile.normalStiffnessNM * effectivePenetration
        + profile.normalDampingNsM * penetrationRate,
    );
    if (normalForce === 0 && penetration > CONTACT_TOLERANCE_M && velocity[1] >= 0) {
      const separationLoss = 0.5 * profile.normalStiffnessNM * penetration ** 2;
      return result({
        status: "separated", stepCount: step - 1, peakPenetrationM: peakPenetration,
        impulse, velocity, massKg, initialEnergyJ, dissipatedEnergyJ: dissipatedEnergy,
        separationLossEnergyJ: separationLoss, timeStepS,
      });
    }
    const tangentialSpeed = Math.hypot(velocity[0], velocity[2]);
    const denominator = Math.hypot(tangentialSpeed, profile.frictionRegularizationMps);
    const frictionScale = -profile.frictionCoefficient * normalForce / denominator;
    const force: Vec3 = [
      frictionScale * velocity[0], normalForce, frictionScale * velocity[2],
    ];
    const dissipatedPower = profile.normalDampingNsM * penetrationRate ** 2
      - force[0] * velocity[0] - force[2] * velocity[2];
    velocity = velocity.map((value, index) =>
      value + force[index] * timeStepS / massKg) as Vec3;
    force.forEach((value, index) => { impulse[index] += value * timeStepS; });
    dissipatedEnergy += dissipatedPower * timeStepS;
    penetration = Math.max(0, penetration - velocity[1] * timeStepS);
    peakPenetration = Math.max(peakPenetration, penetration);
    if (penetration <= CONTACT_TOLERANCE_M && velocity[1] >= 0) {
      return result({
        status: "separated", stepCount: step, peakPenetrationM: peakPenetration,
        impulse, velocity, massKg, initialEnergyJ, dissipatedEnergyJ: dissipatedEnergy,
        timeStepS,
      });
    }
  }
  return result({
    status: "step_limit", stepCount: stepLimit, peakPenetrationM: peakPenetration,
    impulse, velocity, massKg, initialEnergyJ, dissipatedEnergyJ: dissipatedEnergy,
    timeStepS,
  });
}

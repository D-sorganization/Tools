/** Strict request-bound validation for structured-cloned swing Worker results. */

import { isGlobalSpec, stableSpecId, type VariationPlanTs } from "./variationSchema";
import { resolvedBase } from "./variationSampling";
import { swingVariationInputForValues } from "./variationSwingInput";
import { LOCALIZED_TORQUE_PROVENANCE, LOCALIZED_TORQUE_UNIT } from "./localizedTorque";
import { localizedTorqueJointId } from "./variationRegistry";

type RecordValue = Record<string, unknown>;

const record = (value: unknown): RecordValue | null =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as RecordValue
    : null;
const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const vec = (value: unknown, length: number): boolean =>
  Array.isArray(value) && value.length === length && value.every(finite);
const vec3 = (value: unknown): boolean => vec(value, 3);

const jsonDomain = (value: unknown, seen = new Set<unknown>()): boolean => {
  if (value === null || typeof value === "string" || typeof value === "boolean") return true;
  if (finite(value)) return true;
  if (typeof value !== "object" || seen.has(value)) return false;
  seen.add(value);
  const valid = Array.isArray(value)
    ? value.every((item) => jsonDomain(item, seen))
    : Object.values(value as RecordValue).every((item) => jsonDomain(item, seen));
  seen.delete(value);
  return valid;
};

const exactJson = (actual: unknown, expected: unknown): boolean =>
  jsonDomain(actual) && JSON.stringify(actual) === JSON.stringify(expected);

const validateInputDomain = (value: unknown): boolean => {
  const input = record(value); const parameters = input && record(input.pendulumParameters);
  const club = input && record(input.club); const setup = input && record(input.ballSetup);
  const config = input && record(input.doublePendulumRun);
  if (!input || input.sourceKind !== "double_pendulum" || !vec3(input.omegaDps) ||
      ![input.clubheadSpeedMph, input.loftDeg, input.impactOffsetToeMm,
        input.impactOffsetHighMm, input.planeYawDeg, input.planeSideTiltDeg,
        input.planeForwardTiltDeg, input.impactTimeOffsetS].every(finite) ||
      !finite(input.swingDurationS) || input.swingDurationS <= 0 ||
      !(input.impactTimeS === null || finite(input.impactTimeS)) ||
      !parameters || !["m1", "l1", "lc1", "i1", "m2", "l2", "lc2", "i2", "d1", "d2"]
        .every((name) => finite(parameters[name])) ||
      !club || ![club.headMassKg, club.moiAboutShaftKgM2,
        club.coefficientOfRestitution].every(finite) ||
      !setup || (setup.supportMode !== "ground" && setup.supportMode !== "tee") ||
      !finite(setup.teeHeightM) || !config ||
      (config.mode !== "passive" && config.mode !== "prescribed") ||
      !record(config.jointLocks) || !Array.isArray(config.commandedTorqueOffsets)) return false;
  return jsonDomain(input);
};

const validateSwingSample = (value: unknown, previousTime: number): number | null => {
  const sample = record(value);
  if (!sample || !finite(sample.t) || sample.t < 0 || sample.t < previousTime ||
      !vec3(sample.position) || !vec3(sample.velocity) ||
      !vec3(sample.angularVelocity) || !Array.isArray(sample.rotation) ||
      sample.rotation.length !== 3 || !sample.rotation.every((row) => vec(row, 3)) ||
      !Array.isArray(sample.joints) || sample.joints.length < 3 ||
      !sample.joints.every(vec3)) return null;
  return sample.t;
};

const validateImpact = (value: unknown, expectedStatus: "hit" | "miss"): boolean => {
  const impact = record(value);
  return impact !== null && impact.status === expectedStatus &&
    (impact.mode === "delivery_inspection" || impact.mode === "fixed_ball_contact") &&
    [impact.candidateTimeS, impact.closestApproachM, impact.contactThresholdM,
      impact.contactMarginM].every(finite) &&
    vec3(impact.ballPositionM) &&
    impact.frame === "app_frame:x_target,y_up,z_right" &&
    typeof impact.geometryModel === "string" && typeof impact.geometryLimitations === "string";
};

const validateTorqueRun = (value: unknown, swing: unknown[]): boolean => {
  const torque = record(value);
  if (!torque || torque.mode !== "passive" || torque.profileId !== null ||
      !Array.isArray(torque.lockedJointIds) || torque.lockedJointIds.length !== 0 ||
      !Array.isArray(torque.appliedTorqueHistory) ||
      torque.appliedTorqueHistory.length !== swing.length) return false;
  return torque.appliedTorqueHistory.every((raw, index) => {
    const sample = record(raw); const swingSample = record(swing[index]);
    const values = sample && record(sample.torquesNm);
    return sample !== null && swingSample !== null && sample.timeS === swingSample.t &&
      values !== null && Object.keys(values).sort().join("|") ===
        "joint.shoulder|joint.wrist" &&
      finite(values["joint.shoulder"]) && finite(values["joint.wrist"]);
  });
};

const validateLaunch = (value: unknown): boolean => {
  const launch = record(value);
  return launch !== null && [
    launch.ballSpeedMph, launch.launchAngleDeg, launch.launchAzimuthDeg,
    launch.spinRpm, launch.carryM, launch.maxHeightM, launch.flightTimeS,
    launch.landingAngleDeg,
  ].every(finite);
};

const validateFlight = (value: unknown, requireSamples: boolean): boolean =>
  Array.isArray(value) && (!requireSamples || value.length > 0) &&
  value.every((raw, index) => {
    const point = record(raw);
    const previous = index === 0 ? -Infinity : record(value[index - 1])?.time;
    return point !== null && finite(point.time) && point.time >= 0 &&
      (index === 0 || (finite(previous) && point.time >= previous)) &&
      vec3(point.position) && vec3(point.velocity);
  });

const validateRun = (value: unknown, status: unknown): boolean => {
  const run = record(value); const hit = status === "evaluated_hit";
  if (!run || run.sourceKind !== "double_pendulum" || !Array.isArray(run.swing) ||
      run.swing.length < 2 || !finite(run.totalDurationS) || run.totalDurationS < 0) return false;
  let prior = -Infinity;
  for (const sample of run.swing) {
    const next = validateSwingSample(sample, prior);
    if (next === null) return false;
    prior = next;
  }
  const setup = record(run.ballSetup); const manual = record(run.manualDelivery);
  const setupValid = setup !== null &&
    (setup.supportMode === "ground" || setup.supportMode === "tee") &&
    finite(setup.teeHeightM) && setup.teeHeightM >= 0;
  const manualValid = manual !== null &&
    [manual.manualAttackAngleDeg, manual.manualClubPathDeg,
      manual.manualForwardShaftLeanDeg].every(finite) &&
    (manual.shaftAxisDatum === "tracked_reference" ||
      manual.shaftAxisDatum === "generated_hosel");
  if (!validateTorqueRun(run.torqueRun, run.swing) ||
      !validateImpact(run.impactOutcome, hit ? "hit" : "miss") ||
      !vec3(run.ballPositionM) || !setupValid || !manualValid || run.totalDurationS < prior) {
    return false;
  }
  if (hit) {
    return finite(run.impactTimeS) && validateLaunch(run.launch) &&
      validateFlight(run.flight, true);
  }
  return run.impactTimeS === null && run.launch === null &&
    validateFlight(run.flight, false) && (run.flight as unknown[]).length === 0;
};

export function validateSwingTrialPayload(
  trial: RecordValue,
  trialIndex: number,
  plan: VariationPlanTs,
  inputRow: unknown,
  bindDefaultInput = true,
): boolean {
  if (!Array.isArray(inputRow) || inputRow.length !== plan.noise.length ||
      !inputRow.every(finite)) return false;
  const values = { ...resolvedBase(plan) };
  plan.noise.forEach((spec, index) => { values[spec.variableKey] = inputRow[index]; });
  if (!validateInputDomain(trial.input)) return false;
  if (bindDefaultInput) {
    let expectedInput: unknown;
    try { expectedInput = swingVariationInputForValues(plan, values).input; } catch { return false; }
    if (!exactJson(trial.input, expectedInput)) return false;
  }
  if (trial.status === "numerical_failure") {
    return trial.run === null && typeof trial.error === "string" &&
      trial.error.length > 0 && trial.error.length <= 512;
  }
  return trial.error === null && validateRun(trial.run, trial.status) &&
    trial.trialIndex === trialIndex;
}

export function validateLocalizedTrialCommands(
  trial: RecordValue,
  plan: VariationPlanTs,
  inputNames: unknown,
  inputRow: unknown,
): boolean {
  const specs = plan.noise.filter((spec) => !isGlobalSpec(spec));
  const commands = trial.localizedTorqueCommands;
  if (!Array.isArray(commands) || !Array.isArray(inputNames) ||
      !inputNames.every((name) => typeof name === "string") ||
      !Array.isArray(inputRow) || commands.length !== specs.length) return false;
  return specs.every((spec, commandIndex) => {
    const command = record(commands[commandIndex]);
    const inputIndex = inputNames.indexOf(spec.variableKey);
    return command !== null && inputIndex >= 0 &&
      command.specId === stableSpecId(spec) && command.variableKey === spec.variableKey &&
      command.jointId === localizedTorqueJointId(spec.variableKey) &&
      Array.isArray(command.timeWindowS) && command.timeWindowS.length === 2 &&
      command.timeWindowS[0] === spec.timeWindowS?.[0] &&
      command.timeWindowS[1] === spec.timeWindowS?.[1] &&
      command.torqueNm === inputRow[inputIndex] && command.unit === LOCALIZED_TORQUE_UNIT &&
      command.provenance === LOCALIZED_TORQUE_PROVENANCE;
  });
}

export function assertJsonFinite(value: unknown, label: string): void {
  if (!jsonDomain(value)) throw new Error(`${label} must contain only finite JSON values`);
}

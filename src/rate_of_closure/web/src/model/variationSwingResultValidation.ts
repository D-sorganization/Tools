/** Strict request-bound validation for structured-cloned swing Worker results. */

import {
  ballCenterPosition, GOLF_BALL_RADIUS_M, resolveBallSetup,
} from "./ballSetup";
import { assessFixedContact, deliveryInspectionOutcome } from "./contact";
import {
  DOUBLE_PENDULUM_DT_S, effectiveDoublePendulumDurationS,
  summarizeDoublePendulumRun,
} from "./doublePendulum";
import { DOUBLE_PENDULUM_JOINT_IDS } from "./jointLocks";
import { resolveManualDelivery } from "./manualDelivery";
import type { SimulationInput } from "./simulation";
import { isGlobalSpec, stableSpecId, type VariationPlanTs } from "./variationSchema";
import { resolvedBase } from "./variationSampling";
import { swingVariationInputForValues } from "./variationSwingInput";
import {
  LOCALIZED_TORQUE_PROVENANCE, LOCALIZED_TORQUE_UNIT,
  normalizeLocalizedTorqueOffsets,
} from "./localizedTorque";
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
const exactFields = (value: RecordValue, fields: readonly string[]): boolean =>
  Object.keys(value).sort().join("|") === [...fields].sort().join("|");

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

const INPUT_FIELDS = new Set([
  "sourceKind", "clubheadSpeedMph", "omegaDps", "loftDeg", "impactOffsetToeMm",
  "impactOffsetHighMm", "planeYawDeg", "planeSideTiltDeg", "planeForwardTiltDeg",
  "impactTimeS", "swingDurationS", "pendulumParameters", "impactTimeOffsetS", "club",
  "contactMode", "doublePendulumRun", "doublePendulumInitialState", "ballSetup",
  "manualAttackAngleDeg", "manualClubPathDeg", "manualForwardShaftLeanDeg", "shaftAxisDatum",
]);
const PENDULUM_FIELDS = [
  "m1", "l1", "lc1", "i1", "m2", "l2", "lc2", "i2", "d1", "d2",
] as const;

const validateRunConfig = (value: unknown, durationS: number): boolean => {
  const config = record(value); const locks = config && record(config.jointLocks);
  if (!config || !exactFields(config, ["mode", "jointLocks", "commandedTorqueOffsets"]) ||
      config.mode !== "passive" || !locks || !exactFields(locks, ["lockedJointIds"]) ||
      !Array.isArray(locks.lockedJointIds) ||
      locks.lockedJointIds.some((jointId) => typeof jointId !== "string") ||
      new Set(locks.lockedJointIds).size !== locks.lockedJointIds.length ||
      locks.lockedJointIds.some((jointId) => !DOUBLE_PENDULUM_JOINT_IDS.includes(
        jointId as (typeof DOUBLE_PENDULUM_JOINT_IDS)[number],
      )) || !Array.isArray(config.commandedTorqueOffsets)) return false;
  if (!config.commandedTorqueOffsets.every((raw) => {
    const offset = record(raw);
    return offset !== null && exactFields(offset, ["jointId", "timeWindowS", "torqueNm"]);
  })) return false;
  try {
    const normalized = normalizeLocalizedTorqueOffsets(
      config.commandedTorqueOffsets, durationS,
    );
    return exactJson(config.commandedTorqueOffsets, normalized);
  } catch {
    return false;
  }
};

const validateInputDomain = (value: unknown): boolean => {
  const input = record(value); const parameters = input && record(input.pendulumParameters);
  const club = input && record(input.club); const setup = input && record(input.ballSetup);
  if (!input || Object.keys(input).some((field) => !INPUT_FIELDS.has(field)) ||
      input.sourceKind !== "double_pendulum" || !vec3(input.omegaDps) ||
      ![input.clubheadSpeedMph, input.loftDeg, input.impactOffsetToeMm,
        input.impactOffsetHighMm, input.planeYawDeg, input.planeSideTiltDeg,
        input.planeForwardTiltDeg, input.impactTimeOffsetS].every(finite) ||
      !finite(input.swingDurationS) || input.swingDurationS <= 0 ||
      !(input.impactTimeS === null || finite(input.impactTimeS)) ||
      !parameters || !exactFields(parameters, PENDULUM_FIELDS) ||
      !PENDULUM_FIELDS.every((name) => finite(parameters[name])) ||
      !club || ![club.headMassKg, club.moiAboutShaftKgM2,
        club.coefficientOfRestitution].every(finite) ||
      !exactFields(club, ["headMassKg", "moiAboutShaftKgM2", "coefficientOfRestitution"]) ||
      !setup || !exactFields(setup, ["supportMode", "teeHeightM"]) ||
      (input.contactMode !== undefined && input.contactMode !== "delivery_inspection" &&
        input.contactMode !== "fixed_ball_contact") ||
      (input.doublePendulumInitialState !== undefined &&
        !vec(input.doublePendulumInitialState, 4)) ||
      (input.manualAttackAngleDeg !== undefined && !finite(input.manualAttackAngleDeg)) ||
      (input.manualClubPathDeg !== undefined && !finite(input.manualClubPathDeg)) ||
      (input.manualForwardShaftLeanDeg !== undefined &&
        !finite(input.manualForwardShaftLeanDeg)) ||
      (input.shaftAxisDatum !== undefined && input.shaftAxisDatum !== "tracked_reference" &&
        input.shaftAxisDatum !== "generated_hosel")) return false;
  try {
    resolveBallSetup(setup as unknown as SimulationInput["ballSetup"]);
    resolveManualDelivery(input as unknown as SimulationInput);
    if (!validateRunConfig(input.doublePendulumRun, effectiveDoublePendulumDurationS(
      input.swingDurationS,
    ))) return false;
  } catch {
    return false;
  }
  return jsonDomain(input);
};

const validateSwingSample = (value: unknown, expectedTime: number): boolean => {
  const sample = record(value);
  return Boolean(sample && sample.t === expectedTime &&
      vec3(sample.position) && vec3(sample.velocity) &&
      vec3(sample.angularVelocity) && Array.isArray(sample.rotation) &&
      sample.rotation.length === 3 && sample.rotation.every((row) => vec(row, 3)) &&
      Array.isArray(sample.joints) && sample.joints.length >= 3 &&
      sample.joints.every(vec3));
};

type ValidatedSwingSample = {
  t: number;
  position: [number, number, number];
  velocity: [number, number, number];
};

const selectedDeliveryTime = (
  input: SimulationInput,
  swing: readonly ValidatedSwingSample[],
): number => {
  let index: number;
  if (input.impactTimeS === null) {
    const midpoint = swing[swing.length - 1].t / 2;
    index = swing.reduce((best, sample, sampleIndex) => {
      const speed = Math.hypot(...sample.velocity);
      const bestSpeed = Math.hypot(...swing[best].velocity);
      const higher = speed > bestSpeed + 1e-12;
      const centralTie = Math.abs(speed - bestSpeed) <= 1e-12 &&
        Math.abs(sample.t - midpoint) < Math.abs(swing[best].t - midpoint);
      return higher || centralTie ? sampleIndex : best;
    }, 0);
  } else {
    const clamped = Math.max(0, Math.min(input.impactTimeS, swing[swing.length - 1].t));
    index = Math.round(clamped / DOUBLE_PENDULUM_DT_S);
  }
  const shifted = swing[index].t + (input.impactTimeOffsetS ?? 0);
  const clamped = Math.max(0, Math.min(shifted, swing[swing.length - 1].t));
  return swing[Math.round(clamped / DOUBLE_PENDULUM_DT_S)].t;
};

const validateImpact = (
  value: unknown,
  input: SimulationInput,
  swing: ValidatedSwingSample[],
  ballPositionM: [number, number, number],
): boolean => {
  const impact = record(value);
  const mode = input.contactMode ?? "delivery_inspection";
  if (!impact || impact.mode !== mode || !finite(impact.candidateTimeS) ||
      !swing.some((sample) => sample.t === impact.candidateTimeS)) return false;
  const expected = mode === "fixed_ball_contact"
    ? assessFixedContact(swing, ballPositionM, GOLF_BALL_RADIUS_M)
    : deliveryInspectionOutcome(
        selectedDeliveryTime(input, swing), ballPositionM, GOLF_BALL_RADIUS_M,
      );
  return exactJson(impact, expected);
};

const validateTorqueRun = (
  value: unknown,
  input: SimulationInput,
  sampleTimes: readonly number[],
): boolean => exactJson(
  value,
  summarizeDoublePendulumRun(input.doublePendulumRun, sampleTimes),
);

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
      (index === 0 || (finite(previous) && point.time > previous)) &&
      vec3(point.position) && vec3(point.velocity);
  });

const validateRun = (
  value: unknown,
  status: unknown,
  input: SimulationInput,
): boolean => {
  const run = record(value); const hit = status === "evaluated_hit";
  const effectiveDuration = effectiveDoublePendulumDurationS(input.swingDurationS);
  const expectedSamples = Math.round(effectiveDuration / DOUBLE_PENDULUM_DT_S) + 1;
  if (!run || run.sourceKind !== "double_pendulum" || !Array.isArray(run.swing) ||
      run.swing.length !== expectedSamples || !finite(run.totalDurationS)) return false;
  for (let index = 0; index < run.swing.length; index += 1) {
    if (!validateSwingSample(run.swing[index], index * DOUBLE_PENDULUM_DT_S)) return false;
  }
  const sampleTimes = run.swing.map((sample) => record(sample)?.t as number);
  const setup = resolveBallSetup(input.ballSetup);
  const manual = resolveManualDelivery(input);
  const ball = ballCenterPosition(setup);
  const typedSwing = run.swing as ValidatedSwingSample[];
  if (!validateTorqueRun(run.torqueRun, input, sampleTimes) ||
      !validateImpact(run.impactOutcome, input, typedSwing, ball) ||
      !exactJson(run.ballSetup, setup) || !exactJson(run.ballPositionM, ball) ||
      !exactJson(run.manualDelivery, manual)) {
    return false;
  }
  const impact = record(run.impactOutcome);
  if (hit) {
    const launch = record(run.launch);
    return impact?.status === "hit" && run.impactTimeS === impact.candidateTimeS &&
      validateLaunch(run.launch) && launch !== null &&
      run.totalDurationS === effectiveDuration + (launch.flightTimeS as number) &&
      validateFlight(run.flight, true);
  }
  return impact?.status === "miss" && run.impactTimeS === null && run.launch === null &&
    run.totalDurationS === effectiveDuration &&
    validateFlight(run.flight, false) && (run.flight as unknown[]).length === 0;
};

const valuesForInputRow = (
  plan: VariationPlanTs,
  inputRow: readonly number[],
): Record<string, number> => {
  const values = { ...resolvedBase(plan) };
  plan.noise.forEach((spec, index) => { values[spec.variableKey] = inputRow[index]; });
  return values;
};

const expectedInputForRow = (
  plan: VariationPlanTs,
  inputRow: readonly number[],
  baseInput?: SimulationInput,
): SimulationInput => baseInput === undefined
  ? swingVariationInputForValues(plan, valuesForInputRow(plan, inputRow)).input
  : swingVariationInputForValues(plan, valuesForInputRow(plan, inputRow), baseInput).input;

/** Reconstruct the document-owned invariant input before applying sampled values. */
export function documentSwingInputAuthority(
  value: unknown,
  plan: VariationPlanTs,
  inputRow: unknown,
): SimulationInput | null {
  if (!validateInputDomain(value) || !Array.isArray(inputRow) || !inputRow.every(finite)) {
    return null;
  }
  const input = value as SimulationInput;
  const config = record(input.doublePendulumRun);
  const offsets = config?.commandedTorqueOffsets;
  const localizedCount = plan.noise.filter((spec) => !isGlobalSpec(spec)).length;
  if (!Array.isArray(offsets) || offsets.length < localizedCount) return null;
  const baseOffsetCount = offsets.length - localizedCount;
  const baseInput = {
    ...input,
    doublePendulumRun: {
      ...input.doublePendulumRun,
      commandedTorqueOffsets: offsets.slice(0, baseOffsetCount),
    },
  } as SimulationInput;
  try {
    return exactJson(value, expectedInputForRow(plan, inputRow, baseInput))
      ? baseInput
      : null;
  } catch {
    return null;
  }
}

export function validateSwingTrialPayload(
  trial: RecordValue,
  trialIndex: number,
  plan: VariationPlanTs,
  inputRow: unknown,
  baseInput?: SimulationInput,
): boolean {
  if (!Array.isArray(inputRow) || inputRow.length !== plan.noise.length ||
      !inputRow.every(finite)) return false;
  if (!validateInputDomain(trial.input)) return false;
  let expectedInput: SimulationInput;
  try { expectedInput = expectedInputForRow(plan, inputRow, baseInput); } catch { return false; }
  if (!exactJson(trial.input, expectedInput)) return false;
  if (trial.status === "numerical_failure") {
    return trial.run === null && typeof trial.error === "string" &&
      trial.error.length > 0 && trial.error.length <= 512;
  }
  return trial.error === null && validateRun(
    trial.run, trial.status, trial.input as SimulationInput,
  ) &&
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

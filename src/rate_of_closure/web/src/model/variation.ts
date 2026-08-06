/**
 * Public web variation facade and scalar physics executor.
 *
 * Schema/migration contracts live in variationSchema.ts; deterministic
 * independent and grouped sampling lives in variationSampling.ts. This file
 * preserves the original import surface while keeping the scalar pipeline's
 * capability boundary explicit.
 */

import { deriveLaunch, simulateFlight, type Launch } from "./flight";
import { MPH_PER_MPS, solveImpact, toFlightFrame, type Vec3 } from "./simulation";
import { resolvedBase, sampleInputs } from "./variationSampling";
import {
  isGlobalSpec,
  stableSpecId,
  validatePlan,
  type VariationPlanTs,
} from "./variationSchema";
import { TEE_HEIGHT_VARIATION_KEY, type VariationMode } from "./variationRegistry";
import { runSwingVariation } from "./variationSwingEnsemble";

export {
  CATEGORY_DELIVERY,
  CATEGORY_LAUNCH,
  CATEGORY_SWING,
  keysForMode,
  variableDef,
  variableLabel,
  VARIABLE_REGISTRY,
  type VariableDefTs,
  type VariationMode,
} from "./variationRegistry";
export {
  MAX_RUNS,
  planFromJson,
  planToJson,
  SCHEMA_VERSION,
  validatePlan,
  type Distribution,
  type MatrixKindTs,
  type NoiseSpecTs,
  type PerturbationGroupTs,
  type VariationPlanTs,
} from "./variationSchema";
export { fnv1a, mulberry32, sampleInputs } from "./variationSampling";

export const DELIVERY_OUTPUTS = [
  "club_path_deg",
  "face_angle_deg",
  "attack_angle_deg",
  "dynamic_loft_deg",
] as const;
export const LAUNCH_OUTPUTS = [
  "ball_speed_mph",
  "launch_angle_deg",
  "launch_azimuth_deg",
  "spin_rpm",
  "spin_axis_deg",
] as const;
export const FLIGHT_OUTPUTS = [
  "carry_m",
  "lateral_m",
  "apex_m",
  "landing_angle_deg",
  "flight_time_s",
] as const;

export function outputsForMode(mode: VariationMode): string[] {
  if (mode === "swing") return [
    "candidate_time_s", "closest_approach_m", "contact_margin_m",
    "impact_time_s", "clubhead_speed_mps", "ball_speed_mph",
    "launch_angle_deg", "launch_azimuth_deg", "spin_rpm", "carry_m",
    "lateral_m", "max_height_m", "flight_time_s", "landing_angle_deg",
  ];
  return mode === "launch"
    ? [...LAUNCH_OUTPUTS, ...FLIGHT_OUTPUTS]
    : [...DELIVERY_OUTPUTS, ...LAUNCH_OUTPUTS, ...FLIGHT_OUTPUTS];
}

const RAD = Math.PI / 180;
const clampAngle = (value: number): number => Math.max(-89, Math.min(89, value));
const short = (key: string): string => key.slice(key.lastIndexOf(".") + 1);

/** Convert imperial launch variables into the flight model's frame. */
const launchFromImperial = (variables: Record<string, number>): Launch => {
  // Registry azimuth/spin-axis are +right/+fade; flight is +left.
  const azimuth = -variables.launch_azimuth_deg * RAD;
  const axisAngle = -variables.spin_axis_deg * RAD;
  const backspin = Math.cos(axisAngle);
  const sidespin = Math.sin(axisAngle);
  const axisRaw: Vec3 = [
    sidespin * Math.sin(azimuth),
    -backspin,
    sidespin * Math.cos(azimuth),
  ];
  const axisNorm = Math.hypot(...axisRaw) || 1;
  return {
    ballSpeedMps: variables.ball_speed_mph / MPH_PER_MPS,
    launchAngleRad: clampAngle(variables.launch_angle_deg) * RAD,
    azimuthRad: azimuth,
    spinRpm: Math.max(variables.spin_rpm, 0),
    spinAxis: axisRaw.map((value) => value / axisNorm) as Vec3,
  };
};

const spinAxisTiltDeg = (spin: Vec3): number => {
  const magnitude = Math.hypot(...spin);
  if (magnitude < 1e-12) return 0;
  const axis = spin.map((value) => value / magnitude) as Vec3;
  return Math.atan2(-axis[1], Math.hypot(axis[0], axis[2])) / RAD;
};

/** Evaluate one sampled variable set through launch or delivery physics. */
export function evaluateRun(
  variables: Record<string, number>,
  mode: VariationMode,
): Record<string, number> {
  const values: Record<string, number> = {};
  for (const [key, value] of Object.entries(variables)) values[short(key)] = value;

  if (mode === "launch") {
    if (!(values.ball_speed_mph >= 0)) throw new Error("ball_speed_mph must be >= 0");
    const launch = launchFromImperial(values);
    const flight = simulateFlight(launch);
    return {
      ball_speed_mph: values.ball_speed_mph,
      launch_angle_deg: values.launch_angle_deg,
      launch_azimuth_deg: values.launch_azimuth_deg,
      spin_rpm: values.spin_rpm,
      spin_axis_deg: values.spin_axis_deg,
      carry_m: flight.carryM,
      lateral_m: -flight.lateralM,
      apex_m: flight.maxHeightM,
      landing_angle_deg: flight.landingAngleDeg,
      flight_time_s: flight.flightTimeS,
    };
  }

  const impact = solveImpact({
    clubheadSpeedMps: Math.max(values.clubhead_speed_mps, 1e-3),
    clubPathDeg: clampAngle(values.club_path_deg),
    faceAngleDeg: clampAngle(values.face_angle_deg),
    attackAngleDeg: clampAngle(values.attack_angle_deg),
    dynamicLoftDeg: clampAngle(values.dynamic_loft_deg),
    impactOffsetToeMm: values.impact_offset_toe_mm,
    impactOffsetHighMm: values.impact_offset_high_mm,
  });
  const launch = deriveLaunch(
    toFlightFrame(impact.ballVelocity),
    toFlightFrame(impact.ballAngularVelocity),
  );
  const flight = simulateFlight(launch);
  return {
    club_path_deg: clampAngle(values.club_path_deg),
    face_angle_deg: clampAngle(values.face_angle_deg),
    attack_angle_deg: clampAngle(values.attack_angle_deg),
    dynamic_loft_deg: clampAngle(values.dynamic_loft_deg),
    ball_speed_mph: launch.ballSpeedMps * MPH_PER_MPS,
    launch_angle_deg: launch.launchAngleRad / RAD,
    launch_azimuth_deg: -launch.azimuthRad / RAD,
    spin_rpm: launch.spinRpm,
    spin_axis_deg: spinAxisTiltDeg(impact.ballAngularVelocity),
    carry_m: flight.carryM,
    lateral_m: -flight.lateralM,
    apex_m: flight.maxHeightM,
    landing_angle_deg: flight.landingAngleDeg,
    flight_time_s: flight.flightTimeS,
  };
}

export interface VariationDatasetTs {
  plan: VariationPlanTs;
  inputNames: string[];
  inputs: number[][];
  outputNames: string[];
  outputs: (number | null)[][];
  success: boolean[];
}

/** Execute a plan synchronously through the browser's scalar evaluator. */
export function runVariation(plan: VariationPlanTs): VariationDatasetTs {
  validatePlan(plan);
  if (plan.mode === "swing") return runSwingVariation(plan).dataset;
  if (plan.noise.some((spec) => spec.variableKey === TEE_HEIGHT_VARIATION_KEY)) {
    throw new Error(
      "Tee Height variation requires the complete Rate simulation ensemble; " +
      "the scalar browser delivery evaluator has no contact geometry.",
    );
  }
  const localized = plan.noise.filter((spec) => !isGlobalSpec(spec)).map(stableSpecId);
  if (localized.length > 0) {
    throw new Error(
      `scalar evaluator supports only global perturbations: ${localized.join(", ")}`,
    );
  }

  const inputs = sampleInputs(plan);
  const base = resolvedBase(plan);
  const inputNames = plan.noise.map((spec) => spec.variableKey);
  const outputNames = outputsForMode(plan.mode);
  const outputs: (number | null)[][] = [];
  const success: boolean[] = [];
  for (let runIndex = 0; runIndex < plan.nRuns; runIndex += 1) {
    const variables = { ...base };
    inputNames.forEach((key, inputIndex) => {
      variables[key] = inputs[runIndex][inputIndex];
    });
    try {
      const result = evaluateRun(variables, plan.mode);
      outputs.push(outputNames.map((name) => result[name]));
      success.push(true);
    } catch {
      outputs.push(outputNames.map(() => null));
      success.push(false);
    }
  }
  return { plan, inputNames, inputs, outputNames, outputs, success };
}

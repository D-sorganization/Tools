/** Complete browser swing ensembles with retained traces, misses, and failures. */

import { DRIVER_TEE_HEIGHT_M, type BallSetup } from "./ballSetup";
import { golfDefaultParams } from "./doublePendulum";
import {
  DEFAULT_IMPACT_CLUB,
  runSimulation,
  type SimulationInput,
  type SimulationRunTs,
} from "./simulation";
import { deliveryDiagnostics } from "./impactPhysics";
import { resolvedBase, sampleInputs } from "./variationSampling";
import { validatePlan, type VariationPlanTs } from "./variationSchema";
import {
  CATEGORY_CLUB,
  CATEGORY_DELIVERY,
  CATEGORY_SWING,
  TEE_HEIGHT_VARIATION_KEY,
} from "./variationRegistry";
import type { VariationDatasetTs } from "./variation";

export type SwingTrialStatusTs =
  | "evaluated_hit"
  | "evaluated_no_impact"
  | "numerical_failure";

export interface SwingVariationTrialTs {
  trialIndex: number;
  status: SwingTrialStatusTs;
  input: SimulationInput;
  run: SimulationRunTs | null;
  error: string | null;
}

export interface SwingVariationResultTs {
  dataset: VariationDatasetTs;
  runs: SwingVariationTrialTs[];
  coordinateFrame: "app_frame:x_target,y_up,z_right";
}

export const SWING_ENSEMBLE_EXPORT_SCHEMA_VERSION = 1;

export function swingEnsembleToJson(result: SwingVariationResultTs): string {
  return JSON.stringify({
    schemaVersion: SWING_ENSEMBLE_EXPORT_SCHEMA_VERSION,
    coordinateFrame: result.coordinateFrame,
    positionUnit: "m",
    timeUnit: "s",
    dataset: result.dataset,
    trials: result.runs,
  }, null, 2);
}

export function swingTracesToCsv(result: SwingVariationResultTs): string {
  const rows = [[
    "trial", "status", "sample", "time_s", "point_id",
    "x_target_m", "y_up_m", "z_right_m", "is_impact_sample", "coordinate_frame",
  ]];
  result.runs.forEach((trial) => {
    trial.run?.swing.forEach((sample, sampleIndex) => {
      const impact = trial.run?.impactTimeS !== null
        && Math.abs(sample.t - (trial.run?.impactTimeS ?? 0)) <= 0.0005;
      sample.joints.forEach((position, pointIndex) => {
        const pointId = pointIndex === 0
          ? "swing.pivot"
          : pointIndex === sample.joints.length - 1
            ? "swing.clubhead.reference"
            : pointIndex === sample.joints.length - 2
              ? "swing.wrist"
              : `swing.joint.${pointIndex}`;
        rows.push([
          String(trial.trialIndex), trial.status, String(sampleIndex), String(sample.t),
          pointId, ...position.map(String), impact ? "1" : "0", result.coordinateFrame,
        ]);
      });
    });
  });
  return rows.map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}

const csvCell = (value: string): string =>
  /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;

const key = (category: string, name: string): string => `${category}.${name}`;
const YAW = key(CATEGORY_SWING, "yaw_deg");
const SIDE_TILT = key(CATEGORY_SWING, "side_tilt_deg");
const FORWARD_TILT = key(CATEGORY_SWING, "forward_tilt_deg");
const IMPACT_TIME_OFFSET = key(CATEGORY_SWING, "impact_time_offset_s");
const DAMPING_SHOULDER = key(CATEGORY_SWING, "damping_shoulder");
const DAMPING_WRIST = key(CATEGORY_SWING, "damping_wrist");
const TOE_OFFSET = key(CATEGORY_DELIVERY, "impact_offset_toe_mm");
const HIGH_OFFSET = key(CATEGORY_DELIVERY, "impact_offset_high_mm");
const HEAD_MASS = key(CATEGORY_CLUB, "head_mass_kg");
const HEAD_MOI = key(CATEGORY_CLUB, "head_moi_kg_m2");
const COR = key(CATEGORY_CLUB, "cor");

const OUTPUT_NAMES = [
  "candidate_time_s",
  "closest_approach_m",
  "contact_margin_m",
  "impact_time_s",
  "clubhead_speed_mps",
  "spin_loft_deg",
  "face_to_path_deg",
  "spin_axis_tilt_deg",
  "ball_speed_mph",
  "launch_angle_deg",
  "launch_azimuth_deg",
  "spin_rpm",
  "carry_m",
  "lateral_m",
  "max_height_m",
  "flight_time_s",
  "landing_angle_deg",
] as const;

export function defaultSwingVariationInput(ballSetup?: BallSetup): SimulationInput {
  return {
    sourceKind: "double_pendulum",
    clubheadSpeedMph: 30,
    omegaDps: [0, 0, 0],
    loftDeg: 10.5,
    impactOffsetToeMm: 0,
    impactOffsetHighMm: 0,
    planeYawDeg: 0,
    planeSideTiltDeg: -45,
    planeForwardTiltDeg: 0,
    impactTimeS: null,
    impactTimeOffsetS: 0,
    swingDurationS: 1.5,
    pendulumParameters: golfDefaultParams(),
    club: { ...DEFAULT_IMPACT_CLUB },
    ballSetup: ballSetup ?? {
      supportMode: "tee",
      teeHeightM: DRIVER_TEE_HEIGHT_M,
    },
  };
}

export function runSwingVariation(
  plan: VariationPlanTs,
  baseInput: SimulationInput = defaultSwingVariationInput(plan.ballSetup),
  onTrialComplete?: () => void,
): SwingVariationResultTs {
  validatePlan(plan);
  if (plan.mode !== "swing") throw new Error("complete swing ensemble requires swing mode");
  if (baseInput.sourceKind !== "double_pendulum") {
    throw new Error("complete swing ensemble requires the double_pendulum source");
  }
  const localized = plan.noise.filter((spec) =>
    spec.timeWindowS != null || (spec.pointIds?.length ?? 0) > 0,
  );
  if (localized.length > 0) throw new Error("swing ensemble supports global perturbations only");
  const inputs = sampleInputs(plan);
  const base = resolvedBase(plan);
  const inputNames = plan.noise.map((spec) => spec.variableKey);
  const runs: SwingVariationTrialTs[] = [];
  const outputs: Array<Array<number | null>> = [];
  const success: boolean[] = [];
  inputs.forEach((row, trialIndex) => {
    const values = { ...base };
    inputNames.forEach((name, column) => { values[name] = row[column]; });
    const input = applyValues(baseInput, values);
    try {
      const run = runSimulation(input);
      const status = run.impactOutcome.status === "hit"
        ? "evaluated_hit"
        : "evaluated_no_impact";
      runs.push({ trialIndex, status, input, run, error: null });
      outputs.push(outputRow(run, input));
      success.push(true);
    } catch (error) {
      runs.push({
        trialIndex,
        status: "numerical_failure",
        input,
        run: null,
        error: error instanceof Error ? error.message : String(error),
      });
      outputs.push(OUTPUT_NAMES.map(() => null));
      success.push(false);
    }
    onTrialComplete?.();
  });
  return {
    dataset: {
      plan,
      inputNames,
      inputs,
      outputNames: [...OUTPUT_NAMES],
      outputs,
      success,
    },
    runs,
    coordinateFrame: "app_frame:x_target,y_up,z_right",
  };
}

function applyValues(
  base: SimulationInput,
  values: Record<string, number>,
): SimulationInput {
  const parameters = base.pendulumParameters ?? golfDefaultParams();
  const setup = base.ballSetup ?? { supportMode: "ground" as const, teeHeightM: 0 };
  const teeHeight = values[TEE_HEIGHT_VARIATION_KEY];
  if (teeHeight !== undefined && setup.supportMode !== "tee") {
    throw new Error("Tee Height variation requires Tee support");
  }
  return {
    ...base,
    planeYawDeg: values[YAW] ?? base.planeYawDeg,
    planeSideTiltDeg: values[SIDE_TILT] ?? base.planeSideTiltDeg,
    planeForwardTiltDeg: values[FORWARD_TILT] ?? base.planeForwardTiltDeg,
    impactTimeOffsetS: values[IMPACT_TIME_OFFSET] ?? base.impactTimeOffsetS ?? 0,
    impactOffsetToeMm: values[TOE_OFFSET] ?? base.impactOffsetToeMm,
    impactOffsetHighMm: values[HIGH_OFFSET] ?? base.impactOffsetHighMm,
    pendulumParameters: {
      ...parameters,
      d1: values[DAMPING_SHOULDER] ?? parameters.d1,
      d2: values[DAMPING_WRIST] ?? parameters.d2,
    },
    club: {
      headMassKg: values[HEAD_MASS] ?? base.club?.headMassKg
        ?? DEFAULT_IMPACT_CLUB.headMassKg,
      moiAboutShaftKgM2: values[HEAD_MOI] ?? base.club?.moiAboutShaftKgM2
        ?? DEFAULT_IMPACT_CLUB.moiAboutShaftKgM2,
      coefficientOfRestitution: values[COR] ?? base.club?.coefficientOfRestitution
        ?? DEFAULT_IMPACT_CLUB.coefficientOfRestitution,
    },
    ballSetup: teeHeight === undefined
      ? setup
      : { supportMode: "tee", teeHeightM: teeHeight },
  };
}

function outputRow(run: SimulationRunTs, input: SimulationInput): Array<number | null> {
  const outcome = run.impactOutcome;
  if (run.impactTimeS === null || run.launch === null) {
    return [
      outcome.candidateTimeS,
      outcome.closestApproachM,
      outcome.contactMarginM,
      ...OUTPUT_NAMES.slice(3).map(() => null),
    ];
  }
  const impactSample = run.swing.reduce((best, sample) =>
    Math.abs(sample.t - run.impactTimeS!) < Math.abs(best.t - run.impactTimeS!)
      ? sample
      : best,
  );
  const landing = run.flight[run.flight.length - 1];
  const velocity = impactSample.velocity;
  const clubPathDeg = Math.atan2(velocity[2], velocity[0]) * 180 / Math.PI;
  const attackAngleDeg = Math.atan2(
    velocity[1], Math.hypot(velocity[0], velocity[2]),
  ) * 180 / Math.PI;
  const diagnostics = deliveryDiagnostics({
    clubheadSpeedMps: Math.hypot(...velocity),
    clubPathDeg,
    faceAngleDeg: 0,
    attackAngleDeg,
    dynamicLoftDeg: input.loftDeg,
    impactOffsetToeMm: input.impactOffsetToeMm,
    impactOffsetHighMm: input.impactOffsetHighMm,
    club: input.club,
  });
  return [
    outcome.candidateTimeS,
    outcome.closestApproachM,
    outcome.contactMarginM,
    run.impactTimeS,
    Math.hypot(...impactSample.velocity),
    diagnostics.spinLoftDeg,
    diagnostics.faceToPathDeg,
    diagnostics.spinAxisTiltDeg,
    run.launch.ballSpeedMph,
    run.launch.launchAngleDeg,
    run.launch.launchAzimuthDeg,
    run.launch.spinRpm,
    run.launch.carryM,
    landing?.position[2] ?? 0,
    run.launch.maxHeightM,
    run.launch.flightTimeS,
    run.launch.landingAngleDeg,
  ];
}

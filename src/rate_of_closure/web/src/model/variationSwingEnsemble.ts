/** Complete browser swing ensembles with retained traces, misses, and failures. */

import {
  runSimulation,
  type SimulationInput,
  type SimulationRunTs,
} from "./simulation";
import { deliveryDiagnostics } from "./impactPhysics";
import { resolvedBase, sampleInputs } from "./variationSampling";
import { validatePlan, type VariationPlanTs } from "./variationSchema";
import type { VariationDatasetTs } from "./variation";
import type { LocalizedTorqueCommandTs } from "./localizedTorque";
import { spreadsheetSafeCsvCell } from "./csvSecurity";
import {
  defaultSwingVariationInput, swingVariationInputForValues,
} from "./variationSwingInput";
import {
  SWING_ENSEMBLE_EXPORT_SCHEMA_VERSION, swingEnsembleFromJson,
  assertSwingEnsembleResult, swingEnsembleToJsonDocument,
} from "./variationSwingDocument";

export { defaultSwingVariationInput, swingVariationInputForValues } from "./variationSwingInput";
export { SWING_ENSEMBLE_EXPORT_SCHEMA_VERSION, swingEnsembleFromJson };

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
  localizedTorqueCommands: readonly LocalizedTorqueCommandTs[];
}

export interface SwingVariationResultTs {
  dataset: VariationDatasetTs;
  runs: SwingVariationTrialTs[];
  coordinateFrame: "app_frame:x_target,y_up,z_right";
}

export function swingEnsembleToJson(result: SwingVariationResultTs): string {
  return swingEnsembleToJsonDocument(result);
}

export function swingTracesToCsv(result: SwingVariationResultTs): string {
  assertSwingEnsembleResult(result);
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
  return rows.map((row) => row.map(spreadsheetSafeCsvCell).join(",")).join("\n") + "\n";
}

export function localizedTorqueSourcesToCsv(result: SwingVariationResultTs): string {
  assertSwingEnsembleResult(result);
  const rows = [[
    "trial", "status", "spec_id", "variable_key", "joint_id",
    "window_start_s", "window_end_s", "torque_nm", "unit", "provenance",
  ]];
  result.runs.forEach((trial) => trial.localizedTorqueCommands.forEach((command) => {
    rows.push([
      String(trial.trialIndex), trial.status, command.specId, command.variableKey,
      command.jointId, String(command.timeWindowS[0]), String(command.timeWindowS[1]),
      String(command.torqueNm), command.unit, command.provenance,
    ]);
  }));
  return rows.map((row) => row.map(spreadsheetSafeCsvCell).join(",")).join("\n") + "\n";
}

export const SWING_VARIATION_OUTPUT_NAMES = [
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
  const inputs = sampleInputs(plan);
  const base = resolvedBase(plan);
  swingVariationInputForValues(plan, base, baseInput);
  const inputNames = plan.noise.map((spec) => spec.variableKey);
  const runs: SwingVariationTrialTs[] = [];
  const outputs: Array<Array<number | null>> = [];
  const success: boolean[] = [];
  inputs.forEach((row, trialIndex) => {
    const values = { ...base };
    inputNames.forEach((name, column) => { values[name] = row[column]; });
    const { input, localized } = swingVariationInputForValues(plan, values, baseInput);
    try {
      const run = runSimulation(input);
      const status = run.impactOutcome.status === "hit"
        ? "evaluated_hit"
        : "evaluated_no_impact";
      runs.push({
        trialIndex, status, input, run, error: null,
        localizedTorqueCommands: localized.commands,
      });
      outputs.push(outputRow(run, input));
      success.push(true);
    } catch (error) {
      runs.push({
        trialIndex,
        status: "numerical_failure",
        input,
        run: null,
        error: error instanceof Error ? error.message : String(error),
        localizedTorqueCommands: localized.commands,
      });
      outputs.push(SWING_VARIATION_OUTPUT_NAMES.map(() => null));
      success.push(false);
    }
    onTrialComplete?.();
  });
  return {
    dataset: {
      plan,
      inputNames,
      inputs,
      outputNames: [...SWING_VARIATION_OUTPUT_NAMES],
      outputs,
      success,
    },
    runs,
    coordinateFrame: "app_frame:x_target,y_up,z_right",
  };
}

function outputRow(run: SimulationRunTs, input: SimulationInput): Array<number | null> {
  const outcome = run.impactOutcome;
  if (run.impactTimeS === null || run.launch === null) {
    return [
      outcome.candidateTimeS,
      outcome.closestApproachM,
      outcome.contactMarginM,
      ...SWING_VARIATION_OUTPUT_NAMES.slice(3).map(() => null),
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

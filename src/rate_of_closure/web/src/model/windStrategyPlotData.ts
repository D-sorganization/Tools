/** Pure adapter from completed wind-strategy analyses to scalar plot rows. */

import {
  createScalarEnsemble,
  scalarEnsembleRowId,
  type ScalarEnsembleResult,
  type ScalarVariableDefinition,
} from "./scalarEnsembleContract";
import {
  WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
  WIND_UNCERTAINTY_SCHEMA_VERSION,
  sampleWindTrials,
  type PerfectInformationCounterfactual,
  type WindOutcomeStatus,
  type WindStrategy,
  type WindStrategyAnalysis,
  type WindStrategyOutcome,
  type WindStrategyRequest,
  type WindTrial,
} from "./windUncertainty";
import type { WindScenario } from "./wind";

export const WIND_STRATEGY_PLOT_ADAPTER_ID = "wind-strategy-plot-adapter/v1" as const;
const AGREEMENT_TOLERANCE = 1e-9;
const DEGREES_TO_RADIANS = Math.PI / 180;

const stages = [
  { key: "input", label: "Strategy Inputs" },
  { key: "environment", label: "Wind and Target" },
  { key: "actual", label: "Estimated-Wind Decision" },
  { key: "perfect_information", label: "Perfect-Information Counterfactual" },
  { key: "comparison", label: "Information Comparison" },
] as const;

const categories = [
  { key: "wind", label: "Wind" },
  { key: "launch", label: "Launch and Aim" },
  { key: "target", label: "Target" },
  { key: "actual", label: "Actual Outcome" },
  { key: "perfect_information", label: "Perfect-Information Outcome" },
  { key: "information", label: "Information Value" },
] as const;

export const WIND_STRATEGY_PLOT_VARIABLES: readonly ScalarVariableDefinition[] = [
  { key: "true_wind_speed_mps", label: "True Wind Speed", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "true_wind_from_bearing_deg", label: "True Wind From Bearing", unit: "deg", stage_key: "environment", category_key: "wind" },
  { key: "true_wind_forward_mps", label: "True Wind Forward Component", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "true_wind_left_mps", label: "True Wind Left Component", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "true_wind_up_mps", label: "True Wind Up Component", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "estimated_wind_speed_mps", label: "Estimated Wind Speed", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "estimated_wind_from_bearing_deg", label: "Estimated Wind From Bearing", unit: "deg", stage_key: "environment", category_key: "wind" },
  { key: "estimated_wind_forward_mps", label: "Estimated Wind Forward Component", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "estimated_wind_left_mps", label: "Estimated Wind Left Component", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "estimated_wind_up_mps", label: "Estimated Wind Up Component", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "wind_speed_error_mps", label: "Wind Speed Estimate Error", unit: "m/s", stage_key: "environment", category_key: "wind" },
  { key: "wind_bearing_error_deg", label: "Wind Bearing Estimate Error", unit: "deg", stage_key: "environment", category_key: "wind" },
  { key: "launch_ball_speed_mps", label: "Launch Ball Speed", unit: "m/s", stage_key: "input", category_key: "launch" },
  { key: "launch_angle_rad", label: "Launch Angle", unit: "rad", stage_key: "input", category_key: "launch" },
  { key: "launch_azimuth_rad", label: "Base Launch Direction", unit: "rad", stage_key: "input", category_key: "launch" },
  { key: "launch_spin_rpm", label: "Launch Spin Rate", unit: "rpm", stage_key: "input", category_key: "launch" },
  { key: "launch_spin_axis_forward", label: "Spin Axis Forward", unit: "1", stage_key: "input", category_key: "launch" },
  { key: "launch_spin_axis_left", label: "Spin Axis Left", unit: "1", stage_key: "input", category_key: "launch" },
  { key: "launch_spin_axis_up", label: "Spin Axis Up", unit: "1", stage_key: "input", category_key: "launch" },
  { key: "crosswind_aim_gain_rad_per_mps", label: "Crosswind Aim Gain", unit: "rad/(m/s)", stage_key: "input", category_key: "launch" },
  { key: "actual_aim_azimuth_rad", label: "Estimated-Wind Aim Direction", unit: "rad", stage_key: "actual", category_key: "launch" },
  { key: "perfect_information_aim_azimuth_rad", label: "Perfect-Information Aim Direction", unit: "rad", stage_key: "perfect_information", category_key: "launch" },
  { key: "target_forward_m", label: "Target Forward Coordinate", unit: "m", stage_key: "environment", category_key: "target" },
  { key: "target_right_m", label: "Target Right Coordinate", unit: "m", stage_key: "environment", category_key: "target" },
  { key: "actual_landing_forward_m", label: "Actual Landing Forward", unit: "m", stage_key: "actual", category_key: "actual" },
  { key: "actual_landing_right_m", label: "Actual Landing Right", unit: "m", stage_key: "actual", category_key: "actual" },
  { key: "actual_miss_distance_m", label: "Actual Miss Distance", unit: "m", stage_key: "actual", category_key: "actual" },
  { key: "actual_cost", label: "Actual Strategy Cost", unit: "1", stage_key: "actual", category_key: "actual" },
  { key: "perfect_information_landing_forward_m", label: "Perfect-Information Landing Forward", unit: "m", stage_key: "perfect_information", category_key: "perfect_information" },
  { key: "perfect_information_landing_right_m", label: "Perfect-Information Landing Right", unit: "m", stage_key: "perfect_information", category_key: "perfect_information" },
  { key: "perfect_information_miss_distance_m", label: "Perfect-Information Miss Distance", unit: "m", stage_key: "perfect_information", category_key: "perfect_information" },
  { key: "perfect_information_cost", label: "Perfect-Information Cost", unit: "1", stage_key: "perfect_information", category_key: "perfect_information" },
  { key: "information_cost_delta", label: "Information Cost Delta", unit: "1", stage_key: "comparison", category_key: "information" },
];

const cohorts = [
  { key: "completed", label: "Completed" },
  { key: "nonconverged", label: "Nonconverged" },
  { key: "invalid", label: "Invalid" },
] as const;

const finite = (value: number, name: string): number => {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
};

const agrees = (left: number, right: number): boolean =>
  Math.abs(left - right) <= AGREEMENT_TOLERANCE * Math.max(1, Math.abs(left), Math.abs(right));

const normalizeBearing = (value: number): number => ((value + 180) % 360 + 360) % 360 - 180;

interface OutcomeValidity {
  readonly status: WindOutcomeStatus;
  readonly landingForwardM: number | null;
  readonly landingRightM: number | null;
  readonly missDistanceM: number | null;
  readonly failureReason: string | null;
}

function validateTrial(trial: WindTrial, trialIndex: number): void {
  if (trial.trial_index !== trialIndex) throw new RangeError("wind trial indices must be contiguous");
  const scalars = [
    trial.true_speed_mps, trial.true_from_bearing_deg, trial.estimated_speed_mps,
    trial.estimated_from_bearing_deg, trial.speed_error_mps, trial.bearing_error_deg,
  ];
  scalars.forEach((value) => finite(value, `wind trial ${trialIndex}`));
  if (trial.true_speed_mps < 0 || trial.estimated_speed_mps < 0) {
    throw new RangeError("wind speeds must be nonnegative");
  }
  if (!agrees(trial.estimated_speed_mps - trial.true_speed_mps, trial.speed_error_mps)) {
    throw new RangeError("wind speed error must agree with true and estimated wind");
  }
  const estimatedBearing = normalizeBearing(trial.true_from_bearing_deg + trial.bearing_error_deg);
  if (!agrees(normalizeBearing(trial.estimated_from_bearing_deg), estimatedBearing)) {
    throw new RangeError("wind bearing error must agree with true and estimated wind");
  }
}

function validateSampleAgreement(observed: WindTrial, expected: WindTrial): void {
  const pairs = [
    [observed.true_speed_mps, expected.true_speed_mps],
    [observed.true_from_bearing_deg, expected.true_from_bearing_deg],
    [observed.estimated_speed_mps, expected.estimated_speed_mps],
    [observed.estimated_from_bearing_deg, expected.estimated_from_bearing_deg],
    [observed.speed_error_mps, expected.speed_error_mps],
    [observed.bearing_error_deg, expected.bearing_error_deg],
  ];
  if (observed.trial_index !== expected.trial_index ||
      !pairs.every(([left, right]) => agrees(left, right))) {
    throw new RangeError("analysis wind trials must agree with the request sampling contract");
  }
}

function validateScenario(
  scenario: WindScenario,
  speedMps: number,
  bearingDeg: number,
  expectedProvenance: string,
  name: string,
): void {
  const bearingRad = bearingDeg * DEGREES_TO_RADIANS;
  const expected = [-speedMps * Math.cos(bearingRad), speedMps * Math.sin(bearingRad), 0];
  if (scenario.schemaVersion !== "wind-scenario/v1" || scenario.baseVelocityMps.length !== 3) {
    throw new RangeError(`${name} has an unsupported wind scenario`);
  }
  scenario.baseVelocityMps.forEach((value, index) => finite(value, `${name} component ${index}`));
  if (!scenario.baseVelocityMps.every((value, index) => agrees(value, expected[index]))) {
    throw new RangeError(`${name} does not agree with its wind trial`);
  }
  if (scenario.provenance !== expectedProvenance || scenario.shearFractionPer10m !== 0 ||
      scenario.turbulenceIntensityMps !== 0 || scenario.seed !== 0 || scenario.gusts.length !== 0) {
    throw new RangeError(`${name} does not agree with its deterministic scenario contract`);
  }
}

function validateSimulationResult(result: OutcomeValidity): void {
  const scalars = [result.landingForwardM, result.landingRightM, result.missDistanceM];
  const complete = result.status === "completed";
  if (complete && (scalars.some((value) => value === null || !Number.isFinite(value)) || result.failureReason !== null)) {
    throw new RangeError("completed outcome must contain finite landing data and no failure reason");
  }
  if (!complete && (scalars.some((value) => value !== null) || !result.failureReason?.trim())) {
    throw new RangeError("failed outcome must contain null landing data and a failure reason");
  }
}

function validatePerfectInformation(result: PerfectInformationCounterfactual): void {
  validateSimulationResult({
    status: result.status,
    landingForwardM: result.landing_forward_m,
    landingRightM: result.landing_right_m,
    missDistanceM: result.miss_distance_m,
    failureReason: result.failure_reason,
  });
  if (finite(result.cost, "perfect-information cost") < 0) {
    throw new RangeError("perfect-information cost must be nonnegative");
  }
}

function validateOutcome(
  outcome: WindStrategyOutcome,
  trial: WindTrial,
  sourceProvenance: string,
): void {
  validateScenario(
    outcome.true_wind, trial.true_speed_mps, trial.true_from_bearing_deg,
    `${sourceProvenance}/true/trial-${trial.trial_index}`, "true wind",
  );
  validateScenario(
    outcome.estimated_wind, trial.estimated_speed_mps,
    trial.estimated_from_bearing_deg,
    `${sourceProvenance}/estimated/trial-${trial.trial_index}`, "estimated wind",
  );
  validateSimulationResult({
    status: outcome.status,
    landingForwardM: outcome.landing_forward_m,
    landingRightM: outcome.landing_right_m,
    missDistanceM: outcome.miss_distance_m,
    failureReason: outcome.failure_reason,
  });
  if (finite(outcome.cost, "actual cost") < 0) throw new RangeError("actual cost must be nonnegative");
  validatePerfectInformation(outcome.perfect_information);
  finite(outcome.information_cost_delta, "information cost delta");
  const expectedDelta = outcome.cost - outcome.perfect_information.cost;
  if (!agrees(outcome.information_cost_delta, expectedDelta)) {
    throw new RangeError("information cost delta must agree with outcome costs");
  }
}

function validateAnalysisAgreement(
  request: WindStrategyRequest,
  analysis: WindStrategyAnalysis,
): void {
  if (request.uncertainty.schema_version !== WIND_UNCERTAINTY_SCHEMA_VERSION ||
      analysis.schema_version !== WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION) {
    throw new RangeError("unsupported wind strategy schema version");
  }
  if (analysis.provenance !== request.uncertainty.provenance) {
    throw new RangeError("request and analysis provenance must agree");
  }
  if (!agrees(analysis.target.forward_m, request.target.forward_m) ||
      !agrees(analysis.target.right_m, request.target.right_m)) {
    throw new RangeError("request and analysis target must agree");
  }
  if (analysis.wind_trials.length !== request.uncertainty.trials) {
    throw new RangeError("wind trial coverage must agree with the request");
  }
  const expectedTrials = sampleWindTrials(request.uncertainty);
  analysis.wind_trials.forEach((trial, trialIndex) => {
    validateTrial(trial, trialIndex);
    validateSampleAgreement(trial, expectedTrials[trialIndex]);
  });
  validateOutcomeCoverage(request, analysis);
}

function validateOutcomeCoverage(
  request: WindStrategyRequest,
  analysis: WindStrategyAnalysis,
): void {
  const strategyIds = request.strategies.map(({ id }) => id);
  if (new Set(strategyIds).size !== strategyIds.length || strategyIds.some((id) => !id.trim())) {
    throw new RangeError("request strategy ids must be nonempty and unique");
  }
  const expected = request.uncertainty.trials * request.strategies.length;
  const keys = analysis.outcomes.map((item) => scalarEnsembleRowId(item.trial_index, item.strategy_id));
  if (analysis.outcomes.length !== expected || new Set(keys).size !== expected) {
    throw new RangeError("analysis outcome coverage must match every strategy and trial");
  }
  const trialByIndex = new Map(analysis.wind_trials.map((trial) => [trial.trial_index, trial]));
  analysis.outcomes.forEach((item) => {
    const trial = trialByIndex.get(item.trial_index);
    if (!strategyIds.includes(item.strategy_id) || !trial) {
      throw new RangeError("analysis outcome coverage contains an unknown strategy or trial");
    }
    validateOutcome(item, trial, request.uncertainty.provenance);
  });
}

function aimAzimuth(strategy: WindStrategy, wind: WindScenario): number {
  return strategy.launch.azimuthRad -
    strategy.crosswind_aim_gain_rad_per_mps * wind.baseVelocityMps[1];
}

function rowValues(
  request: WindStrategyRequest,
  strategy: WindStrategy,
  trial: WindTrial,
  outcome: WindStrategyOutcome,
): Record<string, number | null> {
  const actual = outcome;
  const perfect = outcome.perfect_information;
  const trueVelocity = outcome.true_wind.baseVelocityMps;
  const estimatedVelocity = outcome.estimated_wind.baseVelocityMps;
  const spinAxis = strategy.launch.spinAxis;
  return {
    true_wind_speed_mps: trial.true_speed_mps,
    true_wind_from_bearing_deg: trial.true_from_bearing_deg,
    true_wind_forward_mps: trueVelocity[0], true_wind_left_mps: trueVelocity[1],
    true_wind_up_mps: trueVelocity[2],
    estimated_wind_speed_mps: trial.estimated_speed_mps,
    estimated_wind_from_bearing_deg: trial.estimated_from_bearing_deg,
    estimated_wind_forward_mps: estimatedVelocity[0],
    estimated_wind_left_mps: estimatedVelocity[1], estimated_wind_up_mps: estimatedVelocity[2],
    wind_speed_error_mps: trial.speed_error_mps, wind_bearing_error_deg: trial.bearing_error_deg,
    launch_ball_speed_mps: strategy.launch.ballSpeedMps,
    launch_angle_rad: strategy.launch.launchAngleRad,
    launch_azimuth_rad: strategy.launch.azimuthRad, launch_spin_rpm: strategy.launch.spinRpm,
    launch_spin_axis_forward: spinAxis[0], launch_spin_axis_left: spinAxis[1],
    launch_spin_axis_up: spinAxis[2],
    crosswind_aim_gain_rad_per_mps: strategy.crosswind_aim_gain_rad_per_mps,
    actual_aim_azimuth_rad: aimAzimuth(strategy, outcome.estimated_wind),
    perfect_information_aim_azimuth_rad: aimAzimuth(strategy, outcome.true_wind),
    target_forward_m: request.target.forward_m, target_right_m: request.target.right_m,
    actual_landing_forward_m: actual.landing_forward_m,
    actual_landing_right_m: actual.landing_right_m,
    actual_miss_distance_m: actual.miss_distance_m, actual_cost: actual.cost,
    perfect_information_landing_forward_m: perfect.landing_forward_m,
    perfect_information_landing_right_m: perfect.landing_right_m,
    perfect_information_miss_distance_m: perfect.miss_distance_m,
    perfect_information_cost: perfect.cost,
    information_cost_delta: actual.information_cost_delta,
  };
}

interface WindRowContext {
  readonly strategy: WindStrategy;
  readonly trial: WindTrial;
  readonly outcome: WindStrategyOutcome;
}

function canonicalRowContexts(
  request: WindStrategyRequest,
  analysis: WindStrategyAnalysis,
): WindRowContext[] {
  const outcomes = new Map(analysis.outcomes.map((outcome) => [
    scalarEnsembleRowId(outcome.trial_index, outcome.strategy_id), outcome,
  ]));
  return analysis.wind_trials.flatMap((trial) => request.strategies.map((strategy) => {
    const rowId = scalarEnsembleRowId(trial.trial_index, strategy.id);
    const outcome = outcomes.get(rowId);
    if (!outcome) throw new RangeError("validated outcome lookup failed");
    return { strategy, trial, outcome };
  }));
}

/** Adapt an existing analysis; this function deliberately performs no simulation. */
export function buildWindStrategyPlotData(
  request: WindStrategyRequest,
  analysis: WindStrategyAnalysis,
): ScalarEnsembleResult<WindOutcomeStatus> {
  validateAnalysisAgreement(request, analysis);
  const rows = canonicalRowContexts(request, analysis).map(({ strategy, trial, outcome }) => {
    return {
      row_id: scalarEnsembleRowId(outcome.trial_index, outcome.strategy_id),
      trial_index: outcome.trial_index,
      series_id: outcome.strategy_id,
      cohort: outcome.status,
      values: rowValues(request, strategy, trial, outcome),
      attributes: {
        actual_status: outcome.status,
        perfect_information_status: outcome.perfect_information.status,
        actual_failure_reason: outcome.failure_reason,
        perfect_information_failure_reason: outcome.perfect_information.failure_reason,
        strategy_label: strategy.label,
      },
    };
  });
  return createScalarEnsemble({
    result_id: `wind-strategy:${analysis.provenance}`,
    provenance: {
      adapter_id: WIND_STRATEGY_PLOT_ADAPTER_ID,
      source_schema_version: analysis.schema_version,
      source_provenance: analysis.provenance,
    },
    stages, categories, variables: WIND_STRATEGY_PLOT_VARIABLES, cohorts, rows,
  });
}

/** Deterministic true/estimated-wind ensembles and paired strategy risk. */

import { simulateFlight, type Launch } from "./flight";
import { meteorologicalWind, type WindScenario } from "./wind";
import { summarizeStrategyOutcomes } from "./windStrategyMetrics";

export const WIND_UNCERTAINTY_SCHEMA_VERSION = "wind-uncertainty/v1" as const;
export const WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION = "wind-strategy-analysis/v2" as const;
const UINT32_SCALE = 2 ** 32;
const UINT32_MAX = 0xffffffff;
const MIN_UNIFORM = 1 / UINT32_SCALE;
const PRECISION_DIGITS = 9;
const GROUND_TOLERANCE_M = 1e-5;

export type DistributionKind = "fixed" | "normal" | "uniform";

export interface ScalarDistribution {
  readonly kind: DistributionKind;
  readonly center: number;
  readonly spread?: number;
  readonly minimum?: number;
  readonly maximum?: number;
}

export interface WindEstimateError {
  readonly speed_bias_mps: number;
  readonly speed_std_mps: number;
  readonly bearing_bias_deg: number;
  readonly bearing_std_deg: number;
  readonly correlation: number;
}

export interface WindUncertaintySpec {
  readonly schema_version: typeof WIND_UNCERTAINTY_SCHEMA_VERSION;
  readonly trials: number;
  readonly seed: number;
  readonly true_speed_mps: ScalarDistribution;
  readonly true_from_bearing_deg: ScalarDistribution;
  readonly estimate_error: WindEstimateError;
  readonly provenance: string;
}

export interface WindTrial {
  readonly trial_index: number;
  readonly true_speed_mps: number;
  readonly true_from_bearing_deg: number;
  readonly estimated_speed_mps: number;
  readonly estimated_from_bearing_deg: number;
  readonly speed_error_mps: number;
  readonly bearing_error_deg: number;
}

export interface WindStrategy {
  readonly id: string;
  readonly label: string;
  readonly launch: Launch;
  readonly crosswind_aim_gain_rad_per_mps: number;
}

export interface WindStrategyRequest {
  readonly uncertainty: WindUncertaintySpec;
  readonly strategies: readonly WindStrategy[];
  readonly target: { readonly forward_m: number; readonly right_m: number };
  readonly analysis: {
    readonly model_name: "waterloo_penner";
    readonly max_time_s: number;
    readonly time_step_s: number;
    readonly miss_scale_m: number;
    readonly failure_cost: number;
    readonly target_radius_m: number;
    readonly miss_distance_cvar_alpha: number;
  };
}

export type WindOutcomeStatus = "completed" | "nonconverged" | "invalid";

export interface PerfectInformationCounterfactual {
  readonly status: WindOutcomeStatus;
  readonly landing_forward_m: number | null;
  readonly landing_right_m: number | null;
  readonly miss_distance_m: number | null;
  readonly cost: number;
  readonly failure_reason: string | null;
}

export interface WindStrategyOutcome {
  readonly trial_index: number;
  readonly strategy_id: string;
  readonly status: WindOutcomeStatus;
  readonly true_wind: WindScenario;
  readonly estimated_wind: WindScenario;
  readonly landing_forward_m: number | null;
  readonly landing_right_m: number | null;
  readonly miss_distance_m: number | null;
  readonly cost: number;
  readonly failure_reason: string | null;
  readonly perfect_information: PerfectInformationCounterfactual;
  readonly information_cost_delta: number;
}

export interface DirectionalRisk {
  readonly probability: number;
  readonly mean_excess_m: number;
  readonly conditional_mean_excess_m: number;
}

export interface WindStrategySummary {
  readonly strategy_id: string;
  readonly label: string;
  readonly completed_trials: number;
  readonly failed_trials: number;
  readonly expected_cost: number;
  readonly expected_perfect_information_cost: number;
  readonly expected_information_cost_delta: number;
  readonly expected_preset_oracle_regret: number;
  readonly preset_oracle_probability_best: number;
  readonly expected_regret: number;
  readonly probability_best: number;
  readonly target_hold_probability: number;
  readonly miss_distance_cvar_m: number;
  readonly miss_distance_cvar_alpha: number;
  readonly short_risk: DirectionalRisk;
  readonly long_risk: DirectionalRisk;
  readonly left_risk: DirectionalRisk;
  readonly right_risk: DirectionalRisk;
  readonly mean_landing_forward_m: number | null;
  readonly mean_landing_right_m: number | null;
}

export interface WindStrategyAnalysis {
  readonly schema_version: typeof WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION;
  readonly provenance: string;
  readonly target: WindStrategyRequest["target"];
  readonly wind_trials: readonly WindTrial[];
  readonly outcomes: readonly WindStrategyOutcome[];
  readonly summaries: readonly WindStrategySummary[];
}

interface TrialContext {
  readonly request: WindStrategyRequest;
  readonly trial: WindTrial;
  readonly strategy: WindStrategy;
  readonly trueWind: WindScenario;
  readonly estimate: WindScenario;
}

interface SimulationOutcome {
  readonly status: WindOutcomeStatus;
  readonly landingForwardM: number | null;
  readonly landingRightM: number | null;
  readonly missDistanceM: number | null;
  readonly cost: number;
  readonly failureReason: string | null;
}

const finite = (value: number, name: string): number => {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
};

const rounded = (value: number): number => Number(value.toFixed(PRECISION_DIGITS));
const normalizeBearing = (value: number): number => rounded(((value + 180) % 360 + 360) % 360 - 180);

class Mulberry32 {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  uniform(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / UINT32_SCALE;
  }

  standardNormal(): number {
    const first = Math.max(MIN_UNIFORM, this.uniform());
    const second = this.uniform();
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
  }
}

function validateDistribution(distribution: ScalarDistribution, name: string): void {
  if (!["fixed", "normal", "uniform"].includes(distribution.kind)) {
    throw new RangeError(`${name}.kind is unsupported`);
  }
  finite(distribution.center, `${name}.center`);
  const spread = distribution.spread ?? 0;
  finite(spread, `${name}.spread`);
  if (spread < 0) throw new RangeError(`${name}.spread must be nonnegative`);
  if (distribution.kind === "fixed" && spread !== 0) {
    throw new RangeError(`${name} fixed distribution requires zero spread`);
  }
  if (distribution.minimum !== undefined) finite(distribution.minimum, `${name}.minimum`);
  if (distribution.maximum !== undefined) finite(distribution.maximum, `${name}.maximum`);
  if (distribution.minimum !== undefined && distribution.maximum !== undefined &&
      distribution.minimum > distribution.maximum) {
    throw new RangeError(`${name}.minimum must not exceed maximum`);
  }
}

function draw(distribution: ScalarDistribution, generator: Mulberry32): number {
  const spread = distribution.spread ?? 0;
  let value = distribution.center;
  if (distribution.kind === "uniform") {
    value += spread * (2 * generator.uniform() - 1);
  } else if (distribution.kind === "normal") {
    value += spread * generator.standardNormal();
  }
  if (distribution.minimum !== undefined) value = Math.max(distribution.minimum, value);
  if (distribution.maximum !== undefined) value = Math.min(distribution.maximum, value);
  return rounded(value);
}

function validateUncertainty(spec: WindUncertaintySpec): void {
  if (spec.schema_version !== WIND_UNCERTAINTY_SCHEMA_VERSION) {
    throw new RangeError(`unsupported uncertainty schema: ${spec.schema_version}`);
  }
  if (!Number.isInteger(spec.trials) || spec.trials < 1 || spec.trials > 100000) {
    throw new RangeError("trials must be an integer in [1, 100000]");
  }
  if (!Number.isInteger(spec.seed) || spec.seed < 0 || spec.seed > UINT32_MAX) {
    throw new RangeError("seed must be a uint32 integer");
  }
  validateDistribution(spec.true_speed_mps, "true_speed_mps");
  validateDistribution(spec.true_from_bearing_deg, "true_from_bearing_deg");
  if (spec.true_speed_mps.minimum === undefined || spec.true_speed_mps.minimum < 0) {
    throw new RangeError("true_speed_mps requires a nonnegative minimum");
  }
  const error = spec.estimate_error;
  Object.entries(error).forEach(([name, value]) => finite(value, `estimate_error.${name}`));
  if (error.speed_std_mps < 0 || error.bearing_std_deg < 0) {
    throw new RangeError("estimate standard deviations must be nonnegative");
  }
  if (error.correlation < -1 || error.correlation > 1) {
    throw new RangeError("correlation must be in [-1, 1]");
  }
  if (!spec.provenance.trim()) throw new RangeError("provenance must be nonempty");
}

export function sampleWindTrials(spec: WindUncertaintySpec): WindTrial[] {
  validateUncertainty(spec);
  const generator = new Mulberry32(spec.seed);
  const correlationScale = Math.sqrt(Math.max(0, 1 - spec.estimate_error.correlation ** 2));
  return Array.from({ length: spec.trials }, (_, trialIndex) => {
    const trueSpeed = draw(spec.true_speed_mps, generator);
    const trueBearing = normalizeBearing(draw(spec.true_from_bearing_deg, generator));
    const speedNormal = generator.standardNormal();
    const independentNormal = generator.standardNormal();
    const bearingNormal = spec.estimate_error.correlation * speedNormal +
      correlationScale * independentNormal;
    const speedError = rounded(spec.estimate_error.speed_bias_mps +
      spec.estimate_error.speed_std_mps * speedNormal);
    const bearingError = rounded(spec.estimate_error.bearing_bias_deg +
      spec.estimate_error.bearing_std_deg * bearingNormal);
    const estimatedSpeed = rounded(Math.max(0, trueSpeed + speedError));
    return {
      trial_index: trialIndex,
      true_speed_mps: trueSpeed,
      true_from_bearing_deg: trueBearing,
      estimated_speed_mps: estimatedSpeed,
      estimated_from_bearing_deg: normalizeBearing(trueBearing + bearingError),
      speed_error_mps: rounded(estimatedSpeed - trueSpeed),
      bearing_error_deg: bearingError,
    };
  });
}

function scenario(speed: number, bearing: number, provenance: string): WindScenario {
  return { ...meteorologicalWind(speed, bearing), provenance };
}

function validateRequest(request: WindStrategyRequest): void {
  finite(request.target.forward_m, "target.forward_m");
  finite(request.target.right_m, "target.right_m");
  if (!request.strategies.length) throw new RangeError("at least one strategy is required");
  const identifiers = request.strategies.map((strategy) => strategy.id);
  if (new Set(identifiers).size !== identifiers.length) {
    throw new RangeError("strategy ids must be unique");
  }
  request.strategies.forEach((strategy) => {
    if (!strategy.id.trim() || !strategy.label.trim()) throw new RangeError("strategy id and label are required");
    finite(strategy.crosswind_aim_gain_rad_per_mps, "crosswind aim gain");
    if (strategy.launch.windScenario) throw new RangeError("strategy launch must not contain wind");
    const launchScalars = [strategy.launch.ballSpeedMps, strategy.launch.launchAngleRad,
      strategy.launch.azimuthRad, strategy.launch.spinRpm, ...strategy.launch.spinAxis];
    if (!launchScalars.every(Number.isFinite)) throw new RangeError("strategy launch must be finite");
    if (strategy.launch.ballSpeedMps < 0 || strategy.launch.spinRpm < 0) {
      throw new RangeError("strategy launch speed and spin must be nonnegative");
    }
  });
  const analysis = request.analysis;
  [analysis.max_time_s, analysis.time_step_s, analysis.miss_scale_m].forEach((value) => {
    if (!(finite(value, "analysis value") > 0)) throw new RangeError("analysis scales must be positive");
  });
  if (finite(analysis.failure_cost, "failure_cost") < 0) throw new RangeError("failure_cost must be nonnegative");
  if (finite(analysis.target_radius_m, "target_radius_m") < 0) {
    throw new RangeError("target_radius_m must be nonnegative");
  }
  const alpha = finite(analysis.miss_distance_cvar_alpha, "miss_distance_cvar_alpha");
  if (alpha <= 0 || alpha >= 1) {
    throw new RangeError("miss_distance_cvar_alpha must be in (0, 1)");
  }
}

function failureSimulation(
  request: WindStrategyRequest,
  status: Exclude<WindOutcomeStatus, "completed">,
  reason: string,
): SimulationOutcome {
  return {
    status, landingForwardM: null, landingRightM: null, missDistanceM: null,
    cost: request.analysis.failure_cost, failureReason: reason,
  };
}

function simulatePolicy(context: TrialContext, decisionWind: WindScenario): SimulationOutcome {
  const { request, strategy, trueWind } = context;
  const correction = strategy.crosswind_aim_gain_rad_per_mps * decisionWind.baseVelocityMps[1];
  try {
    const result = simulateFlight({
      ...strategy.launch,
      azimuthRad: strategy.launch.azimuthRad - correction,
      windScenario: trueWind,
    }, request.analysis.max_time_s, request.analysis.time_step_s, 10);
    const landing = result.trajectory[result.trajectory.length - 1];
    if (!landing || landing.position[2] > GROUND_TOLERANCE_M) {
      return failureSimulation(request, "nonconverged", "ground not reached");
    }
    const forwardM = landing.position[0];
    const rightM = -landing.position[1];
    const missDistanceM = Math.hypot(
      forwardM - request.target.forward_m,
      rightM - request.target.right_m,
    );
    const cost = (missDistanceM / request.analysis.miss_scale_m) ** 2;
    if (![forwardM, rightM, cost].every(Number.isFinite)) {
      return failureSimulation(request,
        "invalid", "simulation produced nonfinite landing data");
    }
    return {
      status: "completed", landingForwardM: forwardM, landingRightM: rightM,
      missDistanceM, cost, failureReason: null,
    };
  } catch (error: unknown) {
    const reason = error instanceof Error ? error.message : "unknown simulation error";
    return failureSimulation(request, "invalid", reason);
  }
}

function counterfactual(result: SimulationOutcome): PerfectInformationCounterfactual {
  return {
    status: result.status, landing_forward_m: result.landingForwardM,
    landing_right_m: result.landingRightM, miss_distance_m: result.missDistanceM,
    cost: result.cost, failure_reason: result.failureReason,
  };
}

function outcome(request: WindStrategyRequest, trial: WindTrial, strategy: WindStrategy): WindStrategyOutcome {
  const provenance = request.uncertainty.provenance;
  const trueWind = scenario(trial.true_speed_mps, trial.true_from_bearing_deg,
    `${provenance}/true/trial-${trial.trial_index}`);
  const estimate = scenario(trial.estimated_speed_mps, trial.estimated_from_bearing_deg,
    `${provenance}/estimated/trial-${trial.trial_index}`);
  const context = { request, trial, strategy, trueWind, estimate };
  const actual = simulatePolicy(context, estimate);
  const perfectInformation = simulatePolicy(context, trueWind);
  return {
    trial_index: trial.trial_index, strategy_id: strategy.id, status: actual.status,
    true_wind: trueWind, estimated_wind: estimate,
    landing_forward_m: actual.landingForwardM, landing_right_m: actual.landingRightM,
    miss_distance_m: actual.missDistanceM, cost: actual.cost,
    failure_reason: actual.failureReason,
    perfect_information: counterfactual(perfectInformation),
    information_cost_delta: actual.cost - perfectInformation.cost,
  };
}

export function analyzeWindStrategies(
  request: WindStrategyRequest,
  onProgress?: (completed: number, total: number) => void,
): WindStrategyAnalysis {
  validateRequest(request);
  const windTrials = sampleWindTrials(request.uncertainty);
  const total = windTrials.length * request.strategies.length;
  let completed = 0;
  onProgress?.(0, total);
  const outcomes = windTrials.flatMap((trial) =>
    request.strategies.map((strategy) => {
      const result = outcome(request, trial, strategy);
      onProgress?.(++completed, total);
      return result;
    }),
  );
  return {
    schema_version: WIND_STRATEGY_ANALYSIS_SCHEMA_VERSION,
    provenance: request.uncertainty.provenance,
    target: request.target,
    wind_trials: windTrials,
    outcomes,
    summaries: summarizeStrategyOutcomes(request, outcomes),
  };
}

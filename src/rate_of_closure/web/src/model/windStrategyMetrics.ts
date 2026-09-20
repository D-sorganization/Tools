/** Pure summary metrics for paired wind-strategy outcomes. */

import type {
  DirectionalRisk,
  WindStrategyOutcome,
  WindStrategyRequest,
  WindStrategySummary,
} from "./windUncertainty";

const BEST_COST_TOLERANCE = 1e-12;

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function optionalMean(values: readonly (number | null)[]): number | null {
  const available = values.filter((value): value is number => value !== null);
  return available.length ? mean(available) : null;
}

function directionalRisk(excesses: readonly number[]): DirectionalRisk {
  const positive = excesses.filter((value) => value > 0);
  return {
    probability: positive.length / excesses.length,
    mean_excess_m: mean(excesses),
    conditional_mean_excess_m: positive.length ? mean(positive) : 0,
  };
}

function cvar(values: readonly number[], alpha: number): number {
  const tailCount = Math.max(1, Math.ceil((1 - alpha) * values.length));
  return mean([...values].sort((left, right) => right - left).slice(0, tailCount));
}

function effectiveMisses(
  request: WindStrategyRequest,
  cohort: readonly WindStrategyOutcome[],
): number[] {
  const failureDistance = request.analysis.miss_scale_m *
    Math.sqrt(request.analysis.failure_cost);
  return cohort.map((item) => item.miss_distance_m ?? failureDistance);
}

interface DirectionalExcesses {
  readonly short: readonly number[];
  readonly long: readonly number[];
  readonly left: readonly number[];
  readonly right: readonly number[];
}

function directionalExcesses(
  request: WindStrategyRequest,
  cohort: readonly WindStrategyOutcome[],
): DirectionalExcesses {
  const values = cohort.map((item) => {
    if (item.landing_forward_m === null || item.landing_right_m === null) {
      return { short: 0, long: 0, left: 0, right: 0 };
    }
    const forwardError = item.landing_forward_m - request.target.forward_m;
    const rightError = item.landing_right_m - request.target.right_m;
    return {
      short: Math.max(-forwardError, 0), long: Math.max(forwardError, 0),
      left: Math.max(-rightError, 0), right: Math.max(rightError, 0),
    };
  });
  return {
    short: values.map((item) => item.short), long: values.map((item) => item.long),
    left: values.map((item) => item.left), right: values.map((item) => item.right),
  };
}

function bestCosts(request: WindStrategyRequest, outcomes: readonly WindStrategyOutcome[]): number[] {
  return Array.from({ length: request.uncertainty.trials }, (_, trialIndex) =>
    Math.min(...outcomes.filter((item) => item.trial_index === trialIndex)
      .map((item) => item.cost)));
}

function bestCredit(
  cohort: readonly WindStrategyOutcome[],
  outcomes: readonly WindStrategyOutcome[],
  bestByTrial: readonly number[],
): number {
  return cohort.reduce((credit, item) => {
    const best = bestByTrial[item.trial_index];
    if (Math.abs(item.cost - best) > BEST_COST_TOLERANCE) return credit;
    const ties = outcomes.filter((peer) => peer.trial_index === item.trial_index &&
      Math.abs(peer.cost - best) <= BEST_COST_TOLERANCE).length;
    return credit + 1 / ties;
  }, 0);
}

function summarizeStrategy(
  request: WindStrategyRequest,
  strategy: WindStrategyRequest["strategies"][number],
  outcomes: readonly WindStrategyOutcome[],
  bestByTrial: readonly number[],
): WindStrategySummary {
  const cohort = outcomes.filter((item) => item.strategy_id === strategy.id);
  const completed = cohort.filter((item) => item.status === "completed");
  const presetRegret = mean(cohort.map((item) => item.cost - bestByTrial[item.trial_index]));
  const presetProbability = bestCredit(cohort, outcomes, bestByTrial) / cohort.length;
  const direction = directionalExcesses(request, cohort);
  const holdCount = cohort.filter((item) => item.miss_distance_m !== null &&
    item.miss_distance_m <= request.analysis.target_radius_m).length;
  return {
    strategy_id: strategy.id, label: strategy.label, completed_trials: completed.length,
    failed_trials: cohort.length - completed.length,
    expected_cost: mean(cohort.map((item) => item.cost)),
    expected_perfect_information_cost: mean(cohort.map((item) => item.perfect_information.cost)),
    expected_information_cost_delta: mean(cohort.map((item) => item.information_cost_delta)),
    expected_preset_oracle_regret: presetRegret,
    preset_oracle_probability_best: presetProbability,
    expected_regret: presetRegret, probability_best: presetProbability,
    target_hold_probability: holdCount / cohort.length,
    miss_distance_cvar_m: cvar(effectiveMisses(request, cohort),
      request.analysis.miss_distance_cvar_alpha),
    miss_distance_cvar_alpha: request.analysis.miss_distance_cvar_alpha,
    short_risk: directionalRisk(direction.short), long_risk: directionalRisk(direction.long),
    left_risk: directionalRisk(direction.left), right_risk: directionalRisk(direction.right),
    mean_landing_forward_m: optionalMean(completed.map((item) => item.landing_forward_m)),
    mean_landing_right_m: optionalMean(completed.map((item) => item.landing_right_m)),
  };
}

export function summarizeStrategyOutcomes(
  request: WindStrategyRequest,
  outcomes: readonly WindStrategyOutcome[],
): WindStrategySummary[] {
  const bestByTrial = bestCosts(request, outcomes);
  return request.strategies.map((strategy) =>
    summarizeStrategy(request, strategy, outcomes, bestByTrial));
}

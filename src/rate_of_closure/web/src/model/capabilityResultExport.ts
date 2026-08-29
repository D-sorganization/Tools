/** Stable, versioned exports for ranked capability alternatives. */

import type {
  OptimizationAlternative,
  OptimizationResult,
} from "./capabilityContract";
import type { ScalarEnsembleResult } from "./scalarEnsembleContract";
import { spreadsheetCsvCell } from "./scalarEnsembleCsv";

export const CAPABILITY_RESULT_EXPORT_SCHEMA = "capability-result-export/v1";

const parameterUnits = (
  ensemble: ScalarEnsembleResult<string>,
): Readonly<Record<string, string>> => {
  const entries = ensemble.variables
    .filter(({ key }) => key.startsWith("nominal."))
    .map(({ key, unit }) => [key.slice("nominal.".length), unit] as const);
  if (!entries.length) throw new RangeError("capability result export requires parameter units");
  if (new Set(entries.map(([parameterId]) => parameterId)).size !== entries.length) {
    throw new RangeError("capability result export requires unique parameter units");
  }
  return Object.freeze(Object.fromEntries(entries));
};

const requireMatchedResult = (
  result: OptimizationResult,
  ensemble: ScalarEnsembleResult<string>,
): void => {
  if (result.problemId !== ensemble.result_id) {
    throw new RangeError("result and ensemble IDs must match");
  }
};

const alternativeWire = (item: OptimizationAlternative) => ({
  club_id: item.clubId,
  confidence: item.confidence,
  cvar_miss_m: item.cvarMissM,
  dispersion_rms_m: item.dispersionRmsM,
  downside_carry_m: item.downsideCarryM,
  expected_miss_m: item.expectedMissM,
  extrapolated: item.extrapolated,
  failed_count: item.failedCount,
  failure_fraction: item.failureFraction,
  limiting_constraints: item.limitingConstraints,
  mean_carry_m: item.meanCarryM,
  no_impact_count: item.noImpactCount,
  parameters: item.parameters.map(({ parameterId, value }) => ({
    parameter_id: parameterId, value,
  })),
  pareto_efficient: item.paretoEfficient,
  rank: item.rank,
  sample_count: item.sampleCount,
  score: item.score,
  successful_count: item.successfulCount,
  target_hold_probability: item.targetHoldProbability,
});

const resultWire = (result: OptimizationResult) => ({
  alternatives: result.alternatives.map(alternativeWire),
  evaluations_attempted: result.evaluationsAttempted,
  evaluations_completed: result.evaluationsCompleted,
  failed_count: result.failedCount,
  no_impact_count: result.noImpactCount,
  problem_id: result.problemId,
  provenance: Object.entries(result.provenance).map(([key, value]) => ({ key, value })),
  schema_version: result.schemaVersion,
  status: result.status,
});

const stable = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(stable);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, item]) => [key, stable(item)]));
  }
  return value;
};

const parameterText = (
  item: OptimizationAlternative,
  units: Readonly<Record<string, string>>,
): string => item.parameters.map(({ parameterId, value }) => {
  const unit = units[parameterId];
  if (unit === undefined) throw new RangeError(`missing unit for ${parameterId}`);
  return `${parameterId}=${value.toPrecision(12)} ${unit}`;
}).join("; ");

const alternativeRow = (
  item: OptimizationAlternative,
  units: Readonly<Record<string, string>>,
): readonly (string | number | boolean)[] => [
  item.rank, item.clubId, parameterText(item, units), item.score, item.meanCarryM,
  item.expectedMissM, item.dispersionRmsM, item.targetHoldProbability,
  item.cvarMissM, item.downsideCarryM, item.sampleCount, item.successfulCount,
  item.noImpactCount, item.failedCount, item.failureFraction, item.confidence,
  item.extrapolated, item.paretoEfficient, item.limitingConstraints.join("; "),
];

const HEADERS = [
  "rank", "club_id", "parameters", "score", "mean_carry_m",
  "expected_miss_m", "dispersion_rms_m", "target_hold_probability",
  "cvar_miss_m", "downside_carry_m", "sample_count", "successful_count",
  "no_impact_count", "failed_count", "failure_fraction", "confidence",
  "extrapolated", "pareto_efficient", "limiting_constraints",
] as const;

/** Export every ranked diagnostic with unambiguous parameter units. */
export const capabilityAlternativesCsv = (
  result: OptimizationResult,
  ensemble: ScalarEnsembleResult<string>,
): string => {
  requireMatchedResult(result, ensemble);
  const units = parameterUnits(ensemble);
  // ⚡ Bolt Optimization: Replace chained array .map().join() with a single-pass loop
  // to eliminate intermediate array allocations and reduce GC pressure for large dataset exports.
  const allRows = [HEADERS, ...result.alternatives.map((item) => alternativeRow(item, units))];
  let csvText = "";
  for (let i = 0; i < allRows.length; i++) {
    if (i > 0) csvText += "\n";
    const row = allRows[i];
    for (let j = 0; j < row.length; j++) {
      if (j > 0) csvText += ",";
      csvText += spreadsheetCsvCell(row[j]);
    }
  }
  return csvText;
};

/** Export the strict result and its external unit declarations. */
export const stableCapabilityResultExportJson = (
  result: OptimizationResult,
  ensemble: ScalarEnsembleResult<string>,
): string => {
  requireMatchedResult(result, ensemble);
  return JSON.stringify(stable({
    parameter_units: Object.entries(parameterUnits(ensemble))
      .map(([parameter_id, unit]) => ({ parameter_id, unit })),
    result: resultWire(result),
    schema_version: CAPABILITY_RESULT_EXPORT_SCHEMA,
  }));
};

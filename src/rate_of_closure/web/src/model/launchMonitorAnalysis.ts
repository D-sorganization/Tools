/** UI-neutral, provenance-preserving launch-monitor statistical analysis. */

import { canonicalFingerprint, uniqueStrings } from "./launchMonitorFingerprint";
import { calculateCorrelations, calculateRegression } from "./launchMonitorAnalysisStatistics";
import {
  finiteLaunchMonitorScalar,
  LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION,
  type GroupAnalysis,
  type LaunchMonitorAnalysisRequest,
  type LaunchMonitorAnalysisResult,
  type LaunchMonitorRow,
} from "./launchMonitorAnalysisTypes";

export { parseLaunchMonitorFile, readLaunchMonitorFile } from "./launchMonitorFileParsing";
export { sha256Text } from "./launchMonitorFingerprint";
export { LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION } from "./launchMonitorAnalysisTypes";
export type {
  AnalysisMode,
  CoefficientEstimate,
  CorrelationEstimate,
  CorrelationMethod,
  GroupAnalysis,
  LaunchMonitorAnalysisRequest,
  LaunchMonitorAnalysisResult,
  LaunchMonitorRow,
  LaunchMonitorScalar,
  MissingPolicy,
  RegressionEstimate,
} from "./launchMonitorAnalysisTypes";

export function numericLaunchMonitorColumns(rows: LaunchMonitorRow[]): string[] {
  // ⚡ Bolt Optimization: Single-pass column counting avoids O(rows * columns) iterations and large flatMap allocations
  const counts = new Map<string, number>();
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    for (const key in row) {
      if ((counts.get(key) || 0) >= 3) continue;
      if (finiteLaunchMonitorScalar(row[key]) !== null) {
        counts.set(key, (counts.get(key) || 0) + 1);
      }
    }
  }
  const result: string[] = [];
  for (const [key, count] of counts.entries()) {
    if (count >= 3) result.push(key);
  }
  return result.sort();
}

const validate = (rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest): void => {
  if (!rows.length) throw new RangeError("At least one observation is required");
  if (!request.outcome || !request.predictors.length) {
    throw new RangeError("Select an outcome and predictors");
  }
  if (request.predictors.includes(request.outcome)) {
    throw new RangeError("outcome cannot also be a predictor");
  }
  if (new Set(request.predictors).size !== request.predictors.length) {
    throw new RangeError("predictors must be unique");
  }
  if (!(request.confidenceLevel > 0.5 && request.confidenceLevel < 1)) {
    throw new RangeError("confidenceLevel must be between 0.5 and 1");
  }
  if (request.minSamples < 3) throw new RangeError("minSamples must be at least 3");
  const selected = [request.outcome, ...request.predictors];
  const missing = [...selected, ...(request.groupBy ? [request.groupBy] : [])]
    .filter((column) => !rows.some((row) => column in row));
  if (missing.length) {
    throw new RangeError(`Columns not present: ${[...new Set(missing)].join(", ")}`);
  }
  const constants = selected.filter((column) => new Set(rows
    .map((row) => finiteLaunchMonitorScalar(row[column]))
    .filter((value) => value !== null)).size < 2);
  if (constants.length) {
    throw new RangeError(`Constant variables cannot be analyzed: ${constants.join(", ")}`);
  }
  if (request.missingPolicy === "fail" && rows.some((row) => selected.some(
    (column) => finiteLaunchMonitorScalar(row[column]) === null,
  ))) {
    throw new RangeError("Selected variables contain missing or non-numeric values");
  }
};

const observationScope = (
  rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest, selected: string[],
): { vendors: string[]; observationKinds: string[]; warnings: string[] } => {
  const vendors = uniqueStrings(rows, "monitor_vendor");
  if (selected.some((column) => column.startsWith("source::")) && vendors.length > 1) {
    throw new RangeError("source fields cannot be pooled across multiple monitors");
  }
  const observationKinds = uniqueStrings(rows, "observation_kind");
  if (!observationKinds.length) observationKinds.push("shot");
  const aggregate = observationKinds.some((kind) => kind.toLowerCase() !== "shot");
  if (aggregate && request.analysisMode !== "correlation") {
    throw new RangeError("Aggregate observations cannot enter regression");
  }
  if (aggregate && !request.allowAggregate) {
    throw new RangeError("Aggregate observations require allowAggregate=true");
  }
  const warnings: string[] = [];
  if (aggregate) {
    warnings.push("Aggregate correlations are descriptive only and may exhibit ecological bias.");
  }
  if (request.correlationMethod !== "pearson" && request.analysisMode !== "regression") {
    warnings.push("Analytical confidence intervals are only reported for Pearson correlation.");
  }
  return { vendors, observationKinds, warnings };
};

const groupedAnalyses = (
  rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest,
): GroupAnalysis[] => {
  if (!request.groupBy) return [];
  return uniqueStrings(rows, request.groupBy).map((value) => {
    const groupRows = rows.filter((row) => String(row[request.groupBy as string]) === value);
    try {
      const result = analyzeLaunchMonitorData(groupRows, { ...request, groupBy: undefined });
      return { groupValue: value, rowCount: groupRows.length,
        correlations: result.correlations, regression: result.regression, warnings: result.warnings };
    } catch (error) {
      return { groupValue: value, rowCount: groupRows.length, correlations: [], regression: null,
        warnings: [error instanceof Error ? error.message : String(error)] };
    }
  });
};

export function analyzeLaunchMonitorData(
  rows: LaunchMonitorRow[], request: LaunchMonitorAnalysisRequest,
): LaunchMonitorAnalysisResult {
  validate(rows, request);
  const selected = [request.outcome, ...request.predictors];
  const { vendors, observationKinds, warnings } = observationScope(rows, request, selected);
  const correlations = request.analysisMode === "regression"
    ? [] : calculateCorrelations(rows, request);
  const regression = request.analysisMode === "correlation"
    ? null : calculateRegression(rows, request);
  return {
    contractVersion: LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION,
    request: { ...request, predictors: [...request.predictors] },
    dataset: {
      rowCount: rows.length,
      completeRowCount: rows.filter((row) => selected.every(
        (column) => finiteLaunchMonitorScalar(row[column]) !== null,
      )).length,
      selectedColumns: selected,
      monitorVendors: vendors,
      sessionIds: uniqueStrings(rows, "session_id"),
      observationKinds,
      fingerprintSha256: canonicalFingerprint(rows, selected),
    },
    correlations,
    regression,
    groups: groupedAnalyses(rows, request),
    warnings,
  };
}

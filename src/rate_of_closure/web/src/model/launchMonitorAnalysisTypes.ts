/** Shared contracts for browser launch-monitor statistical analysis. */

export const LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION = "1.0.0" as const;

export type LaunchMonitorScalar = string | number | boolean | null;
export type LaunchMonitorRow = Record<string, LaunchMonitorScalar>;
export type AnalysisMode = "correlation" | "regression" | "comprehensive";
export type CorrelationMethod = "pearson" | "spearman" | "kendall";
export type MissingPolicy = "pairwise" | "listwise" | "fail";

export interface LaunchMonitorAnalysisRequest {
  outcome: string;
  predictors: string[];
  analysisMode: AnalysisMode;
  correlationMethod: CorrelationMethod;
  missingPolicy: MissingPolicy;
  groupBy?: string;
  confidenceLevel: number;
  minSamples: number;
  allowAggregate?: boolean;
}

export interface CorrelationEstimate {
  predictor: string;
  coefficient: number | null;
  pValue: number | null;
  adjustedPValue: number | null;
  ciLower: number | null;
  ciUpper: number | null;
  sampleCount: number;
  method: CorrelationMethod;
}

export interface CoefficientEstimate {
  estimate: number;
  standardError: number;
  tStatistic: number;
  pValue: number;
  ciLower: number;
  ciUpper: number;
}

export interface RegressionEstimate {
  sampleCount: number;
  rSquared: number;
  adjustedRSquared: number;
  coefficients: Record<string, CoefficientEstimate>;
  residualDiagnostics: {
    rmse: number;
    mae: number;
    residualMean: number;
    residualStd: number;
    durbinWatson: number | null;
    influentialCount: number;
  };
}

export interface GroupAnalysis {
  groupValue: string;
  rowCount: number;
  correlations: CorrelationEstimate[];
  regression: RegressionEstimate | null;
  warnings: string[];
}

export interface LaunchMonitorAnalysisResult {
  contractVersion: typeof LAUNCH_MONITOR_ANALYSIS_CONTRACT_VERSION;
  request: LaunchMonitorAnalysisRequest;
  dataset: {
    rowCount: number;
    completeRowCount: number;
    selectedColumns: string[];
    monitorVendors: string[];
    sessionIds: string[];
    observationKinds: string[];
    fingerprintSha256: string;
  };
  correlations: CorrelationEstimate[];
  regression: RegressionEstimate | null;
  groups: GroupAnalysis[];
  warnings: string[];
}

export const finiteLaunchMonitorScalar = (
  value: LaunchMonitorScalar | undefined,
): number | null => {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return null;
  }
  if (typeof value === "string" &&
      !/^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/.test(value.trim())) {
    return null;
  }
  const converted = typeof value === "number" ? value : Number(value.trim());
  return Number.isFinite(converted) ? converted : null;
};

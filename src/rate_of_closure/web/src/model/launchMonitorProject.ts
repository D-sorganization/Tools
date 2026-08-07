import type { AnalysisMode, CorrelationMethod, LaunchMonitorRow, MissingPolicy } from "./launchMonitorAnalysis";
import type { CovariationUiSettings } from "./launchMonitorCovariation";

export const PROJECT_CONTRACT_VERSION = "1.0.0" as const;

export interface LaunchMonitorProject {
  contractVersion: typeof PROJECT_CONTRACT_VERSION;
  savedAt: string;
  sourceName: string;
  rows: LaunchMonitorRow[];
  settings: {
    outcome: string;
    predictors: string[];
    mode: AnalysisMode;
    method: CorrelationMethod;
    missing: MissingPolicy;
    groupBy: string;
    confidence: number;
    minSamples: number;
    targetDistanceYards: number;
    covariation?: CovariationUiSettings;
  };
}

export function parseLaunchMonitorProject(text: string): LaunchMonitorProject {
  const project: unknown = JSON.parse(text);
  if (!project || typeof project !== "object" || Array.isArray(project)) {
    throw new RangeError("Project must be a JSON object");
  }
  const candidate = project as Partial<LaunchMonitorProject>;
  if (candidate.contractVersion !== PROJECT_CONTRACT_VERSION) {
    throw new RangeError(`Unsupported project contract: ${String(candidate.contractVersion)}`);
  }
  if (!Array.isArray(candidate.rows) || !candidate.settings || typeof candidate.sourceName !== "string") {
    throw new RangeError("Project is missing rows, settings, or source name");
  }
  return candidate as LaunchMonitorProject;
}

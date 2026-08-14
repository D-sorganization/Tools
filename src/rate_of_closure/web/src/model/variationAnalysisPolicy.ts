import { runVariation, type VariationDatasetTs, type VariationPlanTs } from "./variation";
import {
  oneAtATimeSensitivity,
  type SensitivityResultTs,
} from "./variationAnalysis";

/** UI execution policy; deliberately excluded from the physical plan schema. */
export type VariationAnalysisExecution = "all_together" | "individual" | "both";

export interface VariationAnalysisExecutionResult {
  dataset: VariationDatasetTs | null;
  sensitivity: SensitivityResultTs | null;
}

export interface VariationAnalysisExecutors {
  runTogether: (plan: VariationPlanTs) => VariationDatasetTs;
  runIndividually: (plan: VariationPlanTs) => SensitivityResultTs;
}

const DEFAULT_EXECUTORS: VariationAnalysisExecutors = {
  runTogether: runVariation,
  runIndividually: oneAtATimeSensitivity,
};

export function executeVariationAnalyses(
  plan: VariationPlanTs,
  execution: VariationAnalysisExecution,
  executors: VariationAnalysisExecutors = DEFAULT_EXECUTORS,
): VariationAnalysisExecutionResult {
  const runTogether = execution === "all_together" || execution === "both";
  const runIndividually = execution === "individual" || execution === "both";
  return {
    dataset: runTogether ? executors.runTogether(plan) : null,
    sensitivity: runIndividually ? executors.runIndividually(plan) : null,
  };
}

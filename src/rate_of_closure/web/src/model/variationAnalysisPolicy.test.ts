import { describe, expect, it, vi } from "vitest";

import { CATEGORY_LAUNCH, type VariationDatasetTs, type VariationPlanTs } from "./variation";
import {
  executeVariationAnalyses,
  type VariationAnalysisExecutors,
} from "./variationAnalysisPolicy";
import type { SensitivityResultTs } from "./variationAnalysis";

const BALL = `${CATEGORY_LAUNCH}.ball_speed_mph`;
const plan: VariationPlanTs = {
  mode: "launch",
  baseVariables: {},
  noise: [
    { variableKey: BALL, distribution: "normal", scale: 1, lower: null, upper: null },
  ],
  nRuns: 2,
  seed: 0,
  flightModel: "waterloo_penner",
};

const dataset = { plan } as VariationDatasetTs;
const sensitivity = { inputKeys: [BALL] } as SensitivityResultTs;

const executors = (): VariationAnalysisExecutors => ({
  runTogether: vi.fn(() => dataset),
  runIndividually: vi.fn(() => sensitivity),
});

describe("variation analysis execution policy", () => {
  it("runs only the jointly enabled analysis when requested", () => {
    const implementations = executors();
    expect(executeVariationAnalyses(plan, "all_together", implementations)).toEqual({
      dataset,
      sensitivity: null,
    });
    expect(implementations.runTogether).toHaveBeenCalledOnce();
    expect(implementations.runIndividually).not.toHaveBeenCalled();
  });

  it("runs only one-at-a-time analyses when requested", () => {
    const implementations = executors();
    expect(executeVariationAnalyses(plan, "individual", implementations)).toEqual({
      dataset: null,
      sensitivity,
    });
    expect(implementations.runTogether).not.toHaveBeenCalled();
    expect(implementations.runIndividually).toHaveBeenCalledOnce();
  });

  it("runs both analyses only for the explicit both policy", () => {
    const implementations = executors();
    expect(executeVariationAnalyses(plan, "both", implementations)).toEqual({
      dataset,
      sensitivity,
    });
    expect(implementations.runTogether).toHaveBeenCalledOnce();
    expect(implementations.runIndividually).toHaveBeenCalledOnce();
  });
});

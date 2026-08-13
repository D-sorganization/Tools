import { describe, expect, it } from "vitest";

import { executeVariationAnalyses } from "./variationAnalysisPolicy";
import {
  executeVariationWork,
  plannedVariationRuns,
  type VariationExecutionProgress,
} from "./variationExecutionService";
import { CATEGORY_LAUNCH, type VariationPlanTs } from "./variation";

const BALL_SPEED = `${CATEGORY_LAUNCH}.ball_speed_mph`;
const LAUNCH_ANGLE = `${CATEGORY_LAUNCH}.launch_angle_deg`;

const plan: VariationPlanTs = {
  mode: "launch",
  baseVariables: {
    [BALL_SPEED]: 154,
    [LAUNCH_ANGLE]: 13,
    [`${CATEGORY_LAUNCH}.launch_azimuth_deg`]: 0,
    [`${CATEGORY_LAUNCH}.spin_rpm`]: 2400,
    [`${CATEGORY_LAUNCH}.spin_axis_deg`]: 0,
  },
  noise: [
    { variableKey: BALL_SPEED, distribution: "normal", scale: 1, lower: null, upper: null },
    { variableKey: LAUNCH_ANGLE, distribution: "normal", scale: 0.5, lower: null, upper: null },
  ],
  nRuns: 4,
  seed: 93,
  flightModel: "waterloo_penner",
};

describe("variation execution authority", () => {
  it("preserves the existing deterministic results and reports completed evaluations", () => {
    const progress: VariationExecutionProgress[] = [];
    const result = executeVariationWork(
      { plan, analysisExecution: "both" },
      (value) => progress.push(value),
    );
    const expected = executeVariationAnalyses(plan, "both");

    expect(result.dataset).toEqual(expected.dataset);
    expect(result.sensitivity).toEqual(expected.sensitivity);
    expect(result.ensemble).toBeNull();
    expect(plannedVariationRuns(plan, "both")).toBe(12);
    expect(progress[progress.length - 1]).toEqual({
      completedRuns: 12,
      totalRuns: 12,
      phase: "individual",
    });
    expect(progress.map(({ completedRuns }) => completedRuns)).toEqual(
      Array.from({ length: 12 }, (_unused, index) => index + 1),
    );
  });
});

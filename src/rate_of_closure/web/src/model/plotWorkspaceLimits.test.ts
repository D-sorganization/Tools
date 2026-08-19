import { describe, expect, it } from "vitest";

import {
  MAX_MANAGED_PLOTS,
  MAX_SWEEP_EVALUATIONS,
  plotEvaluationCount,
  validatePlotWorkspace,
} from "./plotWorkspaceLimits";
import { BUILTIN_PLOTS, type PlotSpec } from "./plotspec";

const builtin = (name: string): PlotSpec => {
  const entry = BUILTIN_PLOTS.find((candidate) => candidate.name === name);
  if (!entry) throw new Error(`missing builtin ${name}`);
  return entry.make(1.5);
};
const sweep = (count: number): PlotSpec => ({
  ...builtin("closure_sweep"),
  x_count: count,
});

describe("managed plot workspace resource authority", () => {
  it("accepts the exact plot and sweep boundaries", () => {
    const series = builtin("swing_time_series");
    const specs = [sweep(256), sweep(256), ...Array<PlotSpec>(6).fill(series)];

    expect(specs).toHaveLength(MAX_MANAGED_PLOTS);
    expect(plotEvaluationCount(specs)).toBe(MAX_SWEEP_EVALUATIONS);
    expect(() => validatePlotWorkspace(specs)).not.toThrow();
  });

  it("fails closed above either resource boundary", () => {
    const series = builtin("swing_time_series");
    expect(() => validatePlotWorkspace(Array<PlotSpec>(9).fill(series)))
      .toThrow("at most 8 managed plots");
    expect(() => validatePlotWorkspace([sweep(257), sweep(256)]))
      .toThrow("at most 512 sweep evaluations");
  });

  it("does not charge series plots as simulation evaluations", () => {
    const series = builtin("swing_time_series");
    expect(plotEvaluationCount(Array<PlotSpec>(MAX_MANAGED_PLOTS).fill(series)))
      .toBe(0);
  });

  it("rejects a ninth series before workspace computation", () => {
    const series = builtin("swing_time_series");
    expect(() => validatePlotWorkspace([
      { ...series, y_keys: Array<string>(9).fill(series.y_keys[0]) },
    ])).toThrow(/at most 8 series/);
  });
});

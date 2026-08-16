/** Deterministic resource contract for the managed Plots workspace. */
import { type PlotSpec, validateSpec } from "./plotspec";
import { MAX_PLOT_SERIES } from "./plotPointInspector";

export const MAX_MANAGED_PLOTS = 8;
export const MAX_SWEEP_EVALUATIONS = 512;

export function plotEvaluationCount(specs: readonly PlotSpec[]): number {
  let total = 0;
  for (const spec of specs) {
    validateSpec(spec);
    if (spec.y_keys.length > MAX_PLOT_SERIES) {
      throw new Error(`plot supports at most ${MAX_PLOT_SERIES} series`);
    }
    if (spec.kind === "sweep") total += spec.x_count;
  }
  return total;
}

export function validatePlotWorkspace(specs: readonly PlotSpec[]): void {
  if (specs.length > MAX_MANAGED_PLOTS) {
    throw new Error(`workspace supports at most ${MAX_MANAGED_PLOTS} managed plots`);
  }
  const evaluations = plotEvaluationCount(specs);
  if (evaluations > MAX_SWEEP_EVALUATIONS) {
    throw new Error(
      `workspace supports at most ${MAX_SWEEP_EVALUATIONS} sweep evaluations`,
    );
  }
}

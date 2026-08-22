import performanceDocument from "../vendored/visualization_performance.v1.json" with { type: "json" };

import { visualizationTabs, type VisualizationSurface } from "./visualizationTabManifest";

export interface VisualizationPerformanceBudget {
  tabOpenBudgetMs: number;
  resizeSettleBudgetMs: number;
  stableFrameCount: number;
  stabilityTolerancePx: number;
  maxPostSettleShiftPx: number;
  maxLayoutShiftScoreMicrounits: number | null;
}

export interface VisualizationPerformanceTab {
  surface: VisualizationSurface;
  tabId: string;
  workload: "initial-production-state";
}

const exactRecord = (value: unknown, keys: readonly string[], context: string) => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${context} must be an object`);
  }
  const result = Object.fromEntries(Object.entries(value));
  if (Object.keys(result).sort().join("|") !== [...keys].sort().join("|")) {
    throw new Error(`${context} fields must be exact`);
  }
  return result;
};

const boundedInteger = (
  value: unknown, context: string, minimum: number, maximum: number,
): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) ||
      value < minimum || value > maximum) {
    throw new Error(`${context} is outside its integer domain`);
  }
  return value;
};

const text = (value: unknown, context: string): string => {
  if (typeof value !== "string" || value.trim().length === 0 || value.length > 200) {
    throw new Error(`${context} must be bounded nonempty text`);
  }
  return value;
};

const deepFreeze = <T>(value: T): T => {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.values(value).forEach((child) => deepFreeze(child));
    Object.freeze(value);
  }
  return value;
};

const parseBudget = (
  value: unknown, surface: VisualizationSurface,
): VisualizationPerformanceBudget => {
  const budget = exactRecord(value, [
    "tab_open_budget_ms", "resize_settle_budget_ms", "stable_frame_count",
    "stability_tolerance_px", "max_post_settle_shift_px",
    "max_layout_shift_score_microunits",
  ], "surface budget");
  const rawCls = budget.max_layout_shift_score_microunits;
  const cls = rawCls === null ? null
    : boundedInteger(rawCls, "layout-shift score", 0, 1_000_000);
  if (surface === "react" && cls === null) throw new Error("React must declare browser CLS");
  if (surface === "pyqt" && cls !== null) throw new Error("PyQt cannot declare browser CLS");
  return {
    tabOpenBudgetMs: boundedInteger(budget.tab_open_budget_ms, "tab-open budget", 1, 10_000),
    resizeSettleBudgetMs: boundedInteger(
      budget.resize_settle_budget_ms, "resize budget", 1, 10_000,
    ),
    stableFrameCount: boundedInteger(budget.stable_frame_count, "stable frames", 2, 10),
    stabilityTolerancePx: boundedInteger(
      budget.stability_tolerance_px, "stability tolerance", 0, 10,
    ),
    maxPostSettleShiftPx: boundedInteger(
      budget.max_post_settle_shift_px, "post-settle shift", 0, 20,
    ),
    maxLayoutShiftScoreMicrounits: cls,
  };
};

export const parseVisualizationPerformanceManifest = (value: unknown) => {
  const document = exactRecord(value, [
    "schema_id", "schema_version", "measurement_policy", "surfaces", "tabs",
  ], "manifest");
  if (document.schema_id !== "rate-of-closure/visualization-performance-budgets" ||
      document.schema_version !== 1 || document.measurement_policy !==
      "protected-diagnostic-not-user-hardware-qualification") {
    throw new Error("unsupported performance manifest");
  }
  const surfaces = exactRecord(document.surfaces, ["react", "pyqt"], "surfaces");
  if (!Array.isArray(document.tabs)) throw new Error("tabs must be an array");
  const tabs = document.tabs.map((raw): VisualizationPerformanceTab => {
    const entry = exactRecord(raw, ["surface", "tab_id", "workload"], "tab");
    if (entry.surface !== "react" && entry.surface !== "pyqt") {
      throw new Error("unknown performance surface");
    }
    if (entry.workload !== "initial-production-state") {
      throw new Error("unsupported performance workload");
    }
    return { surface: entry.surface, tabId: text(entry.tab_id, "tab id"), workload: entry.workload };
  });
  const identities = tabs.map((entry) => `${entry.surface}:${entry.tabId}`);
  if (new Set(identities).size !== identities.length) {
    throw new Error("duplicate performance tab identity");
  }
  const expected = (["react", "pyqt"] as const).flatMap((surface) =>
    visualizationTabs(surface).map((entry) => `${surface}:${entry.tabId}`));
  if (identities.join("|") !== expected.join("|")) {
    throw new Error("performance tabs must exactly match visibility authority");
  }
  return deepFreeze({
    schemaId: document.schema_id,
    schemaVersion: document.schema_version,
    measurementPolicy: document.measurement_policy,
    surfaces: {
      react: parseBudget(surfaces.react, "react"),
      pyqt: parseBudget(surfaces.pyqt, "pyqt"),
    },
    tabs,
  });
};

const parsed = parseVisualizationPerformanceManifest(performanceDocument);
export const visualizationPerformanceBudgets = parsed.surfaces;
export const visualizationPerformanceTabs = (
  surface: VisualizationSurface,
): readonly VisualizationPerformanceTab[] => parsed.tabs.filter((entry) => entry.surface === surface);

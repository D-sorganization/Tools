import { describe, expect, it } from "vitest";
import performanceDocument from "../../../visualization_performance.v1.json" with { type: "json" };

import { PRIMARY_VIEW_IDS } from "./viewPreferences";
import {
  parseVisualizationPerformanceManifest,
  visualizationPerformanceBudgets,
  visualizationPerformanceTabs,
} from "./visualizationPerformanceManifest";

describe("visualization performance manifest", () => {
  it("exactly covers every React tab with bounded initial-state evidence", () => {
    expect(visualizationPerformanceTabs("react").map((entry) => entry.tabId))
      .toEqual(PRIMARY_VIEW_IDS);
    expect(visualizationPerformanceTabs("react").every(
      (entry) => entry.workload === "initial-production-state",
    )).toBe(true);
    expect(visualizationPerformanceBudgets.react).toEqual({
      tabOpenBudgetMs: 2500,
      resizeSettleBudgetMs: 1500,
      stableFrameCount: 3,
      stabilityTolerancePx: 1,
      maxPostSettleShiftPx: 2,
      maxLayoutShiftScoreMicrounits: 100000,
    });
    expect(visualizationPerformanceBudgets.pyqt).toEqual({
      tabOpenBudgetMs: 5000,
      resizeSettleBudgetMs: 4000,
      stableFrameCount: 3,
      stabilityTolerancePx: 1,
      maxPostSettleShiftPx: 2,
      maxLayoutShiftScoreMicrounits: null,
    });
  });

  it.each([
    ["zero open budget", "tab_open_budget_ms", 0],
    ["unbounded settle budget", "resize_settle_budget_ms", 10001],
    ["one stable frame", "stable_frame_count", 1],
    ["negative tolerance", "stability_tolerance_px", -1],
    ["boolean shift", "max_post_settle_shift_px", true],
    ["CLS above one", "max_layout_shift_score_microunits", 1000001],
  ])("rejects %s", (_label, field, value) => {
    const document = structuredClone(performanceDocument) as {
      surfaces: { react: Record<string, unknown> };
    };
    document.surfaces.react[field] = value;
    expect(() => parseVisualizationPerformanceManifest(document)).toThrow();
  });

  it("rejects coverage drift and returns deeply immutable evidence", () => {
    const missing = structuredClone(performanceDocument);
    missing.tabs.pop();
    expect(() => parseVisualizationPerformanceManifest(missing)).toThrow(/visibility authority/);

    const duplicate = structuredClone(performanceDocument);
    duplicate.tabs.push(structuredClone(duplicate.tabs[0]));
    expect(() => parseVisualizationPerformanceManifest(duplicate)).toThrow(/duplicate/);

    const parsed = parseVisualizationPerformanceManifest(performanceDocument);
    expect(Object.isFrozen(parsed)).toBe(true);
    expect(Object.isFrozen(parsed.tabs)).toBe(true);
    expect(Object.isFrozen(parsed.surfaces.react)).toBe(true);
  });
});

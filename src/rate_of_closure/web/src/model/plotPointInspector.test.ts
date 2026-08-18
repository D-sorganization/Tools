import { describe, expect, it } from "vitest";

import golden from "./__fixtures__/plot_point_inspector_golden_v1.json";
import {
  MAX_PLOT_SAMPLES,
  MAX_PLOT_VERTICES,
  histogramBinAtData,
  navigatePlotSelection,
  nearestSeriesPoint,
  planPlotInspection,
  type PlotNavigation,
  type PlotSelection,
} from "./plotPointInspector";

const selection = (value: Record<string, unknown> | null): PlotSelection | null => {
  if (value === null) return null;
  return value.kind === "series"
    ? { kind: "series", seriesIndex: Number(value.series_index), rawIndex: Number(value.raw_index) }
    : { kind: "histogram", binIndex: Number(value.bin_index) };
};

describe("plot point inspector", () => {
  it("matches the shared exact series pick and navigation", () => {
    const test = golden.series;
    const plan = planPlotInspection(test.kind, test.x, test.series);
    expect(plan.rawCount).toBe(4);
    expect(plan.series.map((item) => item.label)).toEqual(["Alpha", "Beta"]);
    expect(nearestSeriesPoint(plan, test.projected, test.tie_pointer)).toEqual(
      selection(test.tie_selection),
    );
    expect(nearestSeriesPoint(plan, test.projected, [0, 10])).toEqual({
      kind: "series", seriesIndex: 1, rawIndex: 0,
    });
    expect(nearestSeriesPoint(plan, test.projected, [0, 22.1])).toBeNull();
    for (const row of test.navigation) {
      const [current, command, expected] = row as unknown as [
        Record<string, unknown> | null,
        PlotNavigation,
        Record<string, unknown> | null,
      ];
      expect(navigatePlotSelection(plan, selection(current), command)).toEqual(
        selection(expected),
      );
    }
  });

  it("matches deterministic histogram bins and navigation", () => {
    const test = golden.histogram;
    const plan = planPlotInspection("histogram", test.x, []);
    expect(plan.bins.filter((item) => item.count).map((item) => [item.index, item.count]))
      .toEqual(test.nonzero_bins);
    expect(histogramBinAtData(plan, 1, 1)).toEqual({ kind: "histogram", binIndex: 5 });
    expect(histogramBinAtData(plan, 1, 2)).toBeNull();
    for (const row of test.navigation) {
      const [current, command, expected] = row as unknown as [
        Record<string, unknown> | null,
        PlotNavigation,
        Record<string, unknown> | null,
      ];
      expect(navigatePlotSelection(plan, selection(current), command)).toEqual(
        selection(expected),
      );
    }
  });

  it("rejects oversized and nonfinite evidence", () => {
    expect(() => planPlotInspection("histogram", new Array(MAX_PLOT_SAMPLES + 1), []))
      .toThrow(/samples/);
    expect(() => planPlotInspection("line", [0, Infinity], [{ label: "Y", values: [1, 2] }]))
      .toThrow(/finite bounded/);
    const x = Array.from({ length: MAX_PLOT_VERTICES / 2 + 1 }, (_, index) => index);
    expect(() => planPlotInspection("line", x, [
      { label: "A", values: x }, { label: "B", values: x },
    ])).toThrow(/vertices/);
  });
});

import { describe, expect, it } from "vitest";

import type { DatasetResponse } from "../../api/explorerSchemas";
import {
  columnValues,
  defaultPlotConfig,
  linePoints,
  reconcilePlotColumns,
  relativeSeconds,
  scatterPoints,
} from "./explorerState";

const dataset: DatasetResponse = {
  index: [0, 1000, 2000],
  columns: [
    { name: "a", values: [1, 2, 3] },
    { name: "b", values: [4, null, 6] },
  ],
  row_count: 3,
  truncated: false,
  sample_rate_hz: 1,
};

describe("relativeSeconds", () => {
  it("returns small indices unchanged", () => {
    expect(relativeSeconds([0, 1, 2])).toEqual([0, 1, 2]);
  });

  it("converts epoch-ms to seconds-from-start", () => {
    const base = 1_700_000_000_000;
    expect(relativeSeconds([base, base + 1000, base + 2000])).toEqual([0, 1, 2]);
  });

  it("handles empty input", () => {
    expect(relativeSeconds([])).toEqual([]);
  });
});

describe("linePoints", () => {
  it("pairs index with values and maps null -> NaN gap", () => {
    const pts = linePoints([0, 1, 2], [1, null, 3]);
    expect(pts[0]).toEqual([0, 1]);
    expect(Number.isNaN(pts[1][1])).toBe(true);
    expect(pts[2]).toEqual([2, 3]);
  });
});

describe("scatterPoints", () => {
  it("drops pairs with any null/non-finite member", () => {
    const pts = scatterPoints([1, 2, null, 4], [10, null, 30, 40]);
    expect(pts).toEqual([
      [1, 10],
      [4, 40],
    ]);
  });
});

describe("columnValues", () => {
  it("returns the named column values", () => {
    expect(columnValues(dataset, "b")).toEqual([4, null, 6]);
  });
  it("returns [] for a missing column or null dataset", () => {
    expect(columnValues(dataset, "zzz")).toEqual([]);
    expect(columnValues(null, "a")).toEqual([]);
  });
});

describe("reconcilePlotColumns", () => {
  it("keeps valid selections and repairs stale ones", () => {
    const cfg = {
      ...defaultPlotConfig(),
      columns: ["a", "gone"],
      xColumn: "gone",
      yColumn: "b",
    };
    const out = reconcilePlotColumns(cfg, ["a", "b"]);
    expect(out.columns).toEqual(["a"]);
    expect(out.xColumn).toBe("a"); // repaired to first
    expect(out.yColumn).toBe("b"); // still valid
  });

  it("is a no-op when there are no columns", () => {
    const cfg = defaultPlotConfig();
    expect(reconcilePlotColumns(cfg, [])).toBe(cfg);
  });
});

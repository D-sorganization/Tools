import { describe, it, expect } from "vitest";
import {
  resolveRange,
  formatTagIds,
  buildExportUrl,
  RANGE_PRESETS,
} from "./dataExport";

const NOW = Date.UTC(2026, 5, 29, 18, 30, 0); // fixed reference

describe("resolveRange", () => {
  it("computes trailing windows ending at now", () => {
    expect(resolveRange("15m", NOW)).toEqual({ startMs: NOW - 900_000, endMs: NOW });
    expect(resolveRange("1h", NOW)).toEqual({ startMs: NOW - 3_600_000, endMs: NOW });
    expect(resolveRange("6h", NOW)).toEqual({ startMs: NOW - 21_600_000, endMs: NOW });
  });

  it("'today' starts at local midnight and ends now", () => {
    const r = resolveRange("today", NOW);
    const midnight = new Date(NOW);
    midnight.setHours(0, 0, 0, 0);
    expect(r.startMs).toBe(midnight.getTime());
    expect(r.endMs).toBe(NOW);
    expect(r.startMs).toBeLessThanOrEqual(NOW);
  });

  it("'all' starts at the epoch", () => {
    expect(resolveRange("all", NOW)).toEqual({ startMs: 0, endMs: NOW });
  });

  it("validates inputs (DbC)", () => {
    expect(() => resolveRange("15m", Number.NaN)).toThrow(TypeError);
    // @ts-expect-error — exercising the runtime guard for an unknown preset
    expect(() => resolveRange("nope", NOW)).toThrow(RangeError);
  });
});

describe("formatTagIds", () => {
  it("joins, trims, and de-duplicates preserving order", () => {
    expect(formatTagIds([0, 1, 10])).toBe("0,1,10");
    expect(formatTagIds([" 0 ", "0", 1, "", 1])).toBe("0,1");
    expect(formatTagIds([])).toBe("");
  });
});

describe("buildExportUrl", () => {
  it("builds an /api/export URL with encoded params", () => {
    const url = buildExportUrl([0, 1], { startMs: NOW - 60_000, endMs: NOW });
    expect(url.startsWith("/api/export?")).toBe(true);
    expect(url).toContain("tag_ids=0%2C1");
    expect(url).toContain("start_time=");
    expect(url).toContain("end_time=");
    expect(url).toContain(encodeURIComponent(new Date(NOW).toISOString()));
  });

  it("rejects an empty tag set and an inverted range (DbC)", () => {
    expect(() => buildExportUrl([], { startMs: 0, endMs: NOW })).toThrow(RangeError);
    expect(() => buildExportUrl([0], { startMs: NOW, endMs: NOW })).toThrow(RangeError);
    expect(() => buildExportUrl([0], { startMs: NOW, endMs: NOW - 1 })).toThrow(RangeError);
  });
});

describe("RANGE_PRESETS", () => {
  it("exposes the selectable presets", () => {
    expect(RANGE_PRESETS.map((p) => p.id)).toEqual(["15m", "1h", "6h", "today", "all"]);
  });
});

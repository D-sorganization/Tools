import { describe, it, expect } from "vitest";
import { naiveUtcIso, parseHistorianTs } from "./useTrendBackfill";

describe("historian timestamp helpers", () => {
  const ms = Date.UTC(2026, 5, 29, 21, 0, 0); // 2026-06-29T21:00:00Z

  it("naiveUtcIso drops the zone suffix", () => {
    expect(naiveUtcIso(ms)).toBe("2026-06-29T21:00:00.000");
  });

  it("parseHistorianTs treats a zoneless stamp as UTC", () => {
    expect(parseHistorianTs("2026-06-29T21:00:00.000")).toBe(ms);
  });

  it("parseHistorianTs respects an explicit zone", () => {
    expect(parseHistorianTs("2026-06-29T21:00:00.000Z")).toBe(ms);
    expect(parseHistorianTs("2026-06-29T21:00:00.000+00:00")).toBe(ms);
  });

  it("round-trips naive UTC", () => {
    const t = Date.UTC(2026, 0, 2, 3, 4, 5);
    expect(parseHistorianTs(naiveUtcIso(t))).toBe(t);
  });
});

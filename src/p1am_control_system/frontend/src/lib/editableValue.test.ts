import { describe, it, expect } from "vitest";
import { decideCommit, parseNumber } from "./editableValue";

describe("parseNumber", () => {
  it("parses plain numbers and rejects blanks / trailing garbage", () => {
    expect(parseNumber("42")).toBe(42);
    expect(parseNumber(" -3.5 ")).toBe(-3.5);
    expect(Number.isNaN(parseNumber(""))).toBe(true);
    expect(Number.isNaN(parseNumber("   "))).toBe(true);
    expect(Number.isNaN(parseNumber("12abc"))).toBe(true);
  });
});

describe("decideCommit", () => {
  it("accepts a finite change", () => {
    expect(decideCommit("120", 100)).toEqual({ ok: true, value: 120, changed: true });
  });

  it("reports no change when the value is unchanged", () => {
    expect(decideCommit("100", 100)).toEqual({ ok: true, value: 100, changed: false });
  });

  it("clamps to [min, max]", () => {
    expect(decideCommit("5000", 100, { max: 1400 })).toEqual({
      ok: true,
      value: 1400,
      changed: true,
    });
    expect(decideCommit("-10", 100, { min: 0 })).toEqual({
      ok: true,
      value: 0,
      changed: true,
    });
  });

  it("treats a clamp back to the current value as unchanged", () => {
    // Typed above max but max IS the current value -> no write needed.
    expect(decideCommit("9999", 1400, { max: 1400 })).toEqual({
      ok: true,
      value: 1400,
      changed: false,
    });
  });

  it("rejects non-finite / non-numeric input without throwing", () => {
    const r = decideCommit("abc", 100);
    expect(r.ok).toBe(false);
    expect(r.changed).toBe(false);
    expect(r.value).toBe(100); // keeps the old value
    expect(decideCommit("", 100).ok).toBe(false);
  });

  it("honors a custom parser", () => {
    const parse = (raw: string): number => Number.parseInt(raw, 10) * 2;
    expect(decideCommit("10", 0, { parse })).toEqual({
      ok: true,
      value: 20,
      changed: true,
    });
  });

  it("throws on an inverted min/max range (DbC / caller bug)", () => {
    expect(() => decideCommit("1", 0, { min: 10, max: 5 })).toThrow(RangeError);
  });
});

import { describe, it, expect } from "vitest";
import { fmtNumber, fmtBytes, fmtDuration } from "./format";

describe("fmtNumber", () => {
  it("formats finite numbers to fixed decimals", () => {
    expect(fmtNumber(3.14159)).toBe("3.14");
    expect(fmtNumber(3.14159, 1)).toBe("3.1");
    expect(fmtNumber(0)).toBe("0.00");
  });

  it("returns the fallback for nullish or non-finite values", () => {
    expect(fmtNumber(undefined)).toBe("—");
    expect(fmtNumber(null)).toBe("—");
    expect(fmtNumber(NaN)).toBe("—");
    expect(fmtNumber(Infinity)).toBe("—");
    expect(fmtNumber(null, 2, "n/a")).toBe("n/a");
  });
});

describe("fmtBytes", () => {
  it("scales to the right unit", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(1024)).toBe("1.0 KB");
    expect(fmtBytes(1024 * 1024)).toBe("1.0 MB");
    expect(fmtBytes(1024 * 1024 * 1024)).toBe("1.0 GB");
  });

  it("returns em-dash for invalid input", () => {
    expect(fmtBytes(undefined)).toBe("—");
    expect(fmtBytes(NaN)).toBe("—");
  });
});

describe("fmtDuration", () => {
  it("formats seconds, minutes, and hours", () => {
    expect(fmtDuration(5)).toBe("5s");
    expect(fmtDuration(65)).toBe("1m 05s");
    expect(fmtDuration(3723)).toBe("1h 02m 03s");
  });

  it("clamps negatives and rejects invalid input", () => {
    expect(fmtDuration(-10)).toBe("0s");
    expect(fmtDuration(undefined)).toBe("—");
  });
});

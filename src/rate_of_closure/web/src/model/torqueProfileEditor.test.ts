import { describe, expect, it } from "vitest";

import {
  fitTorqueRows,
  loadTorqueProfileLibrary,
  saveTorqueProfileLibrary,
  starterTorqueProfile,
} from "./torqueProfileEditor";

class MemoryStorage implements Storage {
  private values = new Map<string, string>();
  get length() { return this.values.size; }
  clear() { this.values.clear(); }
  getItem(key: string) { return this.values.get(key) ?? null; }
  key(index: number) { return [...this.values.keys()][index] ?? null; }
  removeItem(key: string) { this.values.delete(key); }
  setItem(key: string, value: string) { this.values.set(key, value); }
}

describe("torque profile point fitting", () => {
  it("fits deterministic c0-first polynomials and preserves the sample domain", () => {
    const text = "0,5,-1\n1,4,3\n2,7,7\n3,14,11";
    const first = fitTorqueRows(text, 2);
    const second = fitTorqueRows(text, 2);
    expect(first.shoulder.coefficients).toEqual(second.shoulder.coefficients);
    expect(first.shoulder.coefficients).toEqual(expect.arrayContaining([
      expect.closeTo(5, 10), expect.closeTo(-3, 10), expect.closeTo(2, 10),
    ]));
    expect(first.wrist.coefficients).toEqual(expect.arrayContaining([
      expect.closeTo(-1, 10), expect.closeTo(4, 10), expect.closeTo(0, 10),
    ]));
    expect(first.rows.map((row) => row.timeS)).toEqual([0, 1, 2, 3]);
    expect(first.shoulder.fitMetadata?.degree).toBe(2);
  });

  it("rejects invalid degree, insufficient rows, and unordered samples", () => {
    expect(() => fitTorqueRows("0,0,0\n1,1,1", 4)).toThrow(/degree/i);
    expect(() => fitTorqueRows("0,0,0\n1,1,1", 2)).toThrow(/requires/i);
    expect(() => fitTorqueRows("1,0,0\n0,1,1", 1)).toThrow(/increasing/i);
    expect(() => fitTorqueRows("0,0,0\n0.00000000000001,1,1\n0.00000000000002,2,2\n1,3,3", 3))
      .toThrow(/condition number/i);
  });

  it("normalizes time before fitting and returns physical-time coefficients", () => {
    const fit = fitTorqueRows("1000,5,-1\n1001,4,3\n1002,7,7\n1003,14,11", 2);
    expect(fit.shoulder.evaluate(1000)).toBeCloseTo(5, 7);
    expect(fit.shoulder.evaluate(1003)).toBeCloseTo(14, 7);
    expect(fit.shoulder.fitMetadata?.conditionNumber).toBeLessThan(10);
  });

  it("round-trips the persistent profile library", () => {
    const storage = new MemoryStorage();
    const profile = starterTorqueProfile();
    saveTorqueProfileLibrary([profile], storage);
    expect(loadTorqueProfileLibrary(storage)[0].toJsonObject()).toEqual(profile.toJsonObject());
  });
});

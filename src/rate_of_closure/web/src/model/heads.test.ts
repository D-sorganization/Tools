/**
 * Type-specific head + hosel parity tests (H1, #4125) — mirrors
 * `tests/rate_of_closure/test_club_heads.py`.
 */

import { describe, expect, it } from "vitest";

import { buildParametricHead, getClub, type ClubSpec } from "./club";
import {
  BLADE_PUTTER_PROFILE,
  faceCenterPoint,
  hoselPoint,
  massScale,
  MALLET_PROFILE,
  PLUMBER_NECK_OFFSET_M,
  profileFor,
  resolvedStyle,
} from "./clubHeads";
import { CLUB_LIBRARY } from "./club";

function extents(club: ClubSpec): [number, number, number] {
  const flat = buildParametricHead(club).flat();
  const span = (k: number) =>
    Math.max(...flat.map((v) => v[k])) - Math.min(...flat.map((v) => v[k]));
  return [span(0), span(1), span(2)];
}

describe("type-specific proportions", () => {
  it("makes iron depth much less than wood depth", () => {
    expect(extents(getClub("7-Iron"))[0]).toBeLessThan(
      0.4 * extents(getClub("3-Wood"))[0],
    );
  });

  it("makes the hybrid intermediate between iron and wood", () => {
    const iron = extents(getClub("7-Iron"))[0];
    const hybrid = extents(getClub("3-Hybrid"))[0];
    const wood = extents(getClub("3-Wood"))[0];
    expect(iron).toBeLessThan(hybrid);
    expect(hybrid).toBeLessThan(wood);
  });

  it("makes the blade putter much shallower than the mallet", () => {
    expect(extents(getClub("Blade Putter"))[0]).toBeLessThan(
      0.5 * extents(getClub("Mallet Putter"))[0],
    );
  });

  it("keeps every library head deterministic", () => {
    for (const club of CLUB_LIBRARY) {
      expect(buildParametricHead(club)).toEqual(buildParametricHead(club));
    }
  });

  it("resolves putter styles to distinct profiles", () => {
    const blade = getClub("Blade Putter");
    const mallet = getClub("Mallet Putter");
    expect(resolvedStyle(blade)).toBe("Blade");
    expect(resolvedStyle(mallet)).toBe("Mallet");
    expect(profileFor(blade)).toBe(BLADE_PUTTER_PROFILE);
    expect(profileFor(mallet)).toBe(MALLET_PROFILE);
    expect(profileFor({ ...blade, headStyle: "Auto" })).toBe(
      BLADE_PUTTER_PROFILE,
    );
  });
});

describe("hosel points", () => {
  it("puts the hosel on the heel side for every club", () => {
    for (const club of CLUB_LIBRARY) {
      expect(hoselPoint(club)[2]).toBeLessThan(0);
    }
  });

  it("gives the blade putter its plumber's-neck set-back", () => {
    const blade = getClub("Blade Putter");
    const setback = faceCenterPoint(blade)[0] - hoselPoint(blade)[0];
    expect(setback).toBeCloseTo(PLUMBER_NECK_OFFSET_M * massScale(blade), 12);
  });

  it("pins the blade putter hosel against pytest", () => {
    const [x, y, z] = hoselPoint(getClub("Blade Putter"));
    expect(x).toBeCloseTo(0.0025, 12);
    expect(y).toBeCloseTo(0.0125, 12);
    expect(z).toBeCloseTo(-0.046, 12);
  });

  it("scales the hosel with head mass", () => {
    const wood = getClub("3-Wood");
    const heavy = { ...wood, headMassKg: wood.headMassKg * 2 };
    const ratio = massScale(heavy) / massScale(wood);
    const a = hoselPoint(heavy);
    const b = hoselPoint(wood);
    for (let k = 0; k < 3; k += 1) {
      expect(a[k]).toBeCloseTo(b[k] * ratio, 12);
    }
  });
});

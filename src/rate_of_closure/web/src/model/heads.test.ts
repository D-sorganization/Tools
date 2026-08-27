/**
 * Type-specific head + hosel parity tests (H1, #4125) — mirrors
 * `tests/rate_of_closure/test_club_heads.py`.
 */

import { describe, expect, it } from "vitest";

import { buildParametricHead, faceSagitta, getClub, type ClubSpec, type Vec3 } from "./club";
import {
  BLADE_PUTTER_PROFILE,
  faceCenterPoint,
  hoselPoint,
  leadingEdgeHeight,
  leanPoint,
  massScale,
  MALLET_PROFILE,
  PLUMBER_NECK_OFFSET_M,
  profileFor,
  resolvedStyle,
} from "./clubHeads";
import { CLUB_LIBRARY } from "./club";
import { isWatertight, meshVolumeCentroid } from "./volumetrics";

function extents(club: ClubSpec): [number, number, number] {
  const flat = buildParametricHead(club).flat();
  const span = (k: number) =>
    Math.max(...flat.map((v) => v[k])) - Math.min(...flat.map((v) => v[k]));
  return [span(0), span(1), span(2)];
}

function flatVertices(club: ClubSpec): Vec3[] {
  return buildParametricHead(club).flat();
}

function axisMax(flat: Vec3[], k: number): number {
  let best = -Infinity;
  for (const v of flat) if (v[k] > best) best = v[k];
  return best;
}

function axisMin(flat: Vec3[], k: number): number {
  let best = Infinity;
  for (const v of flat) if (v[k] < best) best = v[k];
  return best;
}

const BLADES = CLUB_LIBRARY.filter(
  (c) => c.clubType === "Iron" || c.clubType === "Wedge",
);
const DRIVERS = CLUB_LIBRARY.filter((c) => c.clubType === "Driver");

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

describe("leading-edge loft lean (#4799 G1)", () => {
  // Mirrors tests/rate_of_closure/test_club_heads.py::TestLeadingEdgeLean
  // test-for-test over the whole 16-club library.

  it("keeps the leading edge at the authored face station (no onset)", () => {
    for (const club of CLUB_LIBRARY) {
      const scale = massScale(club);
      const [x0, hh] = profileFor(club).sections[0];
      const expectedLe = [
        x0 * scale - faceSagitta(club, 0, -hh * scale),
        leadingEdgeHeight(club),
      ];
      const flat = flatVertices(club);
      const near = flat.some(
        (v) =>
          Math.abs(v[0] - expectedLe[0]) + Math.abs(v[1] - expectedLe[1]) <
          1e-6,
      );
      expect(near, `${club.name}: no vertex at the leading edge`).toBe(true);
      const unlofted = flatVertices({ ...club, loftDeg: 0 });
      expect(axisMax(flat, 0)).toBeLessThanOrEqual(
        axisMax(unlofted, 0) + 1e-9,
      );
    }
  });

  it("sets the topline back by slant height times sin(loft)", () => {
    for (const club of CLUB_LIBRARY) {
      const scale = massScale(club);
      const [x0, hh] = profileFor(club).sections[0];
      const lam = (club.loftDeg * Math.PI) / 180;
      const height = 2 * hh * scale;
      const expectedTop = [
        x0 * scale - faceSagitta(club, 0, hh * scale) - height * Math.sin(lam),
        leadingEdgeHeight(club) + height * Math.cos(lam),
      ];
      const flat = flatVertices(club);
      const top = flat.filter(
        (v) =>
          Math.abs(v[0] - expectedTop[0]) + Math.abs(v[1] - expectedTop[1]) <
          1e-6,
      );
      expect(top.length, `${club.name}: no vertex at the face top`).toBeGreaterThan(0);
      const setback = axisMax(flat, 0) - Math.min(...top.map((v) => v[0]));
      const expected = height * Math.sin(lam);
      expect(Math.abs(setback - expected)).toBeLessThanOrEqual(
        0.01 * expected + 1e-9,
      );
    }
  });

  it("keeps the sole height loft-invariant within 0.5 mm", () => {
    for (const club of CLUB_LIBRARY) {
      const lofted = axisMin(flatVertices(club), 1);
      const unlofted = axisMin(flatVertices({ ...club, loftDeg: 0 }), 1);
      expect(Math.abs(lofted - unlofted)).toBeLessThanOrEqual(5e-4);
    }
  });

  it("compresses the vertical extent by cos(loft)", () => {
    for (const club of CLUB_LIBRARY) {
      const flat = flatVertices(club);
      const unlofted = flatVertices({ ...club, loftDeg: 0 });
      const expected =
        (axisMax(unlofted, 1) - axisMin(unlofted, 1)) *
        Math.cos((club.loftDeg * Math.PI) / 180);
      expect(axisMax(flat, 1) - axisMin(flat, 1)).toBeCloseTo(expected, 9);
    }
  });

  it("leaves z untouched by loft", () => {
    for (const club of CLUB_LIBRARY) {
      const flat = flatVertices(club);
      const unlofted = flatVertices({ ...club, loftDeg: 0 });
      expect(flat.map((v) => v[2])).toEqual(unlofted.map((v) => v[2]));
    }
  });

  it("preserves watertightness and positive volume under the lean", () => {
    for (const club of CLUB_LIBRARY) {
      const triangles = buildParametricHead(club);
      expect(isWatertight(triangles), club.name).toBe(true);
      expect(meshVolumeCentroid(triangles).volumeM3).toBeGreaterThan(0);
    }
  });

  it("keeps the triangle count loft-invariant", () => {
    for (const club of CLUB_LIBRARY) {
      const sections = 3 * (profileFor(club).sections.length - 1) + 1;
      const expected = (2 * (sections - 1) + 2 * 4 + 2) * 64;
      expect(buildParametricHead(club)).toHaveLength(expected);
      expect(buildParametricHead({ ...club, loftDeg: 0 })).toHaveLength(
        expected,
      );
    }
  });

  it("puts the face-cap fan center at the leaned face center", () => {
    for (const club of CLUB_LIBRARY) {
      const center = faceCenterPoint(club);
      const hit = flatVertices(club).some(
        (v) =>
          Math.abs(v[0] - center[0]) +
            Math.abs(v[1] - center[1]) +
            Math.abs(v[2] - center[2]) <
          1e-12,
      );
      expect(hit, club.name).toBe(true);
    }
  });
});

describe("loft-aware hosel anchors (#4799 G2)", () => {
  // Mirrors tests/rate_of_closure/test_club_heads.py::TestHoselAnchors.

  it("gives every blade offset, never onset, within 8 mm of the LE", () => {
    for (const club of BLADES) {
      const hoselX = hoselPoint(club)[0];
      const leX = axisMax(flatVertices(club), 0);
      expect(hoselX, club.name).toBeLessThanOrEqual(leX);
      expect(leX - hoselX, club.name).toBeLessThanOrEqual(8e-3);
    }
  });

  it("puts the driver leading edge 20-40 mm ahead of the hosel", () => {
    for (const club of DRIVERS) {
      const gap = axisMax(flatVertices(club), 0) - hoselPoint(club)[0];
      expect(gap, club.name).toBeGreaterThanOrEqual(20e-3);
      expect(gap, club.name).toBeLessThanOrEqual(40e-3);
    }
  });

  it("leans the authored anchor for woods, hybrids, and putters", () => {
    for (const club of CLUB_LIBRARY) {
      if (club.clubType === "Iron" || club.clubType === "Wedge") continue;
      const profile = profileFor(club);
      const scale = massScale(club);
      const [ax, ay, az] = profile.hoselAnchor;
      const expected = leanPoint(club, [ax * scale, ay * scale, az * scale]);
      const point = hoselPoint(club);
      for (let k = 0; k < 3; k += 1) {
        expect(point[k], club.name).toBeCloseTo(expected[k], 12);
      }
    }
  });

  it("enters blades at the heel-face height fraction", () => {
    for (const club of BLADES) {
      const scale = massScale(club);
      const hh = profileFor(club).sections[0][1] * scale;
      const lam = (club.loftDeg * Math.PI) / 180;
      const expectedY =
        leadingEdgeHeight(club) + 0.58 * 2 * hh * Math.cos(lam);
      expect(hoselPoint(club)[1], club.name).toBeCloseTo(expectedY, 12);
    }
  });
});

describe("hosel points", () => {
  it("puts the hosel on the heel side for every club", () => {
    for (const club of CLUB_LIBRARY) {
      expect(hoselPoint(club)[2]).toBeLessThan(0);
    }
  });

  it("gives the blade putter its plumber's-neck set-back", () => {
    // #4799 G1/G2: face center and hosel are both leaned about the
    // leading edge; the hosel top rides one face half-height above the
    // face center, so the x gap grows by hh*sin(loft) on top of the
    // authored neck.
    const blade = getClub("Blade Putter");
    const scale = massScale(blade);
    const hh = profileFor(blade).sections[0][1] * scale;
    const setback = faceCenterPoint(blade)[0] - hoselPoint(blade)[0];
    const expected =
      PLUMBER_NECK_OFFSET_M * scale +
      hh * Math.sin((blade.loftDeg * Math.PI) / 180);
    expect(setback).toBeCloseTo(expected, 12);
  });

  it("pins the blade putter hosel against pytest", () => {
    // Repinned for the leaned authored anchor (#4799 G2).
    const [x, y, z] = hoselPoint(getClub("Blade Putter"));
    expect(x).toBeCloseTo(1.1916010939264044e-3, 12);
    expect(y).toBeCloseTo(1.2465738368864346e-2, 12);
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

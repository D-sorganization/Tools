/**
 * Divergence-theorem volumetrics parity tests (H1, #4125) — the pins
 * mirror `tests/rate_of_closure/test_club_heads.py::TestVolumetricsParity`
 * verbatim; the cube validates the algorithm against an analytic solid.
 */

import { describe, expect, it } from "vitest";

import { buildParametricHead, getClub, type Vec3 } from "./club";
import { type Triangle } from "./mesh";
import { headCog, isWatertight, meshVolumeCentroid } from "./volumetrics";

function cubeMesh(side = 2.0, center = 5.0): Triangle[] {
  const h = side / 2;
  const corners: Vec3[] = [];
  for (const sx of [-h, h]) {
    for (const sy of [-h, h]) {
      for (const sz of [-h, h]) corners.push([sx, sy, sz]);
    }
  }
  const faces: [number, number, number, number, Vec3][] = [
    [0, 1, 3, 2, [-1, 0, 0]],
    [4, 6, 7, 5, [1, 0, 0]],
    [0, 4, 5, 1, [0, -1, 0]],
    [2, 3, 7, 6, [0, 1, 0]],
    [0, 2, 6, 4, [0, 0, -1]],
    [1, 5, 7, 3, [0, 0, 1]],
  ];
  const triangles: Triangle[] = [];
  for (const [a, b, c, d, normal] of faces) {
    for (const tri of [
      [a, b, c],
      [a, c, d],
    ]) {
      let pts = tri.map((i) => corners[i]) as Triangle;
      const u = pts[1].map((v, k) => v - pts[0][k]);
      const w = pts[2].map((v, k) => v - pts[0][k]);
      const n: Vec3 = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
      ];
      if (n[0] * normal[0] + n[1] * normal[1] + n[2] * normal[2] < 0) {
        pts = [pts[2], pts[1], pts[0]];
      }
      triangles.push(
        pts.map((p) => p.map((v) => v + center) as Vec3) as Triangle,
      );
    }
  }
  return triangles;
}

describe("meshVolumeCentroid", () => {
  it("recovers the analytic cube volume and centroid exactly", () => {
    const { volumeM3, cog } = meshVolumeCentroid(cubeMesh());
    expect(volumeM3).toBeCloseTo(8.0, 12);
    for (const c of cog) expect(c).toBeCloseTo(5.0, 12);
  });

  it("rejects open meshes", () => {
    const open = cubeMesh().slice(0, -1);
    expect(isWatertight(open)).toBe(false);
    expect(() => meshVolumeCentroid(open)).toThrow(/watertight/);
  });

  it("rejects inward winding", () => {
    const inverted = cubeMesh().map(
      (tri) => [tri[2], tri[1], tri[0]] as Triangle,
    );
    expect(() => meshVolumeCentroid(inverted)).toThrow(/positive/);
  });

  it("finds every generated library head watertight", () => {
    for (const name of ["Driver 10.5°", "7-Iron", "Blade Putter"]) {
      expect(isWatertight(buildParametricHead(getClub(name)))).toBe(true);
    }
  });
});

describe("Python parity pins", () => {
  it("pins the driver head volume and COG", () => {
    // Repinned for the leading-edge loft lean (#4799 G1).
    const { volumeM3, cog } = meshVolumeCentroid(
      buildParametricHead(getClub("Driver 10.5°")),
    );
    expect(volumeM3).toBeCloseTo(5.704668123824279e-4, 15);
    expect(cog[0]).toBeCloseTo(1.9271184917634455e-3, 14);
    expect(cog[1]).toBeCloseTo(-4.688625881900988e-4, 14);
    expect(cog[2]).toBeCloseTo(0, 13);
  });

  it("pins the blade putter volume and COG", () => {
    // Repinned for the leading-edge loft lean (#4799 G1).
    const { volumeM3, cog } = meshVolumeCentroid(
      buildParametricHead(getClub("Blade Putter")),
    );
    expect(volumeM3).toBeCloseTo(4.634707753682171e-5, 15);
    expect(cog[0]).toBeCloseTo(4.6947209615312937e-4, 14);
    expect(cog[1]).toBeCloseTo(-1.9102497659159376e-3, 14);
  });

  it("reports both geometric and spec CG values", () => {
    const report = headCog(getClub("Driver 10.5°"));
    expect(report.specCgDepthM).toBeCloseTo(0.025, 12);
    expect(report.specCgHeightM).toBeCloseTo(0.028, 12);
    expect(report.cgDepthM).toBeGreaterThan(0);
    expect(report.cgHeightM).toBeGreaterThan(0);
    expect(report.volumeM3).toBeGreaterThan(0);
  });
});

/**
 * Profile-view acceptance gates for clubhead realism (#4799 G5) —
 * mirrors `tests/rate_of_closure/test_club_profile_acceptance.py`
 * test-for-test, and pins the same two rendered tables.
 *
 * Every measurement is taken from the toe-side profile view (a camera
 * on the toe side looking along the head frame's z axis) of the mesh
 * `parametricHeadMesh` returns — the public entry point `clubMeshSource`
 * renders — not of the internal builder. The silhouette is the mesh's
 * mid-plane slice (`|z| <= 1e-6 m`): each superellipse ring carries an
 * exact crown vertex (theta = pi/2) and an exact sole vertex
 * (theta = 3pi/2), and both cap fan centers sit on z = 0.
 */

import { describe, expect, it } from "vitest";

import {
  CLUB_LIBRARY,
  faceSagitta,
  parametricHeadMesh,
  type ClubSpec,
  type Vec3,
} from "./club";
import { faceCenterPoint, hoselPoint, massScale, profileFor } from "./clubHeads";
import {
  HEAD_VOLUME_BOUNDS_M3,
  isWatertight,
  meshVolumeCentroid,
} from "./volumetrics";

/** Half-width of the mid-plane slice that forms the toe-view outline. */
const PROFILE_Z_TOL_M = 1.0e-6;
/** Sole band, as a fraction of the profile height above the leading edge. */
const SOLE_BAND_FRACTION = 0.03;
/** Heights at which the toe-view front edge is sampled. */
const FRONT_EDGE_STEPS = 5;
/** Millimeter tolerance when a rendered row is compared with its pin. */
const REPORT_TOL_MM = 0.02;
/** Volume column tolerance, cm^3. */
const REPORT_TOL_CM3 = 0.2;

const BLADES = CLUB_LIBRARY.filter(
  (c) => c.clubType === "Iron" || c.clubType === "Wedge",
);
const IRONS = CLUB_LIBRARY.filter((c) => c.clubType === "Iron");
const WEDGES = CLUB_LIBRARY.filter((c) => c.clubType === "Wedge");
const DRIVERS = CLUB_LIBRARY.filter((c) => c.clubType === "Driver");
const BLADE_PUTTER = CLUB_LIBRARY.filter((c) => c.name === "Blade Putter");
/** Clubs with no bulge/roll — the analytic loft normal is exact here. */
const FLAT_FACED = CLUB_LIBRARY.filter(
  (c) => c.faceBulgeRadiusM === null && c.faceRollRadiusM === null,
);
/** Clubs whose topline is the face top (blades and the blade putter). */
const FACE_TOPPED = [...BLADES, ...BLADE_PUTTER];

interface ProfileMetrics {
  readonly name: string;
  readonly loftDeg: number;
  readonly leadingEdgeX: number;
  readonly leadingEdgeY: number;
  readonly authoredLeX: number;
  readonly authoredLeY: number;
  readonly hoselX: number;
  readonly toplineSetback: number;
  readonly expectedSetback: number;
  readonly toplineHeight: number;
  readonly expectedFaceHeight: number;
  readonly soleDepth: number;
  readonly soleFlatness: number;
  readonly soleGap: number;
  readonly soleFrontX: number;
  readonly solePoints: number;
  readonly profilePoints: number;
  readonly frontEdge: readonly number[];
  readonly width: number;
  readonly zSymmetry: number;
  readonly volumeCm3: number;
  readonly watertight: boolean;
  readonly faceNormalDeviation: number;
  readonly centerPivotLeX: number;
}

const unlofted = (club: ClubSpec): ClubSpec => ({ ...club, loftDeg: 0 });

/** Toe-view silhouette vertices of the rendered head, meters. */
function profileSlice(club: ClubSpec): Vec3[] {
  const seen = new Set<string>();
  const points: Vec3[] = [];
  for (const tri of parametricHeadMesh(club).triangles) {
    for (const v of tri) {
      if (Math.abs(v[2]) > PROFILE_Z_TOL_M) continue;
      const key = `${v[0]},${v[1]},${v[2]}`;
      if (seen.has(key)) continue;
      seen.add(key);
      points.push(v);
    }
  }
  return points;
}

/** The authored face height `H` at this club's mass scale [m]. */
function authoredFaceHeightM(club: ClubSpec): number {
  return 2 * profileFor(club).sections[0][1] * massScale(club);
}

/**
 * The unlofted, mass-scaled leading edge `(x, y)` [m]: the bottom of the
 * authored face section, set back by its own curvature sagitta. The
 * fixed line of the lean, and what the center-pivot counterfactual
 * rotates.
 */
function authoredLeadingEdgeM(club: ClubSpec): [number, number] {
  const scale = massScale(club);
  const [xFace, halfHeight, , yCenter] = profileFor(club).sections[0];
  return [
    xFace * scale - faceSagitta(club, 0, -halfHeight * scale),
    (yCenter - halfHeight) * scale,
  ];
}

const cache = new Map<string, ProfileMetrics>();

function metrics(club: ClubSpec): ProfileMetrics {
  const cached = cache.get(club.name);
  if (cached) return cached;

  const mesh = parametricHeadMesh(club);
  const flat = mesh.triangles.flat();
  const profile = profileSlice(club);
  const axis = (points: Vec3[], k: number) => points.map((p) => p[k]);

  const leadingEdgeX = Math.max(...axis(profile, 0));
  const lead = profile.find((p) => p[0] === leadingEdgeX) as Vec3;
  const leadingEdgeY = lead[1];
  const yMax = Math.max(...axis(profile, 1));
  const yMin = Math.min(...axis(profile, 1));
  const toplineX = Math.max(
    ...profile.filter((p) => Math.abs(p[1] - yMax) <= 1e-9).map((p) => p[0]),
  );

  const sole = profile
    .filter((p) => p[1] <= leadingEdgeY + SOLE_BAND_FRACTION * (yMax - yMin))
    .sort((a, b) => a[0] - b[0]);
  const soleXs = axis(sole, 0);
  const soleYs = axis(sole, 1);
  let soleGap = 0;
  for (let i = 1; i < soleXs.length; i += 1) {
    soleGap = Math.max(soleGap, soleXs[i] - soleXs[i - 1]);
  }

  const frontEdge: number[] = [];
  for (let step = 0; step <= FRONT_EDGE_STEPS; step += 1) {
    const cut = leadingEdgeY + ((yMax - leadingEdgeY) * step) / FRONT_EDGE_STEPS;
    const above = profile.filter((p) => p[1] >= cut - 1e-12);
    frontEdge.push(Math.max(...above.map((p) => p[0])) * 1e3);
  }

  const lam = (club.loftDeg * Math.PI) / 180;
  const want: Vec3 = [Math.cos(lam), Math.sin(lam), 0];
  const center = faceCenterPoint(club);
  const onCap = mesh.triangles
    .map((tri, index) => ({ tri, index }))
    .filter(({ tri }) =>
      tri.some(
        (v) =>
          Math.abs(v[0] - center[0]) +
            Math.abs(v[1] - center[1]) +
            Math.abs(v[2] - center[2]) <
          1e-12,
      ),
    );
  // A mesh that no longer realizes its own published face center — the
  // pre-#4799 center-pivot generator, for one — reports an infinite
  // deviation rather than throwing, so the gates fail on geometry.
  const faceNormalDeviation = onCap.length
    ? Math.max(
        ...onCap.map(({ index }) =>
          Math.hypot(
            mesh.normals[index][0] - want[0],
            mesh.normals[index][1] - want[1],
            mesh.normals[index][2] - want[2],
          ),
        ),
      )
    : Number.POSITIVE_INFINITY;

  const [authoredX, authoredY] = authoredLeadingEdgeM(club);
  const unloftedCenter = faceCenterPoint(unlofted(club));
  const centerPivotX =
    unloftedCenter[0] +
    Math.cos(lam) * (authoredX - unloftedCenter[0]) -
    Math.sin(lam) * (authoredY - unloftedCenter[1]);

  const zs = flat.map((v) => v[2]);
  const measured: ProfileMetrics = {
    name: club.name,
    loftDeg: club.loftDeg,
    leadingEdgeX: leadingEdgeX * 1e3,
    leadingEdgeY: leadingEdgeY * 1e3,
    authoredLeX: authoredX * 1e3,
    authoredLeY: authoredY * 1e3,
    hoselX: hoselPoint(club)[0] * 1e3,
    toplineSetback: (leadingEdgeX - toplineX) * 1e3,
    expectedSetback: authoredFaceHeightM(club) * Math.sin(lam) * 1e3,
    toplineHeight: (yMax - leadingEdgeY) * 1e3,
    expectedFaceHeight: authoredFaceHeightM(club) * Math.cos(lam) * 1e3,
    soleDepth: (Math.max(...soleXs) - Math.min(...soleXs)) * 1e3,
    soleFlatness: (Math.max(...soleYs) - Math.min(...soleYs)) * 1e3,
    soleGap: soleGap * 1e3,
    soleFrontX: Math.max(...soleXs) * 1e3,
    solePoints: sole.length,
    profilePoints: profile.length,
    frontEdge,
    width: (Math.max(...zs) - Math.min(...zs)) * 1e3,
    zSymmetry: (Math.max(...zs) + Math.min(...zs)) * 1e3,
    volumeCm3: meshVolumeCentroid(mesh.triangles).volumeM3 * 1e6,
    watertight: isWatertight(mesh.triangles),
    faceNormalDeviation,
    centerPivotLeX: centerPivotX * 1e3,
  };
  cache.set(club.name, measured);
  return measured;
}

const offsetOf = (m: ProfileMetrics) => m.leadingEdgeX - m.hoselX;
const pivotOffsetOf = (m: ProfileMetrics) => m.centerPivotLeX - m.hoselX;
const forwardKickOf = (m: ProfileMetrics) => m.centerPivotLeX - m.authoredLeX;

const pad = (text: string, width: number) =>
  width < 0 ? text.padEnd(-width) : text.padStart(width);
const cell = (value: number, width: number, digits: number) =>
  pad(value.toFixed(digits), width);

const PROFILE_HEADER =
  pad("club", -15) +
  pad("loft", 6) +
  pad("le_x", 7) +
  pad("hos_x", 7) +
  pad("setbk", 7) +
  pad("Hsin", 7) +
  pad("topH", 7) +
  pad("soleD", 7) +
  pad("soleF", 7) +
  pad("vol_cc", 8);

const ONSET_HEADER =
  pad("club", -15) +
  pad("le_x", 7) +
  pad("cp_le_x", 9) +
  pad("kick", 7) +
  pad("hos_x", 7) +
  pad("onset", 7) +
  pad("cp_onset", 10);

function profileRow(m: ProfileMetrics): string {
  return (
    pad(m.name, -15) +
    cell(m.loftDeg, 6, 1) +
    cell(m.leadingEdgeX, 7, 2) +
    cell(m.hoselX, 7, 2) +
    cell(m.toplineSetback, 7, 2) +
    cell(m.expectedSetback, 7, 2) +
    cell(m.toplineHeight, 7, 2) +
    cell(m.soleDepth, 7, 2) +
    cell(m.soleFlatness, 7, 3) +
    cell(m.volumeCm3, 8, 1)
  );
}

function onsetRow(m: ProfileMetrics): string {
  return (
    pad(m.name, -15) +
    cell(m.leadingEdgeX, 7, 2) +
    cell(m.centerPivotLeX, 9, 2) +
    cell(forwardKickOf(m), 7, 2) +
    cell(m.hoselX, 7, 2) +
    cell(offsetOf(m), 7, 2) +
    cell(pivotOffsetOf(m), 10, 2)
  );
}

/** The toe-view geometric report for the whole library, as a table. */
function profileReport(): string {
  return [PROFILE_HEADER, ...CLUB_LIBRARY.map((c) => profileRow(metrics(c)))].join(
    "\n",
  );
}

/** The center-pivot counterfactual table for the whole library. */
function onsetReport(): string {
  return [ONSET_HEADER, ...CLUB_LIBRARY.map((c) => onsetRow(metrics(c)))].join(
    "\n",
  );
}

/**
 * Pinned toe-view geometric report (mm; volume cm^3) — byte-for-byte the
 * table `test_club_profile_acceptance.py` pins. `setbk` is the topline
 * setback behind the leading edge, `Hsin`/`topH` the authored face
 * height times sin/cos(loft), `soleD`/`soleF` the sole line's
 * front-to-back depth and its flatness.
 */
const EXPECTED_PROFILE_REPORT = `club             loft   le_x  hos_x  setbk   Hsin   topH  soleD  soleF  vol_cc
Driver 9.5°       9.5  53.60  20.43  53.33   9.24  58.19  73.87  4.603   572.2
Driver 10.5°     10.5  53.60  19.43  54.35  10.21  58.01  73.90  4.589   570.5
Driver 12°       12.0  53.60  17.94  55.86  11.64  57.71  73.94  4.565   567.5
3-Wood           15.0  54.45  15.23  59.81  14.73  57.92  75.22  4.582   588.2
5-Wood           18.0  54.87  12.37  63.30  17.73  57.48  75.88  4.547   592.8
3-Hybrid         19.0  36.85   6.05  45.12  15.63  47.28  38.85  1.891   259.4
3-Iron           21.0  10.85   5.92  17.68  17.68  46.05  20.72  0.000    49.7
5-Iron           27.0  10.93   5.96  22.55  22.55  44.25  20.86  0.000    48.5
7-Iron           34.0  11.00   6.00  27.96  27.96  41.45  21.00  0.000    46.0
9-Iron           41.0  11.07   6.04  33.02  33.02  37.99  21.14  0.000    42.7
Pitching Wedge   46.0  11.87   8.40  36.99  36.99  35.72  27.97  0.549    54.3
Gap Wedge        52.0  11.93   8.45  40.75  40.75  31.84  28.11  0.490    49.0
Sand Wedge       56.0  12.00   8.50  43.11  43.11  29.08  28.25  0.447    45.3
Lob Wedge        60.0  12.07   8.55  45.28  45.28  26.14  28.40  0.402    41.1
Blade Putter      3.0  12.00   1.19   1.31   1.31  24.97  26.00  0.000    46.3
Mallet Putter     3.0  20.00  12.53  26.49   1.47  28.46  63.37  1.165   215.8`;

/**
 * Pinned center-pivot counterfactual (mm). `cp_le_x` is where the
 * leading edge would land if loft rotated the face about its center;
 * `kick` is the forward throw that costs (epic #4799 measured ~21.6 mm
 * on the sand wedge, ~14 mm on the 7-iron, ~5 mm on the driver), and
 * `cp_onset` the resulting onset past the shaft (~25 mm on a wedge).
 */
const EXPECTED_ONSET_REPORT = `club              le_x  cp_le_x   kick  hos_x  onset  cp_onset
Driver 9.5°      53.60    58.24   4.64  20.43  33.17     37.81
Driver 10.5°     53.60    58.72   5.13  19.43  34.17     39.29
Driver 12°       53.60    59.45   5.85  17.94  35.66     41.51
3-Wood           54.45    61.87   7.42  15.23  39.22     46.63
5-Wood           54.87    63.80   8.94  12.37  42.50     51.43
3-Hybrid         36.85    44.72   7.88   6.05  30.80     38.67
3-Iron           10.85    19.69   8.84   5.92   4.93     13.77
5-Iron           10.93    22.20  11.27   5.96   4.97     16.24
7-Iron           11.00    24.98  13.98   6.00   5.00     18.98
9-Iron           11.07    27.58  16.51   6.04   5.03     21.54
Pitching Wedge   11.87    30.36  18.49   8.40   3.46     21.95
Gap Wedge        11.93    32.31  20.37   8.45   3.48     23.85
Sand Wedge       12.00    33.55  21.55   8.50   3.50     25.05
Lob Wedge        12.07    34.71  22.64   8.55   3.52     26.16
Blade Putter     12.00    12.65   0.65   1.19  10.81     11.46
Mallet Putter    20.00    20.73   0.73  12.53   7.47      8.20`;

function parseReport(text: string): Map<string, number[]> {
  const parsed = new Map<string, number[]>();
  for (const line of text.split("\n").slice(1)) {
    parsed.set(
      line.slice(0, 15).trim(),
      line.slice(15).trim().split(/\s+/).map(Number),
    );
  }
  return parsed;
}

function expectReportMatches(
  actual: string,
  expected: string,
  volumeColumn: number,
): void {
  const got = parseReport(actual);
  const want = parseReport(expected);
  expect([...got.keys()].sort()).toEqual([...want.keys()].sort());
  for (const [name, wanted] of want) {
    const measured = got.get(name) as number[];
    expect(measured).toHaveLength(wanted.length);
    wanted.forEach((value, index) => {
      const tol = index === volumeColumn ? REPORT_TOL_CM3 : REPORT_TOL_MM;
      expect(
        Math.abs(measured[index] - value),
        `${name} column ${index}: ${measured[index]} vs pinned ${value}`,
      ).toBeLessThanOrEqual(tol);
    });
  }
}

describe("profile report (#4799 G5)", () => {
  it("matches the pinned toe-view geometric report", () => {
    expectReportMatches(profileReport(), EXPECTED_PROFILE_REPORT, 8);
  });

  it("matches the pinned center-pivot counterfactual report", () => {
    expectReportMatches(onsetReport(), EXPECTED_ONSET_REPORT, -1);
  });

  it("covers every library club with fixed-width rows", () => {
    const rows = profileReport().split("\n");
    expect(rows).toHaveLength(CLUB_LIBRARY.length + 1);
    for (const row of rows) expect(row).toHaveLength(PROFILE_HEADER.length);
  });

  it.each(CLUB_LIBRARY)("measures $name deterministically", (club) => {
    expect(profileSlice(club)).toEqual(profileSlice(club));
  });
});

describe("leading-edge station (#4799 G5)", () => {
  it.each(CLUB_LIBRARY)(
    "$name: the leading edge is the head's forward-most point",
    (club) => {
      const forward = Math.max(
        ...parametricHeadMesh(club).triangles.flat().map((v) => v[0]),
      );
      expect(metrics(club).leadingEdgeX).toBeCloseTo(forward * 1e3, 9);
    },
  );

  it.each(BLADES)(
    "$name: the leading edge is a few mm from the hosel station",
    (club) => {
      const offset = offsetOf(metrics(club));
      expect(offset).toBeGreaterThanOrEqual(0);
      expect(offset).toBeLessThanOrEqual(6);
    },
  );

  it.each(BLADES)(
    "$name: loft never leads the authored blade station",
    (club) => {
      const bare = Math.max(...profileSlice(unlofted(club)).map((p) => p[0]));
      expect(metrics(club).leadingEdgeX).toBeLessThanOrEqual(bare * 1e3 + 1e-9);
    },
  );

  it.each(DRIVERS)("$name: the face sits 20-40 mm ahead of the hosel", (club) => {
    const offset = offsetOf(metrics(club));
    expect(offset).toBeGreaterThanOrEqual(20);
    expect(offset).toBeLessThanOrEqual(40);
  });

  it.each(CLUB_LIBRARY)("$name: the hosel never leads the face", (club) => {
    expect(metrics(club).hoselX).toBeLessThanOrEqual(metrics(club).leadingEdgeX);
  });
});

describe("face lean (#4799 G5)", () => {
  it.each(CLUB_LIBRARY)("$name: the front edge recedes strictly", (club) => {
    const front = metrics(club).frontEdge;
    for (let i = 1; i < front.length; i += 1) {
      expect(front[i]).toBeLessThan(front[i - 1] - 1e-6);
    }
  });

  it.each(FACE_TOPPED)("$name: topline sets back by H*sin(loft)", (club) => {
    const m = metrics(club);
    expect(m.toplineSetback).toBeCloseTo(m.expectedSetback, 6);
  });

  it.each(CLUB_LIBRARY)(
    "$name: nothing tops out ahead of the face top",
    (club) => {
      const m = metrics(club);
      expect(m.toplineSetback).toBeGreaterThanOrEqual(m.expectedSetback - 1e-6);
    },
  );

  it.each(FACE_TOPPED)("$name: face height compresses to slant", (club) => {
    const m = metrics(club);
    expect(m.toplineHeight).toBeCloseTo(m.expectedFaceHeight, 6);
  });

  it.each(CLUB_LIBRARY)("$name: the face caps on faceCenterPoint", (club) => {
    expect(Number.isFinite(metrics(club).faceNormalDeviation)).toBe(true);
  });

  it.each(FLAT_FACED)("$name: flat face realizes (cos, sin, 0)", (club) => {
    expect(metrics(club).faceNormalDeviation).toBeLessThanOrEqual(1e-9);
  });

  it.each(CLUB_LIBRARY)(
    "$name: curved faces realize the loft normal to first order",
    (club) => {
      expect(metrics(club).faceNormalDeviation).toBeLessThanOrEqual(0.05);
    },
  );
});

describe("sole (#4799 G5)", () => {
  it.each([...BLADES, ...BLADE_PUTTER])(
    "$name: the sole line runs continuously from the leading edge",
    (club) => {
      const m = metrics(club);
      expect(m.solePoints).toBeGreaterThanOrEqual(8);
      expect(m.soleFrontX).toBeCloseTo(m.leadingEdgeX, 9);
      expect(m.soleGap).toBeLessThanOrEqual(0.25 * m.soleDepth);
    },
  );

  it.each([...IRONS, ...BLADE_PUTTER])("$name: the sole is flat", (club) => {
    expect(metrics(club).soleFlatness).toBeLessThanOrEqual(1e-9);
  });

  it.each(WEDGES)("$name: flat within a sub-mm bounce hint", (club) => {
    const flatness = metrics(club).soleFlatness;
    expect(flatness).toBeGreaterThanOrEqual(0.2);
    expect(flatness).toBeLessThanOrEqual(0.8);
  });

  it("every wedge sole is deeper than every iron sole", () => {
    const widestIron = Math.max(...IRONS.map((c) => metrics(c).soleDepth));
    const narrowestWedge = Math.min(...WEDGES.map((c) => metrics(c).soleDepth));
    expect(narrowestWedge - widestIron).toBeGreaterThanOrEqual(5);
  });

  it.each(BLADES)("$name: sole depth stays in the published span", (club) => {
    const reference = metrics(club).soleDepth / massScale(club);
    const [low, high] = club.clubType === "Iron" ? [18, 24] : [26, 32];
    expect(reference).toBeGreaterThanOrEqual(low);
    expect(reference).toBeLessThanOrEqual(high);
  });
});

describe("silhouette integrity (#4799 G5)", () => {
  it.each(CLUB_LIBRARY)("$name: watertight with a sane volume", (club) => {
    const m = metrics(club);
    expect(m.watertight).toBe(true);
    expect(m.volumeCm3).toBeGreaterThanOrEqual(HEAD_VOLUME_BOUNDS_M3[0] * 1e6);
    expect(m.volumeCm3).toBeLessThanOrEqual(HEAD_VOLUME_BOUNDS_M3[1] * 1e6);
  });

  it.each(CLUB_LIBRARY)("$name: z extents straddle the face center", (club) => {
    const m = metrics(club);
    expect(Math.abs(m.zSymmetry)).toBeLessThanOrEqual(1e-9);
    expect(m.width).toBeGreaterThan(0);
    expect(faceCenterPoint(club)[2]).toBe(0);
  });

  it.each(CLUB_LIBRARY)("$name: the toe view is a closed outline", (club) => {
    const stations = 3 * (profileFor(club).sections.length - 1) + 1;
    expect(metrics(club).profilePoints).toBe(2 * (stations - 1 + 5) + 2);
  });

  it.each(CLUB_LIBRARY)("$name: one unit normal per triangle", (club) => {
    const mesh = parametricHeadMesh(club);
    expect(mesh.normals).toHaveLength(mesh.triangles.length);
    for (const n of mesh.normals) {
      expect(Math.hypot(n[0], n[1], n[2])).toBeCloseTo(1, 12);
    }
  });
});

describe("center-pivot regression (#4799 G5)", () => {
  it.each(CLUB_LIBRARY)(
    "$name: the mesh does not match the center-pivot leading edge",
    (club) => {
      const m = metrics(club);
      expect(
        m.leadingEdgeX,
        `${club.name}: leading edge at ${m.leadingEdgeX} mm matches the ` +
          `center-pivot station ${m.centerPivotLeX} mm — center-pivot loft ` +
          "has been reintroduced",
      ).toBeLessThan(m.centerPivotLeX - 0.1);
    },
  );

  it.each(CLUB_LIBRARY)(
    "$name: the leading edge stays on its authored station",
    (club) => {
      const m = metrics(club);
      expect(m.leadingEdgeX).toBeCloseTo(m.authoredLeX, 9);
      expect(m.leadingEdgeY).toBeCloseTo(m.authoredLeY, 9);
    },
  );

  it.each(CLUB_LIBRARY)("$name: the forward kick is the closed form", (club) => {
    const lam = (club.loftDeg * Math.PI) / 180;
    const m = metrics(club);
    const halfHeight = authoredFaceHeightM(club) * 0.5 * 1e3;
    const sagitta = faceCenterPoint(unlofted(club))[0] * 1e3 - m.authoredLeX;
    const expected =
      halfHeight * Math.sin(lam) + sagitta * (1 - Math.cos(lam));
    expect(forwardKickOf(m)).toBeCloseTo(expected, 6);
    expect(forwardKickOf(m)).toBeGreaterThan(0.1);
  });

  it.each(BLADES)("$name: center pivot would reintroduce onset", (club) => {
    const m = metrics(club);
    expect(pivotOffsetOf(m)).toBeGreaterThanOrEqual(13);
    expect(offsetOf(m)).toBeLessThanOrEqual(6);
    expect(pivotOffsetOf(m)).toBeGreaterThanOrEqual(2.5 * offsetOf(m));
  });

  it("reproduces the epic's measured sand-wedge onset", () => {
    const wedge = CLUB_LIBRARY.find((c) => c.name === "Sand Wedge") as ClubSpec;
    const m = metrics(wedge);
    expect(m.centerPivotLeX).toBeCloseTo(33.6, 1);
    expect(pivotOffsetOf(m)).toBeCloseTo(25.1, 1);
    expect(forwardKickOf(m)).toBeCloseTo(21.6, 1);
    expect(offsetOf(m)).toBeCloseTo(3.5, 1);
  });
});

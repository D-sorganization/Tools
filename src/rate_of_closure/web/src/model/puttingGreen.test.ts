/**
 * Green surface, 2-D surface roll, capture model (#4800 P2).
 *
 * Mirrors `swing_sim/putting/tests/test_surface.py` test-for-test:
 * analytic gates first (flat -> straight, break sign vs cross-slope,
 * uphill/downhill asymmetry, capture window monotone in speed), then
 * the CRITICAL planar-limit regression — the legacy `simulatePutt`
 * delegates to the surface integrator and must stay bit-identical.
 * The numeric pins mirror the Python reference values.
 */

import { describe, expect, it } from "vitest";

import {
  captureSpeedMps,
  effectiveHoleRadiusM,
  gridSurface,
  HOLE_RADIUS_M,
  planarSurface,
  simulatePutt,
  simulatePuttOnSurface,
  surfaceHeightM,
  type GreenConditions,
  type GridGreenSurface,
  type PuttResult,
} from "./puttingGreen";
import { greenSurfaceFromJson, greenSurfaceToJson } from "./puttingGreenWire";
import { strike } from "./putting";

/** The H1 club-library putter (350 g, 3 deg, COR 0.78). */
const PUTTER = { name: "Putter", headMassKg: 0.35, loftDeg: 3.0, cor: 0.78 };

const FLAT = planarSurface(0, 0);

/** Grid heightfield sampled from the parametric plane. */
function planeGrid(
  gradePercent: number,
  aspectDeg: number,
  origin: [number, number] = [-2.0, -4.0],
  spacing = 0.5,
  nx = 41,
  ny = 17,
): GridGreenSurface {
  const plane = planarSurface(gradePercent, aspectDeg);
  const heights: number[][] = [];
  for (let j = 0; j < ny; j++) {
    const row: number[] = [];
    for (let i = 0; i < nx; i++) {
      row.push(surfaceHeightM(plane, origin[0] + i * spacing, origin[1] + j * spacing));
    }
    heights.push(row);
  }
  return gridSurface(origin, spacing, heights);
}

/** Bit-level result comparison (=== per sample; no tolerance). */
function expectSameResult(a: PuttResult, b: PuttResult): void {
  expect(a.pathXM.length).toBe(b.pathXM.length);
  for (let i = 0; i < a.pathXM.length; i++) {
    expect(a.pathXM[i] === b.pathXM[i]).toBe(true);
    expect(a.pathYM[i] === b.pathYM[i]).toBe(true);
    expect(a.speedsMps[i] === b.speedsMps[i]).toBe(true);
    expect(a.timesS[i] === b.timesS[i]).toBe(true);
  }
  expect(a.skidEndIndex).toBe(b.skidEndIndex);
  expect(a.skidDistanceM === b.skidDistanceM).toBe(true);
  expect(a.totalDistanceM === b.totalDistanceM).toBe(true);
  expect(a.timeS === b.timeS).toBe(true);
  expect(a.breakM === b.breakM).toBe(true);
  expect(a.holed).toBe(b.holed);
  expect(a.speedAtHoleMps === b.speedAtHoleMps).toBe(true);
  expect(a.marginMps === b.marginMps).toBe(true);
  expect(a.missDistanceM === b.missDistanceM).toBe(true);
}

describe("planar-limit regression (CRITICAL gate)", () => {
  it.each([
    [1.6, { stimpFt: 10, gradePercent: 0, aspectDeg: 0 }, 3.0],
    [3.2, { stimpFt: 10, gradePercent: 0, aspectDeg: 0 }, 3.0],
    [1.8, { stimpFt: 10, gradePercent: 2, aspectDeg: 90 }, 3.0],
    [1.6, { stimpFt: 13, gradePercent: 2, aspectDeg: 180 }, 20.0],
    [2.0, { stimpFt: 8, gradePercent: 3, aspectDeg: -45 }, 10.0],
  ] as [number, GreenConditions, number][])(
    "legacy API equals the surface integrator bitwise (%s m/s)",
    (clubhead, green, hole) => {
      const launch = strike(PUTTER, clubhead);
      const legacy = simulatePutt(launch, green, hole);
      const direct = simulatePuttOnSurface(
        launch,
        planarSurface(green.gradePercent, green.aspectDeg),
        {
          stimpFt: green.stimpFt,
          holeDistanceM: hole,
          captureModel: "speed_threshold",
        },
      );
      expectSameResult(legacy, direct);
    },
  );

  it("keeps the pre-#4800 reference pins verbatim", () => {
    const result = simulatePutt(
      strike(PUTTER, 1.8),
      { stimpFt: 10, gradePercent: 2, aspectDeg: 90 },
      3.0,
    );
    expect(result.totalDistanceM).toBeCloseTo(4.417405938785078, 10);
    expect(result.skidDistanceM).toBeCloseTo(0.5103817275162047, 10);
    expect(result.breakM).toBeCloseTo(0.8176068791755766, 10);
    expect(result.missDistanceM).toBeCloseTo(1.4994647284222105, 10);
    const holed = simulatePutt(
      strike(PUTTER, 1.6),
      { stimpFt: 10, gradePercent: 0, aspectDeg: 0 },
      3.0,
    );
    expect(holed.holed).toBe(true);
    expect(holed.speedAtHoleMps).toBeCloseTo(0.5903262895096224, 10);
    expect(holed.marginMps).toBeCloseTo(0.2283133618862715, 10);
  });
});

describe("flat -> straight line", () => {
  it("flat planar rolls dead straight", () => {
    const result = simulatePuttOnSurface(strike(PUTTER, 2.0), FLAT, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expect(result.pathYM.every((y) => y === 0)).toBe(true);
    expect(result.breakM).toBe(0);
  });

  it("flat grid matches flat planar bitwise", () => {
    const grid = gridSurface(
      [-2, -2],
      1.0,
      Array.from({ length: 5 }, () => Array.from({ length: 15 }, () => 0)),
    );
    const launch = strike(PUTTER, 2.0);
    const onGrid = simulatePuttOnSurface(launch, grid, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    const onPlane = simulatePuttOnSurface(launch, FLAT, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expectSameResult(onGrid, onPlane);
    expect(onGrid.pathYM.every((y) => y === 0)).toBe(true);
  });
});

describe("grid sampled from a plane matches the parametric form", () => {
  it.each([[90], [-90], [0], [180], [37]])("aspect %s deg", (aspectDeg) => {
    const launch = strike(PUTTER, 2.0);
    const plane = planarSurface(2.0, aspectDeg);
    const grid = planeGrid(2.0, aspectDeg);
    const a = simulatePuttOnSurface(launch, plane, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    const b = simulatePuttOnSurface(launch, grid, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expect(b.breakM).toBeCloseTo(a.breakM, 9);
    expect(b.totalDistanceM).toBeCloseTo(a.totalDistanceM, 8);
    expect(b.timeS).toBeCloseTo(a.timeS, 2);
    expect(b.holed).toBe(a.holed);
  });
});

describe("grid analytic gates", () => {
  it("cross-slope breaks toward the low side", () => {
    const leftLow = planeGrid(2.0, 90);
    const result = simulatePuttOnSurface(strike(PUTTER, 2.0), leftLow, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expect(result.breakM).toBeGreaterThan(0.01);
  });

  it("mirrored cross-slope mirrors the break", () => {
    const left = planeGrid(2.0, 90);
    const right = planeGrid(2.0, -90, [-2.0, -12.0], 0.5, 41, 33);
    const a = simulatePuttOnSurface(strike(PUTTER, 2.0), left, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    const b = simulatePuttOnSurface(strike(PUTTER, 2.0), right, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expect(a.breakM).toBeCloseTo(-b.breakM, 9);
    expect(a.totalDistanceM).toBeCloseTo(b.totalDistanceM, 9);
  });

  it("uphill rolls shorter than downhill for an equal launch", () => {
    const launch = strike(PUTTER, 1.6);
    const uphill = planeGrid(2.0, 180, [-2.0, -2.0], 0.5, 41, 9);
    const downhill = planeGrid(2.0, 0, [-2.0, -2.0], 0.5, 41, 9);
    const up = simulatePuttOnSurface(launch, uphill, {
      stimpFt: 10,
      holeDistanceM: 20,
    });
    const down = simulatePuttOnSurface(launch, downhill, {
      stimpFt: 10,
      holeDistanceM: 20,
    });
    expect(up.totalDistanceM).toBeLessThan(down.totalDistanceM);
  });

  it("continues flat beyond the grid hull (no runaway)", () => {
    const patch = Array.from({ length: 5 }, () =>
      Array.from({ length: 5 }, (_, i) => -0.02 * (i * 0.5)),
    );
    const surface = gridSurface([0, -1], 0.5, patch);
    const result = simulatePuttOnSurface(strike(PUTTER, 2.0), surface, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expect(result.totalDistanceM).toBeGreaterThan(2.0);
    expect(result.timeS).toBeLessThan(60.0);
  });

  it("repeat runs are identical", () => {
    const grid = planeGrid(2.0, 90);
    const a = simulatePuttOnSurface(strike(PUTTER, 2.0), grid, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    const b = simulatePuttOnSurface(strike(PUTTER, 2.0), grid, {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expectSameResult(a, b);
  });
});

describe("capture model", () => {
  it("pins the limiting cases of the effective radius", () => {
    const vC = captureSpeedMps();
    expect(effectiveHoleRadiusM(0)).toBe(HOLE_RADIUS_M);
    expect(effectiveHoleRadiusM(vC)).toBe(0);
    expect(effectiveHoleRadiusM(2 * vC)).toBe(0);
  });

  it("is strictly monotone in approach speed", () => {
    const vC = captureSpeedMps();
    let previous = Number.POSITIVE_INFINITY;
    for (let k = 0; k <= 40; k++) {
      const radius = effectiveHoleRadiusM((k * vC) / 40);
      expect(radius).toBeLessThan(previous);
      previous = radius;
    }
  });

  it("pins the reference radii (Python parity)", () => {
    expect(effectiveHoleRadiusM(0.5)).toBeCloseTo(0.04275766281973086, 12);
    expect(effectiveHoleRadiusM(0.8)).toBeCloseTo(0.011457634498570566, 12);
  });

  it("capture window is nested within the threshold window", () => {
    const effective = new Set<number>();
    const threshold = new Set<number>();
    for (let k = 0; k < 80; k++) {
      const launch = strike(PUTTER, 1.4 + 0.005 * k);
      if (
        simulatePuttOnSurface(launch, FLAT, { stimpFt: 10, holeDistanceM: 3 })
          .holed
      ) {
        effective.add(k);
      }
      if (
        simulatePuttOnSurface(launch, FLAT, {
          stimpFt: 10,
          holeDistanceM: 3,
          captureModel: "speed_threshold",
        }).holed
      ) {
        threshold.add(k);
      }
    }
    expect(effective.size).toBeGreaterThan(0);
    for (const k of effective) expect(threshold.has(k)).toBe(true);
    expect(effective.size).toBeLessThan(threshold.size);
  });

  it("holes a dying putt under both models", () => {
    const launch = strike(PUTTER, 1.6);
    expect(
      simulatePuttOnSurface(launch, FLAT, { stimpFt: 10, holeDistanceM: 3 })
        .holed,
    ).toBe(true);
    expect(
      simulatePuttOnSurface(launch, FLAT, {
        stimpFt: 10,
        holeDistanceM: 3,
        captureModel: "speed_threshold",
      }).holed,
    ).toBe(true);
  });

  it("firm edge pace discriminates the models", () => {
    const launch = strike(PUTTER, 1.69);
    const effective = simulatePuttOnSurface(launch, FLAT, {
      stimpFt: 10,
      holeDistanceM: 3,
    });
    const legacy = simulatePuttOnSurface(launch, FLAT, {
      stimpFt: 10,
      holeDistanceM: 3,
      captureModel: "speed_threshold",
    });
    expect(legacy.holed).toBe(true);
    expect(effective.holed).toBe(false);
  });

  it("pins the effective-capture reference putt (Python parity)", () => {
    const result = simulatePuttOnSurface(strike(PUTTER, 1.6), FLAT, {
      stimpFt: 10,
      holeDistanceM: 3,
    });
    expect(result.holed).toBe(true);
    expect(result.speedAtHoleMps).toBeCloseTo(0.5903262895096224, 10);
    expect(result.marginMps).toBeCloseTo(0.2283133618862715, 10);
    expect(result.totalDistanceM).toBeCloseTo(2.9618566583973425, 7);
    expect(result.timeS).toBeCloseTo(2.368, 2);
  });

  it("pins the grid cross-slope reference putt (Python parity)", () => {
    const result = simulatePuttOnSurface(strike(PUTTER, 2.0), planeGrid(2.0, 90), {
      stimpFt: 10,
      holeDistanceM: 10,
    });
    expect(result.holed).toBe(false);
    expect(result.breakM).toBeCloseTo(1.014167410853712, 7);
    expect(result.totalDistanceM).toBeCloseTo(5.476502851932196, 7);
    expect(result.skidDistanceM).toBeCloseTo(0.626653383578046, 7);
    expect(result.missDistanceM).toBeCloseTo(4.830211424915357, 7);
  });

  it("refuses unknown capture models and bad speeds", () => {
    expect(() =>
      simulatePuttOnSurface(strike(PUTTER, 1.6), FLAT, {
        stimpFt: 10,
        holeDistanceM: 3,
        captureModel: "lip_out" as never,
      }),
    ).toThrow(/capture model/);
    expect(() => effectiveHoleRadiusM(-0.1)).toThrow();
    expect(() => effectiveHoleRadiusM(Number.NaN)).toThrow();
  });
});

describe("surface validation", () => {
  it("rejects out-of-range planar parameters", () => {
    expect(() => planarSurface(-0.1, 0)).toThrow();
    expect(() => planarSurface(50, 0)).toThrow();
    expect(() => planarSurface(2, 361)).toThrow();
  });

  it("rejects malformed grids", () => {
    const flatRow = [0, 0, 0];
    expect(() => gridSurface([0, 0], 1, [flatRow])).toThrow();
    expect(() => gridSurface([0, 0], 1, [flatRow, [0, 0]])).toThrow();
    expect(() => gridSurface([0, 0], 1, [flatRow, [0, Number.NaN, 0]])).toThrow();
    expect(() => gridSurface([0, 0], 0.001, [flatRow, flatRow])).toThrow();
    expect(() => gridSurface([0, 0], 1, [flatRow, [0, 0.5, 0]])).toThrow();
  });

  it("rejects a non-surface argument", () => {
    expect(() =>
      simulatePuttOnSurface(
        strike(PUTTER, 1.6),
        { stimpFt: 10 } as never,
        { stimpFt: 10, holeDistanceM: 3 },
      ),
    ).toThrow(/GreenSurface/);
  });
});

describe("green-surface wire (swing_sim.green_surface/1)", () => {
  // Runtime-local canonical bytes: JS prints integral floats without
  // the trailing ".0" Python uses; the JSON *values* are identical.
  const PLANAR_JSON =
    '{"aspect_deg":90,"format":"swing_sim.green_surface/1",' +
    '"grade_percent":2.5,"kind":"planar"}';

  it("planar round trip is byte-identical", () => {
    const surface = planarSurface(2.5, 90);
    const text = greenSurfaceToJson(surface);
    expect(text).toBe(PLANAR_JSON);
    const parsed = greenSurfaceFromJson(text);
    expect(parsed).toEqual(surface);
    expect(greenSurfaceToJson(parsed)).toBe(text);
  });

  it("grid round trip is byte-identical", () => {
    const surface = gridSurface(
      [-1.0, -1.5],
      0.5,
      [
        [0.0, 0.01, 0.02],
        [0.0, 0.005, 0.01],
      ],
    );
    const text = greenSurfaceToJson(surface);
    const parsed = greenSurfaceFromJson(text);
    expect(parsed).toEqual(surface);
    expect(greenSurfaceToJson(parsed)).toBe(text);
  });

  it("refuses unknown fields", () => {
    expect(() =>
      greenSurfaceFromJson(
        PLANAR_JSON.replace('"kind"', '"stimp_ft":10,"kind"'),
      ),
    ).toThrow(/fields/);
  });

  it("refuses missing fields", () => {
    expect(() =>
      greenSurfaceFromJson(
        '{"format":"swing_sim.green_surface/1","kind":"planar",' +
          '"grade_percent":2.5}',
      ),
    ).toThrow(/fields/);
  });

  it("refuses cross-kind fields", () => {
    expect(() =>
      greenSurfaceFromJson(
        '{"aspect_deg":90,"format":"swing_sim.green_surface/1",' +
          '"grade_percent":2.5,"kind":"planar","spacing_m":0.5}',
      ),
    ).toThrow(/fields/);
  });

  it("refuses a wrong format or kind", () => {
    expect(() => greenSurfaceFromJson(PLANAR_JSON.replace("/1", "/2"))).toThrow(
      /format/,
    );
    expect(() =>
      greenSurfaceFromJson(PLANAR_JSON.replace('"planar"', '"mesh"')),
    ).toThrow(/kind/);
  });

  it("refuses non-finite and boolean numbers", () => {
    expect(() =>
      greenSurfaceFromJson(PLANAR_JSON.replace("2.5", "NaN")),
    ).toThrow();
    expect(() =>
      greenSurfaceFromJson(PLANAR_JSON.replace("2.5", "true")),
    ).toThrow(/number/);
  });

  it("refuses non-object payloads", () => {
    expect(() => greenSurfaceFromJson("[1,2,3]")).toThrow(/object/);
  });

  it("refuses non-finite values at serialization", () => {
    const broken = { ...planarSurface(2.5, 90), gradePercent: Number.NaN };
    expect(() => greenSurfaceToJson(broken)).toThrow(/finite/);
  });
});

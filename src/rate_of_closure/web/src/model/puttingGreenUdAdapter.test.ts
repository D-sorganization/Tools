/**
 * UpstreamDrift topography adapter (#4800 P9, ADR-0045 F2 gate).
 *
 * Mirrors `shared/python/swing_sim/putting/tests/test_ud_adapter.py`
 * test-for-test on the import direction: the fixture
 * (`__fixtures__/ud_green_topography.json`) is a byte-identical copy
 * of the Python suite's `fixtures/ud_green_topography.json` — both
 * synthesized field-for-field from UD's `_load_json_topography`
 * schema (`contours` `[{x, y, elevation}, ...]` plus `hole_position`;
 * UD ships no canned topography JSON to copy). The fixture-import gate
 * additionally *putts* on the imported surface
 * (`simulatePuttOnSurface`), the React-side half of "a UD topography
 * fixture imports and putts on both surfaces."
 */

import { describe, expect, it } from "vitest";

import { greenSurfaceFromUdJson } from "./puttingGreenUdAdapter";
import { simulatePuttOnSurface, type GridGreenSurface, type PuttLaunch } from "./puttingGreen";
import fixture from "./__fixtures__/ud_green_topography.json";

const FIXTURE_TEXT = JSON.stringify(fixture);

//: The fixture's plane: heights = -0.03125 * x (3.125 % downgrade
//: toward +x), 5 x 5 nodes at 1 m spacing, hole at (3, 2).
const FIXTURE_HOLE: [number, number] = [3.0, 2.0];

function launch(speedMps: number): PuttLaunch {
  return {
    ballSpeedMps: speedMps,
    launchAngleDeg: 0,
    horizontalSpeedMps: speedMps,
    spinRadS: 0,
    effectiveLoftDeg: 0,
  };
}

function doc(
  points: [number, number, number][],
  extra: Record<string, unknown> = {},
): string {
  return JSON.stringify({
    contours: points.map(([x, y, elevation]) => ({ x, y, elevation })),
    ...extra,
  });
}

function gridPoints(xs: number[], ys: number[]): [number, number, number][] {
  const points: [number, number, number][] = [];
  for (const y of ys) for (const x of xs) points.push([x, y, 0.0]);
  return points;
}

describe("greenSurfaceFromUdJson: fixture import", () => {
  it("imports the fixture as a grid", () => {
    const parsed = greenSurfaceFromUdJson(FIXTURE_TEXT);
    const surface = parsed.surface;
    expect(surface.kind).toBe("grid");
    expect(surface.originM).toEqual([0.0, 0.0]);
    expect(surface.spacingM).toBe(1.0);
    expect(surface.heightsM.length).toBe(5);
    expect(surface.heightsM[0].length).toBe(5);
    expect(parsed.holePositionM).toEqual(FIXTURE_HOLE);
  });

  it("reproduces the documented planar grade at every node", () => {
    const { surface } = greenSurfaceFromUdJson(FIXTURE_TEXT) as {
      surface: GridGreenSurface;
    };
    for (let j = 0; j < surface.heightsM.length; j++) {
      for (let i = 0; i < surface.heightsM[j].length; i++) {
        expect(surface.heightsM[j][i]).toBeCloseTo(-0.03125 * i, 10);
      }
    }
  });

  it("putts on the imported surface (fixture-driven cross-runtime gate)", () => {
    const { surface } = greenSurfaceFromUdJson(FIXTURE_TEXT);
    const result = simulatePuttOnSurface(launch(1.8), surface, {
      stimpFt: 10,
      holeDistanceM: 3.0,
    });
    expect(result.pathXM.length).toBeGreaterThan(1);
    expect(result.totalDistanceM).toBeGreaterThan(0);
    expect(Number.isFinite(result.totalDistanceM)).toBe(true);
  });
});

describe("greenSurfaceFromUdJson: fail-closed import", () => {
  it("refuses slopes by the adapter's named reason", () => {
    const text = doc(gridPoints([0.0, 1.0], [0.0, 1.0]), {
      slopes: [
        {
          center: [0.5, 0.5],
          radius: 1.0,
          direction: [1.0, 0.0],
          magnitude: 0.02,
        },
      ],
    });
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/slope/);
  });

  it("refuses unknown top-level fields", () => {
    const text = doc(gridPoints([0.0, 1.0], [0.0, 1.0]), {
      turf: { stimp_rating: 10 },
    });
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/unknown/);
  });

  it("refuses a document missing contours", () => {
    expect(() =>
      greenSurfaceFromUdJson(JSON.stringify({ hole_position: [1.0, 1.0] })),
    ).toThrow(/contours/);
  });

  it("refuses a non-object document", () => {
    expect(() => greenSurfaceFromUdJson("[]")).toThrow();
    expect(() => greenSurfaceFromUdJson("42")).toThrow();
  });

  it("refuses malformed JSON", () => {
    expect(() => greenSurfaceFromUdJson("{not json")).toThrow();
  });

  it("refuses a contour missing a required field", () => {
    const text = JSON.stringify({ contours: [{ x: 0.0, y: 0.0 }] });
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/contour fields/);
  });

  it("refuses non-finite elevation", () => {
    const text = JSON.stringify({
      contours: [
        { x: 0.0, y: 0.0, elevation: Number.NaN },
        { x: 1.0, y: 0.0, elevation: 0.0 },
      ],
    });
    expect(() => greenSurfaceFromUdJson(text)).toThrow();
  });

  it("refuses scattered (non-grid) contours", () => {
    const text = doc([
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.5, 0.5, 0.0],
    ]);
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/complete regular grid/);
  });

  it("refuses a duplicate contour node", () => {
    const text = doc([
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 0.0, 0.1],
    ]);
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/duplicate/);
  });

  it("refuses anisotropic x/y spacing", () => {
    const text = doc(gridPoints([0.0, 1.0], [0.0, 2.0]));
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/one spacing/);
  });

  it("refuses an irregular axis", () => {
    const text = doc([
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [2.5, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [1.0, 1.0, 0.0],
      [2.5, 1.0, 0.0],
    ]);
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/evenly spaced/);
  });

  it("refuses a single row or column", () => {
    const text = doc(gridPoints([0.0, 1.0, 2.0], [0.0]));
    expect(() => greenSurfaceFromUdJson(text)).toThrow();
  });

  it("refuses a malformed hole_position", () => {
    const text = doc(gridPoints([0.0, 1.0], [0.0, 1.0]), {
      hole_position: [1.0],
    });
    expect(() => greenSurfaceFromUdJson(text)).toThrow(/hole_position/);
  });
});

/**
 * Parity tests: these pin the same numeric cases as the pytest suite in
 * tests/rate_of_closure/test_model.py, so the TypeScript and Python
 * implementations cannot drift apart silently.
 */

import { describe, expect, it } from "vitest";

import {
  DEFAULT_SCENARIO,
  solve,
  validateScenario,
  type ImpactScenario,
} from "./impact";

function scenario(overrides: Partial<ImpactScenario> = {}): ImpactScenario {
  return { ...DEFAULT_SCENARIO, ...overrides };
}

describe("reference case", () => {
  it("zero rotation means zero deviation", () => {
    const result = solve(scenario({ omegaPlaneDps: 0, omegaShaftDps: 0 }));
    expect(result.pathDeviationDeg).toBeCloseTo(0, 10);
    expect(result.aoaDeviationDeg).toBeCloseTo(0, 10);
    expect(result.speedDeltaMph).toBeCloseTo(0, 10);
  });
});

describe("forum commenter case (35 mm, 2000 deg/s, vertical axis)", () => {
  const forum = scenario({
    omegaPlaneDps: 0,
    omegaShaftDps: 2000,
    lieAngleDeg: 90,
    comToFaceMm: 35,
  });

  it("tangential velocity is 2.733 mph (1.22 m/s misread as mph)", () => {
    expect(solve(forum).tangentialSpeedMph).toBeCloseTo(2.733, 2);
  });

  it("path deviation is -1.30 degrees at 120 mph", () => {
    expect(solve(forum).pathDeviationDeg).toBeCloseTo(-1.3, 2);
  });
});

describe("tour representative case — parity with pytest", () => {
  const legacyTour = scenario({
    omegaPlaneDps: 2200,
    omegaShaftDps: 1700,
    comToFaceMm: 35,
  });

  it("matches the Python model to two decimals", () => {
    const result = solve(legacyTour);
    expect(result.pathDeviationDeg).toBeCloseTo(-1.7, 1);
    expect(result.aoaDeviationDeg).toBeCloseTo(0.63, 1);
  });

  it("closure during contact is about a degree", () => {
    const result = solve(legacyTour);
    expect(result.closureDuringContactDeg).toBeGreaterThan(0.8);
    expect(result.closureDuringContactDeg).toBeLessThan(1.6);
  });
});

describe("AffineDrift dossier alignment — parity with pytest", () => {
  it("default CCV reproduces the ~2,100 deg/s tour mean", () => {
    const result = solve(scenario());
    expect(result.closureRateDps).toBeCloseTo(2100, -1);
  });

  it("normalized closure is omega over v in deg/ft", () => {
    const result = solve(scenario());
    const speedFts = (120.0 / 3600.0) * 5280.0;
    expect(result.normalizedClosureDegPerFt).toBeCloseTo(
      result.closureRateDps / speedFts,
      9,
    );
  });

  it("TrackMan worked example reproduces the ~3 degree gap", () => {
    const result = solve(scenario({ omegaShaftDps: 3575 }));
    expect(result.pathDeviationDeg).toBeCloseTo(-3.0, 1);
  });
});

describe("sign conventions", () => {
  it("closing rotation moves the path left (negative)", () => {
    expect(
      solve(scenario({ omegaPlaneDps: 0 })).pathDeviationDeg,
    ).toBeLessThan(0);
  });

  it("plane rotation shallows the delivery (positive AoA)", () => {
    expect(
      solve(scenario({ omegaShaftDps: 0 })).aoaDeviationDeg,
    ).toBeGreaterThan(0);
  });
});

describe("validation", () => {
  it("rejects non-finite values", () => {
    expect(() =>
      validateScenario(scenario({ clubheadSpeedMph: Number.NaN })),
    ).toThrow(RangeError);
  });

  it("rejects out-of-range values", () => {
    expect(() => validateScenario(scenario({ lieAngleDeg: 5 }))).toThrow(
      RangeError,
    );
  });
});

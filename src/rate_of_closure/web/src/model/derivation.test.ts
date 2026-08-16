/**
 * Parity tests for the derivation content — pins step count, order, and
 * live substitution against tests/rate_of_closure/test_derivation.py so
 * the desktop and web traceability surfaces cannot drift apart.
 */

import { describe, expect, it } from "vitest";

import {
  RESULT_EXPLANATIONS,
  derivationSteps,
} from "./derivation";
import { DEFAULT_SCENARIO } from "./impact";

const STEP_TITLES = [
  "Frame and Sign Conventions",
  "Shaft Axis and Swing-Plane Normal",
  "Angular Velocity Assembly",
  "Lever Arm to the Impact Point",
  "Rigid-Body Point Velocity",
  "Path and Attack-Angle Deviation",
  "Closure Rate — the CCV Identity",
  "Speed-Invariant Closure and the Path Gap",
  "Face Rotation During Contact",
];

describe("derivation steps — parity with pytest", () => {
  it("covers the full chain in the pinned order", () => {
    const titles = derivationSteps(DEFAULT_SCENARIO).map((s) => s.title);
    expect(titles).toEqual(STEP_TITLES);
  });

  it("every step has formula, values, and narrative", () => {
    for (const step of derivationSteps(DEFAULT_SCENARIO)) {
      expect(step.latex.length).toBeGreaterThan(10);
      expect(step.values.length).toBeGreaterThan(10);
      expect(step.narrative.length).toBeGreaterThan(60);
    }
  });

  it("substitutions are live", () => {
    const base = derivationSteps(DEFAULT_SCENARIO);
    const changed = derivationSteps({
      ...DEFAULT_SCENARIO,
      clubheadSpeedMph: 95,
      omegaShaftDps: 2000,
    });
    const differing = base.filter(
      (step, i) => step.values !== changed[i].values,
    );
    expect(differing.length).toBeGreaterThanOrEqual(5);
  });

  it("pins the headline numbers for the default scenario", () => {
    const byTitle = Object.fromEntries(
      derivationSteps(DEFAULT_SCENARIO).map((s) => [s.title, s]),
    );
    expect(byTitle["Path and Attack-Angle Deviation"].values).toContain(
      "-1.56",
    );
    expect(byTitle["Closure Rate — the CCV Identity"].values).toContain(
      "2099",
    );
    expect(
      byTitle["Speed-Invariant Closure and the Path Gap"].values,
    ).toContain("11.93");
  });
});

describe("result explanations", () => {
  it("every result row has a substantial, brand-neutral explanation", () => {
    for (const [key, text] of Object.entries(RESULT_EXPLANATIONS)) {
      expect(text.length, key).toBeGreaterThan(80);
      expect(text, key).not.toContain("TrackMan");
    }
    expect(Object.keys(RESULT_EXPLANATIONS)).toHaveLength(8);
  });
});

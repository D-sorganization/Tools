/**
 * Swing-kinetics parity tests (#4125 H2): the TS inverse-dynamics
 * mirror is pinned against the Python-generated fixture
 * (`__fixtures__/kinetics_parity.json`). Both implementations run the
 * same double-precision RK4 and central differences, so the tolerance
 * is tight (unlike the variation fixture's statistical comparison).
 */

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/kinetics_parity.json";
import {
  computeKinetics,
  gradient,
  kineticsForInput,
  type KineticsSeriesTs,
} from "./kinetics";
import {
  golfDefaultParams,
  simulatePendulum,
  type PendulumState,
  type SimulationInput,
} from "./simulation";

const seriesFromFixture = (): KineticsSeriesTs => {
  const plan = fixture.plan;
  const p = golfDefaultParams();
  const g = plan.gInplane as [number, number];
  const states = simulatePendulum(
    p,
    plan.initialState as PendulumState,
    g,
    plan.dtS,
    plan.nSteps,
  );
  return computeKinetics(p, states, g, plan.dtS);
};

describe("swing kinetics — parity with the Python inverse dynamics", () => {
  it("matches the pytest-generated fixture sample-for-sample", () => {
    const series = seriesFromFixture();
    const keys = [
      "shoulderTorqueNm",
      "wristTorqueNm",
      "shoulderGravityTorqueNm",
      "wristGravityTorqueNm",
      "shoulderDampingTorqueNm",
      "wristDampingTorqueNm",
      "shoulderPowerW",
      "wristPowerW",
      "shoulderForceN",
      "wristForceN",
      "clubheadForceN",
    ] as const;
    for (const sample of fixture.samples) {
      for (const key of keys) {
        const actual = series[key][sample.index];
        const expected = sample[key];
        const scale = Math.max(1.0, Math.abs(expected));
        expect(
          Math.abs(actual - expected),
          `${key}@${sample.index}`,
        ).toBeLessThan(1e-9 * scale);
      }
    }
  });

  it("gradient matches numpy.gradient's central/one-sided scheme", () => {
    const values = [0, 1, 4, 9, 16];
    expect(gradient(values, 1)).toEqual([1, 2, 4, 6, 7]);
  });

  it("returns null for sources without joint states", () => {
    const input: SimulationInput = {
      sourceKind: "manual",
      clubheadSpeedMph: 113,
      omegaDps: [0, 0, 0],
      loftDeg: 10.5,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: 1.5,
    };
    expect(kineticsForInput(input)).toBeNull();
    const series = kineticsForInput({
      ...input,
      sourceKind: "double_pendulum",
    });
    expect(series).not.toBeNull();
    expect(series?.tS.length).toBe(1501);
    // Memoized: same inputs return the same object.
    expect(
      kineticsForInput({ ...input, sourceKind: "double_pendulum" }),
    ).toBe(series);
  });

  it("passive swing torques satisfy the breakdown identity", () => {
    const series = seriesFromFixture();
    // Net torque = gravity + damping + applied; applied ≈ 0 for the
    // passive swing, so net ≈ gravity + damping away from the ends.
    for (let i = 5; i < series.tS.length - 5; i += 250) {
      const residual =
        series.shoulderTorqueNm[i] -
        series.shoulderGravityTorqueNm[i] -
        series.shoulderDampingTorqueNm[i];
      expect(Math.abs(residual)).toBeLessThan(0.05);
    }
  });
});

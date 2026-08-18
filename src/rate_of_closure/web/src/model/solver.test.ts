/**
 * Solver tests — the easy case is pinned against the pytest suite
 * (tests/rate_of_closure/test_solver_gui.py, TestWorker): a 150 mph
 * ball-speed goal over a free clubhead speed in [30, 70] m/s solves to
 * ~45.825 m/s in the Python swing_sim solver; the TS solver must land
 * on the same solution through the parity-ported physics.
 */

import { describe, expect, it } from "vitest";

import {
  achievedQuantities,
  assembleVariables,
  solveGoals,
  validateInputs,
  VARIABLE_DEFAULTS,
} from "./solver";

const easyGoal = { ballSpeedMph: { target: 150.0, weight: 1.0 } };
const easyPartition = {
  free: { clubheadSpeedMps: [30.0, 70.0] as [number, number] },
  fixed: {},
};

describe("solveGoals — pinned parity easy case", () => {
  it("recovers the pytest-pinned clubhead speed for 150 mph ball speed", () => {
    const result = solveGoals(easyGoal, easyPartition);
    expect(result.converged).toBe(true);
    // Python swing_sim solver pins 45.825 m/s (test_solver_gui.py).
    expect(result.variables.clubheadSpeedMps).toBeCloseTo(45.825, 1);
    expect(result.achieved.ballSpeedMph!).toBeCloseTo(150.0, 2);
    expect(result.perGoalErrors.ballSpeedMph!).toBeCloseTo(0.0, 2);
    expect(result.residualNorm).toBeLessThan(1e-2);
  });

  it("respects bounds when the goal is unreachable", () => {
    const result = solveGoals(easyGoal, {
      free: { clubheadSpeedMps: [30.0, 40.0] },
      fixed: {},
    });
    expect(result.variables.clubheadSpeedMps).toBeLessThanOrEqual(40.0);
    expect(result.variables.clubheadSpeedMps).toBeCloseTo(40.0, 3);
    expect(result.achieved.ballSpeedMph!).toBeLessThan(150.0);
  });

  it("solves a two-variable goal (launch angle via loft, speed)", () => {
    const result = solveGoals(
      {
        ballSpeedMph: { target: 150.0, weight: 1.0 },
        launchAngleDeg: { target: 12.0, weight: 1.0 },
      },
      {
        free: {
          clubheadSpeedMps: [30.0, 70.0],
          dynamicLoftDeg: [5.0, 25.0],
        },
        fixed: {},
      },
    );
    expect(result.achieved.ballSpeedMph!).toBeCloseTo(150.0, 1);
    expect(result.achieved.launchAngleDeg!).toBeCloseTo(12.0, 1);
  });

  it("delivery-level goals pass straight through", () => {
    const achieved = achievedQuantities(
      { ...VARIABLE_DEFAULTS, clubPathDeg: 3.0 },
      { clubPathDeg: { target: 3.0, weight: 1.0 } },
    );
    expect(achieved.clubPathDeg).toBeCloseTo(3.0, 9);
    expect(achieved.ballSpeedMph).toBeUndefined();
  });

  it("assembles defaults + fixed + free in the documented order", () => {
    const variables = assembleVariables(
      [50.0],
      ["clubheadSpeedMps"],
      { free: { clubheadSpeedMps: [30, 70] }, fixed: { faceAngleDeg: 2.0 } },
    );
    expect(variables.clubheadSpeedMps).toBe(50.0);
    expect(variables.faceAngleDeg).toBe(2.0);
    expect(variables.dynamicLoftDeg).toBe(VARIABLE_DEFAULTS.dynamicLoftDeg);
  });
});

describe("validateInputs — DbC parity", () => {
  it("rejects an empty goal", () => {
    expect(() => validateInputs({}, easyPartition)).toThrow(/at least one goal/);
  });

  it("rejects an empty free set", () => {
    expect(() =>
      validateInputs(easyGoal, { free: {}, fixed: {} }),
    ).toThrow(/at least one variable/);
  });

  it("rejects inverted bounds", () => {
    expect(() =>
      validateInputs(easyGoal, {
        free: { clubheadSpeedMps: [70.0, 30.0] },
        fixed: {},
      }),
    ).toThrow(/lower < upper/);
  });

  it("rejects overlapping free and fixed variables", () => {
    expect(() =>
      validateInputs(easyGoal, {
        free: { clubheadSpeedMps: [30.0, 70.0] },
        fixed: { clubheadSpeedMps: 45.0 },
      }),
    ).toThrow(/both free and fixed/);
  });

  it("rejects non-positive weights", () => {
    expect(() =>
      validateInputs(
        { ballSpeedMph: { target: 150.0, weight: 0.0 } },
        easyPartition,
      ),
    ).toThrow(/weight/);
  });
});

/**
 * Movement-optimizer swing delivery parity tests.
 *
 * Verifies that the TypeScript implementation of golfer anthropometry and
 * movement optimizer swing synthesis maintains tight numerical parity with
 * the canonical Python implementation captured in movement_optimizer_golden_v1.json.
 */

import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/movement_optimizer_golden_v1.json";
import {
  DEFAULT_GOLFER_HEIGHT_M,
  DEFAULT_GOLFER_MASS_KG,
  golferAnthropometryToParams,
  simulateMovementOptimizerDelivery,
} from "./movementOptimizerSimulation";
import {
  runSimulation,
  type SimulationInput,
} from "./simulation";
import { defaultBallSetupForClub } from "./ballSetup";
import { PASSIVE_DOUBLE_PENDULUM_RUN } from "./doublePendulum";
import { DEFAULT_MANUAL_DELIVERY } from "./manualDelivery";

describe("movement optimizer delivery — parity with Python authority", () => {
  it("computes physical pendulum parameters from golfer anthropometry", () => {
    const params = golferAnthropometryToParams({
      heightM: 1.75,
      bodyMassKg: 75.0,
      armLengthM: 0.735,
      armMassKg: 7.5,
    });

    expect(params.l1).toBeCloseTo(0.735, 4);
    expect(params.m1).toBeCloseTo(7.5, 4);
    expect(params.l2).toBeCloseTo(1.0, 4);
    expect(params.m2).toBeCloseTo(0.35, 4);
    expect(params.d1).toBe(0.4);
    expect(params.d2).toBe(0.25);
  });

  it("throws descriptive error for non-positive height or mass", () => {
    expect(() => golferAnthropometryToParams({ heightM: -1, bodyMassKg: 75 })).toThrow(
      /Golfer height must be finite and positive/,
    );
    expect(() => golferAnthropometryToParams({ heightM: 1.75, bodyMassKg: 0 })).toThrow(
      /Golfer mass must be finite and positive/,
    );
  });

  it("matches the Python-generated golden fixture sample-for-sample", () => {
    const input: SimulationInput = {
      sourceKind: "movement_optimizer",
      clubheadSpeedMph: 110,
      omegaDps: [0, 0, 2000],
      loftDeg: 10.5,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: 0,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: fixture.duration_s,
      contactMode: "delivery_inspection",
      doublePendulumRun: PASSIVE_DOUBLE_PENDULUM_RUN,
      doublePendulumInitialState: [-Math.PI / 2, 0, 0, 0],
      ballSetup: defaultBallSetupForClub(null),
      golferAnthropometry: {
        heightM: fixture.golfer_anthropometry.height_m,
        bodyMassKg: fixture.golfer_anthropometry.mass_kg,
        armLengthM: fixture.golfer_anthropometry.lead_arm_length_m,
        armMassKg: fixture.golfer_anthropometry.lead_arm_mass_kg,
      },
      ...DEFAULT_MANUAL_DELIVERY,
    };

    const samples = simulateMovementOptimizerDelivery(input, fixture.dt_s);
    expect(samples.length).toBe(fixture.samples.length);

    for (let i = 0; i < fixture.samples.length; i++) {
      const actual = samples[i];
      const expected = fixture.samples[i];

      expect(actual.t).toBeCloseTo(expected.t, 4);

      // Position (app frame)
      expect(actual.position[0]).toBeCloseTo(expected.position[0], 3);
      expect(actual.position[1]).toBeCloseTo(expected.position[1], 3);
      expect(actual.position[2]).toBeCloseTo(expected.position[2], 3);

      // Velocity (app frame)
      expect(actual.velocity[0]).toBeCloseTo(expected.velocity[0], 3);
      expect(actual.velocity[1]).toBeCloseTo(expected.velocity[1], 3);
      expect(actual.velocity[2]).toBeCloseTo(expected.velocity[2], 3);

      // Angular velocity (app frame)
      expect(actual.angularVelocity[0]).toBeCloseTo(expected.angular_velocity[0], 3);
      expect(actual.angularVelocity[1]).toBeCloseTo(expected.angular_velocity[1], 3);
      expect(actual.angularVelocity[2]).toBeCloseTo(expected.angular_velocity[2], 3);

      // Rotation matrix (app frame)
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const actualVal = actual.rotation[r][c];
          const expectedVal = expected.rotation[r][c];
          expect(actualVal).toBeCloseTo(expectedVal, 3);
        }
      }

      // Joint positions
      expect(actual.joints.length).toBe(expected.joints.length);
      for (let j = 0; j < expected.joints.length; j++) {
        expect(actual.joints[j][0]).toBeCloseTo(expected.joints[j][0], 3);
        expect(actual.joints[j][1]).toBeCloseTo(expected.joints[j][1], 3);
        expect(actual.joints[j][2]).toBeCloseTo(expected.joints[j][2], 3);
      }
    }
  });

  it("integrates seamlessly into runSimulation pipeline", () => {
    const input: SimulationInput = {
      sourceKind: "movement_optimizer",
      clubheadSpeedMph: 110,
      omegaDps: [0, 0, 2000],
      loftDeg: 10.5,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: 0.5,
      contactMode: "delivery_inspection",
      doublePendulumRun: PASSIVE_DOUBLE_PENDULUM_RUN,
      doublePendulumInitialState: [-Math.PI / 2, 0, 0, 0],
      ballSetup: defaultBallSetupForClub(null),
      golferAnthropometry: {
        heightM: DEFAULT_GOLFER_HEIGHT_M,
        bodyMassKg: DEFAULT_GOLFER_MASS_KG,
      },
      ...DEFAULT_MANUAL_DELIVERY,
    };

    const run = runSimulation(input);
    expect(run.sourceKind).toBe("movement_optimizer");
    expect(run.swing.length).toBeGreaterThan(10);
    expect(run.impactOutcome).toBeDefined();
    expect(run.flight.length).toBeGreaterThan(0);
  });
});

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
import {
  DOUBLE_PENDULUM_MODEL_ID,
  JointLockConfig,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  passiveDoublePendulumRun,
  prescribedDoublePendulumRun,
} from "./doublePendulum";
import {
  JointTorqueAssignment,
  PrescribedTorqueProfile,
  TorquePolynomial,
  TorqueProfileSource,
} from "./torqueProfiles";

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

  it("exposes the state-matched passive-drift ZTCF identity", () => {
    const series = seriesFromFixture();
    for (let i = 0; i < series.tS.length; i += 250) {
      expect(series.shoulderZtcfTorqueNm[i]).toBeCloseTo(
        series.shoulderGravityTorqueNm[i] +
          series.shoulderDampingTorqueNm[i],
        10,
      );
      expect(series.wristZtcfTorqueNm[i]).toBeCloseTo(
        series.wristGravityTorqueNm[i] + series.wristDampingTorqueNm[i],
        10,
      );
      expect(Number.isFinite(series.shoulderZtcfForceN[i])).toBe(true);
      expect(Number.isFinite(series.clubheadZtcfForceN[i])).toBe(true);
    }
  });

  it("rebuilds prescribed rather than passive dynamics for kinetics", () => {
    const selected = new PrescribedTorqueProfile({
      profileId: "profile.kinetics.constant.v1",
      modelId: DOUBLE_PENDULUM_MODEL_ID,
      name: "Kinetics Constant Drive",
      description: "Regression profile proving kinetics use the executed input.",
      source: TorqueProfileSource.DIRECT,
      sourceMetadata: { author: "kinetics-test" },
      createdAtUtc: "2026-08-05T12:00:00Z",
      modifiedAtUtc: "2026-08-05T12:00:00Z",
      timeDomainS: [0, 0.1],
      assignments: [
        new JointTorqueAssignment(
          SHOULDER_JOINT_ID,
          new TorquePolynomial([20]),
        ),
        new JointTorqueAssignment(WRIST_JOINT_ID, new TorquePolynomial([-5])),
      ],
    });
    const input: SimulationInput = {
      sourceKind: "double_pendulum",
      clubheadSpeedMph: 113,
      omegaDps: [0, 0, 0],
      loftDeg: 10.5,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: 0.1,
      doublePendulumRun: prescribedDoublePendulumRun(selected),
    };
    const series = kineticsForInput(input);
    expect(series).not.toBeNull();
    const sample = 50;
    expect(
      series!.shoulderTorqueNm[sample] -
        series!.shoulderGravityTorqueNm[sample] -
        series!.shoulderDampingTorqueNm[sample],
    ).toBeCloseTo(20, 2);
    expect(
      series!.wristTorqueNm[sample] -
        series!.wristGravityTorqueNm[sample] -
        series!.wristDampingTorqueNm[sample],
    ).toBeCloseTo(-5, 2);
  });

  it("keeps ideal constraints active in the pointwise ZTCF", () => {
    const input: SimulationInput = {
      sourceKind: "double_pendulum",
      clubheadSpeedMph: 113,
      omegaDps: [0, 0, 0],
      loftDeg: 10.5,
      impactOffsetToeMm: 0,
      impactOffsetHighMm: 0,
      planeYawDeg: 0,
      planeSideTiltDeg: -45,
      planeForwardTiltDeg: 0,
      impactTimeS: null,
      swingDurationS: 0.1,
      doublePendulumRun: passiveDoublePendulumRun(
        new JointLockConfig([SHOULDER_JOINT_ID]),
      ),
    };
    const series = kineticsForInput(input);
    expect(series).not.toBeNull();
    expect(series!.shoulderZtcfAccelerationRadS2.every((value) => value === 0))
      .toBe(true);
    expect(series!.wristZtcfAccelerationRadS2.some((value) => Math.abs(value) > 0))
      .toBe(true);
  });
});

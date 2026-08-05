import { describe, expect, it } from "vitest";

import {
  DOUBLE_PENDULUM_MODEL_ID,
  DOUBLE_PENDULUM_JOINT_IDS,
  JointLockConfig,
  PASSIVE_DOUBLE_PENDULUM_RUN,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  golfDefaultParams,
  pendulumRk4StepForced,
  prescribedDoublePendulumRun,
  simulateConfiguredPendulum,
} from "./doublePendulum";
import {
  JointTorqueAssignment,
  PrescribedTorqueProfile,
  TorquePolynomial,
  TorqueProfileSource,
} from "./torqueProfiles";
import { runSimulation, type SimulationInput } from "./simulation";

const INPUT: SimulationInput = {
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
};

function profile(options: {
  modelId?: string;
  jointIds?: readonly [string, string];
  domain?: readonly [number, number];
  coefficients?: readonly [readonly number[], readonly number[]];
} = {}): PrescribedTorqueProfile {
  const joints = options.jointIds ?? [SHOULDER_JOINT_ID, WRIST_JOINT_ID];
  const coefficients = options.coefficients ?? [[20], [-5]];
  return new PrescribedTorqueProfile({
    profileId: "profile.web.constant_drive.v1",
    modelId: options.modelId ?? DOUBLE_PENDULUM_MODEL_ID,
    name: "Constant Drive",
    description: "Constant shoulder and wrist torque test profile.",
    source: TorqueProfileSource.DIRECT,
    sourceMetadata: { author: "web-test-suite" },
    createdAtUtc: "2026-08-05T12:00:00Z",
    modifiedAtUtc: "2026-08-05T12:00:00Z",
    timeDomainS: options.domain ?? [0, 0.1],
    assignments: [
      new JointTorqueAssignment(joints[0], new TorquePolynomial(coefficients[0])),
      new JointTorqueAssignment(joints[1], new TorquePolynomial(coefficients[1])),
    ],
  });
}

describe("prescribed double-pendulum integration", () => {
  it("keeps passive dynamics as the explicit and backward-compatible default", () => {
    const implicit = runSimulation(INPUT);
    const explicit = runSimulation({
      ...INPUT,
      doublePendulumRun: PASSIVE_DOUBLE_PENDULUM_RUN,
    });
    expect(explicit.swing).toEqual(implicit.swing);
    expect(explicit.torqueRun).toMatchObject({ mode: "passive", profileId: null });
    expect(explicit.torqueRun.appliedTorqueHistory).toHaveLength(explicit.swing.length);
  });

  it("evaluates torque at every non-autonomous RK4 substep", () => {
    const sampledTimes: number[] = [];
    const torqueAt = (timeS: number) => {
      sampledTimes.push(timeS);
      return [20, -5] as const;
    };
    pendulumRk4StepForced(
      golfDefaultParams(),
      [0, 0, 0, 0],
      [0, 0],
      0.25,
      0.01,
      torqueAt,
    );
    expect(sampledTimes).toEqual([0.25, 0.255, 0.255, 0.26]);
  });

  it("drives the existing dynamics and records that prescribed input ran", () => {
    const passive = runSimulation(INPUT);
    const selected = profile();
    const forced = runSimulation({
      ...INPUT,
      doublePendulumRun: prescribedDoublePendulumRun(selected),
    });
    expect(forced.swing[forced.swing.length - 1].velocity).not.toEqual(
      passive.swing[passive.swing.length - 1].velocity,
    );
    expect(forced.torqueRun).toMatchObject({
      mode: "prescribed",
      profileId: selected.profileId,
    });
    expect(forced.torqueRun.appliedTorqueHistory[0].torquesNm).toEqual({
      "joint.shoulder": 20,
      "joint.wrist": -5,
    });
    expect(runSimulation({
      ...INPUT,
      doublePendulumRun: prescribedDoublePendulumRun(selected),
    }).swing).toEqual(forced.swing);
  });

  it.each([
    ["model_id", profile({ modelId: "model.other.v1" })],
    ["joint assignments", profile({ jointIds: [SHOULDER_JOINT_ID, "joint.elbow"] })],
    ["time domain", profile({ domain: [0.01, 0.1] })],
    ["time domain", profile({ domain: [0, 0.05] })],
  ])("rejects an incompatible profile: %s", (message, selected) => {
    expect(() => runSimulation({
      ...INPUT,
      doublePendulumRun: prescribedDoublePendulumRun(selected),
    })).toThrow(message);
  });

  it("rejects prescribed configuration for a source that cannot execute it", () => {
    expect(() => runSimulation({
      ...INPUT,
      sourceKind: "triple_pendulum",
      doublePendulumRun: prescribedDoublePendulumRun(profile()),
    })).toThrow(/double.pendulum/i);
    expect(() => runSimulation({
      ...INPUT,
      sourceKind: "triple_pendulum",
      doublePendulumRun: {
        mode: "passive",
        jointLocks: new JointLockConfig([WRIST_JOINT_ID]),
      },
    })).toThrow(/joint locks.*double.pendulum/i);
  });

  it("normalizes ideal joint locks into canonical shoulder-wrist order", () => {
    const locks = new JointLockConfig([WRIST_JOINT_ID, SHOULDER_JOINT_ID]);
    expect(locks.lockedJointIds).toEqual(DOUBLE_PENDULUM_JOINT_IDS);
    expect(locks.mask).toEqual([true, true]);
    expect(Object.isFrozen(locks.lockedJointIds)).toBe(true);
    expect(() => new JointLockConfig(["joint.elbow"])).toThrow(/joint/i);
    expect(() => new JointLockConfig([SHOULDER_JOINT_ID, SHOULDER_JOINT_ID]))
      .toThrow(/unique/i);
  });

  it.each([
    [
      SHOULDER_JOINT_ID,
      0,
      2,
      1,
      [0.4, -0.2577261949720987, 0, -0.15052670741601543],
    ],
    [
      WRIST_JOINT_ID,
      1,
      3,
      0,
      [0.3732185954154153, -0.25, -0.5277245656966174, 0],
    ],
  ] as const)(
    "holds the %s coordinate exactly while the free coordinate evolves",
    (jointId, coordinate, velocity, freeCoordinate, pythonReferenceFinal) => {
      const initial = [0.4, -0.25, 0, 0] as const;
      const states = simulateConfiguredPendulum(
        golfDefaultParams(),
        [...initial],
        [0, -9.80665],
        0.001,
        100,
        {
          mode: "passive",
          jointLocks: new JointLockConfig([jointId]),
        },
      );
      for (const state of states) {
        expect(state[coordinate]).toBe(initial[coordinate]);
        expect(state[velocity]).toBe(0);
      }
      expect(states[states.length - 1][freeCoordinate]).not.toBeCloseTo(
        initial[freeCoordinate],
        8,
      );
      states[states.length - 1].forEach((value, index) => {
        expect(value).toBeCloseTo(pythonReferenceFinal[index], 11);
      });
    },
  );

  it("holds both coordinates despite prescribed torques and records the locks", () => {
    const initial = [0.4, -0.25, 0, 0] as const;
    const selected = profile();
    const locks = new JointLockConfig(DOUBLE_PENDULUM_JOINT_IDS);
    const run = runSimulation({
      ...INPUT,
      doublePendulumInitialState: [...initial],
      doublePendulumRun: prescribedDoublePendulumRun(selected, locks),
    });
    expect(run.torqueRun.lockedJointIds).toEqual(DOUBLE_PENDULUM_JOINT_IDS);
    expect(run.torqueRun.appliedTorqueHistory[0].torquesNm).toEqual({
      [SHOULDER_JOINT_ID]: 20,
      [WRIST_JOINT_ID]: -5,
    });
    for (const sample of run.swing) {
      const wrist = sample.joints[1];
      const tip = sample.joints[2];
      expect(wrist[0]).toBeCloseTo(run.swing[0].joints[1][0], 12);
      expect(wrist[1]).toBeCloseTo(run.swing[0].joints[1][1], 12);
      expect(tip[0]).toBeCloseTo(run.swing[0].joints[2][0], 12);
      expect(tip[1]).toBeCloseTo(run.swing[0].joints[2][1], 12);
    }
  });

  it("rejects a lock that would require an unmodelled initial impulse", () => {
    expect(() => runSimulation({
      ...INPUT,
      doublePendulumInitialState: [0.4, -0.25, 0.1, 0],
      doublePendulumRun: {
        mode: "passive",
        jointLocks: new JointLockConfig([SHOULDER_JOINT_ID]),
      },
    })).toThrow(/locked shoulder.*initial.*velocity/i);
  });
});

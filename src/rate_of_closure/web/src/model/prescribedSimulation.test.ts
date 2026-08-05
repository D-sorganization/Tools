import { describe, expect, it } from "vitest";

import {
  DOUBLE_PENDULUM_MODEL_ID,
  PASSIVE_DOUBLE_PENDULUM_RUN,
  SHOULDER_JOINT_ID,
  WRIST_JOINT_ID,
  golfDefaultParams,
  pendulumRk4StepForced,
  prescribedDoublePendulumRun,
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
  });
});

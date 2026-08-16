import { describe, expect, it } from "vitest";

import golden from "./__fixtures__/localized_torque_python_golden_v1.json";
import {
  JointTorqueAssignment,
  PrescribedTorqueProfile,
  TorquePolynomial,
  TorqueProfileSource,
} from "./torqueProfiles";
import {
  passiveDoublePendulumRun,
  prescribedDoublePendulumRun,
  simulateConfiguredPendulum,
  summarizeDoublePendulumRun,
} from "./doublePendulum";
import { DOUBLE_PENDULUM_MODEL_ID, NO_JOINT_LOCKS } from "./jointLocks";

const offsets = golden.commands.map((command) => ({
  jointId: command.joint_id as "joint.shoulder" | "joint.wrist",
  timeWindowS: command.time_window_s as [number, number],
  torqueNm: command.torque_nm,
}));

const profile = (): PrescribedTorqueProfile => new PrescribedTorqueProfile({
  profileId: "profile.localized.golden.v1",
  modelId: DOUBLE_PENDULUM_MODEL_ID,
  name: "Golden constant drive",
  description: "Python cross-runtime constant prescribed torque authority.",
  source: TorqueProfileSource.DIRECT,
  sourceMetadata: { authority: "python-golden" },
  createdAtUtc: "2026-08-13T00:00:00Z",
  modifiedAtUtc: "2026-08-13T00:00:00Z",
  timeDomainS: [0, 0.005],
  assignments: [
    new JointTorqueAssignment("joint.shoulder", new TorquePolynomial([20])),
    new JointTorqueAssignment("joint.wrist", new TorquePolynomial([-5])),
  ],
});

const expectGolden = (
  config: ReturnType<typeof passiveDoublePendulumRun>,
  expected: number[][],
  expectedTorques: number[][],
): void => {
  const states = simulateConfiguredPendulum(
    golden.parameters,
    golden.initial_state as [number, number, number, number],
    golden.gravity_m_s2 as [number, number],
    golden.dt_s,
    golden.n_steps,
    config,
  );
  states.forEach((state, row) => state.forEach((value, column) =>
    expect(value).toBeCloseTo(expected[row][column], 13)));
  const sampleTimes = expectedTorques.map((_row, index) => index * golden.dt_s);
  const history = summarizeDoublePendulumRun(config, sampleTimes).appliedTorqueHistory;
  expect(history.map((sample) => [
    sample.torquesNm["joint.shoulder"], sample.torquesNm["joint.wrist"],
  ])).toEqual(expectedTorques);
};

describe("localized torque Python parity", () => {
  it("matches passive half-open boundary integration at every RK4 substep", () => {
    expectGolden(
      passiveDoublePendulumRun(NO_JOINT_LOCKS, offsets),
      golden.passive.states,
      golden.passive.sampled_torques_nm,
    );
  });

  it("adds offsets to prescribed commands without replacing them", () => {
    expectGolden(
      prescribedDoublePendulumRun(profile(), NO_JOINT_LOCKS, offsets),
      golden.prescribed.states,
      golden.prescribed.sampled_torques_nm,
    );
  });

  it("fails closed for an off-duration or spatial joint command", () => {
    expect(() => simulateConfiguredPendulum(
      golden.parameters,
      [0, 0, 0, 0],
      [0, 0],
      0.001,
      5,
      passiveDoublePendulumRun(NO_JOINT_LOCKS, [{
        jointId: "joint.shoulder", timeWindowS: [0.004, 0.006], torqueNm: 1,
      }]),
    )).toThrow(/window.*duration/i);
    expect(() => passiveDoublePendulumRun(NO_JOINT_LOCKS, [{
      jointId: "swing.wrist" as "joint.wrist", timeWindowS: [0, 0.001], torqueNm: 1,
    }])).toThrow(/jointId/i);
    expect(() => passiveDoublePendulumRun(
      NO_JOINT_LOCKS, "bad" as unknown as [],
    )).toThrow(/offsets must be an array/i);
    expect(() => passiveDoublePendulumRun(NO_JOINT_LOCKS, [null] as unknown as []))
      .toThrow(/offset must be an object/i);
    expect(() => summarizeDoublePendulumRun({
      mode: "passive",
      jointLocks: NO_JOINT_LOCKS,
      commandedTorqueOffsets: [null],
    } as unknown as ReturnType<typeof passiveDoublePendulumRun>)).toThrow(/offset must be an object/i);
  });
});

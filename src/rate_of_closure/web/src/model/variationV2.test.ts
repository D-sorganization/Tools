import { describe, expect, it } from "vitest";

import {
  CATEGORY_LAUNCH,
  planFromJson,
  planToJson,
  runVariation,
  sampleInputs,
  validatePlan,
  type PerturbationGroupTs,
  type VariationPlanTs,
} from "./variation";
import { datasetToJson, oneAtATimeSensitivity } from "./variationAnalysis";
import { DRIVER_TEE_HEIGHT_M } from "./ballSetup";
import { TEE_HEIGHT_VARIATION_KEY } from "./variationRegistry";
import localizedTorqueFixture from "./__fixtures__/localized_torque_authoring_v1.json";

const BALL = `${CATEGORY_LAUNCH}.ball_speed_mph`;
const ANGLE = `${CATEGORY_LAUNCH}.launch_angle_deg`;

const groupedPlan = (
  matrixKind: PerturbationGroupTs["matrixKind"] = "correlation",
): VariationPlanTs => ({
  mode: "launch",
  baseVariables: { [BALL]: 154.25, [ANGLE]: 12.75 },
  noise: [
    {
      variableKey: BALL,
      distribution: "normal",
      scale: 2,
      lower: null,
      upper: null,
      specId: "speed",
      timeWindowS: null,
      pointIds: [],
    },
    {
      variableKey: ANGLE,
      distribution: "normal",
      scale: 1,
      lower: null,
      upper: null,
      specId: "angle",
      timeWindowS: null,
      pointIds: [],
    },
  ],
  groups: [
    {
      groupId: "launch-group",
      specIds: ["speed", "angle"],
      matrixKind,
      matrix:
        matrixKind === "correlation"
          ? [
              [1, 0.7],
              [0.7, 1],
            ]
          : [
              [4, 1.4],
              [1.4, 1],
            ],
    },
  ],
  nRuns: 500,
  seed: 11,
  flightModel: "waterloo_penner",
});

const sampleCorrelation = (rows: number[][]): number => {
  const columns = [0, 1].map((column) => rows.map((row) => row[column]));
  const means = columns.map(
    (column) => column.reduce((sum, value) => sum + value, 0) / column.length,
  );
  const covariance = columns[0].reduce(
    (sum, value, index) =>
      sum + (value - means[0]) * (columns[1][index] - means[1]),
    0,
  );
  const sumsOfSquares = columns.map((column, index) =>
    column.reduce((sum, value) => sum + (value - means[index]) ** 2, 0),
  );
  return covariance / Math.sqrt(sumsOfSquares[0] * sumsOfSquares[1]);
};

describe("variation plan schema v2", () => {
  it("shares the exact localized torque authoring fixture with PyQt", () => {
    const decoded = planFromJson(JSON.stringify(localizedTorqueFixture));
    expect(JSON.parse(planToJson(decoded))).toEqual(localizedTorqueFixture);
  });

  it.each([
    ["missing window", null, ["joint.shoulder"], /finite half-open time window/],
    ["reversed window", [0.4, 0.2], ["joint.shoulder"], /start < end/],
    ["off-duration window", [0.2, 1.6], ["joint.shoulder"], /1\.5 s/],
    ["spatial point ID", [0.2, 0.4], ["swing.wrist"], /topological joint.*spatial/],
  ])("rejects a localized torque %s before use", (_name, window, points, message) => {
    const payload = JSON.parse(JSON.stringify(localizedTorqueFixture)) as {
      noise: Array<{ time_window_s: number[] | null; point_ids: string[] }>;
    };
    payload.noise[0].time_window_s = window as number[] | null;
    payload.noise[0].point_ids = points as string[];
    expect(() => planFromJson(JSON.stringify(payload))).toThrow(message as RegExp);
  });

  it("round-trips Tee Height only for an active Tee setup", () => {
    const teePlan = groupedPlan();
    teePlan.mode = "delivery";
    teePlan.ballSetup = { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M };
    teePlan.baseVariables = { [TEE_HEIGHT_VARIATION_KEY]: DRIVER_TEE_HEIGHT_M };
    teePlan.noise = [{
      variableKey: TEE_HEIGHT_VARIATION_KEY,
      distribution: "normal",
      scale: 0.002,
      lower: 0,
      upper: 0.1,
      specId: TEE_HEIGHT_VARIATION_KEY,
      timeWindowS: null,
      pointIds: [],
    }];
    teePlan.groups = [];
    expect(planFromJson(planToJson(teePlan))).toEqual(teePlan);
    expect(() => runVariation(teePlan)).toThrow(/complete Rate simulation ensemble.*contact geometry/i);

    expect(() => validatePlan({
      ...teePlan,
      ballSetup: { supportMode: "ground", teeHeightM: 0 },
    })).toThrow(/Tee Height.*Ground.*select Tee/i);
  });

  it("migrates v1 defaults while retaining base variables and flight model", () => {
    const migrated = planFromJson(
      JSON.stringify({
        schema_version: 1,
        mode: "launch",
        base_variables: { [BALL]: 161.5 },
        noise: [
          {
            variable_key: BALL,
            distribution: "normal",
            scale: 1.5,
            lower: null,
            upper: null,
          },
        ],
        n_runs: 8,
        seed: 4,
        flight_model: "custom-flight-model",
      }),
    );

    expect(migrated.baseVariables).toEqual({ [BALL]: 161.5 });
    expect(migrated.flightModel).toBe("custom-flight-model");
    expect(migrated.noise[0]).toMatchObject({
      specId: BALL,
      timeWindowS: null,
      pointIds: [],
    });
    expect(migrated.groups).toEqual([]);
    expect(JSON.parse(planToJson(migrated))).toMatchObject({
      schema_version: 2,
      base_variables: { [BALL]: 161.5 },
      flight_model: "custom-flight-model",
    });
  });

  it("round-trips spec IDs, locus metadata, groups, and replay inputs", () => {
    const plan = groupedPlan();
    plan.noise[0].timeWindowS = [0.7, 0.8];
    plan.noise[0].pointIds = ["swing.clubhead"];

    const decoded = planFromJson(planToJson(plan));

    expect(decoded).toEqual(plan);
    expect(decoded.baseVariables).toEqual(plan.baseVariables);
    expect(decoded.flightModel).toBe(plan.flightModel);
    const datasetJson = JSON.parse(
      datasetToJson({
        plan: decoded,
        inputNames: [BALL, ANGLE],
        inputs: [[154, 13]],
        outputNames: [],
        outputs: [[]],
        success: [true],
      }),
    ) as { plan: Record<string, unknown> };
    expect(datasetJson.plan).toMatchObject({
      base_variables: plan.baseVariables,
      flight_model: plan.flightModel,
    });
  });

  it("rejects unsupported future schemas", () => {
    expect(() =>
      planFromJson(JSON.stringify({ schema_version: 3 })),
    ).toThrow(/unsupported schema_version 3/);
  });
});

describe("perturbation group contracts", () => {
  it.each([
    {
      name: "asymmetric",
      matrix: [
        [1, 0.2],
        [0.3, 1],
      ],
      message: /symmetric/,
    },
    {
      name: "non-unit correlation diagonal",
      matrix: [
        [2, 0.2],
        [0.2, 1],
      ],
      message: /unit diagonal/,
    },
    {
      name: "non-positive-semidefinite",
      matrix: [
        [1, 2],
        [2, 1],
      ],
      message: /positive semidefinite/,
    },
  ])("rejects a $name matrix", ({ matrix, message }) => {
    const plan = groupedPlan();
    plan.groups![0].matrix = matrix;
    expect(() => validatePlan(plan)).toThrow(message);
  });

  it("rejects covariance diagonals that disagree with spec scales", () => {
    const plan = groupedPlan("covariance");
    plan.groups![0].matrix[0][0] = 2;
    expect(() => validatePlan(plan)).toThrow(/diagonal.*scale/);
  });

  it("rejects unknown, overlapping, or non-normal group members", () => {
    const unknown = groupedPlan();
    unknown.groups![0].specIds[1] = "missing";
    expect(() => validatePlan(unknown)).toThrow(/unknown specId/);

    const overlap = groupedPlan();
    overlap.groups!.push({
      groupId: "overlap",
      specIds: ["speed", "angle"],
      matrixKind: "correlation",
      matrix: [
        [1, 0],
        [0, 1],
      ],
    });
    expect(() => validatePlan(overlap)).toThrow(/only one group/);

    const nonNormal = groupedPlan();
    nonNormal.noise[0].distribution = "uniform";
    expect(() => validatePlan(nonNormal)).toThrow(/normal distributions/);
  });
});

describe("grouped sampling", () => {
  it.each(["correlation", "covariance"] as const)(
    "samples declared %s semantics reproducibly",
    (matrixKind) => {
      const plan = groupedPlan(matrixKind);
      const first = sampleInputs(plan);
      const repeated = sampleInputs(plan);
      const changed = sampleInputs({ ...plan, seed: plan.seed + 1 });

      expect(repeated).toEqual(first);
      expect(changed).not.toEqual(first);
      expect(sampleCorrelation(first)).toBeCloseTo(0.7, 1);

      const standardDeviations = [0, 1].map((column) => {
        const deviations = first.map(
          (row) => row[column] - plan.baseVariables[plan.noise[column].variableKey],
        );
        const mean = deviations.reduce((sum, value) => sum + value, 0) / deviations.length;
        return Math.sqrt(
          deviations.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
            deviations.length,
        );
      });
      expect(Math.abs(standardDeviations[0] - 2)).toBeLessThan(0.15);
      expect(Math.abs(standardDeviations[1] - 1)).toBeLessThan(0.1);
    },
  );

  it("uses specId as the independent subset-stable stream key", () => {
    const both = groupedPlan();
    both.groups = [];
    both.nRuns = 32;
    const only = { ...both, noise: [both.noise[0]] };
    expect(sampleInputs(both).map((row) => row[0])).toEqual(
      sampleInputs(only).map((row) => row[0]),
    );
  });

  it("reduces grouped plans to independent marginals for one-at-a-time analysis", () => {
    const plan = groupedPlan();
    plan.nRuns = 4;
    const result = oneAtATimeSensitivity(plan);
    expect(result.inputKeys).toEqual([BALL, ANGLE]);
    expect(result.matrix).toHaveLength(2);
  });

  it("allows locus metadata in sampling but rejects scalar execution", () => {
    const plan = groupedPlan();
    plan.groups = [];
    plan.nRuns = 4;
    plan.noise = [
      {
        ...plan.noise[0],
        specId: "speed-at-impact",
        timeWindowS: [0.7, 0.8],
        pointIds: ["swing.clubhead"],
      },
    ];

    expect(sampleInputs(plan)).toHaveLength(4);
    expect(() => runVariation(plan)).toThrow(/global perturbations/);
  });
});

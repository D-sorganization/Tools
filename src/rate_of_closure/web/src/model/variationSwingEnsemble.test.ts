import { describe, expect, it } from "vitest";

import visualizationContract from "./__fixtures__/variation_visualization_contract.json";

import { DRIVER_TEE_HEIGHT_M } from "./ballSetup";
import { golfDefaultParams } from "./doublePendulum";
import type { SimulationInput } from "./simulation";
import { executeVariationWork } from "./variationExecutionService";
import { validateResult } from "./variationExecutionValidation";
import {
  CATEGORY_SWING,
  keysForMode,
  type VariationPlanTs,
} from "./variation";
import {
  runSwingVariation,
  localizedTorqueSourcesToCsv,
  swingEnsembleToJson,
  swingTracesToCsv,
} from "./variationSwingEnsemble";

const YAW = `${CATEGORY_SWING}.yaw_deg`;
const DAMPING = `${CATEGORY_SWING}.damping_shoulder`;
const SHOULDER_TORQUE = `${CATEGORY_SWING}.shoulder_commanded_torque_offset_nm`;
const WRIST_TORQUE = `${CATEGORY_SWING}.wrist_commanded_torque_offset_nm`;

const baseInput = (): SimulationInput => ({
  sourceKind: "double_pendulum",
  clubheadSpeedMph: 30,
  omegaDps: [0, 0, 0],
  loftDeg: 56,
  impactOffsetToeMm: 0,
  impactOffsetHighMm: 0,
  planeYawDeg: 0,
  planeSideTiltDeg: -45,
  planeForwardTiltDeg: 0,
  impactTimeS: null,
  swingDurationS: 0.05,
  ballSetup: { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M },
});

const plan = (): VariationPlanTs => ({
  mode: "swing",
  baseVariables: { [YAW]: 0, [DAMPING]: 0.4 },
  noise: [
    {
      variableKey: YAW,
      distribution: "uniform",
      scale: 2,
      lower: null,
      upper: null,
    },
    {
      variableKey: DAMPING,
      distribution: "uniform",
      scale: 0.02,
      lower: null,
      upper: null,
    },
  ],
  nRuns: 3,
  seed: 9,
  flightModel: "waterloo_penner",
  ballSetup: { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M },
});

describe("web swing variation ensemble", () => {
  it("offers trace-capable swing variables and context-gated tee height", () => {
    const keys = keysForMode("swing", {
      supportMode: "tee",
      teeHeightM: DRIVER_TEE_HEIGHT_M,
    });

    expect(keys).toContain(YAW);
    expect(keys).toContain(DAMPING);
    expect(keys).toContain("swing_sim.ball_setup.tee_height_m");
  });

  it("matches the canonical Python visualization identifiers", () => {
    expect(visualizationContract.coordinate_frame).toBe(
      "app_frame:x_target,y_up,z_right",
    );
    expect(visualizationContract.point_ids).toEqual([
      "swing.pivot",
      "swing.wrist",
      "swing.clubhead.reference",
    ]);
    expect(visualizationContract.trial_statuses).toEqual([
      "evaluated_hit",
      "evaluated_no_impact",
      "numerical_failure",
    ]);
  });

  it("retains every run and maps sampled plane and damping values", () => {
    const result = runSwingVariation(plan(), baseInput());

    expect(result.dataset.inputNames).toEqual([YAW, DAMPING]);
    expect(result.dataset.outputNames).toEqual(visualizationContract.output_names);
    expect(result.runs).toHaveLength(3);
    result.runs.forEach((trial, index) => {
      expect(trial.run).not.toBeNull();
      if (trial.run === null) throw new Error("expected evaluated trial");
      expect(trial.run.swing).toHaveLength(51);
      expect(trial.input.planeYawDeg).toBeCloseTo(result.dataset.inputs[index][0]);
      expect(trial.input.pendulumParameters?.d1).toBeCloseTo(
        result.dataset.inputs[index][1],
      );
      expect(trial.input.pendulumParameters?.d2).toBe(golfDefaultParams().d2);
    });
    expect(new Set(result.runs.map((trial) => trial.input.planeYawDeg)).size).toBe(3);

    const document = JSON.parse(swingEnsembleToJson(result));
    expect(document.coordinateFrame).toBe("app_frame:x_target,y_up,z_right");
    expect(document.trials).toHaveLength(3);
    const csv = swingTracesToCsv(result);
    expect(csv).toContain("x_target_m,y_up_m,z_right_m");
    expect(csv).toContain("swing.clubhead.reference");
  });

  it("executes localized shoulder/wrist commands and exports typed provenance", () => {
    const localizedPlan: VariationPlanTs = {
      ...plan(),
      baseVariables: { [SHOULDER_TORQUE]: 3, [WRIST_TORQUE]: -2 },
      noise: [
        {
          variableKey: SHOULDER_TORQUE, specId: "drive.shoulder", distribution: "uniform",
          scale: 0.1, lower: null, upper: null,
          timeWindowS: [0.001, 0.003], pointIds: ["joint.shoulder"],
        },
        {
          variableKey: WRIST_TORQUE, specId: "drive.wrist", distribution: "uniform",
          scale: 0.1, lower: null, upper: null,
          timeWindowS: [0.002, 0.004], pointIds: ["joint.wrist"],
        },
      ],
    };
    const first = runSwingVariation(localizedPlan, {
      ...baseInput(), contactMode: "fixed_ball_contact",
    });
    const second = runSwingVariation(localizedPlan, {
      ...baseInput(), contactMode: "fixed_ball_contact",
    });

    expect(first).toEqual(second);
    expect(first.runs).toHaveLength(localizedPlan.nRuns);
    expect(first.runs.every((trial) => trial.status === "evaluated_no_impact")).toBe(true);
    expect(first.runs.every((trial) => trial.run?.impactOutcome.status === "miss")).toBe(true);
    const commands = first.runs[0].localizedTorqueCommands;
    expect(commands.map((command) => command.specId)).toEqual(["drive.shoulder", "drive.wrist"]);
    expect(commands.map((command) => command.jointId)).toEqual([
      "joint.shoulder", "joint.wrist",
    ]);
    expect(commands.every((command) => command.unit === "N*m")).toBe(true);
    expect(first.runs[0].run?.torqueRun.appliedTorqueHistory[1].torquesNm)
      .toEqual({ "joint.shoulder": first.dataset.inputs[0][0], "joint.wrist": 0 });
    expect(first.runs[0].run?.torqueRun.appliedTorqueHistory[3].torquesNm)
      .toEqual({ "joint.shoulder": 0, "joint.wrist": first.dataset.inputs[0][1] });
    expect(first.runs[0].run?.torqueRun.appliedTorqueHistory[4].torquesNm)
      .toEqual({ "joint.shoulder": 0, "joint.wrist": 0 });

    const json = JSON.parse(swingEnsembleToJson(first));
    expect(json.trials[0].localizedTorqueCommands[0]).toMatchObject({
      specId: "drive.shoulder",
      jointId: "joint.shoulder",
      unit: "N*m",
      provenance: "variation_plan.v2:additive_commanded_torque",
    });
    const csv = localizedTorqueSourcesToCsv(first);
    expect(csv).toContain("spec_id,variable_key,joint_id,window_start_s,window_end_s,torque_nm,unit,provenance");
    expect(csv).toContain("drive.shoulder");
    expect(csv).not.toContain("swing.clubhead.reference");

    const failures = runSwingVariation(localizedPlan, {
      ...baseInput(), doublePendulumInitialState: [Number.NaN, 0, 0, 0],
    });
    expect(failures.runs.every((trial) =>
      trial.status === "numerical_failure" &&
      trial.run === null &&
      trial.error !== null &&
      trial.localizedTorqueCommands.length === 2)).toBe(true);
    expect(failures.dataset.success).toEqual([false, false, false]);
    expect(failures.dataset.outputs.flat().every((value) => value === null)).toBe(true);

    const request = { plan: localizedPlan, analysisExecution: "all_together" as const };
    const workerResult = executeVariationWork(request, () => undefined);
    const forgedInput = workerResult.ensemble?.runs[0].input as unknown as Record<string, unknown>;
    forgedInput.sourceKind = "manual";
    expect(() => validateResult(workerResult, request)).toThrow(/trials/i);

    expect(() => runSwingVariation(localizedPlan, {
      ...baseInput(), swingDurationS: 0.0025,
    })).toThrow(/window exceeds run/i);
  });
});

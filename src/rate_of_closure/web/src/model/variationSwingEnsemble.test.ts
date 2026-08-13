import { describe, expect, it } from "vitest";

import visualizationContract from "./__fixtures__/variation_visualization_contract.json";

import { DRIVER_TEE_HEIGHT_M } from "./ballSetup";
import { golfDefaultParams } from "./doublePendulum";
import type { SimulationInput } from "./simulation";
import {
  CATEGORY_SWING,
  keysForMode,
  type VariationPlanTs,
} from "./variation";
import {
  runSwingVariation,
  swingEnsembleToJson,
  swingTracesToCsv,
} from "./variationSwingEnsemble";

const YAW = `${CATEGORY_SWING}.yaw_deg`;
const DAMPING = `${CATEGORY_SWING}.damping_shoulder`;

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
});

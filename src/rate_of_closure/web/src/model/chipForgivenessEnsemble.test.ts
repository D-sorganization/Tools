import { describe, expect, it } from "vitest";

import { getClub } from "./club";
import {
  analyzeChipForgivenessEnsemble,
  chipForgivenessStudyToCsv,
  chipForgivenessStudyToJson,
  chipForgivenessVariationDataset,
  defaultChipVariationInput,
} from "./chipForgivenessEnsemble";
import { runSwingVariation } from "./variationSwingEnsemble";
import { buildScalarPlotVariables } from "./variationPlotData";
import type { VariationPlanTs } from "./variation";

const plan: VariationPlanTs = {
  mode: "swing",
  baseVariables: {},
  noise: [{
    variableKey: "swing_sim.swing.yaw_deg",
    distribution: "uniform",
    scale: 0.1,
    lower: null,
    upper: null,
    specId: "yaw",
    timeWindowS: null,
    pointIds: [],
  }],
  nRuns: 3,
  seed: 13,
  flightModel: "waterloo_penner",
  groups: [],
  ballSetup: { supportMode: "ground", teeHeightM: 0 },
};

describe("browser chip forgiveness ensemble adapter", () => {
  it("provides an explicit ground-mode wedge simulation input", () => {
    const input = defaultChipVariationInput();
    const wedge = getClub("Sand Wedge");

    expect(input.ballSetup).toEqual({ supportMode: "ground", teeHeightM: 0 });
    expect(input.loftDeg).toBe(wedge.loftDeg);
    expect(input.club?.headMassKg).toBe(wedge.headMassKg);
  });

  it("retains every trial and exposes ground and shaft metrics", () => {
    const ensemble = runSwingVariation(plan, defaultChipVariationInput());

    const study = analyzeChipForgivenessEnsemble(ensemble, { seed: plan.seed });

    expect(study.records).toHaveLength(plan.nRuns);
    expect(study.summary.sampleCount).toBe(plan.nRuns);
    expect(study.records[0].metrics).toHaveProperty("low_point_clearance_m");
    expect(study.records[0].metrics).toHaveProperty("shaft_counterfactual_aoa_delta_deg");
    expect(study.records[0].metrics).toHaveProperty("shaft_shapley_aoa_deg");
    expect(study.metadata.objectiveId).toBe("chip-target-27.432m-balanced-v1");
    expect(study.inputs.lossModel.includeTurfPenetration).toBe(false);
    expect(study.summary.rankingScope).toMatch(/conditional/i);
    const dataset = chipForgivenessVariationDataset(study);
    expect(dataset.outputNames).toContain("loss");
    expect(dataset.outputNames).toContain("shaft_counterfactual_aoa_delta_deg");
    expect(dataset.outputs).toHaveLength(plan.nRuns);
    expect(() => buildScalarPlotVariables(dataset)).not.toThrow();
  });

  it("retains numerical failures with the declared penalty", () => {
    const ensemble = runSwingVariation(plan, defaultChipVariationInput());
    ensemble.runs[1] = {
      ...ensemble.runs[1],
      status: "numerical_failure",
      run: null,
      error: "planted failure",
    };

    const study = analyzeChipForgivenessEnsemble(ensemble, { seed: plan.seed });

    expect(study.records[1].cohort).toBe("numerical_failure");
    expect(study.records[1].constraintViolated).toBe(true);
    expect(study.records[1].diagnostic).toMatch(/planted failure/i);
    expect(study.summary.cohorts.numerical_failure.count).toBe(1);
    const payload = JSON.parse(chipForgivenessStudyToJson(study));
    expect(payload.schemaVersion).toBe(1);
    expect(payload.metadata.samplingDesign).toBe("iid-monte-carlo-joint");
    expect(payload.inputs.wedge.bounceDeg).toBe(10);
    expect(chipForgivenessStudyToCsv(study)).toContain("numerical_failure");
  });

  it("derives non-aliasing custom objective identity and rejects nonfinite exports", () => {
    const ensemble = runSwingVariation(plan, defaultChipVariationInput());
    const study = analyzeChipForgivenessEnsemble(ensemble, { targetCarryM: 20.0004 });
    const adjacent = analyzeChipForgivenessEnsemble(ensemble, { targetCarryM: 20.0005 });

    expect(study.metadata.objectiveId).toBe("chip-target-20.0004m-balanced-v1");
    expect(adjacent.metadata.objectiveId).toBe("chip-target-20.0005m-balanced-v1");
    study.records[0].metrics.carry_m = Number.NaN;
    expect(() => chipForgivenessStudyToJson(study)).toThrow(/nonfinite/i);
    expect(() => chipForgivenessStudyToCsv(study)).toThrow(/nonfinite/i);
  });
});

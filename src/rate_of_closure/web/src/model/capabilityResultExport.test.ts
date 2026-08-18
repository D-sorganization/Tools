import { describe, expect, it } from "vitest";

import { runCapabilityOptimization } from "./capabilityRun";
import {
  buildCapabilityWorkflow,
  defaultCapabilityWorkflowInputs,
} from "./capabilityWorkflow";
import {
  capabilityAlternativesCsv,
  stableCapabilityResultExportJson,
} from "./capabilityResultExport";

const output = runCapabilityOptimization(buildCapabilityWorkflow({
  ...defaultCapabilityWorkflowInputs(), candidateBudget: 1, ensembleSize: 2,
  alternativesCount: 1,
}));

describe("capability result exports", () => {
  it("exports every ranked diagnostic with parameter units", () => {
    const csv = capabilityAlternativesCsv(output.result, output.ensemble);

    expect(csv.split("\n")[0]).toBe(
      "rank,club_id,parameters,score,mean_carry_m,expected_miss_m,dispersion_rms_m," +
      "target_hold_probability,cvar_miss_m,downside_carry_m,sample_count,successful_count," +
      "no_impact_count,failed_count,failure_fraction,confidence,extrapolated,pareto_efficient," +
      "limiting_constraints",
    );
    expect(csv).toContain("ball_speed=");
    expect(csv).toContain("m/s");
  });

  it("uses a versioned stable envelope compatible with the Python wire shape", () => {
    const payload = JSON.parse(stableCapabilityResultExportJson(output.result, output.ensemble));

    expect(payload.schema_version).toBe("capability-result-export/v1");
    expect(payload.result.alternatives[0]).toEqual(expect.objectContaining({
      cvar_miss_m: expect.any(Number), downside_carry_m: expect.any(Number),
      sample_count: 2, successful_count: expect.any(Number),
      failure_fraction: expect.any(Number), extrapolated: expect.any(Boolean),
      pareto_efficient: expect.any(Boolean),
    }));
    expect(payload.parameter_units).toEqual(expect.arrayContaining([
      { parameter_id: "ball_speed", unit: "m/s" },
    ]));
  });

  it("fails closed when result and ensemble provenance do not match", () => {
    const mismatched = { ...output.ensemble, result_id: "different-problem" };

    expect(() => capabilityAlternativesCsv(output.result, mismatched))
      .toThrow("result and ensemble IDs must match");
    expect(() => stableCapabilityResultExportJson(output.result, mismatched))
      .toThrow("result and ensemble IDs must match");
  });
});

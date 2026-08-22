import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/durable_ensemble_evidence_golden_v1.json";
import {
  DURABLE_ENSEMBLE_ANALYSIS_METHOD,
  DURABLE_ENSEMBLE_EVIDENCE_SCHEMA,
  parseDurableEnsembleEvidence,
} from "./durableEnsembleEvidence";

describe("durable ensemble evidence wire", () => {
  it("parses the shared path-free golden evidence", () => {
    const evidence = parseDurableEnsembleEvidence(fixture);

    expect(evidence.schemaVersion).toBe(DURABLE_ENSEMBLE_EVIDENCE_SCHEMA);
    expect(evidence.analysis.methodId).toBe(DURABLE_ENSEMBLE_ANALYSIS_METHOD);
    expect(evidence.archive).toMatchObject({
      status: "in_progress", trialCount: 5, analyzedTrialCount: 3, failedCount: 1,
    });
    expect(evidence.outputMoments).toHaveLength(17);
    expect(JSON.stringify(fixture)).not.toMatch(/directory|[A-Z]:\\/i);
  });

  it.each([
    ["unknown field", { ...fixture, extra: true }, /fields/i],
    ["cross-plan frame", {
      ...fixture, analysis: { ...fixture.analysis, coordinate_frame: "private.frame" },
    }, /frame/i],
    ["wrong unit", {
      ...fixture,
      output_moments: fixture.output_moments.map((row, index) =>
        index === 0 ? { ...row, unit: "ft" } : row),
    }, /canonical/i],
    ["partial count mismatch", {
      ...fixture, archive: { ...fixture.archive, analyzed_trial_count: 4 },
    }, /status counts/i],
  ])("rejects %s", (_label, value, message) => {
    expect(() => parseDurableEnsembleEvidence(value)).toThrow(message);
  });
});

/** UI-neutral Morris job and report presentation tests. */

import { describe, expect, it } from "vitest";

import reportFixture from "./__fixtures__/morris_global_sensitivity_golden_v1.json";
import { parseMorrisJobEnvelope, type MorrisJobEnvelope } from "./morrisAuthorityContract";
import { parseMorrisReport } from "./morrisGlobalSensitivityContract";
import { presentMorrisJob, presentMorrisReport } from "./morrisPresentation";

const runningJob = (): MorrisJobEnvelope => parseMorrisJobEnvelope({
  schema_id: "rate-of-closure/morris-job",
  schema_version: 1,
  job_id: "job-1",
  request_id: "study-1",
  status: "running",
  completed_samples: 9,
  total_samples: 36,
  cancel_requested: false,
  report: null,
  error: null,
});

describe("Morris job presentation", () => {
  it("derives progress and available actions from the job envelope", () => {
    expect(presentMorrisJob(runningJob())).toEqual({
      status: "running",
      terminal: false,
      completedSamples: 9,
      totalSamples: 36,
      progressFraction: 0.25,
      cancelRequested: false,
      canCancel: true,
      canPresentResults: false,
      message: "Morris study running: 9/36",
      errorCode: null,
      errorMessage: null,
    });
  });

  it("presents failure diagnostics without exposing result actions", () => {
    const failed = parseMorrisJobEnvelope({
      schema_id: "rate-of-closure/morris-job", schema_version: 1,
      job_id: "job-1", request_id: "study-1", status: "failed",
      completed_samples: 8, total_samples: 36, cancel_requested: false,
      report: null, error: { code: "execution_failed", message: "Morris execution failed" },
    });

    expect(presentMorrisJob(failed)).toMatchObject({
      terminal: true, canCancel: false, canPresentResults: false,
      errorCode: "execution_failed", errorMessage: "Morris execution failed",
    });
  });
});

describe("Morris report presentation", () => {
  it("ranks finite effects within one exact target and carries diagnostics", () => {
    const report = parseMorrisReport(structuredClone(reportFixture));
    const presentation = presentMorrisReport(report, "clubhead_x_m");

    expect(presentation.target).toEqual({
      name: "clubhead_x_m", unit: "m", kind: "state-point",
      timeS: 0.03, pointId: "clubhead", coordinateFrame: "app_frame:x_target,y_up,z_right",
      label: "Clubhead X M",
    });
    expect(presentation.rows.map((row) => [row.rank, row.specId, row.muStar])).toEqual([
      [1, "speed-global", 3],
      [2, "face-window", 2],
    ]);
    expect(presentation.rows[0]).toMatchObject({
      variableKey: "swing_sim.impact.delivery.clubhead_speed_mps",
      label: "Clubhead Speed", sourceUnit: "m/s", sourceLower: 0, sourceUpper: 1,
      availability: "available", sampleAdequacy: "adequate",
      totalPairs: 12, validPairs: 12, typedNoImpactPairs: 0,
      noImpactUnavailablePairs: 0, failedPairs: 0, nonfinitePairs: 0,
    });
    expect(Object.isFrozen(presentation)).toBe(true);
    expect(presentation.rows.every(Object.isFrozen)).toBe(true);
  });

  it("rejects an unknown target rather than silently choosing one", () => {
    const report = parseMorrisReport(structuredClone(reportFixture));
    expect(() => presentMorrisReport(report, "missing")).toThrow(/target/);
  });
});

import { describe, expect, it } from "vitest";

import { defaultVariationPlan } from "../components/variationUi";
import fixture from "./__fixtures__/durable_ensemble_evidence_golden_v1.json";
import {
  parseDurableEnsembleJob,
  serializeDurableEnsembleRequest,
} from "./durableEnsembleAuthorityContract";
import { defaultMorrisAuthorityBase } from "./morrisWorkflowDefaults";
import { getClub } from "./club";
import { DEFAULT_SCENARIO } from "./impact";

const base = defaultMorrisAuthorityBase(getClub("Driver 10.5°"), {
  ...DEFAULT_SCENARIO, clubheadSpeedMph: 113,
});
const swingPlan = () => ({
  ...defaultVariationPlan(), mode: "swing" as const,
  baseVariables: { "swing_sim.swing.yaw_deg": 0 },
  noise: [{
    variableKey: "swing_sim.swing.yaw_deg", distribution: "normal" as const,
    scale: 1, lower: null, upper: null, specId: "swing_sim.swing.yaw_deg",
    timeWindowS: null, pointIds: [],
  }],
  ballSetup: { supportMode: "tee" as const, teeHeightM: 0.0381 },
});

describe("durable ensemble authority contract", () => {
  it("serializes the global plan without a client-owned ball setup", () => {
    const document = serializeDurableEnsembleRequest({
      requestId: "request-1", archiveId: "archive-1", plan: swingPlan(),
      base, chunkSize: 128,
    });
    expect(document).toMatchObject({
      request_id: "request-1", archive_id: "archive-1", chunk_size: 128,
      base: { clubhead_speed_mph: 113, contact_mode: "fixed_ball_contact" },
    });
    expect(document.plan).not.toHaveProperty("ball_setup");
    expect(document.plan_sha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("parses incremental path-free jobs and rejects inconsistent progress", () => {
    const document = {
      schema_id: "rate-of-closure/durable-ensemble-job", schema_version: 1,
      job_id: "job-1", request_id: "request-1", archive_id: "archive-1",
      status: "running", completed_trials: 3, total_trials: 5,
      cancel_requested: false, evidence: fixture, error: null,
    };
    expect(parseDurableEnsembleJob(document)).toMatchObject({ completedTrials: 3, status: "running" });
    expect(() => parseDurableEnsembleJob({ ...document, completed_trials: 4 })).toThrow(/inconsistent/i);
  });

  it("rejects localized and oversized work", () => {
    const plan = swingPlan();
    expect(() => serializeDurableEnsembleRequest({
      requestId: "request-1", archiveId: "archive-1",
      plan: { ...plan, noise: plan.noise.map((row, index) => index === 0 ? { ...row, timeWindowS: [0, 1] } : row) },
      base, chunkSize: 128,
    })).toThrow(/global perturbations/i);
    expect(() => serializeDurableEnsembleRequest({
      requestId: "request-1", archiveId: "archive-1", plan, base, chunkSize: 4097,
    })).toThrow(/4096/);
  });
});

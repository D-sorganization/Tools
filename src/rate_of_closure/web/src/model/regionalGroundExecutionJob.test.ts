import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/regional_ground_execution_job_golden_v1.json";
import {
  MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
  parseRegionalGroundExecutionJob,
  regionalGroundExecutionJobFromJson,
  stableRegionalGroundExecutionJobJson,
} from "./regionalGroundExecutionJob";

const canonical = JSON.stringify(fixture.job);
const clone = (): Record<string, unknown> => structuredClone(fixture.job);
const nested = (
  value: Record<string, unknown>,
  key: string,
): Record<string, unknown> => value[key] as Record<string, unknown>;

describe("regional-ground execution job v1", () => {
  it("round-trips the shared Python canonical fixture and all identities", () => {
    const job = regionalGroundExecutionJobFromJson(canonical);

    expect(stableRegionalGroundExecutionJobJson(job)).toBe(canonical);
    expect(job.input_sha256).toBe(fixture.input_sha256);
    expect(job.job_sha256).toBe(fixture.job_sha256);
    expect(job.flight.trajectory_sha256).toBe(
      fixture.job.flight.trajectory_sha256,
    );
  });

  it.each([
    ["root extra", () => ({ ...fixture.job, extra: true }), /fields/i],
    ["boolean capture", () => ({ ...fixture.job, capture_speed_m_s: true }), /capture.*finite/i],
    ["unsafe setting", () => {
      const value = clone();
      nested(nested(value, "flight"), "settings").step_s =
        Number.MAX_SAFE_INTEGER + 1;
      return value;
    }, /safe range/i],
    ["wrong input digest", () => ({ ...fixture.job, input_sha256: "0".repeat(64) }), /input_sha256/i],
    ["wrong job digest", () => ({ ...fixture.job, job_sha256: "0".repeat(64) }), /job_sha256/i],
    ["uppercase digest", () => {
      const value = clone();
      nested(value, "flight").result_sha256 = "A".repeat(64);
      return value;
    }, /lowercase/i],
    ["trial mismatch", () => {
      const value = clone();
      nested(value, "execution_options").max_trials = 5;
      return value;
    }, /n_runs/i],
    ["parallelism bound", () => {
      const value = clone();
      nested(value, "execution_options").max_parallelism = 33;
      return value;
    }, /parallelism/i],
    ["flight model mismatch", () => {
      const value = clone();
      nested(value, "flight").model_id = "nathan";
      return value;
    }, /model_id.*flight_model/i],
    ["launch-relative surface mismatch", () => {
      const value = clone();
      const setup = nested(nested(value, "launch"), "ball_setup");
      setup.tee_height_m = 0.03;
      (setup.ball_center_m as number[])[1] = 0.051335;
      return value;
    }, /launch-relative transfer surface/i],
  ])("rejects %s", (_label, build, message) => {
    expect(() => parseRegionalGroundExecutionJob(build())).toThrow(message);
  });

  it("rejects duplicate fields, surrogate text, and oversized UTF-8", () => {
    expect(() => regionalGroundExecutionJobFromJson(
      canonical.replace('"job_id":"driver-ground-study-1729"',
        '"job_id":"driver-ground-study-1729","job_id":"duplicate"'),
    )).toThrow(/duplicate/i);
    expect(() => regionalGroundExecutionJobFromJson(
      canonical.replace("fixture-4369", "fixture-\\ud800"),
    )).toThrow(/surrogate/i);
    expect(() => regionalGroundExecutionJobFromJson(
      "é".repeat(MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES / 2 + 1),
    )).toThrow(/maximum wire size/i);
  });

  it("deep-freezes imported executable inputs", () => {
    const job = regionalGroundExecutionJobFromJson(canonical);
    expect(Object.isFrozen(job)).toBe(true);
    expect(Object.isFrozen(job.flight.settings)).toBe(true);
    const plan = job.variation_request.regional_plan as Record<string, unknown>;
    expect(Object.isFrozen(plan.regions)).toBe(true);
  });
});

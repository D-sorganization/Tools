import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/regional_ground_execution_result_golden_v1.json";
import jobFixture from "./__fixtures__/regional_ground_execution_job_golden_v1.json";
import { canonicalGroundJson } from "./flightGroundContract";
import {
  MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
  assertRegionalGroundExecutionResultMatchesJob,
  parseRegionalGroundExecutionResult,
  regionalGroundExecutionResultFromJson,
  stableRegionalGroundExecutionResultJson,
} from "./regionalGroundExecutionResult";
import { parseRegionalGroundExecutionJob } from "./regionalGroundExecutionJob";
import { sha256Text } from "./sha256";

const canonical = JSON.stringify(fixture.result);
const clone = (): Record<string, unknown> => structuredClone(fixture.result);
const nested = (
  value: Record<string, unknown>,
  key: string,
): Record<string, unknown> => value[key] as Record<string, unknown>;

describe("regional-ground execution result v1", () => {
  it("round-trips the Python-produced golden and preserves typed nulls", () => {
    const result = regionalGroundExecutionResultFromJson(canonical);

    expect(stableRegionalGroundExecutionResultJson(result)).toBe(canonical);
    expect(result.dataset_sha256).toBe(fixture.dataset_sha256);
    expect(sha256Text(canonical)).toBe(fixture.canonical_sha256);
    expect(result.dataset.rows[1].values["metric.carry_distance"]).toBeNull();
    expect(() => assertRegionalGroundExecutionResultMatchesJob(
      result,
      parseRegionalGroundExecutionJob(jobFixture.job),
    )).not.toThrow();
  });

  it.each([
    ["root extra", () => ({ ...fixture.result, extra: true }), /fields/i],
    ["wrong dataset digest", () => ({
      ...fixture.result, dataset_sha256: "0".repeat(64),
    }), /dataset_sha256/i],
    ["dataset substitution", () => {
      const value = clone();
      const rows = nested(value, "dataset").rows as Record<string, unknown>[];
      nested(rows[0], "values")["metric.carry_distance"] = 999;
      return value;
    }, /dataset_sha256/i],
    ["nested extra", () => {
      const value = clone();
      nested(value, "dataset").extra = true;
      return value;
    }, /scalar ensemble result.*fields/i],
    ["typed null string", () => {
      const value = clone();
      const rows = nested(value, "dataset").rows as Record<string, unknown>[];
      nested(rows[1], "values")["metric.carry_distance"] = "0";
      return value;
    }, /finite/i],
    ["missing typed value", () => {
      const value = clone();
      const rows = nested(value, "dataset").rows as Record<string, unknown>[];
      delete nested(rows[0], "values")["metric.carry_distance"];
      return value;
    }, /variable keys/i],
  ])("rejects %s", (_label, build, message) => {
    expect(() => parseRegionalGroundExecutionResult(build())).toThrow(message);
  });

  it.each(["job_id", "job_sha256", "input_sha256"])(
    "rejects substituted %s against the expected job",
    (field) => {
      const value = clone();
      value[field] = field === "job_id" ? "substituted" : "0".repeat(64);
      const parsed = parseRegionalGroundExecutionResult(value);
      expect(() => assertRegionalGroundExecutionResultMatchesJob(
        parsed,
        parseRegionalGroundExecutionJob(jobFixture.job),
      )).toThrow(new RegExp(field));
    },
  );

  it.each([
    ["result ID", (dataset: Record<string, unknown>) => {
      dataset.result_id = "other-study";
    }, /result_id/i],
    ["trial count", (dataset: Record<string, unknown>) => {
      (dataset.rows as unknown[]).pop();
    }, /trial count/i],
    ["trial ordering", (dataset: Record<string, unknown>) => {
      (dataset.rows as unknown[]).reverse();
    }, /trial ordering/i],
    ["series ID", (dataset: Record<string, unknown>) => {
      const row = (dataset.rows as Record<string, unknown>[])[0];
      row.series_id = "substituted";
      row.row_id = "series:substituted/trial:0";
    }, /series_id/i],
  ])("rejects self-consistent dataset substitution: %s", (_label, mutate, message) => {
    const value = clone();
    const dataset = nested(value, "dataset");
    mutate(dataset);
    value.dataset_sha256 = sha256Text(canonicalGroundJson(dataset));
    const parsed = parseRegionalGroundExecutionResult(value);
    expect(() => assertRegionalGroundExecutionResultMatchesJob(
      parsed,
      parseRegionalGroundExecutionJob(jobFixture.job),
    )).toThrow(message);
  });

  it("rejects duplicate fields and oversized UTF-8 before parsing", () => {
    expect(() => regionalGroundExecutionResultFromJson(canonical.replace(
      '"job_id":"driver-ground-study-1729"',
      '"job_id":"driver-ground-study-1729","job_id":"duplicate"',
    ))).toThrow(/duplicate/i);
    expect(() => regionalGroundExecutionResultFromJson(canonical.replace(
      '"result_id":"seeded-ground-study"',
      '"result_id":"seeded-ground-study","result_id":"duplicate"',
    ))).toThrow(/duplicate/i);
    expect(() => regionalGroundExecutionResultFromJson(
      "é".repeat(MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES / 2 + 1),
    )).toThrow(/maximum wire size/i);
  });

  it("deep-freezes the complete result object", () => {
    const result = regionalGroundExecutionResultFromJson(canonical);
    expect(Object.isFrozen(result)).toBe(true);
    expect(Object.isFrozen(result.dataset)).toBe(true);
    expect(Object.isFrozen(result.dataset.rows[0].values)).toBe(true);
  });
});

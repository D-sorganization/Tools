import { describe, expect, it } from "vitest";

import baseFixture from "./__fixtures__/flight_to_ground_golden_v1.json";
import fixture from "./__fixtures__/ground_regional_execution_golden_v1.json";
import { canonicalGroundJson } from "./flightGroundContract";
import {
  groundRegionalExecutionResultFromJson,
  parseGroundRegionalExecutionResult,
  stableGroundRegionalExecutionJson,
} from "./groundRegionalExecution";
import { sha256Text } from "./sha256";

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

describe("regional ground execution result v1", () => {
  it("round-trips the shared fixture with exact digest and frozen base bytes", () => {
    const parsed = parseGroundRegionalExecutionResult(fixture.result);

    expect(JSON.parse(stableGroundRegionalExecutionJson(parsed))).toEqual(fixture.result);
    expect(sha256Text(stableGroundRegionalExecutionJson(parsed))).toBe(
      fixture.result_sha256,
    );
    expect(canonicalGroundJson(parsed.ground_result)).toBe(
      canonicalGroundJson(baseFixture.result),
    );
    expect(parsed.executor_provenance.input_sha256).toBe(parsed.execution_input_sha256);
  });

  it("rejects field, digest, identity, and transition-ledger fabrication", () => {
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.result,
      unexpected: true,
    })).toThrow(/fields/);
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.result,
      ground_request_sha256: "bad",
    })).toThrow(/ground_request_sha256/);
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.result,
      ground_request_sha256: fixture.result.ground_request_sha256.toUpperCase(),
    })).toThrow(/ground_request_sha256/);
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.result,
      request_id: "different",
    })).toThrow(/identities/);

    const transition = clone(fixture.result) as Record<string, unknown>;
    transition.transitions = [{
      event_sequence: 99,
      time_s: 6,
      position_m: [224, 0.02135, -2.5],
      from_region_id: null,
      to_region_id: "rough-band",
      from_surface_id: "firm-fairway",
      to_surface_id: "regional-rough",
    }];
    expect(() => parseGroundRegionalExecutionResult(transition)).toThrow(
      /transition ledger/,
    );
  });

  it("accepts typed cancellation without fabricating a ground result", () => {
    const cancelled = clone(fixture.result) as Record<string, unknown>;
    cancelled.status = "cancelled";
    cancelled.failure_reason = "cancelled";
    cancelled.ground_result = null;

    const parsed = parseGroundRegionalExecutionResult(cancelled);

    expect(parsed.status).toBe("cancelled");
    expect(parsed.failure_reason).toBe("cancelled");
    expect(parsed.ground_result).toBeNull();
    expect(parsed.plan_id).toBe(fixture.result.plan_id);
  });

  it("rejects duplicate and malformed JSON before partial acceptance", () => {
    expect(() => groundRegionalExecutionResultFromJson(
      '{"schema_version":"ground-regional-execution-result/v1",' +
      '"schema_version":"ground-regional-execution-result/v1"}',
    )).toThrow(/duplicate/);
    expect(() => groundRegionalExecutionResultFromJson("[]")).toThrow(/object/);
  });
});

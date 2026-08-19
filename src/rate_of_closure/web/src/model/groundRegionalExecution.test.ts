import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ground_regional_execution_golden_v1.json";
import {
  groundRegionalExecutionResultFromJson,
  parseGroundRegionalExecutionResult,
  stableGroundRegionalExecutionJson,
} from "./groundRegionalExecution";
import { sha256Text } from "./sha256";

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

describe("regional ground execution result v1", () => {
  it("round-trips the executor-produced shared fixture with exact digest", () => {
    const parsed = parseGroundRegionalExecutionResult(fixture.representable.result);

    expect(JSON.parse(stableGroundRegionalExecutionJson(parsed))).toEqual(
      fixture.representable.result,
    );
    expect(sha256Text(stableGroundRegionalExecutionJson(parsed))).toBe(
      fixture.representable.result_sha256,
    );
    expect(parsed.executor_provenance.input_sha256).toBe(parsed.execution_input_sha256);
  });

  it("rejects field, digest, identity, and transition-ledger fabrication", () => {
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.representable.result,
      unexpected: true,
    })).toThrow(/fields/);
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.representable.result,
      ground_request_sha256: "bad",
    })).toThrow(/ground_request_sha256/);
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.representable.result,
      ground_request_sha256: fixture.representable.result.ground_request_sha256.toUpperCase(),
    })).toThrow(/ground_request_sha256/);
    expect(() => parseGroundRegionalExecutionResult({
      ...fixture.representable.result,
      request_id: "different",
    })).toThrow(/identities/);

    const transition = clone(fixture.representable.result) as Record<string, unknown>;
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
      /transition ledger|regional plan/,
    );
  });

  it("accepts typed cancellation without fabricating a ground result", () => {
    const parsed = parseGroundRegionalExecutionResult(fixture.cancelled.result);

    expect(parsed.status).toBe("cancelled");
    expect(parsed.failure_reason).toBe("cancelled");
    expect(parsed.ground_result).toBeNull();
    expect(parsed.plan_id).toBe(fixture.cancelled.result.plan_id);
    [fixture.cancelled, fixture.failed].forEach((entry) => {
      const outcome = parseGroundRegionalExecutionResult(entry.result);
      expect(JSON.parse(stableGroundRegionalExecutionJson(outcome))).toEqual(
        entry.result,
      );
      expect(sha256Text(stableGroundRegionalExecutionJson(outcome))).toBe(
        entry.result_sha256,
      );
    });
    const invalidCancelled = clone(fixture.cancelled.result) as Record<string, unknown>;
    invalidCancelled.failure_reason = "step_limit";
    expect(() => parseGroundRegionalExecutionResult(invalidCancelled)).toThrow(
      /cancelled failure_reason/,
    );
    const invalidFailed = clone(fixture.failed.result) as Record<string, unknown>;
    invalidFailed.failure_reason = "cancelled";
    expect(() => parseGroundRegionalExecutionResult(invalidFailed)).toThrow(
      /cancelled status/,
    );
    const fabricatedCancelled = clone(fixture.cancelled.result) as Record<string, unknown>;
    fabricatedCancelled.transitions = fixture.representable.result.transitions;
    expect(() => parseGroundRegionalExecutionResult(fabricatedCancelled)).toThrow(
      /null ground_result cannot declare transitions/,
    );
  });

  it("rejects fake executor identity and plan-inconsistent transition mappings", () => {
    const producer = clone(fixture.representable.result) as Record<string, unknown>;
    (producer.executor_provenance as Record<string, unknown>).producer = "lookalike";
    expect(() => parseGroundRegionalExecutionResult(producer)).toThrow(/executor producer/);
    const version = clone(fixture.representable.result) as Record<string, unknown>;
    (version.executor_provenance as Record<string, unknown>).producer_version = "9.9.9";
    expect(() => parseGroundRegionalExecutionResult(version)).toThrow(/executor version/);
    const revision = clone(fixture.representable.result) as Record<string, unknown>;
    (revision.executor_provenance as Record<string, unknown>).source_revision =
      "verified-build-2";
    expect(parseGroundRegionalExecutionResult(revision).executor_provenance.source_revision)
      .toBe("verified-build-2");

    const mapping = clone(fixture.representable.result) as Record<string, unknown>;
    const transitions = mapping.transitions as Array<Record<string, unknown>>;
    transitions[0].from_region_id = "rough-band";
    transitions[0].to_region_id = null;
    expect(() => parseGroundRegionalExecutionResult(mapping)).toThrow(/regional plan/);

    const reversed = clone(fixture.representable.result) as Record<string, unknown>;
    const reversedItem = (reversed.transitions as Array<Record<string, unknown>>)[0];
    [reversedItem.from_region_id, reversedItem.to_region_id] =
      [reversedItem.to_region_id, reversedItem.from_region_id];
    [reversedItem.from_surface_id, reversedItem.to_surface_id] =
      [reversedItem.to_surface_id, reversedItem.from_surface_id];
    expect(() => parseGroundRegionalExecutionResult(reversed)).toThrow(/regional plan/);
  });

  it("matches shared adversarial transition-wire acceptance", async () => {
    const cases: ReadonlyArray<{
      readonly accepted: boolean;
      readonly overrides: Readonly<Record<string, unknown>>;
    }> = (await import(
      "./__fixtures__/ground_regional_execution_adversarial_v1.json"
    )).default.cases;
    cases.forEach((item) => {
      const changed = clone(fixture.representable.result) as Record<string, unknown>;
      Object.assign(
        (changed.transitions as Array<Record<string, unknown>>)[0],
        item.overrides,
      );
      if (item.accepted) {
        expect(parseGroundRegionalExecutionResult(changed).transitions[0].event_sequence)
          .toBe(3);
      } else {
        expect(() => parseGroundRegionalExecutionResult(changed)).toThrow();
      }
    });
  });

  it("rejects duplicate and malformed JSON before partial acceptance", () => {
    expect(() => groundRegionalExecutionResultFromJson(
      '{"schema_version":"ground-regional-execution-result/v1",' +
      '"schema_version":"ground-regional-execution-result/v1"}',
    )).toThrow(/duplicate/);
    expect(() => groundRegionalExecutionResultFromJson("[]")).toThrow(/object/);
  });
});

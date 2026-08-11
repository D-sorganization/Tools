import { describe, expect, it } from "vitest";

import requestFixture from "./__fixtures__/ground_repeated_bounce_request_wire_golden_v1.json";
import resultFixture from "./__fixtures__/ground_repeated_bounce_wire_golden_v1.json";
import {
  MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES,
  pairRepeatedBounceRequestResult,
  parseRepeatedBounceRequest,
  repeatedBounceRequestFromJson,
  stableRepeatedBounceRequestJson,
} from "./repeatedBounceRequestWire";
import { sha256Text } from "./sha256";

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
type MutableRecord = Record<string, unknown>;

describe("repeated-bounce request wire v1", () => {
  it("matches Python canonical JSON and SHA-256 without browser physics", () => {
    const request = parseRepeatedBounceRequest(requestFixture.request);
    const text = stableRepeatedBounceRequestJson(request);

    expect(text).toBe(stableRepeatedBounceRequestJson(requestFixture.request));
    expect(sha256Text(text)).toBe(requestFixture.sha256);
    expect(repeatedBounceRequestFromJson(text)).toEqual(request);
    expect(request.unit_system).toBe("SI");
    expect(request.frame).toBe("target_frame:x_downrange,y_up,z_right");
  });

  it("pairs only exact request, result, model, surface, and digest identity", () => {
    const pair = pairRepeatedBounceRequestResult(
      requestFixture.request,
      resultFixture.result,
    );

    expect(pair.execution_input_sha256).toBe(
      requestFixture.request.execution_input_sha256,
    );
    expect(pair.result.request_fingerprint_sha256).toBe(
      pair.request.ground_request_sha256,
    );
  });

  it.each([
    ["unknown field", (value: MutableRecord) => { value.extra = true; }],
    ["schema", (value: MutableRecord) => { value.schema_version = "future"; }],
    ["units", (value: MutableRecord) => { value.unit_system = "imperial"; }],
    ["frame", (value: MutableRecord) => { value.frame = "world"; }],
    ["request identity", (value: MutableRecord) => { value.request_id = "wrong"; }],
    ["surface identity", (value: MutableRecord) => { value.surface_id = "wrong"; }],
    ["model identity", (value: MutableRecord) => { value.model_id = "wrong"; }],
    ["ground digest", (value: MutableRecord) => { value.ground_request_sha256 = "0".repeat(64); }],
    ["input digest", (value: MutableRecord) => { value.execution_input_sha256 = "0".repeat(64); }],
    ["nonfinite", (value: MutableRecord) => { value.capture_speed_m_s = Number.POSITIVE_INFINITY; }],
    ["nested extra", (value: MutableRecord) => {
      (value.ground_request as MutableRecord).extra = true;
    }],
  ])("rejects %s", (_name, mutate) => {
    const value = clone(requestFixture.request);
    mutate(value);
    expect(() => parseRepeatedBounceRequest(value)).toThrow();
  });

  it("rejects duplicate keys and oversized UTF-8", () => {
    const text = stableRepeatedBounceRequestJson(requestFixture.request);
    const duplicate = text.replace(
      '"request_id":"surface-run-analytic"',
      '"request_id":"duplicate","request_id":"surface-run-analytic"',
    );
    expect(() => repeatedBounceRequestFromJson(duplicate)).toThrow(/duplicate/i);
    expect(() => repeatedBounceRequestFromJson(
      "é".repeat(MAX_REPEATED_BOUNCE_REQUEST_WIRE_BYTES / 2 + 1),
    )).toThrow(/maximum wire size/i);
  });

  it("rejects finite capture-speed changes without a matching input digest", () => {
    const value = clone(requestFixture.request);
    value.capture_speed_m_s = 0.06;

    expect(() => parseRepeatedBounceRequest(value)).toThrow(
      /execution_input_sha256/,
    );
  });

  it.each([
    ["request identity", (value: MutableRecord) => { value.request_id = "wrong"; }],
    ["surface identity", (value: MutableRecord) => { value.surface_id = "wrong"; }],
    ["model identity", (value: MutableRecord) => { value.model_version = "wrong"; }],
    ["fingerprint", (value: MutableRecord) => {
      value.request_fingerprint_sha256 = "0".repeat(64);
    }],
  ])("rejects result %s mismatch", (_name, mutate) => {
    const result = clone(resultFixture.result);
    mutate(result);
    expect(() => pairRepeatedBounceRequestResult(requestFixture.request, result)).toThrow();
  });
});

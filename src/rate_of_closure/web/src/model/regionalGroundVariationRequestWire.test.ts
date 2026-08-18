import { describe, expect, it } from "vitest";

import goldenRequest from "./__fixtures__/regional_ground_variation_request_golden_v1.json";
import {
  MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES,
  regionalGroundVariationRequestFromJson,
  stableRegionalGroundVariationRequestJson,
} from "./regionalGroundVariationRequestWire";

const goldenText = JSON.stringify(goldenRequest);

describe("regional-ground variation request wire", () => {
  it("round-trips the exact Python-canonical v1 fixture", () => {
    const request = regionalGroundVariationRequestFromJson(goldenText);

    expect(stableRegionalGroundVariationRequestJson(request)).toBe(goldenText);
    expect(request.resultId).toBe("seeded-ground-study");
    expect(request.plan.seed).toBe(1729);
  });

  it.each([
    ["unknown root field", { ...goldenRequest, extra: true }, /fields/i],
    ["boolean max_rows", { ...goldenRequest, max_rows: true }, /max_rows.*integer/i],
    ["unsupported schema", { ...goldenRequest, schema_version: 2 }, /schema_version/i],
    ["coercive plan number", {
      ...goldenRequest,
      variation_plan: { ...goldenRequest.variation_plan, seed: "1729" },
    }, /seed.*integer/i],
    ["plan field outside the combined contract", {
      ...goldenRequest,
      variation_plan: { ...goldenRequest.variation_plan, ball_setup: null },
    }, /variation_plan.*fields/i],
  ])("rejects %s", (_label, payload, message) => {
    expect(() => regionalGroundVariationRequestFromJson(JSON.stringify(payload)))
      .toThrow(message);
  });

  it("rejects duplicate keys before applying any request", () => {
    expect(() => regionalGroundVariationRequestFromJson(
      goldenText.replace('"max_rows":8', '"max_rows":8,"max_rows":9'),
    )).toThrow(/duplicate json field: max_rows/i);
  });

  it("rejects invalid UTF-16 and cross-runtime unsafe numbers", () => {
    expect(() => regionalGroundVariationRequestFromJson(
      goldenText.replace("seeded-ground-study", "seeded-\\ud800-study"),
    )).toThrow(/surrogate/i);
    expect(() => regionalGroundVariationRequestFromJson(
      goldenText.replace('"seed":1729', `"seed":${Number.MAX_SAFE_INTEGER + 1}`),
    )).toThrow(/safe range/i);
  });

  it("bounds direct parser input by UTF-8 byte length", () => {
    const oversized = " ".repeat(MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES + 1);

    expect(() => regionalGroundVariationRequestFromJson(oversized))
      .toThrow(/maximum wire size/i);
  });
});

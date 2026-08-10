import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/flight_to_ground_golden_v1.json";
import {
  canonicalGroundJson,
  parseFlightToGroundRequest,
  parseFlightToGroundResult,
  stableFlightToGroundRequestJson,
  stableFlightToGroundResultJson,
} from "./flightGroundContract";

const sha256 = async (value: string): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
};

describe("flight-to-ground v1 contract", () => {
  it("round-trips the shared Python/TypeScript/Rust/WASM fixture exactly", () => {
    const request = parseFlightToGroundRequest(fixture.request);
    const result = parseFlightToGroundResult(fixture.result);

    expect(stableFlightToGroundRequestJson(request)).toBe(canonicalGroundJson(fixture.request));
    expect(stableFlightToGroundResultJson(result)).toBe(canonicalGroundJson(fixture.result));
    expect(JSON.parse(stableFlightToGroundRequestJson(request))).toEqual(fixture.request);
    expect(JSON.parse(stableFlightToGroundResultJson(result))).toEqual(fixture.result);
  });

  it("implements the fixture's cross-runtime numeric policy", () => {
    expect(canonicalGroundJson(fixture.numeric_policy_cases.values)).toBe(
      fixture.numeric_policy_cases.expected_json,
    );
    expect(() => canonicalGroundJson(
      fixture.numeric_policy_cases.rejected_values.unsafe_integer,
    )).toThrow(/safe range/);
    expect(() => canonicalGroundJson(
      fixture.numeric_policy_cases.rejected_values.unpaired_surrogate,
    )).toThrow(/surrogate/);
  });

  it("matches the fixture's exact cross-runtime request and result digests", async () => {
    const request = parseFlightToGroundRequest(fixture.request);
    const result = parseFlightToGroundResult(fixture.result);

    expect(await sha256(stableFlightToGroundRequestJson(request))).toBe(fixture.request_sha256);
    expect(await sha256(stableFlightToGroundResultJson(result))).toBe(fixture.result_sha256);
  });

  it("fails closed on unknown fields and invalid physical contact", () => {
    expect(() => parseFlightToGroundRequest({ ...fixture.request, extra: true })).toThrow(/fields/);
    expect(() => parseFlightToGroundRequest({
      ...fixture.request,
      first_penetrating_state: {
        ...fixture.request.first_penetrating_state,
        position_m: [210, 0.03, -3],
      },
    })).toThrow(/straddle/);
    expect(() => parseFlightToGroundRequest({
      ...fixture.request,
      surface: { ...fixture.request.surface, frame: "flight" },
    })).toThrow(/frame/);
  });
});

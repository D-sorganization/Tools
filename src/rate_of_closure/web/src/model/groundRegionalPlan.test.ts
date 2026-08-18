import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/ground_regional_plan_golden_v1.json";
import {
  groundRegionalMaterialPlanRequestFromJson,
  parseGroundRegionalMaterialPlanRequest,
  parseGroundRegionalMaterialPlanResult,
  stableGroundRegionalMaterialPlanJson,
} from "./groundRegionalPlan";

const sha256 = async (value: string): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
};

describe("ground regional material plan v1", () => {
  it("round-trips the shared fixture with exact cross-runtime digests", async () => {
    const request = parseGroundRegionalMaterialPlanRequest(fixture.request);
    const result = parseGroundRegionalMaterialPlanResult(fixture.result);

    expect(JSON.parse(stableGroundRegionalMaterialPlanJson(request))).toEqual(fixture.request);
    expect(JSON.parse(stableGroundRegionalMaterialPlanJson(result))).toEqual(fixture.result);
    expect(await sha256(stableGroundRegionalMaterialPlanJson(request))).toBe(
      fixture.request_sha256,
    );
    expect(await sha256(stableGroundRegionalMaterialPlanJson(result))).toBe(
      fixture.result_sha256,
    );
  });

  it("fails closed on extra keys, duplicate identity, and non-coplanar geometry", () => {
    expect(() => parseGroundRegionalMaterialPlanRequest({
      ...fixture.request,
      extra: true,
    })).toThrow(/fields/);
    expect(() => parseGroundRegionalMaterialPlanRequest({
      ...fixture.request,
      regions: [fixture.request.regions[0], fixture.request.regions[0]],
    })).toThrow(/region_id values must be unique/);
    expect(() => parseGroundRegionalMaterialPlanRequest({
      ...fixture.request,
      regions: [{
        ...fixture.request.regions[0],
        surface: {
          ...fixture.request.regions[0].surface,
          normal_unit: [0, 0.8, 0.6],
        },
      }],
    })).toThrow(/coplanar static geometry/);

    expect(() => parseGroundRegionalMaterialPlanRequest({
      ...fixture.request,
      base_surface: {
        ...fixture.request.base_surface,
        surface_velocity_m_s: [1, 0, 0],
      },
    })).toThrow(/static surfaces/);

    expect(() => parseGroundRegionalMaterialPlanRequest({
      ...fixture.request,
      regions: [
        fixture.request.regions[0],
        {
          ...fixture.request.regions[1],
          surface: {
            ...fixture.request.regions[1].surface,
            surface_id: fixture.request.regions[0].surface.surface_id,
          },
        },
      ],
    })).toThrow(/surface_id values must be unique/);
  });

  it("rejects result surfaces that are not bound to their region identities", () => {
    const changed = structuredClone(fixture.result);
    changed.ordered_regions[0].surface.surface_id = "invented";

    expect(() => parseGroundRegionalMaterialPlanResult(changed)).toThrow(/surface identity/);

    const wrongDigest = structuredClone(fixture.result);
    wrongDigest.request_sha256 = "0".repeat(64);
    expect(() => parseGroundRegionalMaterialPlanResult(wrongDigest)).toThrow(/request_sha256/);
  });

  it("rejects duplicate JSON keys and oversized wire documents", () => {
    expect(() => groundRegionalMaterialPlanRequestFromJson(
      '{"schema_version":"ground-regional-material-plan-request/v1",' +
      '"schema_version":"ground-regional-material-plan-request/v1"}',
    )).toThrow(/duplicate/);
    expect(() => groundRegionalMaterialPlanRequestFromJson(
      "{" + " ".repeat(1_048_577) + "}",
    )).toThrow(/maximum wire size/);
  });
});

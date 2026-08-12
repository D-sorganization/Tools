import { describe, expect, it, vi } from "vitest";

import {
  fetchRegionalGroundAuthorityCapability,
  parseRegionalGroundAuthorityCapability,
} from "./regionalGroundAuthority";

const unavailable = {
  schema_version: "rate-of-closure/regional-ground-authority-capability/v1",
  authority_id: "rate-of-closure-python-authority",
  authority_version: "1",
  available: false,
  regional_ground_execution: false,
  reason_code: "runner_not_started",
  detail: "Qualified execution runner is not started.",
};

describe("regional-ground authority capability", () => {
  it("parses and freezes the exact Python-owned unavailable state", () => {
    const parsed = parseRegionalGroundAuthorityCapability(unavailable);

    expect(parsed).toEqual(unavailable);
    expect(Object.isFrozen(parsed)).toBe(true);
  });

  it("rejects extras, inconsistent availability, and executable claims", () => {
    expect(() => parseRegionalGroundAuthorityCapability({
      ...unavailable,
      unexpected: true,
    })).toThrow(/fields/i);
    expect(() => parseRegionalGroundAuthorityCapability({
      ...unavailable,
      available: true,
    })).toThrow(/availability/i);
    expect(() => parseRegionalGroundAuthorityCapability({
      ...unavailable,
      regional_ground_execution: true,
    })).toThrow(/not qualified/i);
  });

  it("returns a typed unavailable state when the local authority cannot be reached", async () => {
    const fetcher = vi.fn().mockRejectedValue(new TypeError("network failed"));

    const result = await fetchRegionalGroundAuthorityCapability(fetcher);

    expect(result.available).toBe(false);
    expect(result.regional_ground_execution).toBe(false);
    expect(result.reason_code).toBe("authority_unreachable");
    expect(result.detail).not.toContain("network failed");
  });

  it("distinguishes an invalid authority response from a network failure", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }));

    const result = await fetchRegionalGroundAuthorityCapability(fetcher);

    expect(result.reason_code).toBe("authority_invalid_response");
  });
});

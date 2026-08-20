import { describe, expect, it } from "vitest";
import { validateLaunchMonitorV2Response } from "./launchMonitorV2Client";

const response = { contract_version: "2.0.0", status: "available", analysis: {}, units: {},
  lineage: { dataset_fingerprint_sha256: "a".repeat(64), backing_records: [] },
  missingness: {}, availability: [], uncertainty: {}, player_identity: {}, vendor_provenance: [],
  claims: { vendor_comparison: "descriptive", device_emulation: false, device_certification: false, causal_inference: false }, warnings: [] };

describe("Upstream v2 client", () => {
  it("validates the canonical envelope and reports residuals unavailable", () => {
    expect(validateLaunchMonitorV2Response(response).rowAlignedResiduals).toMatchObject({ state: "unavailable" });
  });
  it("rejects unsafe claims", () => {
    expect(() => validateLaunchMonitorV2Response({ ...response, claims: { ...response.claims, device_emulation: true } }))
      .toThrow(/emulation/i);
  });
});

import { describe, expect, it } from "vitest";
import { validateLaunchMonitorStrokesGainedResponse, validateLaunchMonitorV2Response } from "./launchMonitorV2Client";

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

describe("Upstream source-backed strokes-gained client", () => {
  it("accepts only the canonical scoring contract and safe claims", () => {
    const scoring = {
      contract_version: "launch-monitor-strokes-gained-analysis/1.0.0", status: "available",
      metric_name: "source_backed_strokes_gained", unit: "strokes",
      value_summary: { count: 3, mean: 0.25 }, baseline: { baseline_id: "test" },
      formula: "SG", units: {}, availability: {}, uncertainty: {}, row_results: [],
      excluded_rows: [], exclusions: {}, group_summaries: [], longitudinal_summaries: [],
      analysis_context: {}, dataset_fingerprint_sha256: "b".repeat(64), warnings: [], limitations: [],
      claims: { is_strokes_gained: true, source_backed: true, device_emulation: false, device_certification: false, causal_inference: false },
    };
    expect(validateLaunchMonitorStrokesGainedResponse(scoring).mean).toBe(0.25);
    expect(() => validateLaunchMonitorStrokesGainedResponse({ ...scoring, claims: { ...scoring.claims, source_backed: false } })).toThrow(/source-backed/i);
  });
});
